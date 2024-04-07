import jax
import time
import optax
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from flax.core import frozen_dict

from agents.BaseAgent import BaseAgent, TrainState, TargetState
from components.replay import FiniteReplay
from components.networks import MLPGaussianTanhActor, MLPQCritic, Temperature


class SAC(BaseAgent):
  """
  Implementation of Soft Actor-Critic.
  """
  def __init__(self, cfg):
    cfg['agent'].setdefault('actor_update_steps', 1)
    cfg['agent'].setdefault('critic_update_steps', 1)
    super().__init__(cfg)
    assert self.action_type == 'CONTINUOUS', f'{self.agent_name} only supports continous action spaces.'
    self.cfg['exploration_steps'] = max(int(self.cfg['exploration_steps']), self.cfg['batch_size'])
    # Set replay buffer
    self.replay = FiniteReplay(self.cfg['buffer_size'], keys=['obs', 'action', 'reward', 'mask', 'next_obs'])
    # Set networks
    self.createNN()
  
  def createNN(self):
    # Create train_states and nets of actor, critic, and temperature
    dummy_obs = self.env['Train'].observation_space.sample()[None,]
    dummy_action = self.env['Train'].action_space.sample()[None,]
    self.seed, actor_seed, critic_seed, temp_seed = jax.random.split(self.seed, 4)
    self.actor_net = MLPGaussianTanhActor(
      action_size = self.action_size,
      net_cfg = self.cfg['agent']['actor_net_cfg']
    )
    self.critic_net = nn.vmap(
      MLPQCritic,
      in_axes = None, out_axes = 0,
      variable_axes = {'params': 0},  # Parameters are not shared between critics
      split_rngs = {'params': True},  # Different initializations
      axis_size = self.cfg['agent']['critic_num'],  # Number of critics
    )(
      net_cfg = self.cfg['agent']['critic_net_cfg']
    )
    self.temp_net = Temperature(init_temp=1.0)
    self.actor_state = TrainState.create(
      apply_fn = self.actor_net.apply,
      params = self.actor_net.init(actor_seed, dummy_obs),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'])
    )
    self.critic_state = TargetState.create(
      apply_fn = self.critic_net.apply,
      params = self.critic_net.init(critic_seed, dummy_obs, dummy_action),
      target_params = self.critic_net.init(critic_seed, dummy_obs, dummy_action),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'])
    )
    self.target_entropy = -0.5 * self.action_size
    self.temp_state = TrainState.create(
      apply_fn = self.temp_net.apply,
      params = self.temp_net.init(temp_seed),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'])
    )

  def run_steps(self, mode='Train'):
    self.start_time = time.time()
    obs, info = self.env[mode].reset()
    for step in tqdm(range(self.train_steps), disable=not self.cfg['show_progress']):
      # Get an action
      action = self.get_action(step, obs[None,], mode)['action']
      # Take a env step
      next_obs, reward, terminated, truncated, info = self.env[mode].step(action)
      # Save experience
      mask = self.discount * (1-terminated)
      self.save_experience(obs, action, reward, mask, next_obs)
      # Update observation
      obs = next_obs
      # Record and reset
      if terminated or truncated:
        result_dict = {
          'Env': self.env_name,
          'Agent': self.agent_name,
          'Step': step,
          'Return': info['episode']['r'][0]
        }
        self.result[mode].append(result_dict)
        obs, info = self.env[mode].reset()
      # Update the agent
      if step > self.cfg['exploration_steps']:
        if step % self.cfg['agent']['critic_update_steps'] == 0 or step % self.cfg['agent']['actor_update_steps'] == 0:
          batch = self.replay.sample(self.cfg['batch_size'])
          # batch = frozen_dict.freeze(batch)
          # batch = jax.device_put(batch)
          self.seed, critic_seed, action_seed = jax.random.split(self.seed, 3)
          # Update critic
          if step % self.cfg['agent']['critic_update_steps'] == 0:
            self.critic_state = self.update_critic(self.actor_state, self.critic_state, self.temp_state, batch, critic_seed)
          if self.agent_name not in ['NAF']:
            if step % self.cfg['agent']['actor_update_steps'] == 0:
              # Update actor
              self.actor_state, entropy = self.update_actor(self.actor_state, self.critic_state, self.temp_state, batch, action_seed)
              # Update temperature
              if self.agent_name in ['SAC']:
                self.temp_state = self.update_temperature(self.temp_state, entropy)
      # Display log, test, and save checkpoint
      self.log_test_save(step, self.train_steps, mode)
  
  def get_action(self, step, obs, mode='Train'):
    if mode == 'Train':
      if step <= self.cfg['exploration_steps']:
        action = self.env[mode].action_space.sample()
      else:
        action, self.seed = self.random_action(self.actor_state, self.critic_state, obs, self.seed)
        action = jax.device_get(action)[0]
    else: # mode == 'Test'
      action, self.seed = self.optimal_action(self.actor_state, self.critic_state, obs, self.seed)
      action = jax.device_get(action)[0]
    return dict(action=action)

  @partial(jax.jit, static_argnames=['self'])
  def random_action(self, actor_state, critic_state, obs, seed):
    seed, action_seed = jax.random.split(seed, 2)
    u_mean, u_log_std = actor_state.apply_fn(actor_state.params, obs)
    eps = jax.random.normal(action_seed, shape=u_mean.shape)
    u = u_mean + jnp.exp(u_log_std) * eps
    action = jnp.tanh(u)
    return action, seed

  @partial(jax.jit, static_argnames=['self'])
  def optimal_action(self, actor_state, critic_state, obs, seed):
    u_mean, _ = actor_state.apply_fn(actor_state.params, obs)
    action = jnp.tanh(u_mean)
    return action, seed

  @partial(jax.jit, static_argnames=['self'])
  def update_critic(self, actor_state, critic_state, temp_state, batch, seed):
    next_action, next_logp = self.sample_action_with_logp(actor_state, actor_state.params, batch['next_obs'], seed)
    q_next = critic_state.apply_fn(critic_state.target_params, batch['next_obs'], next_action) # Shape: (critic, batch, 1)
    q_next = jnp.min(q_next, axis=0) # Shape: (batch, 1)
    alpha = temp_state.apply_fn(temp_state.params)
    q_target = batch['reward'].reshape(-1, 1) + batch['mask'].reshape(-1, 1) * (q_next - alpha * next_logp.reshape(-1, 1)) # Shape: (batch, 1)
    # Compute critic loss
    def critic_loss(params):
      qs = critic_state.apply_fn(params, batch['obs'], batch['action']) # Shape: (critic, batch, 1)
      loss = ((qs - q_target)**2).mean(axis=1).sum()
      return loss
    grads = jax.grad(critic_loss)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=grads)
    # Soft-update target network
    critic_state = critic_state.replace(
      target_params = optax.incremental_update(critic_state.params, critic_state.target_params, self.cfg['tau'])
    )
    return critic_state

  @partial(jax.jit, static_argnames=['self'])
  def update_actor(self, actor_state, critic_state, temp_state, batch, seed):
    alpha = temp_state.apply_fn(temp_state.params)
    # Compute actor loss
    def actor_loss(params):
      action, logp = self.sample_action_with_logp(actor_state, params, batch['obs'], seed)
      entropy = -logp.mean()
      qs = critic_state.apply_fn(critic_state.params, batch['obs'], action)
      q_min = jnp.min(qs, axis=0) # Shape: (batch, 1)
      loss = (alpha * logp - q_min).mean()
      return loss, entropy
    grads, entropy = jax.grad(actor_loss, has_aux=True)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    return actor_state, entropy

  @partial(jax.jit, static_argnames=['self'])
  def update_temperature(self, temp_state, entropy):
    def temperature_loss(params):
      alpha = temp_state.apply_fn(params)
      loss = alpha * (entropy - self.target_entropy).mean()
      return loss
    grads = jax.grad(temperature_loss)(temp_state.params)
    temp_state = temp_state.apply_gradients(grads=grads)
    return temp_state

  @partial(jax.jit, static_argnames=['self'])
  def sample_action_with_logp(self, actor_state, params, obs, seed):
    u_mean, u_log_std = actor_state.apply_fn(params, obs)
    eps = jax.random.normal(seed, shape=u_mean.shape)
    u = u_mean + jnp.exp(u_log_std) * eps
    action = jnp.tanh(u)
    # Get log_prob(action): https://github.com/openai/spinningup/issues/279
    logp = (-0.5*(eps**2) - 0.5*jnp.log(2.0*jnp.pi) - u_log_std).sum(axis=-1)
    logp -= (2*(jnp.log(2) - u - nn.softplus(-2*u))).sum(axis=-1)
    return action, logp