import jax
import time
import random
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial

from agents.BaseAgent import BaseAgent, TargetState
from components.replay import FiniteReplay
from components.networks import MLPQNet, MinAtarQNet


class DQN(BaseAgent):
  """
  Implementation of Deep Q-Learning.
  """
  def __init__(self, cfg):
    super().__init__(cfg)
    self.cfg['agent'].setdefault('update_per_step', 1)
    if self.cfg['agent']['update_per_step'] < 1:
      self.update_epochs = int(1/self.cfg['agent']['update_per_step'])
      self.cfg['agent']['update_per_step'] = 1
    else:
      self.update_epochs = 1
      self.cfg['agent']['update_per_step'] = int(self.cfg['agent']['update_per_step'])
    assert self.action_type == 'DISCRETE', f'{self.agent_name} only supports discrete action spaces.'
    self.cfg['exploration_steps'] = max(int(self.cfg['exploration_steps']), self.cfg['batch_size'])
    # Set replay buffer
    self.replay = FiniteReplay(self.cfg['buffer_size'], keys=['obs', 'action', 'reward', 'mask', 'next_obs'])
    # Set networks
    self.createNN()

  def createNN(self):
    # Create train_state and net for critic
    dummy_obs = self.env['Train'].observation_space.sample()[None,]
    self.seed, critic_seed = jax.random.split(self.seed)
    if 'MinAtar' in self.env_name:
      QNet = MinAtarQNet
    else:
      QNet = MLPQNet
    self.critic_net = QNet(
      action_size = self.action_size,
      net_cfg = self.cfg['agent']['net_cfg']
    )
    self.critic_state = TargetState.create(
      apply_fn = self.critic_net.apply,
      params = self.critic_net.init(critic_seed, dummy_obs),
      target_params = self.critic_net.init(critic_seed, dummy_obs),
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
        # Update critic
        if step % self.cfg['agent']['update_per_step'] == 0:
          batch = self.replay.sample(self.cfg['batch_size'])
          # batch = jax.device_put(batch)
          for _ in range(self.update_epochs):
            self.critic_state, self.seed = self.update_critic(self.critic_state, batch, self.seed)
        # Update target network
        if step % self.cfg['agent']['target_update_freq'] == 0: # optax.periodic_update is much slower ...
          self.critic_state = self.critic_state.replace(
            target_params = jax.tree_map(lambda x: x, self.critic_state.params)
          )
      # Display log, test, and save checkpoint
      self.log_test_save(step, self.train_steps, mode)

  def get_action(self, step, obs, mode='Train'):
    eps = self.linear_schedule(step)
    if mode == 'Train' and random.random() < eps: # jax.random.uniform is really slow...
      action = self.env[mode].action_space.sample()
    else:
      action = self.optimal_action(self.critic_state, obs)
      action = jax.device_get(action)[0] # jax.device_get(action[0]) is much slower
    return dict(action=action)
  
  def linear_schedule(self, step):
    slope = (self.cfg['agent']['eps_end'] - self.cfg['agent']['eps_start']) / self.cfg['agent']['eps_steps']
    eps = max(slope * step + self.cfg['agent']['eps_start'], self.cfg['agent']['eps_end'])
    return eps
  
  @partial(jax.jit, static_argnames=['self'])
  def optimal_action(self, critic_state, obs):
    q_values = critic_state.apply_fn(critic_state.params, obs) # Shape: (batch, action)
    action = q_values.argmax(axis=-1) # Shape: (batch,)
    return action

  @partial(jax.jit, static_argnames=['self'])
  def update_critic(self, critic_state, batch, seed):
    # Compute target q values
    q_next = critic_state.apply_fn(critic_state.target_params, batch['next_obs']) # Shape: (batch, action)
    q_next = jnp.max(q_next, axis=-1) # Shape: (batch,)
    q_target = batch['reward'] + batch['mask'] * q_next # Shape: (batch,)
    # Compute critic loss
    def critic_loss(params):
      q = critic_state.apply_fn(params, batch['obs']) # Shape: (batch, action)
      q = q[jnp.arange(q.shape[0]), batch['action']] # Shape: (batch,)
      loss = ((q - q_target)**2).mean()
      return loss
    grads = jax.grad(critic_loss)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=grads)
    return critic_state, seed