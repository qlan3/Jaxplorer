import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from agents.SAC import SAC
from agents.BaseAgent import TargetState
from components.networks import MLPDeterministicActor, MLPQCritic


class DDPG(SAC):
  """
  Implementation of DDPG (Deep Deterministic Policy Gradient)
  """
  def createNN(self):
    # Create train_states and nets of actor, critic, and temperature
    dummy_obs = self.env['Train'].observation_space.sample()[None,]
    dummy_action = self.env['Train'].action_space.sample()[None,]
    self.seed, actor_seed, critic_seed = jax.random.split(self.seed, 3)
    self.actor_net = MLPDeterministicActor(
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
    self.actor_state = TargetState.create(
      apply_fn = jax.jit(self.actor_net.apply),
      params = self.actor_net.init(actor_seed, dummy_obs),
      target_params = self.actor_net.init(actor_seed, dummy_obs),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'])
    )
    self.critic_state = TargetState.create(
      apply_fn = jax.jit(self.critic_net.apply),
      params = self.critic_net.init(critic_seed, dummy_obs, dummy_action),
      target_params = self.critic_net.init(critic_seed, dummy_obs, dummy_action),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'])
    )
    self.temp_state = None

  @partial(jax.jit, static_argnames=['self'])
  def random_action(self, actor_state, critic_state, obs, seed):
    seed, noise_seed = jax.random.split(seed, 2)
    mu = actor_state.apply_fn(actor_state.params, obs)
    action = mu + self.cfg['agent']['action_noise'] * jax.random.normal(noise_seed, shape=mu.shape, dtype=mu.dtype)
    return action, seed

  @partial(jax.jit, static_argnames=['self'])
  def optimal_action(self, actor_state, critic_state, obs, seed):
    action = actor_state.apply_fn(actor_state.params, obs)
    return action, seed

  @partial(jax.jit, static_argnames=['self'])
  def update_critic(self, actor_state, critic_state, temp_state, batch, seed):
    next_action = actor_state.apply_fn(actor_state.target_params, batch['next_obs'])
    q_next = critic_state.apply_fn(critic_state.target_params, batch['next_obs'], next_action) # Shape: (critic, batch, 1)
    q_next = jnp.min(q_next, axis=0) # Shape: (batch, 1)
    q_target = batch['reward'].reshape(-1, 1) + batch['mask'].reshape(-1, 1) * q_next # Shape: (batch, 1)
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
    # Compute actor loss
    def actor_loss(params):
      action = actor_state.apply_fn(params, batch['obs'])
      qs = critic_state.apply_fn(critic_state.params, batch['obs'], action) # Shape: (critic, batch, 1)
      q_min = jnp.min(qs, axis=0) # Shape: (batch, 1)
      loss = -q_min.mean()
      return loss
    grads = jax.grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    # Soft-update target network
    actor_state = actor_state.replace(
      target_params = optax.incremental_update(actor_state.params, actor_state.target_params, self.cfg['tau'])
    )
    return actor_state, None