import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from agents.SAC import SAC
from agents.BaseAgent import TargetState
from components.networks import NAFQuadraticCritic


class NAF(SAC):
  """
  Implementation of Normalized Advantage Function.
  """
  def createNN(self):
    # Create train_states and nets of actor, critic, and temperature
    dummy_obs = self.env['Train'].observation_space.sample()[None,]
    dummy_action = self.env['Train'].action_space.sample()[None,]
    self.seed, critic_seed = jax.random.split(self.seed, 2)
    self.critic_net = nn.vmap(
      NAFQuadraticCritic,
      in_axes = None, out_axes = 0,
      variable_axes = {'params': 0},  # Parameters are not shared between critics
      split_rngs = {'params': True},  # Different initializations
      axis_size = self.cfg['agent']['critic_num'],  # Number of critics
      methods = ['__call__', 'get_mu', 'get_v']
    )(
      V_net_cfg = self.cfg['agent']['V_net_cfg'],
      mu_net_cfg = self.cfg['agent']['mu_net_cfg'],
      L_net_cfg = self.cfg['agent']['L_net_cfg'],
      action_size = self.action_size
    )
    self.actor_state = None
    self.critic_state = TargetState.create(
      apply_fn = jax.jit(self.critic_net.apply),
      params = self.critic_net.init(critic_seed, dummy_obs, dummy_action),
      target_params = self.critic_net.init(critic_seed, dummy_obs, dummy_action),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'])
    )
    self.temp_state = None

  @partial(jax.jit, static_argnames=['self'])
  def random_action(self, actor_state, critic_state, obs, seed):
    seed, action_seed, noise_seed = jax.random.split(seed, 3)
    action = critic_state.apply_fn(critic_state.params, obs, method='get_mu') # Shape: (critic, batch, action)
    action = jax.random.choice(key=action_seed, a=action, axis=0) # Shape: (batch, action)
    action += self.cfg['agent']['action_noise'] * jax.random.normal(noise_seed, shape=action.shape, dtype=action.dtype)
    return action, seed
  
  @partial(jax.jit, static_argnames=['self'])
  def optimal_action(self, actor_state, critic_state, obs, seed):
    seed, action_seed = jax.random.split(seed, 2)
    action = critic_state.apply_fn(critic_state.params, obs, method='get_mu')
    action = jax.random.choice(key=action_seed, a=action, axis=0)
    return action, seed

  @partial(jax.jit, static_argnames=['self'])
  def update_critic(self, actor_state, critic_state, temp_state, batch, seed):
    v_next = critic_state.apply_fn(critic_state.target_params, batch['next_obs'], method='get_v') # Shape: (critic, batch, 1)
    v_next = jnp.min(v_next, axis=0) # Shape: (batch, 1)
    q_target = batch['reward'].reshape(-1, 1) + batch['mask'].reshape(-1, 1) * v_next # Shape: (batch, 1)
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