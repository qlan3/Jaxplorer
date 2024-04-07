import jax
import optax
import jax.numpy as jnp
from functools import partial

from agents.DDPG import DDPG


class TD3(DDPG):
  """
  Implementation of Twin Delayed Deep Deterministic Policy Gradients.
  """
  @partial(jax.jit, static_argnames=['self'])
  def update_critic(self, actor_state, critic_state, temp_state, batch, seed):
    seed, noise_seed = jax.random.split(seed, 2)
    next_action = actor_state.apply_fn(actor_state.target_params, batch['next_obs'])
    noise = self.cfg['agent']['target_noise'] * jax.random.normal(noise_seed, shape=next_action.shape, dtype=next_action.dtype)
    noise = jnp.clip(noise, -self.cfg['agent']['noise_clip'], self.cfg['agent']['noise_clip'])
    next_action = jnp.clip(next_action+noise, -1.0, 1.0)
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
      q1 = qs[0] # Shape: (batch, 1)
      loss = -q1.mean()
      return loss
    grads = jax.grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    # Soft-update target network
    actor_state = actor_state.replace(
      target_params = optax.incremental_update(actor_state.params, actor_state.target_params, self.cfg['tau'])
    )
    return actor_state, None