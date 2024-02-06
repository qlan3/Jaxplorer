import jax
import jax.numpy as jnp
from functools import partial

from agents.DQN import DQN


class DDQN(DQN):
  """
  Implementation of Double DQN.
  """
  @partial(jax.jit, static_argnames=['self'])
  def update_critic(self, critic_state, batch, seed):
    best_action = self.optimal_action(critic_state, batch['next_obs'])
    # Compute target q values
    q_next = critic_state.apply_fn(critic_state.target_params, batch['next_obs']) # Shape: (batch, action)
    q_next = q_next[jnp.arange(q_next.shape[0]), best_action] # Shape: (batch,)
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