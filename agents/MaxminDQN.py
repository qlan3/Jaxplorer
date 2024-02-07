import jax
import time
import random
from tqdm import tqdm
import flax.linen as nn
import jax.numpy as jnp
from functools import partial

from agents.DQN import DQN
from agents.BaseAgent import TargetState
from components.networks import MLPQNet, MinAtarQNet


class MaxminDQN(DQN):
  """
  Implementation of Maxmin DQN. 
    Note that here we update all Q-nets while the version in [Explorer](https://github.com/qlan3/Explorer/) only updates one of Q-nets.
  """
  def createNN(self):
    # Create train_state and net for critic
    dummy_obs = self.env['Train'].observation_space.sample()[None,]
    self.seed, critic_seed = jax.random.split(self.seed)
    if 'MinAtar' in self.env_name:
      QNet = MinAtarQNet
    else:
      QNet = MLPQNet
    self.critic_net = nn.vmap(
      QNet,
      in_axes = None, out_axes = 0,
      variable_axes = {'params': 0},  # Parameters are not shared between critics
      split_rngs = {'params': True},  # Different initializations
      axis_size = self.cfg['agent']['critic_num'],  # Number of critics
    )(
      action_size = self.action_size,
      net_cfg = self.cfg['agent']['net_cfg']
    )
    self.critic_state = TargetState.create(
      apply_fn = self.critic_net.apply,
      params = self.critic_net.init(critic_seed, dummy_obs),
      target_params = self.critic_net.init(critic_seed, dummy_obs),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'])
    )

  @partial(jax.jit, static_argnames=['self'])
  def optimal_action(self, critic_state, obs):
    q_values = critic_state.apply_fn(critic_state.params, obs) # Shape: (critic, batch, action)
    action = q_values.min(axis=0).argmax(axis=-1) # Shape: (batch,)
    return action

  @partial(jax.jit, static_argnames=['self'])
  def update_critic(self, critic_state, batch, seed):
    # Compute target q values
    q_next = critic_state.apply_fn(critic_state.target_params, batch['next_obs']) # Shape: (critic, batch, action)
    q_next = q_next.min(axis=0)  # Shape: (batch, action)
    q_next = q_next.max(axis=-1) # Shape: (batch,)
    q_target = batch['reward'] + batch['mask'] * q_next # Shape: (batch,)
    # Compute critic loss
    def critic_loss(params):
      q = critic_state.apply_fn(params, batch['obs']) # Shape: (critic, batch, action)
      q = q[:, jnp.arange(q.shape[1]), batch['action']] # Shape: (critic, batch,)
      loss = ((q - q_target)**2).mean(axis=1).sum()
      return loss
    grads = jax.grad(critic_loss)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=grads)
    return critic_state, seed