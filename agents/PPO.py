import jax
import time
import math
import optax
import distrax
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

import gymnasium as gym
from gymnasium import wrappers
from envs.wrappers import UniversalSeed

def ppo_make_env(env_name, gamma=0.99, deque_size=1, **kwargs):
  """ Make env for PPO. """
  env = gym.make(env_name, **kwargs)
  # Episode statistics wrapper: set it before reward wrappers
  env = wrappers.RecordEpisodeStatistics(env, deque_size=deque_size)
  # Action wrapper
  env = wrappers.ClipAction(wrappers.RescaleAction(env, min_action=-1, max_action=1))
  # Obs wrapper
  env = wrappers.FlattenObservation(env) # For dm_control
  env = wrappers.NormalizeObservation(env)
  env = wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
  # Reward wrapper
  env = wrappers.NormalizeReward(env, gamma=gamma)
  env = wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
  # Seed wrapper: must be the last wrapper to be effective
  env = UniversalSeed(env)
  return env


from agents.BaseAgent import BaseAgent, TrainState
from components.replay import FiniteReplay
from components.networks import MLPVCritic, MLPGaussianActor, MLPCategoricalActor


class PPO(BaseAgent):
  """
  Implementation of PPO.
  """
  def __init__(self, cfg):
    super().__init__(cfg)
    self.train_iter = math.ceil(self.train_steps // self.cfg['agent']['collect_steps'])
    self.num_batches = self.cfg['agent']['collect_steps'] // self.cfg['batch_size']
    # Remake environment for PPO
    del self.env
    self.env = {
      'Train': ppo_make_env(self.env_name, gamma=self.discount, **self.cfg['env']['kwargs']),
      'Test': ppo_make_env(self.env_name, gamma=self.discount, deque_size=cfg['test_episodes'], **self.cfg['env']['kwargs'])
    }
    self.env['Train'].set_seed(self.cfg['seed'])
    self.env['Test'].set_seed(self.cfg['seed']+42)
    # Set replay buffer
    self.replay = FiniteReplay(self.cfg['agent']['collect_steps'], keys=['obs', 'action', 'reward', 'mask', 'v', 'log_pi'])
    # Set networks
    self.createNN()

  def set_optim(self, optim_name, optim_kwargs, schedule=None):
    optim_kwargs.setdefault('anneal_lr', False)
    optim_kwargs.setdefault('grad_clip', -1)
    optim_kwargs.setdefault('max_grad_norm', -1)
    anneal_lr = optim_kwargs['anneal_lr']
    grad_clip = optim_kwargs['grad_clip']
    max_grad_norm = optim_kwargs['max_grad_norm']
    del optim_kwargs['anneal_lr'], optim_kwargs['grad_clip'], optim_kwargs['max_grad_norm']
    assert not (grad_clip > 0 and max_grad_norm > 0), 'Cannot apply both grad_clip and max_grad_norm at the same time.'
    if anneal_lr and schedule is not None:
      optim_kwargs['learning_rate'] = schedule
    if grad_clip > 0:
      optim = optax.chain(
        optax.clip(grad_clip),
        getattr(optax, optim_name.lower())(**optim_kwargs)
      )
    elif max_grad_norm > 0:
      optim = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        getattr(optax, optim_name.lower())(**optim_kwargs)
      )
    else:
      optim = getattr(optax, optim_name.lower())(**optim_kwargs)
    return optim

  def createNN(self):
    # Create nets and train_states
    lr = self.cfg['optim']['kwargs']['learning_rate']
    def linear_schedule(count):
      iter = count // (self.cfg['agent']['update_epochs'] * self.num_batches)
      frac = 1.0 - iter / self.train_iter
      return frac * lr
    dummy_obs = self.env['Train'].observation_space.sample()[None,]
    self.seed, actor_seed, critic_seed = jax.random.split(self.seed, 3)
    # Set actor network
    if self.action_type == 'DISCRETE':
      actor_net = MLPCategoricalActor
    elif self.action_type == 'CONTINUOUS':
      actor_net = MLPGaussianActor
    self.actor_net = actor_net(
      action_size = self.action_size,
      net_cfg = self.cfg['agent']['actor_net_cfg'],
      last_w_scale = 0.01
    )
    self.critic_net = MLPVCritic(
      net_cfg = self.cfg['agent']['critic_net_cfg'],
      last_w_scale = 1.0
    )
    self.actor_state = TrainState.create(
      apply_fn = self.actor_net.apply,
      params = self.actor_net.init(actor_seed, dummy_obs),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'], schedule=linear_schedule)
    )
    self.critic_state = TrainState.create(
      apply_fn = self.critic_net.apply,
      params = self.critic_net.init(critic_seed, dummy_obs),
      tx = self.set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'], schedule=linear_schedule)
    )

  def run_steps(self, mode='Train'):
    self.start_time = time.time()
    obs, info = self.env[mode].reset()
    for i in tqdm(range(self.train_iter), disable=not self.cfg['show_progress']):
      for step in range(i*self.cfg['agent']['collect_steps'], (i+1)*self.cfg['agent']['collect_steps']):
        # Get an action
        pred = self.get_action(step, obs[None,], mode)
        action, v, log_pi = pred['action'], pred['v'], pred['log_pi']
        # Take a env step
        next_obs, reward, terminated, truncated, info = self.env[mode].step(action)
        # Save experience
        # done = terminated or truncated
        # mask = self.discount * (1-done)
        mask = self.discount * (1-terminated)
        self.save_experience(obs, action, reward, mask, v, log_pi)
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
      # Get all the data and convet to PyTree
      trajectory = self.replay.get(self.cfg['agent']['collect_steps'])
      # Update the agent
      self.actor_state, self.critic_state, self.seed = self.update(self.actor_state, self.critic_state, trajectory, obs[None,], self.seed)
      # Reset storage
      self.replay.clear()
      # Display log, test, and save checkpoint
      self.log_test_save(i, self.train_iter, mode)

  def save_experience(self, obs, action, reward, mask, v, log_pi):
    prediction = {
      'obs': obs,
      'action': action,
      'reward': reward,
      'mask': mask,
      'v': v,
      'log_pi': log_pi
    }
    self.replay.add(prediction)

  def get_action(self, step, obs, mode='Train'):
    if mode == 'Train':
      action, v, log_pi, self.seed = self.random_action(self.actor_state, self.critic_state, obs, self.seed)
    else: # mode == 'Test'
      action, v, log_pi, self.seed = self.optimal_action(self.actor_state, self.critic_state, obs, self.seed)
    action = jax.device_get(action)[0]
    return dict(action=action, v=v, log_pi=log_pi)

  @partial(jax.jit, static_argnames=['self'])
  def random_action(self, actor_state, critic_state, obs, seed):
    seed, action_seed = jax.random.split(seed, 2)
    action_mean, action_log_std = actor_state.apply_fn(actor_state.params, obs)
    pi = distrax.MultivariateNormalDiag(action_mean, jnp.exp(action_log_std))
    v = critic_state.apply_fn(critic_state.params, obs)
    action = pi.sample(seed=action_seed)
    log_pi = pi.log_prob(action)
    return action, v, log_pi, seed

  @partial(jax.jit, static_argnames=['self'])
  def optimal_action(self, actor_state, critic_state, obs, seed):
    action_mean, action_log_std = actor_state.apply_fn(actor_state.params, obs)
    pi = distrax.MultivariateNormalDiag(action_mean, jnp.exp(action_log_std))
    v = critic_state.apply_fn(critic_state.params, obs)
    log_pi = pi.log_prob(action_mean)
    return action_mean, v, log_pi, seed

  @partial(jax.jit, static_argnames=['self'])
  def update(self, actor_state, critic_state, trajectory, last_obs, seed):
    # Compute advantage and v_target, store them in trajectory
    trajectory = self.compute_advantage(critic_state, trajectory, last_obs)
    # Update the agent for mutliple epochs
    carry = (actor_state, critic_state, trajectory, seed)
    carry, _ = jax.lax.scan(
      f = self.update_epoch,
      init = carry,
      xs = None,
      length = self.cfg['agent']['update_epochs']
    )
    actor_state, critic_state, trajectory, seed = carry
    return actor_state, critic_state, seed

  @partial(jax.jit, static_argnames=['self'])
  def compute_advantage(self, critic_state, trajectory, last_obs):
    last_v = critic_state.apply_fn(critic_state.params, last_obs).squeeze()
    # Compute GAE advantage
    def _calculate_gae(gae_and_next_value, transition):
      gae, next_value = gae_and_next_value
      delta = transition['reward'] + transition['mask'] * next_value - transition['v']
      gae = delta + self.cfg['agent']['gae_lambda'] * transition['mask'] * gae
      next_value = transition['v']
      return (gae, next_value), gae

    _, adv = jax.lax.scan(
      f = _calculate_gae,
      init = (0.0, last_v),
      xs = trajectory,
      length = self.cfg['agent']['collect_steps'],
      reverse = True,
    )
    trajectory['adv'] = adv
    trajectory['v_target'] = adv + trajectory['v']
    return trajectory
  
  @partial(jax.jit, static_argnames=['self'])
  def update_epoch(self, carry, _):
    actor_state, critic_state, trajectory, seed = carry
    seed, update_seed = jax.random.split(seed)
    # Shuffle trajectory
    shuffled_trajectory = jax.tree_util.tree_map(
      lambda x: jax.random.permutation(update_seed, x, axis=0),
      trajectory
    )
    # Split trajectory into batches
    batches = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, (-1, self.cfg['batch_size']) + x.shape[1:]),
      shuffled_trajectory,
    )
    carry = (actor_state, critic_state)
    carry, _ = jax.lax.scan(
      f = self.update_batch,
      init = carry,
      xs = batches
    )
    actor_state, critic_state = carry
    carry = (actor_state, critic_state, trajectory, seed)
    return carry, None
  
  @partial(jax.jit, static_argnames=['self'])
  def update_batch(self, carry, batch):
    actor_state, critic_state = carry
    adv = (batch['adv'] - batch['adv'].mean()) / (batch['adv'].std() + 1e-8)
    def compute_loss(params):
      actor_param, critic_param = params
      # Compute critic loss
      v = critic_state.apply_fn(critic_param, batch['obs'])
      v_clipped = batch['v'] + (v - batch['v']).clip(-self.cfg['agent']['clip_ratio'], self.cfg['agent']['clip_ratio'])
      critic_loss_unclipped = jnp.square(v - batch['v_target'])
      critic_loss_clipped = jnp.square(v_clipped - batch['v_target'])
      critic_loss = 0.5 * jnp.maximum(critic_loss_unclipped, critic_loss_clipped).mean()      
      # Compute actor loss
      action_mean, action_log_std = actor_state.apply_fn(actor_param, batch['obs'])
      pi = distrax.MultivariateNormalDiag(action_mean, jnp.exp(action_log_std))
      log_pi = pi.log_prob(batch['action'])
      ratio = jnp.exp(log_pi - batch['log_pi'])
      obj = ratio * adv
      obj_clipped = jnp.clip(ratio, 1.0-self.cfg['agent']['clip_ratio'], 1.0+self.cfg['agent']['clip_ratio']) * adv
      actor_loss = -jnp.minimum(obj, obj_clipped).mean()
      # Compute entropy
      entropy = pi.entropy().mean()
      total_loss = actor_loss + self.cfg['agent']['vf_coef'] * critic_loss - self.cfg['agent']['ent_coef'] * entropy
      return total_loss

    grads = jax.grad(compute_loss)((actor_state.params, critic_state.params))
    actor_grads, critic_grads = grads
    actor_state = actor_state.apply_gradients(grads=actor_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)
    carry = (actor_state, critic_state)
    return carry, None