import jax
import time
import flax
import optax
import numpy as np
import pandas as pd
from gymnasium import spaces
from flax.training.train_state import TrainState

from envs.env import make_env
from utils.logger import Logger


class TargetState(TrainState):
  target_params: flax.core.FrozenDict = None


class BaseAgent(object):
  '''
  Implementation of Base Agent
  '''
  def __init__(self, cfg):
    cfg['agent'].setdefault('actor_update_steps', 1)
    cfg['agent'].setdefault('critic_update_steps', 1)
    cfg['agent'].setdefault('critic_num', 1)
    self.cfg = cfg
    # Set logger
    self.logger = Logger(self.cfg['logs_dir'])
    # Set some varibles
    self.env_name = self.cfg['env']['name']
    self.agent_name = self.cfg['agent']['name']
    self.seed = jax.random.PRNGKey(cfg['seed'])
    self.config_idx = self.cfg['config_idx']
    self.discount = self.cfg['discount']
    self.train_steps = int(cfg['train_steps'])
    self.cfg['test_interval'] = int(cfg['test_interval'])
    self.cfg['display_interval'] = int(cfg['display_interval'])
    self.log_path = {'Train': self.cfg['train_log_path'], 'Test': self.cfg['test_log_path']}
    self.result = {'Train': [], 'Test': []}
    # Make environment
    self.env = {
      'Train': make_env(self.env_name, **self.cfg['env']['kwargs']),
      'Test': make_env(self.env_name, deque_size=cfg['test_episodes'], **self.cfg['env']['kwargs'])
    }
    self.env['Train'].set_seed(self.cfg['seed'])
    self.env['Test'].set_seed(self.cfg['seed']+42)
    self.obs_size = self.get_obs_size(self.env['Train'])
    self.action_type, self.action_size = self.get_action_size(self.env['Train'])

  def set_optim(self, optim_name, optim_kwargs, schedule=None):
    optim_kwargs.setdefault('anneal_lr', False)
    optim_kwargs.setdefault('grad_clip', -1)
    optim_kwargs.setdefault('max_grad_norm', -1)
    anneal_lr = optim_kwargs['anneal_lr']
    grad_clip = optim_kwargs['grad_clip']
    max_grad_norm = optim_kwargs['max_grad_norm']
    del optim_kwargs['anneal_lr'], optim_kwargs['grad_clip'], optim_kwargs['max_grad_norm']
    assert not (grad_clip > 0 and max_grad_norm > 0), 'Either grad_clip or max_grad_norm should be set.'
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

  def get_action_size(self, env):
    if hasattr(env, 'single_action_space'):
      action_space = env.single_action_space
    else:
      action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
      action_type = 'DISCRETE'
      return action_type, action_space.n
    elif isinstance(action_space, spaces.Box):
      action_type = 'CONTINUOUS'
      return action_type, int(np.prod(action_space.shape))
    else:
      action_type = 'UNKNOWN'
      raise ValueError('Unknown action type.')

  def get_obs_size(self, env):
    if hasattr(env, 'single_observation_space'):
      observation_space = env.single_observation_space
    else:
      observation_space = env.observation_space
    if isinstance(observation_space, spaces.Discrete):
      return observation_space.n
    elif isinstance(observation_space, spaces.Box):
      return int(np.prod(observation_space.shape))
    else:
      raise ValueError('Unknown observation type.')

  def save_experience(self, obs, action, reward, mask, next_obs):
    prediction = {
      'obs': obs,
      'action': action,
      'reward': reward,
      'mask': mask,
      'next_obs': next_obs
    }
    self.replay.add(prediction)

  def get_action(self, step, obs, mode='Train'):
    raise NotImplementedError

  def log_test_save(self, iter, train_iter, mode):
    if self.agent_name in ['PPO']:
      step = iter * self.cfg['agent']['collect_steps']
    else:
      step = iter
    # Test for several episodes
    if (self.cfg['test_interval'] > 0) and (iter % self.cfg['test_interval'] == 0 or iter == train_iter-1):
      self.test(step)
    # Save checkpoint
    if (self.cfg['ckpt_interval'] > 0) and ((iter > 0 and iter % self.cfg['ckpt_interval'] == 0) or iter == train_iter-1):
      self.save_checkpoint(step)
    # Display log
    if (self.cfg['display_interval'] > 0) and (iter % self.cfg['display_interval'] == 0 or iter == train_iter-1):
      speed = step / (time.time() - self.start_time)
      eta = (self.train_steps - step) / speed / 60 if speed>0 else -1
      self.logger.info(f'<{self.config_idx}> ({self.agent_name}) ({self.env_name}) {step}/{self.train_steps}: Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)')
      if len(self.result[mode]) > 0:
        self.logger.info(f'<{self.config_idx}> [{mode}] {step}/{self.train_steps}: Return={self.result[mode][-1]["Return"]:.2f}')

  def test(self, step, mode='Test'):
    for _ in range(self.cfg['test_episodes']):
      obs, info = self.env[mode].reset()
      while True:
        action = self.get_action(step, obs[None,], mode)['action']
        obs, reward, terminated, truncated, info = self.env[mode].step(action)
        if terminated or truncated:
          break
    # Gather result
    result_dict = {
      'Env': self.env_name,
      'Agent': self.agent_name,
      'Step': step,
      'Return': np.mean(self.env[mode].return_queue)
    }
    self.result[mode].append(result_dict)
    self.logger.info(f'<{self.config_idx}> [{mode}] Step {step}/{self.train_steps}: Return={result_dict["Return"]:.2f}')

  def save_checkpoint(self, step):
    '''
    # Save model params
    ckpt_dict = {
      'model_param':,
      'optim_state':
    }
    with open(self.ckpt_path, 'wb') as f:
      pickle.dump(ckpt_dict, f)
    '''
    # Save result to files
    if self.cfg['test_interval'] > 0:
      modes = ['Train', 'Test']
    else:
      modes = ['Train']
    for mode in modes:
      result = pd.DataFrame(self.result[mode])
      result['Env'] = result['Env'].astype('category')
      result['Agent'] = result['Agent'].astype('category')
      result.to_feather(self.log_path[mode])