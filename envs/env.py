import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import ClipAction, RescaleAction, RecordEpisodeStatistics
from envs.wrappers import UniversalSeed

import gym_pygame
import gym_minatar
import gym_exploration


def make_env(env_name, deque_size=1, **kwargs):
  """
  Make env for general tasks.
  """
  env = gym.make(env_name, **kwargs)
  # Episode statistics wrapper
  env = RecordEpisodeStatistics(env, deque_size=deque_size)
  # Action wrapper
  if isinstance(env.action_space, spaces.Box): # Continuous action space
    env = ClipAction(RescaleAction(env, min_action=-1, max_action=1))
  # Seed wrapper: must be the last wrapper to be effective
  env = UniversalSeed(env)
  return env


def make_vec_env(env_name, num_envs=1, asynchronous=False, deque_size=1, max_episode_steps=None, **kwargs):
  env = gym.make(env_name, **kwargs)
  wrappers = []
  # Episode statistics wrapper
  wrappers.append(lambda env: RecordEpisodeStatistics(env, deque_size=deque_size))
  # Action wrapper
  if isinstance(env.action_space, spaces.Box): # Continuous action space
    wrappers.append(lambda env: ClipAction(RescaleAction(env, min_action=-1, max_action=1)))
  envs = gym.vector.make(
    env_name,
    num_envs = num_envs,
    asynchronous = asynchronous,
    max_episode_steps = max_episode_steps,
    wrappers = wrappers
  )
  # Seed wrapper: must be the last wrapper to be effective
  envs = UniversalSeed(envs)
  return envs