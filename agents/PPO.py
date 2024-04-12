import jax
import time
import math
import distrax
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial

from envs.env import ppo_make_env
from agents.BaseAgent import BaseAgent, TrainState
from components.replay import FiniteReplay
from components.networks import MLPVCritic, MLPGaussianActor, MLPCategoricalActor


class PPO(BaseAgent):
  """
  Implementation of Proximal Policy Optimization.
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

  def createNN(self):
    # Create nets and train_states
    lr = self.cfg['optim']['kwargs']['learning_rate']
    def linear_schedule(count):
      iter = count // (self.cfg['agent']['update_epochs'] * self.num_batches)
      frac = 1.0 - iter / self.train_iter
      return frac * lr
    dummy_obs = self.env['Train'].observation_space.sample()[None,]
    self.seed, actor_seed, critic_seed = jax.random.split(self.seed, 3)
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
        done = terminated or truncated
        mask = self.discount * (1.0 - done)
        self.save_experience(obs, action, reward, mask, v, log_pi)
        # Update observation
        obs = next_obs
        # Record and reset
        if done:
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
    v = jax.device_get(v)[0]
    log_pi = jax.device_get(log_pi)[0]
    return dict(action=action, v=v, log_pi=log_pi)

  @partial(jax.jit, static_argnames=['self'])
  def random_action(self, actor_state, critic_state, obs, seed):
    seed, action_seed = jax.random.split(seed)
    action_mean, action_std = actor_state.apply_fn(actor_state.params, obs)
    pi = distrax.Normal(loc=action_mean, scale=action_std)
    action = pi.sample(seed=action_seed)
    log_pi = pi.log_prob(action).sum(-1)
    v = critic_state.apply_fn(critic_state.params, obs)
    return action, v, log_pi, seed

  @partial(jax.jit, static_argnames=['self'])
  def optimal_action(self, actor_state, critic_state, obs, seed):
    action_mean, action_std = actor_state.apply_fn(actor_state.params, obs)
    pi = distrax.Normal(loc=action_mean, scale=action_std)
    log_pi = pi.log_prob(action_mean).sum(-1)
    v = critic_state.apply_fn(critic_state.params, obs)
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
      reverse = True
    )
    trajectory['v_target'] = adv + trajectory['v']
    # Normalize advantage
    trajectory['adv'] = (adv - adv.mean()) / (adv.std() + 1e-8)
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
    (actor_state, critic_state), _ = jax.lax.scan(
      f = self.update_batch,
      init = (actor_state, critic_state),
      xs = batches
    )
    carry = (actor_state, critic_state, trajectory, seed)
    return carry, None
  
  @partial(jax.jit, static_argnames=['self'])
  def update_batch(self, carry, batch):
    actor_state, critic_state = carry
    # Set loss function
    def compute_loss(params):
      # Compute critic loss
      v = critic_state.apply_fn(params['critic'], batch['obs'])
      critic_loss = jnp.square(v - batch['v_target']).mean()      
      # Compute actor loss
      action_mean, action_std = actor_state.apply_fn(params['actor'], batch['obs'])
      pi = distrax.Normal(loc=action_mean, scale=action_std)
      log_pi = pi.log_prob(batch['action']).sum(-1)
      ratio = jnp.exp(log_pi - batch['log_pi'])
      obj = ratio * batch['adv']
      obj_clipped = jnp.clip(ratio, 1.0-self.cfg['agent']['clip_ratio'], 1.0+self.cfg['agent']['clip_ratio']) * batch['adv']
      actor_loss = -jnp.minimum(obj, obj_clipped).mean()
      # Compute entropy loss
      entropy_loss = pi.entropy().sum(-1).mean()
      total_loss = actor_loss + self.cfg['agent']['vf_coef'] * critic_loss - self.cfg['agent']['ent_coef'] * entropy_loss
      return total_loss
    # Update train_state and critic_state
    params = {
      'actor': actor_state.params,
      'critic': critic_state.params
    }
    grads = jax.grad(compute_loss)(params)
    actor_state = actor_state.apply_gradients(grads=grads['actor'])
    critic_state = critic_state.apply_gradients(grads=grads['critic'])
    carry = (actor_state, critic_state)
    return carry, None