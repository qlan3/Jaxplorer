import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import variance_scaling, lecun_uniform, he_uniform, constant, zeros_init, orthogonal

from jax.typing import ArrayLike
from typing import Any, Callable, Sequence, Tuple, Dict
Initializer = Callable[[Any, Tuple[int, ...], Any], Any]


activations = {
  'ReLU': nn.relu,
  'ELU': nn.elu,
  'Softplus': nn.softplus,
  'LeakyReLU': nn.leaky_relu,
  'Tanh': jnp.tanh,
  'Sigmoid': nn.sigmoid,
  'Exp': jnp.exp
}


def default_init(scale: float = jnp.sqrt(2)):
  return orthogonal(scale)


class MLP(nn.Module):
  """ Multilayer perceptron. """
  layer_dims: Sequence[int]
  hidden_act: str = 'ReLU'
  output_act: str = 'Linear'
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0
  
  def setup(self):
    layers = []
    for i in range(len(self.layer_dims)):
      layers.append(nn.Dense(self.layer_dims[i], kernel_init=self.kernel_init()))
      layers.append(activations[self.hidden_act])
    layers.pop()
    if self.last_w_scale > 0:
      layers.pop()
      layers.append(nn.Dense(self.layer_dims[-1], kernel_init=self.kernel_init(self.last_w_scale)))

    if self.output_act != 'Linear':
      layers.append(activations[self.output_act])
    self.mlp = nn.Sequential(layers)

  def __call__(self, x):
    return self.mlp(x)


BatchMLP = nn.vmap(
  MLP,
  in_axes=0, out_axes=0,
  variable_axes={'params': None}, # Parameters are shared across one batch
  split_rngs={'params': False}    # The RNG of params is shared
)


class MLPQNet(nn.Module):
  """MLP Q-network: Q(s)"""
  net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  kernel_init: Initializer = default_init
  action_size: int = 10
  last_w_scale: float = -1.0
  
  def setup(self):
    self.Q_net = MLP(
      layer_dims = list(self.net_cfg['hidden_dims']) + [self.action_size],
      hidden_act = self.net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )

  def __call__(self, obs):
    obs = obs.reshape((obs.shape[0], -1))  # flatten
    q = self.Q_net(obs)
    return q


class MinAtarQNet(nn.Module):
  """MinAtar Q-network: Conv2d + MLP"""
  net_cfg: FrozenDict = FrozenDict({'feature_dim': 128, 'hidden_act': 'ReLU'})
  kernel_init: Initializer = default_init
  action_size: int = 10
  
  def setup(self):
    self.Q_net = nn.Sequential([
      nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), kernel_init=self.kernel_init),
      activations[self.net_cfg['hidden_act']],
      lambda x: x.reshape((x.shape[0], -1)), # flatten
      nn.Dense(self.net_cfg['feature_dim'], kernel_init=self.kernel_init),
      activations[self.net_cfg['hidden_act']],
      nn.Dense(self.action_size, kernel_init=self.kernel_init)
    ])

  def __call__(self, obs):
    q = self.Q_net(obs)
    return q


class MLPVCritic(nn.Module):
  """ MLP state value critic: V(s). """
  net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0

  def setup(self):
    self.V_net = MLP(
      layer_dims = list(self.net_cfg['hidden_dims']) + [1],
      hidden_act = self.net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )

  def __call__(self, x):
    return self.V_net(x).squeeze(-1)


class MLPQCritic(nn.Module):
  """ MLP action value Critic: Q(s,a). """
  net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0
  
  def setup(self):
    self.Q_net = MLP(
      layer_dims = list(self.net_cfg['hidden_dims']) + [1],
      hidden_act = self.net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )

  def __call__(self, obs, action):
    x = jnp.concatenate([obs, action], -1)
    q = self.Q_net(x)
    return q


class MLPCategoricalActor(nn.Module):
  """ MLP actor network with discrete categorical policy. """
  action_size: int = 4
  net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0

  def setup(self):
    self.actor_net = MLP(
      layer_dims = list(self.net_cfg['hidden_dims'])+[self.action_size],
      hidden_act = self.net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )

  def __call__(self, obs):
    logits = self.actor_net(obs)
    return logits
  

class MLPGaussianActor(nn.Module):
  """ MLP actor network with Guassian policy N(mu, std). """
  action_size: int = 4
  net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  log_std_min: float = -20.0
  log_std_max: float = 2.0
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0

  def setup(self):
    self.actor_net = MLP(
      layer_dims = list(self.net_cfg['hidden_dims'])+[self.action_size],
      hidden_act = self.net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )
    self.action_log_std = self.param('log_std', zeros_init(), (self.action_size,))

  def __call__(self, obs):
    u_mean = self.actor_net(obs)
    u_log_std = jnp.clip(self.action_log_std, self.log_std_min, self.log_std_max)
    u_log_std = jnp.broadcast_to(u_log_std, u_mean.shape)
    return u_mean, u_log_std


class PPONet(nn.Module):
  action_size: int = 4
  actor_net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  critic_net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  log_std_min: float = -20.0
  log_std_max: float = 2.0
  kernel_init: Initializer = default_init

  def setup(self):
    self.actor_net = MLP(
      layer_dims = list(self.actor_net_cfg['hidden_dims'])+[self.action_size],
      hidden_act = self.actor_net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = 0.01
    )
    self.action_log_std = self.param('log_std', zeros_init(), (self.action_size,))
    self.critic_net = MLP(
      layer_dims = list(self.critic_net_cfg['hidden_dims']) + [1],
      hidden_act = self.critic_net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = 1.0
    )  

  def __call__(self, obs):
    u_mean = self.actor_net(obs)
    u_log_std = jnp.clip(self.action_log_std, self.log_std_min, self.log_std_max)
    return u_mean, u_log_std, self.critic_net(obs).squeeze(-1)
  
  def get_v(self, obs):
    return self.critic_net(obs).squeeze(-1)


class MLPGaussianTanhActor(nn.Module):
  """ MLP actor network with Guassian policy N(mu, std): Tanh is applied outside of this module. """
  action_size: int = 4
  net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  log_std_min: float = -20.0
  log_std_max: float = 2.0
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0

  def setup(self):
    self.actor_net = MLP(
      layer_dims = list(self.net_cfg['hidden_dims'])+[2*self.action_size],
      hidden_act = self.net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )

  def __call__(self, obs):
    u_mean, u_log_std = jnp.split(self.actor_net(obs), indices_or_sections=2, axis=-1)
    u_log_std = jnp.clip(u_log_std, self.log_std_min, self.log_std_max)
    return u_mean, u_log_std


class MLPDeterministicActor(nn.Module):
  action_size: int = 4
  net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0
  
  def setup(self):
    self.actor_net = MLP(
      layer_dims = list(self.net_cfg['hidden_dims'])+[self.action_size],
      hidden_act = self.net_cfg['hidden_act'],
      output_act = 'Tanh',
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )
    
  def __call__(self, obs):
    return self.actor_net(obs)


class Temperature(nn.Module):
  """ Self-tuning temperature for SAC. """
  init_temp: float = 1.0

  def setup(self):
    self.log_temp = self.param('log_temp', init_fn=lambda seed: jnp.full((), jnp.log(self.init_temp)))

  def __call__(self):
    return jnp.exp(self.log_temp)


class NAFQuadraticCritic(nn.Module):
  """ Normalized advantage function with a quadratic critic:
  Q(s,a) = V(s) + A(s,a) where
    A(s,a) = -0.5 * (a - mu(s))^T * P(s) * (a - mu(s)),
    P(s) = L * L^T, and L is lower triangular with non-negative diagonal.
  """
  V_net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'ReLU'})
  mu_net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'Tanh'})
  L_net_cfg: FrozenDict = FrozenDict({'hidden_dims': [32,32], 'hidden_act': 'Tanh'})
  action_size: int = 3
  kernel_init: Initializer = default_init
  last_w_scale: float = -1.0

  def setup(self):
    self.V_net = MLP(
      layer_dims = list(self.V_net_cfg['hidden_dims'])+[1],
      hidden_act = self.V_net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )
    self.mu_net = MLP(
      layer_dims = list(self.mu_net_cfg['hidden_dims'])+[self.action_size],
      hidden_act = self.mu_net_cfg['hidden_act'],
      output_act = 'Tanh',
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )
    self.L_net = MLP(
      layer_dims = list(self.L_net_cfg['hidden_dims'])+[self.action_size*(1+self.action_size)//2],
      hidden_act = self.L_net_cfg['hidden_act'],
      kernel_init = self.kernel_init,
      last_w_scale = self.last_w_scale
    )

  def __call__(self, obs, action):
    V = self.V_net(obs)
    mu = self.mu_net(obs)
    L = jax.vmap(self.vector_to_lower_triangular)(self.L_net(obs))
    L_T = jnp.transpose(L, axes=(0,2,1))
    P = L @ L_T
    a_mu = jnp.expand_dims(action-mu, axis=-1)
    a_mu_T = jnp.transpose(a_mu, axes=(0,2,1))
    A = -0.5 * (a_mu_T @ P @ a_mu).squeeze(axis=-1)
    Q = V + A
    return Q
  
  def get_mu(self, obs):
    return self.mu_net(obs)
  
  def get_v(self, obs):
    return self.V_net(obs)
  
  def vector_to_lower_triangular(self, v):
    # Transform the vector into a lower triangular matrix
    m = jnp.zeros((self.action_size, self.action_size), dtype=v.dtype).at[jnp.tril_indices(self.action_size)].set(v)
    # Apply exp to get non-negative diagonal
    m = m.at[jnp.diag_indices(self.action_size)].set(jnp.exp(jnp.diag(m)))
    return m