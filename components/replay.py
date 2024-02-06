import numpy as np


class FiniteReplay(object):
  '''
  Finite replay buffer to store experiences: FIFO (first in, firt out)
  '''
  def __init__(self, buffer_size, keys=None):
    if keys is None:
      keys = ['action', 'reward', 'mask']
    self.keys = keys
    self.buffer_size = int(buffer_size)
    self.clear()

  def clear(self):
    self.pos = 0
    self.full = False
    self.buffer = dict()

  def add(self, data):
    assert list(data) == self.keys
    if self.is_empty(): # Lazy initialization
      for k, v in data.items():
        if k in ['obs', 'next_obs', 'action']:
          self.buffer[k] = np.empty((self.buffer_size, *v.shape), dtype=v.dtype)
        else:
          self.buffer[k] = np.empty((self.buffer_size,))
    for k, v in data.items():
      self.buffer[k][self.pos] = v
    self.pos = (self.pos + 1) % self.buffer_size
    if self.pos == 0:
      self.full = True

  def sample(self, batch_size, keys=None):
    if keys is None:
      keys = self.keys
    # Sampling with replacement
    idxs = np.random.randint(0, self.size(), size=batch_size)
    batch = {k:self.buffer[k][idxs] for k in keys}
    return batch

  def get(self, batch_size, keys=None):
    if keys is None:
      keys = self.keys
    batch = {k:self.buffer[k][:batch_size] for k in keys}
    return batch
  
  def size(self):
    if self.full:
      return self.buffer_size
    else:
      return self.pos

  def is_empty(self):
    if self.size() == 0:
      return True
    else:
      return False
  
  def is_full(self):
    return self.full
  

class VectorFiniteReplay(FiniteReplay):
  '''
  Vector finite replay buffer to store experiences: for vector environments
  '''
  def clear(self):
    self.pos = 0
    self.full = False
    self.buffer = dict()
    self.num_envs = None

  def add(self, data):
    assert list(data) == self.keys
    if self.is_empty(): # Lazy initialization
      for k, v in data.items():
        if self.num_envs is None:
          self.num_envs = v.shape[0]
        self.buffer[k] = np.empty((self.buffer_size, *v.shape), dtype=v.dtype)
    for k, v in data.items():
      self.buffer[k][self.pos] = v
    self.pos = (self.pos + 1) % self.buffer_size
    if self.pos == 0:
      self.full = True

  def sample(self, batch_size, keys=None):
    if keys is None:
      keys = self.keys
    # Sampling with replacement
    batch_idxs = np.random.randint(0, self.size(), size=batch_size)
    env_idxs = np.random.randint(0, high=self.num_envs, size=batch_size)
    batch = {k:self.buffer[k][batch_idxs][env_idxs] for k in keys}
    return batch