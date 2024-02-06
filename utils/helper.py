import os
import sys
import random
import psutil
import datetime
import numpy as np


def get_time_str():
  return datetime.datetime.now().strftime("%y.%m.%d-%H:%M:%S")

def rss_memory_usage():
  '''
  Return the resident memory usage in MB
  '''
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / float(2 ** 20)
  return mem

def str_to_class(module_name, class_name):
  '''
  Convert string to class
  '''
  return getattr(sys.modules[module_name], class_name)

def set_random_seed(seed):
  '''
  Set all random seeds
  '''
  random.seed(seed)
  np.random.seed(seed)

def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)

def generate_batch_idxs(length, batch_size):
  idxs = np.asarray(np.random.permutation(length))
  batches = idxs[:length // batch_size * batch_size].reshape(-1, batch_size)
  return batches