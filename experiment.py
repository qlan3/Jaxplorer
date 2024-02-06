import os
import jax
import copy
import time
import json
import numpy as np

import agents
from utils.helper import set_random_seed, rss_memory_usage


class Experiment(object):
  '''
  Train the agent to play the game.
  '''
  def __init__(self, cfg):
    self.cfg = copy.deepcopy(cfg)
    # Set default device
    try:
      if not (len(jax.devices(backend='cuda'))>0 and 'cuda' in cfg['device']):
        self.cfg['device'] = 'cpu'
    except:
      self.cfg['device'] = 'cpu'
    if self.cfg['device'] != 'cuda':
      if self.cfg['device'] == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')
      elif 'cuda' in self.cfg['device']:
        backend, idx = self.cfg['device'].split(':')
        device = jax.devices(backend=backend)[int(idx)]
        jax.config.update("jax_default_device", device)
    if self.cfg['generate_random_seed']:
      self.cfg['seed'] = np.random.randint(int(1e6))
    # if not os.path.exists(self.cfg['cfg_path']):
    self.save_config(self.cfg['cfg_path'])

  def run(self):
    '''
    Run the game for multiple times
    '''
    start_time = time.time()
    set_random_seed(self.cfg['seed'])
    agent = getattr(agents, self.cfg['agent']['name'])(self.cfg)
    # Train && Test
    agent.run_steps()
    end_time = time.time()
    agent.logger.info(f'Memory usage: {rss_memory_usage():.2f} MB')
    agent.logger.info(f'Time elapsed: {(end_time-start_time)/60:.2f} minutes')
  
  def save_config(self, cfg_path):
    cfg_json = json.dumps(self.cfg, indent=2)
    with open(cfg_path, 'w') as f:
      f.write(cfg_json)