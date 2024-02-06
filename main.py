import sys
import jax
import argparse

from utils.sweeper import Sweeper
from utils.helper import make_dir
from experiment import Experiment

# jax.config.update('jax_disable_jit', True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
# Set a specific platform
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_default_device", jax.devices()[0])

# Fake devices
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'


def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/test.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  
  # Set config dict default value
  cfg['env'].setdefault('kwargs', {})
  # cfg['env']['kwargs'].setdefault('max_episode_steps', None)
  cfg.setdefault('show_progress', False)
  
  # Set experiment name and log paths
  cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
  if len(args.slurm_dir) > 0:  
    cfg['logs_dir'] = f"{args.slurm_dir}/{cfg['exp']}/{cfg['config_idx']}/"
    make_dir(cfg['logs_dir'])
  else:
    cfg['logs_dir'] = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
  make_dir(f"./logs/{cfg['exp']}/{cfg['config_idx']}/")
  cfg['train_log_path'] = cfg['logs_dir'] + 'result_Train.feather'
  cfg['test_log_path'] = cfg['logs_dir'] + 'result_Test.feather'
  cfg['ckpt_path'] = cfg['logs_dir'] + 'ckpt.pickle'
  cfg['cfg_path'] = cfg['logs_dir'] + 'config.json'

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)