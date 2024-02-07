import os
import math
from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-10:].mean(skipna=False) if mode=='Train' else result['Return'][-5:].mean(skipna=False)
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(skipna=False),
    'Return (se)': result['Return (mean)'].sem(ddof=0)
  }
  return result_dict

cfg = {
  'exp': 'exp_name',
  'merged': True,
  'x_label': 'Step',
  'y_label': 'Return',
  'rolling_score_window': 20,
  # 'rolling_score_window': -1,
  'hue_label': 'Agent',
  'show': False,
  'imgType': 'png',
  'estimator': 'mean',
  # 'estimator': 'median',
  'ci': 'se',
  # 'ci': ('ci', 95),
  'EMA': True,
  'loc': 'best',
  'sweep_keys': ['agent/name', 'optim/kwargs/learning_rate'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  cfg['exp'] = exp
  cfg['runs'] = runs
  sweep_keys_dict = dict(
    dqn = ['optim/kwargs/learning_rate'],
    ddqn = ['optim/kwargs/learning_rate'],
    maxmin = ['optim/kwargs/learning_rate', 'agent/critic_num'],
    ppo = ['optim/kwargs/learning_rate', 'agent/actor_net_cfg/hidden_dims'],
  )
  algo = exp.rstrip('0123456789').split('_')[-1]
  cfg['sweep_keys'] = sweep_keys_dict[algo]
  plotter = Plotter(cfg)

  plotter.csv_merged_results('Train', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Train', indexes='all')

  # plotter.csv_merged_results('Test', get_csv_result_dict, get_process_result_dict)
  # plotter.plot_results(mode='Test', indexes='all')

  # Hyper-parameter Comparison
  # plotter.csv_unmerged_results('Train', get_process_result_dict)
  # plotter.csv_unmerged_results('Test', get_process_result_dict)
  # constraints = [('agent/name', ['NAF'])]
  # # constraints = []
  # for param_name in cfg['sweep_keys']:
  #   plotter.compare_parameter(param_name=param_name, constraints=constraints, mode='Test', kde=False)
  #   plotter.compare_parameter(param_name=param_name, constraints=constraints, mode='Test', kde=True)


if __name__ == "__main__":
  runs = 10
  mujoco_list = ['mujoco_sac', 'mujoco_ddpg', 'mujoco_td3', 'mujoco_naf']
  dqn_list = ['classic_dqn', 'lunar_dqn', 'pygame_dqn', 'minatar_dqn']
  ddqn_list = ['classic_ddqn', 'lunar_ddqn', 'pygame_ddqn', 'minatar_ddqn']
  maxmin_list = ['classic_maxmin', 'lunar_maxmin', 'pygame_maxmin', 'minatar_maxmin']
  for exp in ['mujoco_ppo']:
    # unfinished_index(exp, runs=runs)
    # memory_info(exp, runs=runs)
    # time_info(exp, runs=runs)
    analyze(exp, runs=runs)