import os
import numpy as np
from scipy.stats import bootstrap
from collections import namedtuple

from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-1*int(len(result['Return'])*0.1):].mean(skipna=True), # mean of last 10%
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train', ci=95, method='percentile'):
  perf_mean = result['Return (mean)'].values.tolist()
  if len(perf_mean) > 1:
    CI = bootstrap(
      (perf_mean,),
      np.mean, confidence_level=ci/100,
      method=method
    ).confidence_interval
  else:
    CI = namedtuple('ConfidenceInterval', ['low', 'high'])(low=perf_mean[0], high=perf_mean[0])
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(skipna=True),
    'Return (se)': result['Return (mean)'].sem(ddof=0),
    'Return (bmean)': (CI.high + CI.low) / 2,
    f'Return (ci={ci})': (CI.high - CI.low) / 2,
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
    ppo = ['optim/kwargs/learning_rate'],
  )
  algo = exp.rstrip('0123456789').split('_')[-1]
  cfg['sweep_keys'] = sweep_keys_dict[algo]
  plotter = Plotter(cfg)

  plotter.csv_merged_results('Train', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Train', indexes='all')

  # plotter.csv_unmerged_results('Train', get_process_result_dict)
  # group_keys = ['optim/kwargs/learning_rate', 'agent/critic_num']
  # plotter.get_top1_result(group_keys=group_keys, perf='Return (bmean)', errorbar='Return (ci=95)', mode='Train', nd=2, markdown=False)
  
  # Hyper-parameter Comparison
  # plotter.csv_unmerged_results('Train', get_process_result_dict)
  # plotter.csv_unmerged_results('Test', get_process_result_dict)
  # constraints = [('agent/name', ['NAF'])]
  # constraints = []
  # for param_name in cfg['sweep_keys']:
  #   plotter.compare_parameter(param_name=param_name, constraints=constraints, mode='Train', kde=False)
  #   plotter.compare_parameter(param_name=param_name, constraints=constraints, mode='Train', kde=True)


if __name__ == "__main__":
  runs = 10
  mujoco_list = ['mujoco_sac', 'mujoco_ddpg', 'mujoco_td3', 'mujoco_ppo', 'mujoco_naf']
  dqn_list = ['classic_dqn', 'lunar_dqn', 'pygame_dqn', 'minatar_dqn']
  ddqn_list = ['classic_ddqn', 'lunar_ddqn', 'pygame_ddqn', 'minatar_ddqn']
  maxmin_list = ['classic_maxmin', 'lunar_maxmin', 'pygame_maxmin', 'minatar_maxmin']
  for exp in mujoco_list:
    unfinished_index(exp, runs=runs)
    memory_info(exp, runs=runs)
    time_info(exp, runs=runs)
    analyze(exp, runs=runs)