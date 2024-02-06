import os
import json
from utils.sweeper import Sweeper


def find_one_run():
  agent_config = 'meta_sdl11.json'
  config_file = os.path.join('./configs/', agent_config)
  sweeper = Sweeper(config_file)
  for i in range(1, 1+sweeper.config_dicts['num_combinations']):
    cfg = sweeper.generate_config_for_idx(i)
    if cfg['agent_optim']['kwargs']['mlp_dims'] == [32,32] and cfg['agent_optim']['kwargs']['hidden_size'] == 8 and cfg['agent_optim']['kwargs']['learning_rate'] == 1e-1:
      print(i, end='\n')
  print()


def find_many_runs():
  l = [24,29,9,28,23,18,30,25,20,26,16,11,17,22,12]
  l.sort()
  print('len(l)=', len(l))
  ll = []
  for r in range(1,5):
    for x in l:
      ll.append(x+30*r)
  print('len(ll)=', len(ll))
  # print(*ll, sep=',')
  print(*ll, sep=' ')


def print_cfg():
  exp = 'meta_sdl11'
  agent_config = f'{exp}.json'
  config_file = os.path.join('./configs/', agent_config)
  sweeper = Sweeper(config_file)
  l = {}
  for i in range(1, 1+sweeper.config_dicts['num_combinations']):
    cfg = sweeper.generate_config_for_idx(i)
    k = f"{cfg['agent_optim']['kwargs']['mlp_dims']}/{cfg['agent_optim']['kwargs']['hidden_size']}/{cfg['agent_optim']['kwargs']['learning_rate']}"
    if k in l.keys():
      l[k].append(i)
    else:
      l[k] = [i]
  for k, v in l.items():
    cfg = sweeper.generate_config_for_idx(v[0])
    cfg = cfg['agent_optim']['kwargs']
    cfg['param_load_path'] = []
    for i in v:
      cfg['param_load_path'].append(f"./logs/meta_sdl11/{i}/param.pickle")
    cfg_json = json.dumps(cfg, indent=2)
    print(cfg_json, end='\n')


if __name__ == "__main__":
  # find_one_run()
  find_many_runs()
  # print_cfg()