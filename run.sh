clear
gpu_num=1
export gpu_num

# PPO jobs
parallel --eta --ungroup --jobs 2 python main.py --config_file ./configs/mujoco_ppo.json --config_idx {1} ::: $(seq 1 120)
# python main.py --config_file ./configs/test.json --config_idx 1
# python main.py --config_file ./configs/test.json --config_idx 2
# python main.py --config_file ./configs/mujoco_sac.json --config_idx 1
# python main.py --config_file ./configs/mujoco_ppo.json --config_idx 1

# DQN jobs
# parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/classic_dqn.json --config_idx {1} ::: $(seq 1 90)
# parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/lunar_dqn.json --config_idx {1} ::: $(seq 1 30)
# parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/pygame_dqn.json --config_idx {1} ::: $(seq 1 60)
# parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/minatar_dqn.json --config_idx {1} ::: $(seq 1 30)

# MaxminDQN jobs
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/classic_maxmin.json --config_idx {1} ::: $(seq 1 720)
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/lunar_maxmin.json --config_idx {1} ::: $(seq 1 240)
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/pygame_maxmin.json --config_idx {1} ::: $(seq 1 480)
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/minatar_maxmin.json --config_idx {1} ::: $(seq 1 120)

# DDQN jobs
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/classic_ddqn.json --config_idx {1} ::: $(seq 1 90)
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/lunar_ddqn.json --config_idx {1} ::: $(seq 1 30)
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/pygame_ddqn.json --config_idx {1} ::: $(seq 1 60)
# taskset -c 0-59 parallel --eta --ungroup --jobs procfile python main.py --config_file ./configs/minatar_ddqn.json --config_idx {1} ::: $(seq 1 30)

# MujoCo jobs
# parallel --eta --ungroup --jobs 1 python main.py --config_file ./configs/mujoco_sac.json --config_idx {1} ::: $(seq 1 30)
# parallel --eta --ungroup --jobs 1 python main.py --config_file ./configs/mujoco_ddpg.json --config_idx {1} ::: $(seq 1 60)
# parallel --eta --ungroup --jobs 1 python main.py --config_file ./configs/mujoco_td3.json --config_idx {1} ::: $(seq 1 30)
# parallel --eta --ungroup --jobs 1 python main.py --config_file ./configs/mujoco_td31.json --config_idx {1} ::: $(seq 1 30)

python slacker_msger.py