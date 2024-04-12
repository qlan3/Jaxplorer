This document records the best hyper-parameter setup for each (task, algorithm) pair, in terms of training performance.

## Gym Classic Control

- Performance metric: average returns of last 10 (for training) or 5 (for testing) episodes.
- Reported performance: average performance over 5 runs with 1 standard deviation.

### MountainCar-v0

| Algo\Param |  train perf  |   test perf   |   cfg file     | cfg index |   lr   | critic_num |
| ---------- | ------------ | ------------- | -------------- | --------- | ------ | ---------- |
|    DQN     | -125 $\pm$ 6 | -246 $\pm$ 31 | classic_dqn    |     4     | 0.003  |     1      |
|    DDQN    | -134 $\pm$ 4 | -164 $\pm$ 13 | classic_ddqn   |     7     | 0.001  |     1      |
|  MaxminDQN | -154 $\pm$ 13| -210 $\pm$ 38 | classic_maxmin |     13    | 0.003  |     2      |
|  MaxminDQN |-628 $\pm$ 168| -681 $\pm$ 168| classic_maxmin |     40    | 0.0003 |     4      |
|  MaxminDQN |-816 $\pm$ 146| -898 $\pm$ 86 | classic_maxmin |     31    | 0.001  |     6      |
|  MaxminDQN |-830 $\pm$ 86 | -943 $\pm$ 43 | classic_maxmin |     22    | 0.003  |     8      |


### Acrobot-v1

| Algo\Param |  train perf  |   test perf   |   cfg file     | cfg index |   lr   | critic_num |
| ---------- | ------------ | ------------- | -------------- | --------- | ------ | ---------- |
|    DQN     | -91 $\pm$ 2  | -246 $\pm$ 31 | classic_dqn    |     8     |  0.001 |     1      |
|    DDQN    | -94 $\pm$ 3  | -103 $\pm$ 8  | classic_ddqn   |     8     |  0.001 |     1      |
|  MaxminDQN | -98 $\pm$ 4  | -218 $\pm$ 31 | classic_maxmin |     26    |  0.001 |     2      |
|  MaxminDQN | -375 $\pm$ 68| -410 $\pm$ 54 | classic_maxmin |     65    |  3E-05 |     4      |
|  MaxminDQN | -498 $\pm$ 2 | -500 $\pm$ 0  | classic_maxmin |     68    |  3E-05 |     6      |
|  MaxminDQN | -500 $\pm$ 0 | -500 $\pm$ 0  | classic_maxmin |     23    |  0.003 |     8      |


### CartPole-v1

| Algo\Param |  train perf  |   test perf   |   cfg file     | cfg index |   lr   | critic_num |
| ---------- | ------------ | ------------- | -------------- | --------- | ------ | ---------- |
|    DQN     | 460 $\pm$ 29 | 476 $\pm$ 14  | classic_dqn    |     9     | 0.001  |     1      |
|    DDQN    | 455 $\pm$ 23 | 478 $\pm$ 10  | classic_ddqn   |     9     | 0.001  |     1      |
|  MaxminDQN | 500 $\pm$ 0  | 499 $\pm$ 1   | classic_maxmin |     51    | 0.0001 |     2      |
|  MaxminDQN | 500 $\pm$ 0  | 500 $\pm$ 0   | classic_maxmin |     54    | 0.0001 |     4      |
|  MaxminDQN | 500 $\pm$ 0  | 500 $\pm$ 0   | classic_maxmin |     21    | 0.003  |     6      |
|  MaxminDQN | 500 $\pm$ 0  | 500 $\pm$ 0   | classic_maxmin |     36    | 0.001  |     8      |



## Gym Box2D

- Performance metric: average returns of last 10 (for training) or 5 (for testing) episodes.
- Reported performance: average performance over 5 runs with 1 standard deviation.

### LunarLander-v2

| Algo\Param |  train perf  |   test perf   |  cfg file    | cfg index |   lr   | critic_num |
| ---------- | ------------ | ------------- | ------------ | --------- | ------ | ---------- |
|    DQN     | 241 $\pm$ 11 | 235 $\pm$ 11  | lunar_dqn    |     4     | 0.0001 |     1      |
|    DDQN    | 183 $\pm$ 17 | 162 $\pm$ 17  | lunar_ddqn   |     3     | 0.0003 |     1      |
|  MaxminDQN | 263 $\pm$ 4  | 248 $\pm$ 11  | lunar_maxmin |     5     | 0.001  |     2      |
|  MaxminDQN | 275 $\pm$ 4  | 264 $\pm$ 5   | lunar_maxmin |     6     | 0.001  |     4      |
|  MaxminDQN | 219 $\pm$ 54 | 216 $\pm$ 54  | lunar_maxmin |     7     | 0.001  |     6      |
|  MaxminDQN |  65 $\pm$ 81 |  60 $\pm$ 82  | lunar_maxmin |     8     | 0.001  |     8      |



## PyGame Learning Environment

- Performance metric: average returns of last 10 (for training) or 5 (for testing) episodes.
- Reported performance: average performance over 5 runs with 1 standard deviation.

### Catcher-PLE-v0

| Algo\Param |  train perf  |  test perf  |    cfg file   | cfg index |   lr   | critic_num |
| ---------- | ------------ | ----------- | ------------- | --------- | ------ | ---------- |
|    DQN     |  54 $\pm$ 0  | 54 $\pm$ 1  | pygame_dqn    |    11     |  1E-05 |     1      |
|    DDQN    |  52 $\pm$ 2  | 52 $\pm$ 1  | pygame_ddqn   |     7     | 0.0001 |     1      |
|  MaxminDQN |  55 $\pm$ 0  | 55 $\pm$ 0  | pygame_maxmin |    33     |  3E-05 |     2      |
|  MaxminDQN |  55 $\pm$ 0  | 55 $\pm$ 0  | pygame_maxmin |    35     |  3E-05 |     4      |
|  MaxminDQN |  55 $\pm$ 0  | 55 $\pm$ 1  | pygame_maxmin |    37     |  3E-05 |     6      |
|  MaxminDQN |  55 $\pm$ 0  | 55 $\pm$ 0  | pygame_maxmin |    39     |  3E-05 |     8      |


### Pixelcopter-PLE-v0

| Algo\Param |  train perf  |  test perf  |   cfg file    | cfg index |   lr   | critic_num |
| ---------- | ------------ | ----------- | ------------- | --------- | ------ | ---------- |
|    DQN     |  40 $\pm$ 2  | 57 $\pm$ 1  | pygame_dqn    |    12     |  1E-05 |     1      |
|    DDQN    |  44 $\pm$ 4  | 51 $\pm$ 1  | pygame_ddqn   |    12     |  1E-05 |     1      |
|  MaxminDQN |  45 $\pm$ 3  | 54 $\pm$ 5  | pygame_maxmin |    34     |  3E-05 |     2      |
|  MaxminDQN |  48 $\pm$ 3  | 53 $\pm$ 2  | pygame_maxmin |    28     | 0.0001 |     4      |
|  MaxminDQN |  52 $\pm$ 2  | 64 $\pm$ 3  | pygame_maxmin |    30     | 0.0001 |     6      |
|  MaxminDQN |  56 $\pm$ 3  | 56 $\pm$ 1  | pygame_maxmin |    32     | 0.0001 |     8      |


## MuJoCo

- Performance metric: average training returns of last 10% episodes.
- Reported performance: bootstrapped average performance over 5 runs with 95% confidence interval.

|   Task\Algo    |       PPO      |       SAC       |       DDPG      |       TD3      |
| -------------- | -------------- | --------------- | --------------- | -------------- |
| Ant-v4         | 1920 $\pm$ 393 |  4989 $\pm$ 332 | 1411 $\pm$ 576  | 2780 $\pm$ 244 |
| HalfCheetah-v4 | 3868 $\pm$ 585 | 10469 $\pm$ 269 | 9441 $\pm$ 553  | 8587 $\pm$ 453 |
| Hopper-v4      | 2334 $\pm$ 147 |  2459 $\pm$ 352 | 1750 $\pm$ 283  | 2437 $\pm$ 730 |
| Humanoid-v4    |  670 $\pm$ 23  |  5141 $\pm$ 125 | 3190 $\pm$ 1013 | 4948 $\pm$ 299 |
| Swimmer-v4     |   68 $\pm$ 11  |    62 $\pm$ 7   |   99 $\pm$ 23   |   83 $\pm$ 20  |
| Walker2d-v4    | 2857 $\pm$ 379 |  4285 $\pm$ 538 | 2424 $\pm$ 595  | 3959 $\pm$ 444 |


## MinAtar

### Asterix-MinAtar-v1


### Breakout-MinAtar-v1


### Freeway-MinAtar-v1


### SpaceInvaders-MinAtar-v1


### Seaquest-MinAtar-v1