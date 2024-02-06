# Jaxplorer

Jaxplorer is a Jax reinforcement learning (RL) framework for **exploring** new ideas.  

> [!WARNING]
> This project is still experimental, APIs could change without notice.  
> Pure Jax RL tasks are not supported at this stage.

> [!NOTE]
> For PyTorch version, please check [Explorer](https://github.com/qlan3/Explorer).


## Requirements

- Python: 3.11
- [Jax](https://jax.readthedocs.io/en/latest/installation.html): >=0.4.20
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium): pip install gymnasium==0.29.1
- [Mujoco](https://github.com/google-deepmind/mujoco): pip install mujoco==2.3.7
- [Gymnasium(mujoco)](https://gymnasium.farama.org/environments/mujoco/): pip install gymnasium[mujoco]
- Others: Please check `requirements.txt`.


## Implemented algorithms

- [Deep Q-Learning (DQN)](https://users.cs.duke.edu/~pdinesh/sources/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Double Deep Q-learning (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Maxmin Deep Q-learning (MaxminDQN)](https://arxiv.org/pdf/2002.06487.pdf)
- [Proximal Policy Optimisation (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Continuous Deep Q-Learning with Model-based Acceleration (NAF)](https://arxiv.org/pdf/1603.00748.pdf): model-free version; a different exploration strategy is applied for simplicity.

TODO: add more algorithms and improve the performance of PPO.

## Experiments

### Train && Test

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including log files are saved in directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `classic_dqn.json` and configuration index `1`:

```python main.py --config_file ./configs/classic_dqn.json --config_idx 1```

The models are tested for one episode after every `test_per_episodes` training episodes which can be set in the configuration file.


### Grid Search (Optional)

First, we calculate the number of total combinations in a configuration file (e.g. `classic_dqn.json`):

`python utils/sweeper.py`

The output will be:

`Number of total combinations in classic_dqn.json: 18`

Then we run through all configuration indexes from `1` to `18`. The simplest way is using a bash script:

``` bash
for index in {1..18}
do
  python main.py --config_file ./configs/classic_dqn.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule a large number of jobs:

``` bash
parallel --eta --ungroup python main.py --config_file ./configs/classic_dqn.json --config_idx {1} ::: $(seq 1 18)
```

Any configuration index that has the same remainder (divided by the number of total combinations) should have the same configuration dict. So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```
for index in 1 19 37 55 73
do
  python main.py --config_file ./configs/classic_dqn.json --config_idx $index
done
```

Or a simpler way:
```
parallel --eta --ungroup python main.py --config_file ./configs/classic_dqn.json --config_idx {1} ::: $(seq 1 18 90)
```

### Slurm (Optional)

Slurm is supported as well. Please check `submit.py`.
TODO: add more details.


### Analysis (Optional)

To analyze the experimental results, just run:

`python analysis.py`

Inside `analysis.py`, `unfinished_index` will print out the configuration indexes of unfinished jobs based on the existence of the result file. `memory_info` will print out the memory usage information and generate a histogram to show the distribution of memory usages in directory `logs/classic_dqn/0`. Similarly, `time_info` will print out the time information and generate a histogram to show the distribution of time in directory `logs/classic_dqn/0`. Finally, `analyze` will generate `csv` files that store training and test results. Please check `analysis.py` for more details. More functions are available in `utils/plotter.py`.

TODO: add more details about hyper-parameter comparison.


## Cite

If you find this repo useful to your research, please cite my paper if related. Otherwise, please cite this repo:

~~~bibtex
@misc{Jaxplorer,
  author = {Lan, Qingfeng},
  title = {A Jax Reinforcement Learning Framework for Exploring New Ideas},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/qlan3/Jaxplorer}}
}
~~~

# Acknowledgements

- [Explorer](https://github.com/qlan3/Explorer)
- [Jax RL](https://github.com/ikostrikov/jaxrl)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [PureJaxRL](https://github.com/luchris429/purejaxrl)