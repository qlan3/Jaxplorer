import os
import sys
import argparse
import numpy as np
from math import ceil
from utils.submitter import Submitter


def make_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)


def main(argv):
  # python submit.py --job_type S
  # python submit.py --job_type M
  parser = argparse.ArgumentParser(description="Submit jobs")
  parser.add_argument('--job_type', type=str, default='M', help='Run single (S) or multiple (M) jobs in one experiment: S, M')
  args = parser.parse_args()

  sbatch_cfg = {
    # Account name
    # 'account': 'def-ashique',
    'account': 'rrg-ashique',
    # Job name
    # 'job-name': 'naf1', # 1/2GPU, 8G, 75min
    'job-name': 'sac1', # 1/2GPU, 8G, 75min
    # Job time
    'time': '0-01:15:00',
    # Email notification
    'mail-user': 'qlan3@ualberta.ca'
  }
  general_cfg = {
    # User name
    'user': 'qlan3',
    # Check time interval in minutes
    'check-time-interval': 5,
    # Clusters info: name & capacity
    'cluster_capacity': 996,
    # Job indexes list
    # 'job-list': np.array([1,2])
    # 'job-list': np.array(range(3, 40+1))
    'job-list': np.array(range(1, 5+1))
  }
  make_dir(f"output/{sbatch_cfg['job-name']}")

  if args.job_type == 'M':
    # The number of total jobs for one task
    jobs_per_task = 2
    # Max number of parallel jobs in one task
    max_parallel_jobs = 2
    mem_per_job = 8  # in GB
    cpu_per_job = 4  # Larger cpus_per_job increases speed
    mem_per_cpu = int(ceil(max_parallel_jobs*mem_per_job/cpu_per_job))
    # Write to procfile for Parallel
    with open('procfile', 'w') as f:
      f.write(str(max_parallel_jobs))
    sbatch_cfg['gres'] = 'gpu:1' # GPU type
    sbatch_cfg['cpus-per-task'] = cpu_per_job*max_parallel_jobs
    sbatch_cfg['mem-per-cpu'] = f'{mem_per_cpu}G' # Memory
    # Sbatch script path
    general_cfg['script-path'] = './sbatch_m.sh'
    # Max number of jobs for Parallel
    general_cfg['jobs-per-task'] = jobs_per_task
    submitter = Submitter(general_cfg, sbatch_cfg)
    submitter.multiple_submit()
  elif args.job_type == 'S':
    mem_per_job = 8  # in GB
    cpu_per_job = 4  # Larger cpus_per_job increases speed
    mem_per_cpu = int(ceil(mem_per_job/cpu_per_job))
    sbatch_cfg['gres'] = 'gpu:1' # GPU type
    sbatch_cfg['cpus-per-task'] = cpu_per_job
    sbatch_cfg['mem-per-cpu'] = f'{mem_per_cpu}G' # Memory
    # Sbatch script path
    general_cfg['script-path'] = './sbatch_s.sh'
    submitter = Submitter(general_cfg, sbatch_cfg)
    submitter.single_submit()


if __name__=='__main__':
  main(sys.argv)
