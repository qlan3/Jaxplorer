#!/bin/bash
# Ask SLURM to send the USR1 signal 300 seconds before end of the time limit
#SBATCH --signal=B:USR1@60
#SBATCH --output=output/%x/%a.txt
#SBATCH --mail-type=ALL
#SBATCH --exclude=bc11202,bc11203,bc11357,bc11322,bc11234

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
# ---------------------------------------------------------------------
cleanup()
{
    echo "Copy log files from temporary directory"
    sour=$SLURM_TMPDIR/$SLURM_JOB_NAME/.
    dest=./logs/$SLURM_JOB_NAME/
    echo "Source directory: $sour"
    echo "Destination directory: $dest"
    cp -rf $sour $dest
}
# Call `cleanup` once we receive USR1 or EXIT signal
trap 'cleanup' USR1 EXIT
# ---------------------------------------------------------------------
# export OMP_NUM_THREADS=1
module load StdEnv/2023 gcc/12.3 cudacore/.12.2.2 cudnn/8.9 cuda/12.2 mujoco/3.0.1 python/3.11 scipy-stack arrow
source ~/envs/invert/bin/activate

python main.py --config_file ./configs/${SLURM_JOB_NAME}.json --config_idx $SLURM_ARRAY_TASK_ID --slurm_dir $SLURM_TMPDIR
# python main.py --config_file ./configs/${SLURM_JOB_NAME}.json --config_idx $SLURM_ARRAY_TASK_ID

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------