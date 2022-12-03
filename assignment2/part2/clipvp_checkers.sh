#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=custom_clipvp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/%u/job_logs/%x_%A_%a_%u.out
#SBATCH --array=0

module purge
module load 2021
module load Anaconda3/2021.05

# activate the environment
source activate dl2022

root="/scratch/$USER"
mkdir -p $root

code_dir="/home/$USER/uvadlc_practicals_2022/assignment2/part2"

datasets=(
    # cifar10
    cifar100
	 )

i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[i]}

echo "Training CLIPVP with checkers pattern on both CIFAR datasets..."
python $code_dir/main.py --dataset $dataset \
	    --epochs 10 \
	    --method checkers \
	    --prompt_size 2 \
	    --root "./data" \
	    --num_workers 3 \
	    --print_freq 50 \
	    --patience 5 \
	    --visualize
