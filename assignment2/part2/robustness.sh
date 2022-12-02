#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=robustness_CLIPVP_padding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=16000M
#SBATCH --output=/home/%u/job_logs/%x_%A_%a_%u.out
#SBATCH --array=0-1

module purge
module load 2021
module load Anaconda3/2021.05

# activate the environment
source activate dl2022

root="/scratch/$USER"
mkdir -p $root

code_dir="/home/$USER/uvadlc_practicals_2022/assignment2/part2"

datasets=(cifar10 cifar100)
models=("$code_dir/save/models/padding_30_cifar10_clip_ViT-B" \
	    "$code_dir/save/models/padding_30_cifar100_clip_ViT-B")

i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[i]}
model=${models[i]}

echo "CLIPVP padding distributional shift on $dataset"
python robustness.py --dataset $dataset \
       --method padding \
       --test_noise \
       --evaluate \
       --resume "$model/32_sgd_lr_40_decay_0_bsz_128_warmup_1000_trial_1/model_best.pth.tar"
