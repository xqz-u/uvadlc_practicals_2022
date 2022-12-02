#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=resnet18_imagenet_cifar100_augmented
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=16000M
#SBATCH --output=/home/%u/job_logs/%x_%A_%a_%u.out
#SBATCH --array=0-4

module purge
module load 2021
module load Anaconda3/2021.05

# activate the environment
source activate dl2022

root="/scratch/$USER"
mkdir -p $root

code_dir="/home/$USER/uvadlc_practicals_2022/assignment2/part1"

augmentations=(rand_hflip rand_crop color_jitter all)

i=$SLURM_ARRAY_TASK_ID
aug=${augmentations[i]}
echo "Resnet18 Imagenet-1K -> CIFAR100, method: $aug"
python $code_dir/train.py --data_dir $root --augmentation_name $aug
