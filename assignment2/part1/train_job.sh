#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=resnet18_imagenet_cifar100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

module purge
module load 2021
module load Anaconda3/2021.05

# activate the environment
source activate dl2022

root="/scratch/$USER"
mkdir -p $root

code_dir="/home/$USER/uvadlc_practicals_2022/assignment2/part1"

time (python $code_dir/train.py --data_dir $root)
