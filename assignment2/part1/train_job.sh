#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=resnet18_imagenet_cifar100_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --mem=32000M
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

module purge
module load 2021
module load Anaconda3/2021.05

# activate the environment
source activate dl2022

root="/scratch/$USER"
#mkdir -p $root

code_dir="/home/$USER/uvadlc_practicals_2022/assignment2/part1"

augmentations=(rand_hflip rand_crop color_jitter all)

if [ -z $1 ]; then
    echo "Resnet18 Imagenet-1K -> CIFAR100"
    python $code_dir/train.py --data_dir $root
else
    for aug in "${augmentations[@]}"; do
	echo "Resnet18 Imagenet-1K -> CIFAR100, method: $aug"
	python $code_dir/train.py --data_dir $root --augmentation_name $aug
    done
fi
