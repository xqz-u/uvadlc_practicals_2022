#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=cross_datasets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=16000M
#SBATCH --output=/home/%u/job_logs/%x_%A_%a_%u.out
#SBATCH --array=0-5

module purge
module load 2021
module load Anaconda3/2021.05

# activate the environment
source activate dl2022

root="/scratch/$USER"
mkdir -p $root

code_dir="/home/$USER/uvadlc_practicals_2022/assignment2/part2"

datasets=$(cifar10 cifar100)
methods=(padding fixed_patch random_patch)
models=("$code_dir/save/models/padding_30_cifar10_clip_ViT-B" \
	    "$code_dir/save/models/padding_30_cifar100_clip_ViT-B" \
	    "$code_dir/save/models/fixed_patch_1_cifar10_clip_ViT-B" \
	    "$code_dir/save/models/fixed_patch_1_cifar100_clip_ViT-B" \
	    "$code_dir/save/models/random_patch_1_cifar10_clip_ViT-B" \
	    "$code_dir/save/models/random_patch_1_cifar100_clip_ViT-B")

i=$SLURM_ARRAY_TASK_ID
model=${models[i]}
method_idx=$(( $i / 2 ))
method=${methods[method_idx]}
# shouldn't be necessary but still
dataset_idx=$(( $i / 3 ))
dataset=${datasets[dataset_idx]}

echo "CLIPVP cross-datasets, method $method model $model"
echo "$method $dataset $model"
exit 0
python cross_datasets.py --dataset $dataset \
       --method $method \
       --evaluate \
       --resume "$model/32_sgd_lr_40_decay_0_bsz_128_warmup_1000_trial_1/model_best.pth.tar"
