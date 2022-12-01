#!/bin/bash

# TODO actual flags to avoid positional args

[ ! -z $3 ] && gpu_node_flag="-w $3"

echo "node: $gpu_node_flag time: $1 mem: $2"

srun --partition=gpu_shared_course \
	--gres=gpu:1 \
       	--mem=$2  \
	--time=$1 \
	$gpu_node_flag \
	--pty bash -i
