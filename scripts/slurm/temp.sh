#!/bin/bash

#SBATCH --job-name=temp
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --time=12:00:00

#SBATCH --output=logs/temp%j.out
#SBATCH --error=logs/temp%j.err

#SBATCH --no-requeue

set -e
set -o pipefail

module load 2025
module load CUDA/12.8.0
source activate ms2

bash configs/mustard/grpo_rm/greedy_linear_1gpu.sh