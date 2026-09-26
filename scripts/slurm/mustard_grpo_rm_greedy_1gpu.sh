#!/bin/bash

#SBATCH --job-name=en_greedy_grpo_rm
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=12:00:00

#SBATCH --output=logs/mustard_greedy_grpo_rm_%j.out
#SBATCH --error=logs/mustard_greedy_grpo_rm_%j.err

#SBATCH --no-requeue

set -e

source activate ms2

bash configs/mustard/grpo_rm/greedy_1gpu.sh
