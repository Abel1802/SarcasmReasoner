#!/bin/bash

#SBATCH --job-name=valid_all_checkpoints
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --time=5:00:00

#SBATCH --output=logs/valid_all_checkpoints_%j.out
#SBATCH --error=logs/valid_all_checkpoints_%j.err

#SBATCH --no-requeue

set -e
set -o pipefail

module load 2025
module load CUDA/12.8.0
source activate ms2


bash scripts/evaluation/eval_all_checkpoints.sh \
    mustard \
    grpo_base \
    results/mustard/grpo/base/v2-20260924-114119


bash scripts/evaluation/eval_all_checkpoints.sh \
    mustard \
    grpo_best_of_8 \
    results/mustard/grpo/best_of_8/v0-20260924-165734


bash scripts/evaluation/eval_all_checkpoints.sh \
    mustard \
    grpo_diverse_8 \
    results/mustard/grpo/diverse_8/v0-20260924-170016


bash scripts/evaluation/eval_all_checkpoints.sh \
    mustard \
    grporm_best_of_8 \
    results/mustard/grpo_rm/best_of_8/v0-20260926-151027


bash scripts/evaluation/eval_all_checkpoints.sh \
    mustard \
    grporm_diverse_8 \
    results/mustard/grpo_rm/diverse_8/v0-20260926-151025