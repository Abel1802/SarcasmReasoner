#!/bin/bash

#SBATCH --job-name=valid_all_checkpoints
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --time=2:00:00

#SBATCH --output=logs/valid_all_checkpoints_%j.out
#SBATCH --error=logs/valid_all_checkpoints_%j.err

#SBATCH --no-requeue

set -e
set -o pipefail

module load 2025
module load CUDA/12.8.0
source activate ms2


# bash scripts/evaluation/eval_all_checkpoints.sh \
#     mustard \
#     grpo_greedy \
#     results/mustard/grpo/greedy/v0-20260924-122118


# bash scripts/evaluation/eval_all_checkpoints.sh \
#     mustard \
#     grpo_best_of_8 \
#     results/mustard/grpo/best_of_8/v0-20260924-165734


# bash scripts/evaluation/eval_all_checkpoints.sh \
#     mustard \
#     grpo_diverse_8 \
#     results/mustard/grpo/diverse_8/v0-20260924-170016


bash scripts/evaluation/eval_checkpoint.sh \
    mustard \
    grpo_greedy \
    results/mustard/grpo/greedy/v0-20260924-122118/checkpoint-200 \
    test


bash scripts/evaluation/eval_checkpoint.sh \
    mustard \
    grpo_best_of_8 \
    results/mustard/grpo/best_of_8/v0-20260924-165734/checkpoint-600 \
    test


bash scripts/evaluation/eval_checkpoint.sh \
    mustard \
    grpo_diverse_8 \
    results/mustard/grpo/diverse_8/v0-20260924-170016/checkpoint-841 \
    test



bash scripts/evaluation/eval_checkpoint.sh \
    mustard \
    grporm_greedy \
    results/mustard/grpo_rm/greedy/v0-20260926-125827/checkpoint-800 \
    test


bash scripts/evaluation/eval_checkpoint.sh \
    mustard \
    grporm_best_of_8 \
    results/mustard/grpo_rm/best_of_8/v0-20260926-151027/checkpoint-841 \
    test


bash scripts/evaluation/eval_checkpoint.sh \
    mustard \
    grporm_diverse_8 \
    results/mustard/grpo_rm/diverse_8/v0-20260926-151025/checkpoint-200 \
    test