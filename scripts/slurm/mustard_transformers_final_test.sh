#!/bin/bash

#SBATCH --job-name=mustard_tf_test
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=02:00:00

# 10 checkpoints, maximum 4 running simultaneously.
#
# Change %4 -> %10 if you want all 10 submitted concurrently
# and your allocation / queue policy permits it.
#SBATCH --array=0-9%10

#SBATCH --output=logs/mustard_tf_test_%A_%a.out
#SBATCH --error=logs/mustard_tf_test_%A_%a.err

#SBATCH --no-requeue

set -euo pipefail


# ============================================================
# Environment
# ============================================================

source activate ms2

cd /gpfs/work3/0/prjs1171/zhu_workspace/SarcasmReasoner


# ============================================================
# Formal MUStARD++ checkpoints
#
# IMPORTANT:
#   Each array task evaluates exactly one checkpoint.
#
#   Formal evaluator:
#       Transformers
#       temperature = 0
#       batch size = 1
#       inference seed = 42
#
# ============================================================

VARIANTS=(
    "sft_greedy"
    "sft_best_of_8"
    "sft_diverse_8"

    "grpo_base"
    "grpo_greedy"
    "grpo_best_of_8"
    "grpo_diverse_8"

    "grporm_greedy"
    "grporm_best_of_8"
    "grporm_diverse_8"
)


CHECKPOINTS=(
    "results/mustard/sft/greedy/v0-20260923-221644/checkpoint-105"
    "results/mustard/sft/best_of_8/v0-20260924-103030/checkpoint-144"
    "results/mustard/sft/diverse_8/v0-20260924-103353/checkpoint-834"

    "results/mustard/grpo/base/v2-20260924-114119/checkpoint-841"
    "results/mustard/grpo/greedy/v0-20260924-122118/checkpoint-841"
    "results/mustard/grpo/best_of_8/v0-20260924-165734/checkpoint-841"
    "results/mustard/grpo/diverse_8/v0-20260924-170016/checkpoint-841"

    "results/mustard/grpo_rm/greedy/v0-20260926-125827/checkpoint-841"
    "results/mustard/grpo_rm/best_of_8/v0-20260926-151027/checkpoint-841"
    "results/mustard/grpo_rm/diverse_8/v0-20260926-151025/checkpoint-841"
)


# ============================================================
# Resolve this array task
# ============================================================

IDX="${SLURM_ARRAY_TASK_ID}"

VARIANT="${VARIANTS[$IDX]}"
CHECKPOINT="${CHECKPOINTS[$IDX]}"


echo
echo "============================================================"
echo "MUStARD++ Formal Transformers Test"
echo "============================================================"

echo "SLURM job:       ${SLURM_JOB_ID}"
echo "Array task:      ${SLURM_ARRAY_TASK_ID}"

echo
echo "Variant:"
echo "  ${VARIANT}"

echo
echo "Checkpoint:"
echo "  ${CHECKPOINT}"

echo
echo "Host:"
echo "  $(hostname)"

echo "============================================================"
echo


# ============================================================
# Sanity checks
# ============================================================

if [ ! -d "$CHECKPOINT" ]; then

    echo "ERROR: checkpoint does not exist:"
    echo "  $CHECKPOINT"

    exit 1
fi


if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then

    echo "ERROR: adapter_config.json missing:"
    echo "  $CHECKPOINT/adapter_config.json"

    exit 1
fi


if [ ! -x scripts/evaluation/eval_checkpoint_transformers.sh ]; then

    echo "ERROR: formal Transformers evaluator missing:"
    echo "  scripts/evaluation/eval_checkpoint_transformers.sh"

    exit 1
fi


# ============================================================
# Formal TEST evaluation
# ============================================================

bash scripts/evaluation/eval_checkpoint_transformers.sh \
    mustard \
    "$VARIANT" \
    "$CHECKPOINT" \
    test


echo
echo "============================================================"
echo "ARRAY TASK FINISHED"
echo "============================================================"

echo "Variant:"
echo "  $VARIANT"

echo
echo "Checkpoint:"
echo "  $CHECKPOINT"

echo "============================================================"
