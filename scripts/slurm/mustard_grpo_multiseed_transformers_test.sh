#!/bin/bash

#SBATCH --job-name=grpo_seed_tf_test
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=02:00:00

# 5 checkpoints, each task uses one H100.
#SBATCH --array=0-4%5

#SBATCH --output=logs/grpo_seed_tf_test_%A_%a.out
#SBATCH --error=logs/grpo_seed_tf_test_%A_%a.err

#SBATCH --no-requeue


set -euo pipefail


# ============================================================
# Environment
# ============================================================

source activate ms2

cd /gpfs/work3/0/prjs1171/zhu_workspace/SarcasmReasoner


# ============================================================
# MUStARD++ GRPO multi-seed final checkpoints
#
# Formal TEST evaluation:
#
#   backend        = Transformers
#   temperature    = 0
#   max batch size = 1
#   inference seed = 42
#
# Training seeds:
#
#   Greedy:
#       123
#       456
#
#   Diverse-8:
#       42
#       123
#       456
#
# ============================================================


METHODS=(
    "greedy"
    "greedy"

    "diverse_8"
    "diverse_8"
    "diverse_8"
)


TRAIN_SEEDS=(
    123
    456

    42
    123
    456
)


VARIANTS=(
    "grpo_greedy"
    "grpo_greedy"

    "grpo_diverse_8"
    "grpo_diverse_8"
    "grpo_diverse_8"
)


CHECKPOINTS=(
    "results/mustard/grpo/greedy/seed123/v0-20260928-114059/checkpoint-841"
    "results/mustard/grpo/greedy/seed456/v0-20260928-114058/checkpoint-841"

    "results/mustard/grpo/diverse_8/seed42/v0-20260928-114059/checkpoint-841"
    "results/mustard/grpo/diverse_8/seed123/v0-20260928-114058/checkpoint-841"
    "results/mustard/grpo/diverse_8/seed456/v0-20260928-114058/checkpoint-841"
)


# ============================================================
# Resolve array task
# ============================================================

IDX="${SLURM_ARRAY_TASK_ID}"

METHOD="${METHODS[$IDX]}"
TRAIN_SEED="${TRAIN_SEEDS[$IDX]}"
VARIANT="${VARIANTS[$IDX]}"
CHECKPOINT="${CHECKPOINTS[$IDX]}"


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ GRPO Multi-seed Transformers TEST"
echo "============================================================"

echo "SLURM job:       ${SLURM_JOB_ID}"
echo "Array task:      ${SLURM_ARRAY_TASK_ID}"

echo
echo "Method:          ${METHOD}"
echo "Training seed:   ${TRAIN_SEED}"
echo "Variant:         ${VARIANT}"

echo
echo "Checkpoint:"
echo "  ${CHECKPOINT}"

echo
echo "Host:"
echo "  $(hostname)"

echo
echo "CUDA_VISIBLE_DEVICES:"
echo "  ${CUDA_VISIBLE_DEVICES:-not-set}"

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


EVALUATOR="scripts/evaluation/eval_checkpoint_transformers.sh"


if [ ! -x "$EVALUATOR" ]; then
    echo "ERROR: Transformers evaluator missing or not executable:"
    echo "  $EVALUATOR"
    exit 1
fi


# ============================================================
# GPU information
# ============================================================

nvidia-smi

echo


# ============================================================
# Formal deterministic TEST
# ============================================================

bash "$EVALUATOR" \
    mustard \
    "$VARIANT" \
    "$CHECKPOINT" \
    test


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "TRANSFORMERS TEST FINISHED"
echo "============================================================"

echo "Method:        $METHOD"
echo "Training seed: $TRAIN_SEED"

echo
echo "Checkpoint:"
echo "  $CHECKPOINT"

echo "============================================================"
