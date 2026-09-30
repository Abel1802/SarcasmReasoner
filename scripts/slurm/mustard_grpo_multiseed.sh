#!/bin/bash

#SBATCH --job-name=mustard_grpo_seeds
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=08:00:00

#SBATCH --array=0-4%5

#SBATCH --output=logs/mustard_grpo_seed_%A_%a.out
#SBATCH --error=logs/mustard_grpo_seed_%A_%a.err

#SBATCH --no-requeue


set -euo pipefail


# ============================================================
# Environment
# ============================================================

source activate ms2

cd /gpfs/work3/0/prjs1171/zhu_workspace/SarcasmReasoner


# ============================================================
# Paired SFT -> GRPO multi-seed experiment
#
# Existing:
#
#   Greedy seed42:
#     already trained
#
# New runs:
#
#   Greedy:
#     seed123
#     seed456
#
#   Diverse:
#     seed42
#     seed123
#     seed456
#
# IMPORTANT:
#   Each GRPO run starts from the SFT final checkpoint trained
#   with the SAME training seed.
# ============================================================


METHODS=(
    "greedy"
    "greedy"

    "diverse_8"
    "diverse_8"
    "diverse_8"
)


SEEDS=(
    123
    456

    42
    123
    456
)


CONFIGS=(
    "configs/mustard/grpo/greedy_1gpu.sh"
    "configs/mustard/grpo/greedy_1gpu.sh"

    "configs/mustard/grpo/diverse_8_1gpu.sh"
    "configs/mustard/grpo/diverse_8_1gpu.sh"
    "configs/mustard/grpo/diverse_8_1gpu.sh"
)


SFT_CKPTS=(
    "results/mustard/sft/greedy/seed123/v0-20260927-200013/checkpoint-105"
    "results/mustard/sft/greedy/seed456/v0-20260927-200011/checkpoint-105"

    "results/mustard/sft/diverse_8/v0-20260924-103353/checkpoint-834"
    "results/mustard/sft/diverse_8/seed123/v0-20260927-200010/checkpoint-834"
    "results/mustard/sft/diverse_8/seed456/v0-20260927-200009/checkpoint-834"
)


# ============================================================
# Resolve array task
# ============================================================

IDX="${SLURM_ARRAY_TASK_ID}"

METHOD="${METHODS[$IDX]}"
TRAIN_SEED="${SEEDS[$IDX]}"
CONFIG="${CONFIGS[$IDX]}"
SFT_CKPT="${SFT_CKPTS[$IDX]}"

OUTPUT_DIR="results/mustard/grpo/${METHOD}/seed${TRAIN_SEED}"


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ GRPO multi-seed"
echo "============================================================"

echo "Array task:      $IDX"
echo "Method:          $METHOD"
echo "Training seed:   $TRAIN_SEED"

echo
echo "SFT initialization:"
echo "  $SFT_CKPT"

echo
echo "GRPO config:"
echo "  $CONFIG"

echo
echo "Output root:"
echo "  $OUTPUT_DIR"

echo
echo "Host:"
echo "  $(hostname)"

echo "============================================================"
echo


# ============================================================
# Sanity checks
# ============================================================

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config missing:"
    echo "  $CONFIG"
    exit 1
fi


if [ ! -d "$SFT_CKPT" ]; then
    echo "ERROR: SFT checkpoint missing:"
    echo "  $SFT_CKPT"
    exit 1
fi


if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json missing:"
    echo "  $SFT_CKPT/adapter_config.json"
    exit 1
fi


if [ -d "$OUTPUT_DIR" ] \
   && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]
then
    echo "ERROR: output directory already contains files:"
    echo "  $OUTPUT_DIR"
    echo
    echo "Refusing accidental duplicate training."
    exit 1
fi


mkdir -p "$OUTPUT_DIR"


# ============================================================
# GRPO
#
# Changes across runs:
#
#   1. SFT initialization
#   2. training seed
#
# Everything else is identical:
#
#   epoch             = 1
#   G                 = 8
#   LR                = 1e-5
#   beta              = 0.04
#   reward            = accuracy + 0.2 format
#   temperature       = 1.0
#   top-p             = 0.95
# ============================================================

SEED="$TRAIN_SEED" \
SFT_CKPT="$SFT_CKPT" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash "$CONFIG"


# ============================================================
# Final checkpoint check
# ============================================================

FINAL_CKPT=$(find "$OUTPUT_DIR" \
    -type d \
    -name "checkpoint-841" \
    -print \
    | head -n 1)


if [ -z "$FINAL_CKPT" ]; then

    echo
    echo "ERROR: final checkpoint-841 not found under:"
    echo "  $OUTPUT_DIR"

    exit 1
fi


echo
echo "============================================================"
echo "GRPO seed finished"
echo "============================================================"

echo "Method:       $METHOD"
echo "Seed:         $TRAIN_SEED"

echo
echo "SFT init:"
echo "  $SFT_CKPT"

echo
echo "Final GRPO checkpoint:"
echo "  $FINAL_CKPT"

echo "============================================================"
