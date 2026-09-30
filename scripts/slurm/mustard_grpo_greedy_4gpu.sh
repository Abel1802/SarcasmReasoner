#!/bin/bash

#SBATCH --job-name=en_greedy_grpo_4gpu
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --mem=180G
#SBATCH --time=12:00:00

#SBATCH --output=logs/mustard_greedy_grpo_4gpu_%j.out
#SBATCH --error=logs/mustard_greedy_grpo_4gpu_%j.err

#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# MUStARD++ Greedy-SFT -> GRPO
#
# Initialization:
#   Qwen/Qwen2.5-Omni-7B
#   + Greedy-SFT LoRA checkpoint-105
#
# Train:
#   841 sources
#
# Validation:
#   180 sources
#   Offline checkpoint evaluation
#
# GRPO:
#   G = 8
#   1 epoch
#
# Hardware:
#   4 x H100 on one node
# ============================================================


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "MUStARD++ Greedy-SFT -> GRPO"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo "============================================================"
echo


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Project root:"
pwd
echo


# ============================================================
# Logs
# ============================================================

mkdir -p logs


# ============================================================
# Conda environment
# ============================================================

source activate ms2


echo "Python:"
which python
python --version
echo


echo "Swift / TRL / Torch:"
python - <<'PY'
import swift
import trl
import torch

print("swift:", swift.__version__)
print("trl:", trl.__version__)
print("torch:", torch.__version__)
PY

echo


# ============================================================
# GPU information
# ============================================================

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo

nvidia-smi

echo


# ============================================================
# Environment sanity check
# ============================================================

python - <<'PY'
import torch
import trl
from packaging.version import Version

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "CUDA is not available"

assert torch.cuda.device_count() == 4, (
    f"Expected exactly 4 visible GPUs, "
    f"but found {torch.cuda.device_count()}"
)

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))

assert Version(trl.__version__) >= Version("0.26"), (
    f"TRL >= 0.26 required, found {trl.__version__}"
)

print("Environment check passed.")
PY

echo


# ============================================================
# Required files / directories
# ============================================================

TRAIN_SCRIPT="configs/mustard/grpo/greedy_4gpu.sh"

TRAIN_DATA="data/mustard/processed/zero_shot_train.jsonl"
VALID_DATA="data/mustard/processed/zero_shot_valid.jsonl"

PLUGIN="src/plugins/sarcasm_grpo_reward.py"

SFT_CKPT="results/mustard/sft/greedy/v0-20260923-221644/checkpoint-105"


for FILE in \
    "$TRAIN_SCRIPT" \
    "$TRAIN_DATA" \
    "$VALID_DATA" \
    "$PLUGIN"
do
    if [ ! -f "$FILE" ]; then
        echo "ERROR: required file not found:"
        echo "  $FILE"
        exit 1
    fi
done


if [ ! -d "$SFT_CKPT" ]; then
    echo "ERROR: SFT checkpoint directory not found:"
    echo "  $SFT_CKPT"
    exit 1
fi


if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found:"
    echo "  $SFT_CKPT/adapter_config.json"
    exit 1
fi


echo "Required files and SFT checkpoint found."
echo


# ============================================================
# SFT checkpoint information
# ============================================================

echo "Greedy-SFT checkpoint:"
echo "  $SFT_CKPT"
echo

echo "Checkpoint contents:"
ls -lh "$SFT_CKPT"
echo


# ============================================================
# Dataset checks
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
VALID_SIZE=$(wc -l < "$VALID_DATA")


echo "Dataset statistics:"
echo "  train: $TRAIN_SIZE"
echo "  valid: $VALID_SIZE"
echo


if [ "$TRAIN_SIZE" -ne 841 ]; then
    echo "ERROR: expected 841 MUStARD++ training sources."
    echo "Actual: $TRAIN_SIZE"
    exit 1
fi


if [ "$VALID_SIZE" -ne 180 ]; then
    echo "ERROR: expected 180 MUStARD++ validation sources."
    echo "Actual: $VALID_SIZE"
    exit 1
fi


echo "Dataset checks passed."
echo


# ============================================================
# Run GRPO
# ============================================================

echo "============================================================"
echo "Starting formal MUStARD++ Greedy-SFT -> GRPO"
echo "============================================================"
echo
echo "Policy initialization:"
echo "  Base model:  Qwen/Qwen2.5-Omni-7B"
echo "  SFT adapter: $SFT_CKPT"
echo
echo "Reference policy:"
echo "  Same Greedy-SFT adapter"
echo
echo "GRPO optimizer and scheduler start fresh."
echo "============================================================"
echo


bash "$TRAIN_SCRIPT"

STATUS=$?


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ Greedy-SFT -> GRPO job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"
