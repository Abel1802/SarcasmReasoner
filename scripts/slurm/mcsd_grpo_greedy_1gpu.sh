#!/bin/bash

#SBATCH --job-name=mcsd_greedy_grpo
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=406G
#SBATCH --time=24:00:00

#SBATCH --output=logs/mcsd_greedy_grpo_%j.out
#SBATCH --error=logs/mcsd_greedy_grpo_%j.err

#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# MCSD Greedy-SFT -> GRPO
#
# Initialization:
#   Qwen/Qwen2.5-Omni-7B
#   + Greedy-SFT LoRA checkpoint-228
#
# Train:
#   1893 sources
#
# Validation:
#   406 sources
#   GRPO reward evaluation every 1000 steps
#
# GRPO:
#   G = 8
#   1 epoch
#
# Hardware:
#   1 x H100
# ============================================================


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "MCSD Greedy-SFT -> GRPO"
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

PROJECT_ROOT="/gpfs/work3/0/prjs1171/zhu_workspace/SarcasmReasoner"

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

assert torch.cuda.device_count() == 1, (
    f"Expected exactly 1 visible GPU, "
    f"but found {torch.cuda.device_count()}"
)

print("GPU:", torch.cuda.get_device_name(0))

assert Version(trl.__version__) >= Version("0.26"), (
    f"TRL >= 0.26 required, found {trl.__version__}"
)

print("Environment check passed.")
PY

echo


# ============================================================
# Required files / directories
# ============================================================

TRAIN_SCRIPT="configs/mcsd/grpo/greedy_1gpu.sh"

TRAIN_DATA="data/mcsd/processed/zero_shot_train.jsonl"
VALID_DATA="data/mcsd/processed/zero_shot_valid.jsonl"

PLUGIN="src/plugins/sarcasm_grpo_reward.py"

SFT_CKPT="results/mcsd/sft/greedy/v6-20260924-101848/checkpoint-228"


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


if [ "$TRAIN_SIZE" -ne 1893 ]; then
    echo "ERROR: expected 1893 MCSD training sources."
    echo "Actual: $TRAIN_SIZE"
    exit 1
fi


if [ "$VALID_SIZE" -ne 406 ]; then
    echo "ERROR: expected 406 MCSD validation sources."
    echo "Actual: $VALID_SIZE"
    exit 1
fi


echo "Dataset checks passed."
echo


# ============================================================
# Run GRPO
# ============================================================

echo "============================================================"
echo "Starting MCSD Greedy-SFT -> GRPO"
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
echo "MCSD Greedy-SFT -> GRPO job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"
