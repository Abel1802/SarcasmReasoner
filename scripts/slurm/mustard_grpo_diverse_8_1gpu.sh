#!/bin/bash

#SBATCH --job-name=en_div8_grpo
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=12:00:00

#SBATCH --output=logs/mustard_div8_grpo_%j.out
#SBATCH --error=logs/mustard_div8_grpo_%j.err

#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# MUStARD++ Diverse-8-SFT -> GRPO
# ============================================================


echo "============================================================"
echo "MUStARD++ Diverse-8-SFT -> GRPO"
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
# Environment
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

TRAIN_SCRIPT="configs/mustard/grpo/diverse_8_1gpu.sh"

TRAIN_DATA="data/mustard/processed/zero_shot_train.jsonl"
VALID_DATA="data/mustard/processed/zero_shot_valid.jsonl"

PLUGIN="src/plugins/sarcasm_grpo_reward.py"

SFT_CKPT="results/mustard/sft/diverse_8/v0-20260924-103353/checkpoint-560"


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


echo "Required files and Diverse-8 checkpoint found."
echo


# ============================================================
# Checkpoint information
# ============================================================

echo "Diverse-8-SFT checkpoint:"
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
echo "Starting MUStARD++ Diverse-8-SFT -> GRPO"
echo "============================================================"
echo

echo "Policy initialization:"
echo "  Base model:  Qwen/Qwen2.5-Omni-7B"
echo "  SFT adapter: $SFT_CKPT"
echo

echo "Reference policy:"
echo "  Same Diverse-8-SFT adapter"
echo

echo "GRPO optimizer and scheduler start fresh."
echo


bash "$TRAIN_SCRIPT"

STATUS=$?


echo
echo "============================================================"
echo "MUStARD++ Diverse-8-SFT -> GRPO job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"