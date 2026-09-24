#!/bin/bash

#SBATCH --job-name=mustard_best8_sft
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=04:00:00

#SBATCH --output=logs/mustard_best8_sft_%j.out
#SBATCH --error=logs/mustard_best8_sft_%j.err

#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "MUStARD++ Best-of-8 SFT"
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


echo "Swift:"
python - <<'PY'
import swift
print(swift.__version__)
PY

echo


# ============================================================
# GPU information
# ============================================================

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo

nvidia-smi

echo


# ============================================================
# Environment sanity check
# ============================================================

echo "Checking environment..."

python - <<'PY'
import torch
import swift

print("torch:", torch.__version__)
print("swift:", swift.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))

assert torch.cuda.device_count() == 1, (
    f"Expected exactly 1 visible GPU, "
    f"but found {torch.cuda.device_count()}"
)
PY

echo
echo "Environment check passed."
echo


# ============================================================
# Required files
# ============================================================

TRAIN_SCRIPT="configs/mustard/sft/best_of_8_1gpu.sh"

TRAIN_DATA="data/mustard/processed/sft/sft_best_of_8_train.jsonl"
VALID_DATA="data/mustard/processed/sft/sft_best_of_8_valid.jsonl"


if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: training script not found:"
    echo "  $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: training data not found:"
    echo "  $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VALID_DATA" ]; then
    echo "ERROR: validation data not found:"
    echo "  $VALID_DATA"
    exit 1
fi


echo "Required files found."
echo


# ============================================================
# Dataset statistics
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
VALID_SIZE=$(wc -l < "$VALID_DATA")

echo "Best-of-8 dataset:"
echo "  train: $TRAIN_SIZE"
echo "  valid: $VALID_SIZE"
echo


if [ "$TRAIN_SIZE" -ne 757 ]; then
    echo "ERROR: expected 757 training examples, got $TRAIN_SIZE"
    exit 1
fi

if [ "$VALID_SIZE" -ne 163 ]; then
    echo "ERROR: expected 163 validation examples, got $VALID_SIZE"
    exit 1
fi


# ============================================================
# Run training
# ============================================================

echo
echo "============================================================"
echo "Starting MUStARD++ Best-of-8 SFT"
echo "============================================================"
echo


bash "$TRAIN_SCRIPT"

STATUS=$?


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ Best-of-8 SFT job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"