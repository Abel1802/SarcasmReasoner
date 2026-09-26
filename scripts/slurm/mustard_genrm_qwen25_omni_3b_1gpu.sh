#!/bin/bash

#SBATCH --job-name=mustard_genrm_3b
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=12:00:00

#SBATCH --output=logs/mustard_genrm_3b_%j.out
#SBATCH --error=logs/mustard_genrm_3b_%j.err

#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# MUStARD++ Qwen2.5-Omni-3B GenRM SFT
# ============================================================

echo "============================================================"
echo "MUStARD++ Qwen2.5-Omni-3B GenRM SFT"
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
# Environment
# ============================================================

source activate ms2

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


echo "Python:"
which python
python --version
echo

echo "Swift / Torch:"
python - <<'PY'
import swift
import torch

print("swift:", swift.__version__)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "CUDA is not available"

assert torch.cuda.device_count() == 1, (
    f"Expected exactly 1 visible GPU, "
    f"but found {torch.cuda.device_count()}"
)

print("GPU:", torch.cuda.get_device_name(0))
PY

echo


# ============================================================
# GPU information
# ============================================================

nvidia-smi
echo


# ============================================================
# Required files
# ============================================================

TRAIN_SCRIPT="configs/mustard/genrm/qwen25_omni_3b_1gpu.sh"

TRAIN_DATA="data/mustard/processed/genrm/sft/genrm_train.jsonl"
VALID_DATA="data/mustard/processed/genrm/sft/genrm_valid.jsonl"


for FILE in \
    "$TRAIN_SCRIPT" \
    "$TRAIN_DATA" \
    "$VALID_DATA"
do
    if [ ! -f "$FILE" ]; then
        echo "ERROR: required file not found:"
        echo "  $FILE"
        exit 1
    fi
done

echo "Required files found."
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


if [ "$TRAIN_SIZE" -ne 6448 ]; then
    echo "ERROR: expected 6448 GenRM training examples."
    echo "Actual: $TRAIN_SIZE"
    exit 1
fi

if [ "$VALID_SIZE" -ne 1373 ]; then
    echo "ERROR: expected 1373 GenRM validation examples."
    echo "Actual: $VALID_SIZE"
    exit 1
fi

echo "Dataset checks passed."
echo


# ============================================================
# Run full GenRM SFT
# ============================================================

echo "============================================================"
echo "Starting MUStARD++ GenRM full training"
echo "============================================================"
echo
echo "Model:       Qwen/Qwen2.5-Omni-3B"
echo "Train size:  6448"
echo "Valid size:  1373"
echo "GPU:         1 x H100"
echo "Mode:        full"
echo
echo "============================================================"
echo


bash "$TRAIN_SCRIPT" full

STATUS=$?


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ GenRM SFT job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"
