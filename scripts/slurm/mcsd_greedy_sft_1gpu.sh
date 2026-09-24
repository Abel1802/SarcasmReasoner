#!/bin/bash

#SBATCH --job-name=mcsd_greedy_sft
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=04:00:00

#SBATCH --output=logs/mcsd_greedy_sft_%j.out
#SBATCH --error=logs/mcsd_greedy_sft_%j.err

#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "MCSD Greedy-SFT"
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
# CUDA / GPU information
# ============================================================

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo

nvidia-smi

echo


# ============================================================
# Sanity checks
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

TRAIN_SCRIPT="configs/mcsd/sft/greedy_1gpu.sh"
SAFE_VIDEO_LAUNCHER="src/training/swift_sft_safe_video.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: training script not found:"
    echo "  $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$SAFE_VIDEO_LAUNCHER" ]; then
    echo "ERROR: safe video launcher not found:"
    echo "  $SAFE_VIDEO_LAUNCHER"
    exit 1
fi


# ============================================================
# Run training
# ============================================================

echo "============================================================"
echo "Starting MCSD Greedy-SFT"
echo "============================================================"
echo


bash "$TRAIN_SCRIPT"

STATUS=$?


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "MCSD Greedy-SFT job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"