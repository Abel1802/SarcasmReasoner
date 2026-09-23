#!/bin/bash

#SBATCH --job-name=mcsd_n8
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --time=24:00:00

#SBATCH --output=logs/mcsd_n8_%j.out
#SBATCH --error=logs/mcsd_n8_%j.err

# Do not automatically requeue.
# The inner sampling script supports resume.
#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "MCSD N=8 Teacher Sampling"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "Visible GPUs:  ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo "============================================================"
echo


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT="/gpfs/work3/0/prjs0864/phd_projects/SarcasmReasoner"

cd "$PROJECT_ROOT"

echo "Project root:"
pwd
echo


# ============================================================
# Conda environment
# ============================================================

source activate ms2

echo "Python:"
which python
echo

python --version
echo


# ============================================================
# Runtime environment
# ============================================================

export PYTHONUNBUFFERED=1

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
import vllm

print("torch:", torch.__version__)
print("swift:", swift.__version__)
print("vllm:", vllm.__version__)
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

if [ ! -f "src/generation/swift_sample_local_video.py" ]; then
    echo "ERROR: sampling wrapper missing:"
    echo "  src/generation/swift_sample_local_video.py"
    exit 1
fi

if [ ! -f "configs/mcsd/teacher/qwen3_omni_30b_sample_n8.sh" ]; then
    echo "ERROR: MCSD sampling script missing:"
    echo "  configs/mcsd/teacher/qwen3_omni_30b_sample_n8.sh"
    exit 1
fi


# ============================================================
# Dataset sanity check
# ============================================================

echo "Checking MCSD zero-shot datasets..."

wc -l \
    data/mcsd/processed/zero_shot_valid.jsonl \
    data/mcsd/processed/zero_shot_test.jsonl \
    data/mcsd/processed/zero_shot_train.jsonl

echo


# ============================================================
# Run MCSD sampling
#
# Inner script:
#
#   valid -> test -> train
#
# Expected raw trajectories:
#
#   valid:  406 x 8 = 3248
#   test:   406 x 8 = 3248
#   train: 1893 x 8 = 15144
#
#   total = 21640
#
# Resume is handled by ms-swift checkpoint files.
# ============================================================

echo
echo "============================================================"
echo "Starting MCSD N=8 sampling"
echo "============================================================"
echo


bash configs/mcsd/teacher/qwen3_omni_30b_sample_n8.sh


STATUS=$?


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "MCSD sampling job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"
