#!/bin/bash

#SBATCH --job-name=mustard_n8
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --time=24:00:00

#SBATCH --output=logs/mustard_n8_%j.out
#SBATCH --error=logs/mustard_n8_%j.err

# Do not automatically requeue a partially completed job.
# The sampling script itself supports resume.
#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "MUStARD++ N=8 Teacher Sampling"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Working dir:   ${SLURM_SUBMIT_DIR}"
echo "============================================================"
echo


# ============================================================
# Move to project root
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
# Environment information
# ============================================================

export PYTHONUNBUFFERED=1

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
    echo "ERROR: sampling wrapper missing."
    exit 1
fi

if [ ! -f "configs/mustard/teacher/qwen3_omni_30b_sample_n8.sh" ]; then
    echo "ERROR: sampling script missing."
    exit 1
fi


# ============================================================
# Run sampling
#
# The inner script performs:
#
#   valid -> test -> train
#
# and supports resume through ms-swift checkpoint files.
# ============================================================

echo
echo "============================================================"
echo "Starting MUStARD++ sampling"
echo "============================================================"
echo


bash configs/mustard/teacher/qwen3_omni_30b_sample_n8.sh


STATUS=$?


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "Job finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"
