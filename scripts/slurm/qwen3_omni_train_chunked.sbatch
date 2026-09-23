#!/bin/bash
#SBATCH --job-name=qwen3_teacher_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=results/mcsd/teacher/logs/slurm_train_%j.out
#SBATCH --error=results/mcsd/teacher/logs/slurm_train_%j.err

set -euo pipefail

cd /scratch/p311104/SarcasmReasoner

mkdir -p results/mcsd/teacher/logs

module load Miniconda3
source activate ms2

echo "============================================================"
echo "SLURM JOB INFO"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo

# ------------------------------------------------------------
# Local cache on compute node
# ------------------------------------------------------------

JOB_TMP="${TMPDIR:-/tmp}/${USER}/${SLURM_JOB_ID}"

mkdir -p "$JOB_TMP/modelscope"
mkdir -p "$JOB_TMP/huggingface"

export MODELSCOPE_CACHE="$JOB_TMP/modelscope"
export HF_HOME="$JOB_TMP/huggingface"

echo "JOB_TMP:          $JOB_TMP"
echo "MODELSCOPE_CACHE: $MODELSCOPE_CACHE"
echo "HF_HOME:          $HF_HOME"
df -h "${TMPDIR:-/tmp}"
echo

# ------------------------------------------------------------
# Multimodal / CPU settings
# ------------------------------------------------------------

export OMP_NUM_THREADS=8
export USE_AUDIO_IN_VIDEO=False
export FORCE_QWENVL_VIDEO_READER=decord
export FPS_MAX_FRAMES=12

# ------------------------------------------------------------
# GPU check
# ------------------------------------------------------------

echo "============================================================"
echo "GPU INFO"
echo "============================================================"
nvidia-smi
echo

# ------------------------------------------------------------
# Run chunked train generation
# ------------------------------------------------------------

echo "============================================================"
echo "Starting chunked train inference"
echo "============================================================"

bash configs/mcsd/teacher/qwen3_omni_30b_train_chunked.sh

STATUS=$?

echo
echo "============================================================"
echo "JOB FINISHED"
echo "============================================================"
echo "Exit code: $STATUS"
echo "End time:  $(date)"

if [ -f results/mcsd/teacher/qwen3_omni_30b_train.jsonl ]; then
    echo
    wc -l results/mcsd/teacher/qwen3_omni_30b_train.jsonl
fi

exit "$STATUS"
