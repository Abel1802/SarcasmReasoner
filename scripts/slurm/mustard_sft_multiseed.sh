#!/bin/bash

#SBATCH --job-name=mustard_sft_seeds
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=06:00:00

#SBATCH --array=0-5%6

#SBATCH --output=logs/mustard_sft_seed_%A_%a.out
#SBATCH --error=logs/mustard_sft_seed_%A_%a.err

#SBATCH --no-requeue


set -e
set -o pipefail


# ============================================================
# Environment
# ============================================================

source activate ms2

cd /gpfs/work3/0/prjs1171/zhu_workspace/SarcasmReasoner


# ============================================================
# Experiment grid
#
# Existing:
#   seed 42
#
# New:
#   seed 123
#   seed 456
# ============================================================

METHODS=(
    "greedy"
    "greedy"

    "best_of_8"
    "best_of_8"

    "diverse_8"
    "diverse_8"
)

SEEDS=(
    123
    456

    123
    456

    123
    456
)

CONFIGS=(
    "configs/mustard/sft/greedy_1gpu.sh"
    "configs/mustard/sft/greedy_1gpu.sh"

    "configs/mustard/sft/best_of_8_1gpu.sh"
    "configs/mustard/sft/best_of_8_1gpu.sh"

    "configs/mustard/sft/diverse_8_1gpu.sh"
    "configs/mustard/sft/diverse_8_1gpu.sh"
)


IDX="${SLURM_ARRAY_TASK_ID}"

METHOD="${METHODS[$IDX]}"
TRAIN_SEED="${SEEDS[$IDX]}"
CONFIG="${CONFIGS[$IDX]}"

OUTPUT_DIR="results/mustard/sft/${METHOD}/seed${TRAIN_SEED}"


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ SFT multi-seed training"
echo "============================================================"

echo "Array task:      $IDX"
echo "Method:          $METHOD"
echo "Training seed:   $TRAIN_SEED"

echo
echo "Config:"
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
# Safety
#
# Do not silently create v1/v2 from accidental resubmission.
# ============================================================

if [ -d "$OUTPUT_DIR" ] \
   && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]
then
    echo "ERROR: output directory already contains files:"
    echo "  $OUTPUT_DIR"
    echo
    echo "Refusing to overwrite / create another version."
    exit 1
fi


mkdir -p "$OUTPUT_DIR"


if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config missing:"
    echo "  $CONFIG"
    exit 1
fi


# ============================================================
# Train
#
# Both --seed and --data_seed inside the config inherit SEED.
# All other hyperparameters remain unchanged.
# ============================================================

SEED="$TRAIN_SEED" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash "$CONFIG"


# ============================================================
# Validate final checkpoint
# ============================================================

case "$METHOD" in

    greedy)
        EXPECTED_STEP=105
        ;;

    best_of_8)
        EXPECTED_STEP=144
        ;;

    diverse_8)
        EXPECTED_STEP=834
        ;;

    *)
        echo "ERROR: unknown method: $METHOD"
        exit 1
        ;;
esac


FINAL_CKPT=$(find "$OUTPUT_DIR" \
    -type d \
    -name "checkpoint-${EXPECTED_STEP}" \
    -print \
    | head -n 1)


if [ -z "$FINAL_CKPT" ]; then

    echo
    echo "ERROR: expected final checkpoint not found:"
    echo "  checkpoint-${EXPECTED_STEP}"
    echo
    echo "Under:"
    echo "  $OUTPUT_DIR"

    exit 1
fi


echo
echo "============================================================"
echo "SFT seed finished successfully"
echo "============================================================"

echo "Method:        $METHOD"
echo "Seed:          $TRAIN_SEED"

echo
echo "Final checkpoint:"
echo "  $FINAL_CKPT"

echo "============================================================"
