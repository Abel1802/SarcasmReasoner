#!/bin/bash

set -euo pipefail


# ============================================================
# Formal deterministic checkpoint evaluation
#
# Backend:
#   Transformers
#
# Intended use:
#   Final validation / test results for the paper.
#
# Variant format:
#
#   sft_<method>
#   grpo_<method>
#   grporm_<method>
#
# Usage:
#
# bash scripts/evaluation/eval_checkpoint_transformers.sh \
#     <dataset> \
#     <variant> \
#     <adapter_path> \
#     <split> [split ...]
#
# Example:
#
# bash scripts/evaluation/eval_checkpoint_transformers.sh \
#     mustard \
#     grporm_greedy \
#     results/mustard/grpo_rm/greedy/v0-20260926-125827/checkpoint-800 \
#     test
#
# IMPORTANT:
#   split must be explicitly supplied.
#   This avoids accidentally evaluating TEST.
# ============================================================


# ============================================================
# Arguments
# ============================================================

if [ "$#" -lt 4 ]; then

    echo "Usage:"
    echo
    echo "bash $0 \\"
    echo "  <dataset> \\"
    echo "  <variant> \\"
    echo "  <adapter_path> \\"
    echo "  <split> [split ...]"
    echo
    echo "Variant:"
    echo "  sft_<method>"
    echo "  grpo_<method>"
    echo "  grporm_<method>"
    echo

    exit 1
fi


DATASET="$1"
VARIANT="$2"
ADAPTER="$3"

shift 3

SPLITS=("$@")


# ============================================================
# Validate splits
# ============================================================

for SPLIT in "${SPLITS[@]}"; do

    case "$SPLIT" in
        valid|test)
            ;;
        *)
            echo "ERROR: unsupported split:"
            echo "  $SPLIT"
            echo
            echo "Allowed:"
            echo "  valid"
            echo "  test"
            exit 1
            ;;
    esac

done


# ============================================================
# Parse variant
# ============================================================

case "$VARIANT" in

    sft_*)
        FAMILY="sft"
        METHOD="${VARIANT#sft_}"
        ;;

    grpo_*)
        FAMILY="grpo"
        METHOD="${VARIANT#grpo_}"
        ;;

    grporm_*)
        FAMILY="grpo_rm"
        METHOD="${VARIANT#grporm_}"
        ;;

    *)
        echo "ERROR: invalid variant:"
        echo "  $VARIANT"
        echo
        echo "Expected:"
        echo "  sft_<method>"
        echo "  grpo_<method>"
        echo "  grporm_<method>"
        exit 1
        ;;

esac


if [ -z "$METHOD" ]; then
    echo "ERROR: method cannot be empty."
    exit 1
fi


# ============================================================
# Formal evaluation configuration
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"

SAFE_INFER="src/evaluation/swift_infer_safe_video.py"

METRIC_SCRIPT="src/evaluation/evaluate_sarcasm_predictions.py"


# Greedy deterministic decoding.
TEMPERATURE=0

MAX_NEW_TOKENS=4096

INFER_SEED=42


# Transformers processes one example at a time.
MAX_BATCH_SIZE=1


# ============================================================
# Multimodal environment
# ============================================================

export ENABLE_AUDIO_OUTPUT=0

export USE_AUDIO_IN_VIDEO=False

export FPS_MAX_FRAMES=12

export VIDEO_MAX_PIXELS=50176

export MAX_PIXELS=1003520

export FORCE_QWENVL_VIDEO_READER=decord

export TOKENIZERS_PARALLELISM=false

export PYTHONUNBUFFERED=1


# ============================================================
# Reproducibility environment
# ============================================================

export PYTHONHASHSEED="$INFER_SEED"

export OMP_NUM_THREADS=1

export MKL_NUM_THREADS=1

export NUMEXPR_NUM_THREADS=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8


# ============================================================
# Validate files
# ============================================================

if [ ! -d "$ADAPTER" ]; then

    echo "ERROR: adapter directory not found:"
    echo "  $ADAPTER"

    exit 1
fi


if [ ! -f "$ADAPTER/adapter_config.json" ]; then

    echo "ERROR: adapter_config.json not found:"
    echo "  $ADAPTER/adapter_config.json"

    exit 1
fi


if [ ! -f "$SAFE_INFER" ]; then

    echo "ERROR: safe inference wrapper not found:"
    echo "  $SAFE_INFER"

    exit 1
fi


if [ ! -f "$METRIC_SCRIPT" ]; then

    echo "ERROR: metric script not found:"
    echo "  $METRIC_SCRIPT"

    exit 1
fi


# ============================================================
# Recover checkpoint step
# ============================================================

CKPT_NAME=$(basename "$ADAPTER")


if [[ "$CKPT_NAME" =~ ^checkpoint-([0-9]+)$ ]]; then

    STEP="${BASH_REMATCH[1]}"

else

    echo "ERROR: adapter path must end with checkpoint-<step>:"
    echo "  $ADAPTER"

    exit 1
fi


# ============================================================
# Recover training run name
#
# Example:
#
# results/.../v0-20260926-125827/checkpoint-800
#
# ->
#
# RUN_NAME=v0-20260926-125827
#
# This prevents different training seeds / runs from
# overwriting one another.
# ============================================================

RUN_DIR=$(dirname "$ADAPTER")

RUN_NAME=$(basename "$RUN_DIR")


# ============================================================
# Output
# ============================================================

OUTPUT_DIR="results/${DATASET}/evaluation/transformers/${FAMILY}/${METHOD}/${RUN_NAME}/${METHOD}_ckpt${STEP}"

mkdir -p "$OUTPUT_DIR"


# ============================================================
# Validate Python
# ============================================================

python -m py_compile "$SAFE_INFER"

python -m py_compile "$METRIC_SCRIPT"


# ============================================================
# Configuration summary
# ============================================================

echo
echo "============================================================"
echo "Formal Transformers Sarcasm Evaluation"
echo "============================================================"

echo "Dataset:             $DATASET"
echo "Variant:             $VARIANT"
echo "Family:              $FAMILY"
echo "Method:              $METHOD"

echo "Training run:        $RUN_NAME"
echo "Checkpoint:          checkpoint-${STEP}"

echo
echo "Model:               $MODEL"

echo "Adapter:"
echo "  $ADAPTER"

echo
echo "Splits:              ${SPLITS[*]}"

echo
echo "Backend:             transformers"
echo "Temperature:         $TEMPERATURE"
echo "Max new tokens:      $MAX_NEW_TOKENS"
echo "Max batch size:      $MAX_BATCH_SIZE"
echo "Inference seed:      $INFER_SEED"

echo
echo "Video backend:       decord"
echo "Decord threads:      1"

echo
echo "Output:"
echo "  $OUTPUT_DIR"

echo "============================================================"
echo


# ============================================================
# Hardware / environment
# ============================================================

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo

nvidia-smi
echo


python - <<'PY'
import sys

import decord
import torch
import swift
import transformers

print("Python:", sys.version.split()[0])

print("decord:", decord.__version__)
print("swift:", swift.__version__)
print("transformers:", transformers.__version__)
print("torch:", torch.__version__)

print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo


# ============================================================
# Temporary-directory cleanup
# ============================================================

CURRENT_TMP_DIR=""


cleanup() {

    if [ -n "${CURRENT_TMP_DIR:-}" ] \
       && [ -d "$CURRENT_TMP_DIR" ]; then

        rm -rf "$CURRENT_TMP_DIR"

    fi
}


trap cleanup EXIT


# ============================================================
# Evaluate each explicitly requested split
# ============================================================

for SPLIT in "${SPLITS[@]}"; do

    GOLD_DATA="data/${DATASET}/processed/zero_shot_${SPLIT}.jsonl"

    PRED_PATH="${OUTPUT_DIR}/${SPLIT}_predictions.jsonl"

    METRICS_PATH="${OUTPUT_DIR}/${SPLIT}_metrics.json"

    SCORED_PATH="${OUTPUT_DIR}/${SPLIT}_scored.jsonl"

    MANIFEST_PATH="${OUTPUT_DIR}/${SPLIT}_reproducibility.txt"


    echo
    echo "============================================================"
    echo "Evaluating"
    echo "============================================================"

    echo "Dataset:     $DATASET"
    echo "Family:      $FAMILY"
    echo "Method:      $METHOD"
    echo "Run:         $RUN_NAME"
    echo "Checkpoint:  checkpoint-${STEP}"
    echo "Split:       $SPLIT"

    echo
    echo "Output:"
    echo "  $OUTPUT_DIR"

    echo "============================================================"
    echo


    # ========================================================
    # Dataset
    # ========================================================

    if [ ! -f "$GOLD_DATA" ]; then

        echo "ERROR: dataset not found:"
        echo "  $GOLD_DATA"

        exit 1
    fi


    GOLD_COUNT=$(wc -l < "$GOLD_DATA")


    echo "Gold dataset:"
    echo "  $GOLD_DATA"

    echo
    echo "Gold examples: $GOLD_COUNT"
    echo


    # ========================================================
    # Existing-result protection
    #
    # OVERWRITE=1 permits replacement, but old formal files
    # remain untouched until the new run has fully succeeded.
    # ========================================================

    EXISTING=0

    for FILE in \
        "$PRED_PATH" \
        "$METRICS_PATH" \
        "$SCORED_PATH" \
        "$MANIFEST_PATH"
    do

        if [ -e "$FILE" ]; then
            EXISTING=1
        fi

    done


    if [ "$EXISTING" -eq 1 ] \
       && [ "${OVERWRITE:-0}" != "1" ]; then

        echo "ERROR: formal evaluation output already exists."
        echo
        echo "Output directory:"
        echo "  $OUTPUT_DIR"
        echo
        echo "To intentionally rerun:"
        echo
        echo "OVERWRITE=1 bash $0 \\"
        echo "  $DATASET \\"
        echo "  $VARIANT \\"
        echo "  $ADAPTER \\"
        echo "  $SPLIT"

        exit 1
    fi


    # ========================================================
    # Temporary output
    #
    # Never destroy a successful existing formal evaluation
    # before the replacement run has completed.
    # ========================================================

    CURRENT_TMP_DIR=$(mktemp -d \
        "${OUTPUT_DIR}/.${SPLIT}.tmp.XXXXXX")


    TMP_PRED="${CURRENT_TMP_DIR}/${SPLIT}_predictions.jsonl"

    TMP_METRICS="${CURRENT_TMP_DIR}/${SPLIT}_metrics.json"

    TMP_SCORED="${CURRENT_TMP_DIR}/${SPLIT}_scored.jsonl"

    TMP_MANIFEST="${CURRENT_TMP_DIR}/${SPLIT}_reproducibility.txt"


    # ========================================================
    # Transformers inference
    # ========================================================

    python "$SAFE_INFER" \
        --model "$MODEL" \
        --adapters "$ADAPTER" \
        --val_dataset "$GOLD_DATA" \
        --infer_backend transformers \
        --stream false \
        --torch_dtype bfloat16 \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_batch_size "$MAX_BATCH_SIZE" \
        --load_from_cache_file false \
        --dataset_num_proc 1 \
        --seed "$INFER_SEED" \
        --data_seed "$INFER_SEED" \
        --result_path "$TMP_PRED"


    # ========================================================
    # Validate generated output
    # ========================================================

    if [ ! -f "$TMP_PRED" ]; then

        echo
        echo "ERROR: prediction file was not created:"
        echo "  $TMP_PRED"

        exit 1
    fi


    PRED_COUNT=$(wc -l < "$TMP_PRED")


    echo
    echo "Gold count:       $GOLD_COUNT"
    echo "Prediction count: $PRED_COUNT"
    echo


    if [ "$GOLD_COUNT" -ne "$PRED_COUNT" ]; then

        echo "ERROR: prediction count mismatch."
        echo
        echo "Gold:        $GOLD_COUNT"
        echo "Predictions: $PRED_COUNT"

        exit 1
    fi


    # ========================================================
    # Classification metrics
    # ========================================================

    python "$METRIC_SCRIPT" \
        --gold "$GOLD_DATA" \
        --pred "$TMP_PRED" \
        --metrics "$TMP_METRICS" \
        --scored "$TMP_SCORED"


    # ========================================================
    # Reproducibility manifest
    # ========================================================

    {
        echo "============================================================"
        echo "Formal Sarcasm Evaluation Manifest"
        echo "============================================================"

        echo "timestamp=$(date --iso-8601=seconds)"
        echo "hostname=$(hostname)"

        echo
        echo "[evaluation]"

        echo "dataset=$DATASET"
        echo "variant=$VARIANT"
        echo "family=$FAMILY"
        echo "method=$METHOD"
        echo "training_run=$RUN_NAME"
        echo "checkpoint_step=$STEP"
        echo "split=$SPLIT"

        echo
        echo "[model]"

        echo "model=$MODEL"
        echo "adapter=$(readlink -f "$ADAPTER")"

        echo
        echo "[generation]"

        echo "backend=transformers"
        echo "temperature=$TEMPERATURE"
        echo "max_new_tokens=$MAX_NEW_TOKENS"
        echo "max_batch_size=$MAX_BATCH_SIZE"
        echo "inference_seed=$INFER_SEED"
        echo "torch_dtype=bfloat16"

        echo
        echo "[multimodal]"

        echo "USE_AUDIO_IN_VIDEO=$USE_AUDIO_IN_VIDEO"
        echo "FPS_MAX_FRAMES=$FPS_MAX_FRAMES"
        echo "VIDEO_MAX_PIXELS=$VIDEO_MAX_PIXELS"
        echo "MAX_PIXELS=$MAX_PIXELS"
        echo "FORCE_QWENVL_VIDEO_READER=$FORCE_QWENVL_VIDEO_READER"
        echo "decord_threads=1"

        echo
        echo "[determinism]"

        echo "PYTHONHASHSEED=$PYTHONHASHSEED"
        echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
        echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
        echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
        echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"

        echo
        echo "[software]"

        python - <<'PYVERSIONS'
import sys

import decord
import torch
import swift
import transformers

print(f"python={sys.version.split()[0]}")
print(f"decord={decord.__version__}")
print(f"torch={torch.__version__}")
print(f"swift={swift.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
PYVERSIONS

        echo
        echo "[gpu]"

        nvidia-smi \
            --query-gpu=name,driver_version \
            --format=csv,noheader \
            2>/dev/null || true

        echo
        echo "[git]"

        echo -n "commit="

        git rev-parse HEAD \
            2>/dev/null \
            || echo "unknown"

        echo "working_tree:"

        git status --short \
            2>/dev/null \
            || true

        echo
        echo "[sha256]"

        echo -n "gold="
        sha256sum "$GOLD_DATA" \
            | awk '{print $1}'

        echo -n "predictions="
        sha256sum "$TMP_PRED" \
            | awk '{print $1}'

        echo -n "metrics="
        sha256sum "$TMP_METRICS" \
            | awk '{print $1}'

        echo -n "scored="
        sha256sum "$TMP_SCORED" \
            | awk '{print $1}'

        echo -n "evaluation_script="
        sha256sum "$0" \
            | awk '{print $1}'

        echo -n "safe_infer="
        sha256sum "$SAFE_INFER" \
            | awk '{print $1}'

        echo -n "metric_script="
        sha256sum "$METRIC_SCRIPT" \
            | awk '{print $1}'

        echo -n "adapter_config="
        sha256sum "$ADAPTER/adapter_config.json" \
            | awk '{print $1}'

        echo
        echo "[adapter_weights]"

        find "$ADAPTER" \
            -maxdepth 1 \
            -type f \
            \( \
                -name "*.safetensors" \
                -o \
                -name "*.bin" \
            \) \
            -print0 \
            | sort -z \
            | while IFS= read -r -d '' WEIGHT_FILE
        do
            sha256sum "$WEIGHT_FILE"
        done

        echo
        echo "============================================================"

    } > "$TMP_MANIFEST"


    # ========================================================
    # Atomic publication of successful result
    # ========================================================

    mv -f "$TMP_PRED" \
        "$PRED_PATH"

    mv -f "$TMP_METRICS" \
        "$METRICS_PATH"

    mv -f "$TMP_SCORED" \
        "$SCORED_PATH"

    mv -f "$TMP_MANIFEST" \
        "$MANIFEST_PATH"


    rm -rf "$CURRENT_TMP_DIR"

    CURRENT_TMP_DIR=""


    # ========================================================
    # Final summary
    # ========================================================

    echo
    echo "============================================================"
    echo "Finished"
    echo "============================================================"

    echo "Dataset:     $DATASET"
    echo "Family:      $FAMILY"
    echo "Method:      $METHOD"
    echo "Run:         $RUN_NAME"
    echo "Checkpoint:  checkpoint-${STEP}"
    echo "Split:       $SPLIT"

    echo
    echo "Predictions:"
    echo "  $PRED_PATH"

    echo
    echo "Metrics:"
    echo "  $METRICS_PATH"

    echo
    echo "Scored:"
    echo "  $SCORED_PATH"

    echo
    echo "Manifest:"
    echo "  $MANIFEST_PATH"

    echo
    echo "Prediction SHA256:"

    sha256sum "$PRED_PATH"

    echo "============================================================"
    echo

done


echo
echo "============================================================"
echo "All requested formal evaluations finished"
echo "============================================================"

echo "Dataset:     $DATASET"
echo "Family:      $FAMILY"
echo "Method:      $METHOD"
echo "Run:         $RUN_NAME"
echo "Checkpoint:  checkpoint-${STEP}"

echo
echo "Results:"
echo "  $OUTPUT_DIR"

echo "============================================================"
