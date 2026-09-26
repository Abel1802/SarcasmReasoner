#!/bin/bash

set -e
set -o pipefail


# ============================================================
# Generic multimodal checkpoint evaluation with vLLM
#
# Variant format:
#
#   sft_<method>
#   grpo_<method>
#   grporm_<method>
#
# <method> may be any non-empty string.
#
#
# Output hierarchy:
#
#   sft_greedy
#   + checkpoint-105
#   ->
#   results/mustard/evaluation/
#       sft/greedy/greedy_ckpt105/
#
#
#   grpo_best_of_8
#   + checkpoint-800
#   ->
#   results/mustard/evaluation/
#       grpo/best_of_8/best_of_8_ckpt800/
#
#
#   grporm_diverse_8
#   + checkpoint-800
#   ->
#   results/mustard/evaluation/
#       grpo_rm/diverse_8/diverse_8_ckpt800/
#
#
# Usage:
#
# bash scripts/evaluation/eval_checkpoint.sh \
#     <dataset> \
#     <variant> \
#     <adapter_path> \
#     [split ...]
#
#
# Examples
# ------------------------------------------------------------
#
# SFT:
#
# bash scripts/evaluation/eval_checkpoint.sh \
#     mustard \
#     sft_greedy \
#     results/mustard/sft/greedy/v0-20260923-221644/checkpoint-105 \
#     test
#
#
# Ordinary GRPO:
#
# bash scripts/evaluation/eval_checkpoint.sh \
#     mustard \
#     grpo_greedy \
#     results/mustard/grpo/greedy/v0-20260924-122118/checkpoint-800 \
#     valid
#
#
# GRPO + GenRM:
#
# bash scripts/evaluation/eval_checkpoint.sh \
#     mustard \
#     grporm_greedy \
#     results/mustard/grpo_rm/greedy/v0-20260926-125827/checkpoint-800 \
#     test
#
#
# If split is omitted:
#
#   valid test
# ============================================================


# ============================================================
# Arguments
# ============================================================

if [ "$#" -lt 3 ]; then

    echo "Usage:"
    echo
    echo "bash $0 <dataset> <variant> <adapter_path> [split ...]"
    echo
    echo "Variant format:"
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


if [ "$#" -eq 0 ]; then
    SPLITS=("valid" "test")
else
    SPLITS=("$@")
fi


# ============================================================
# Parse variant
#
# Important:
#
# The part after the prefix is arbitrary.
#
# Examples:
#
#   sft_greedy
#   grpo_best_of_8
#   grporm_diverse_8
#   grpo_any_future_method
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
        echo "Expected format:"
        echo "  sft_<method>"
        echo "  grpo_<method>"
        echo "  grporm_<method>"
        echo

        exit 1
        ;;

esac


if [ -z "$METHOD" ]; then

    echo "ERROR: method cannot be empty:"
    echo "  $VARIANT"

    exit 1
fi


# ============================================================
# Base model
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"


# ============================================================
# Safe inference wrapper
# ============================================================

SAFE_INFER="src/evaluation/swift_infer_safe_video.py"


# ============================================================
# Generation
#
# Formal evaluation uses deterministic greedy decoding.
# ============================================================

TEMPERATURE=0

MAX_NEW_TOKENS=4096

SEED=42


# ============================================================
# vLLM
# ============================================================

VLLM_TP=1

VLLM_GPU_MEMORY_UTILIZATION=0.90

VLLM_MAX_MODEL_LEN=16384

VLLM_MAX_NUM_SEQS=4

VLLM_MAX_LORA_RANK=16


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
# Validate adapter
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


# ============================================================
# Recover checkpoint step
#
# Required adapter basename:
#
#   checkpoint-105
#   checkpoint-200
#   checkpoint-800
#   ...
# ============================================================

CKPT_NAME=$(basename "$ADAPTER")


if [[ "$CKPT_NAME" =~ ^checkpoint-([0-9]+)$ ]]; then

    STEP="${BASH_REMATCH[1]}"

else

    echo "ERROR: adapter path must end with checkpoint-<step>:"
    echo "  $ADAPTER"
    echo
    echo "Example:"
    echo "  .../checkpoint-800"

    exit 1
fi


# ============================================================
# Output directory
#
# sft_<method>
# ->
# results/<dataset>/evaluation/
#     sft/<method>/<method>_ckpt<step>
#
#
# grpo_<method>
# ->
# results/<dataset>/evaluation/
#     grpo/<method>/<method>_ckpt<step>
#
#
# grporm_<method>
# ->
# results/<dataset>/evaluation/
#     grpo_rm/<method>/<method>_ckpt<step>
# ============================================================

OUTPUT_DIR="results/${DATASET}/evaluation/${FAMILY}/${METHOD}/${METHOD}_ckpt${STEP}"

mkdir -p "$OUTPUT_DIR"


# ============================================================
# Other sanity checks
# ============================================================

if [ ! -f "$SAFE_INFER" ]; then

    echo "ERROR: safe inference wrapper missing:"
    echo "  $SAFE_INFER"

    exit 1
fi


METRIC_SCRIPT="src/evaluation/evaluate_sarcasm_predictions.py"


if [ ! -f "$METRIC_SCRIPT" ]; then

    echo "ERROR: metric script missing:"
    echo "  $METRIC_SCRIPT"

    exit 1
fi


# ============================================================
# Validate Python syntax
# ============================================================

python -m py_compile "$SAFE_INFER"

python -m py_compile "$METRIC_SCRIPT"


# ============================================================
# Configuration summary
# ============================================================

echo
echo "============================================================"
echo "Multimodal Sarcasm Evaluation"
echo "============================================================"
echo "Dataset:             $DATASET"
echo "Variant:             $VARIANT"
echo "Family:              $FAMILY"
echo "Method:              $METHOD"
echo "Checkpoint step:     $STEP"
echo
echo "Model:               $MODEL"
echo "Adapter:"
echo "  $ADAPTER"
echo
echo "Splits:              ${SPLITS[*]}"
echo
echo "Inference backend:   vLLM"
echo "Safe video wrapper:  $SAFE_INFER"
echo "Video backend:       decord"
echo "Decord threads:      1"
echo
echo "Temperature:         $TEMPERATURE"
echo "Max new tokens:      $MAX_NEW_TOKENS"
echo
echo "vLLM TP:             $VLLM_TP"
echo "vLLM max model len:  $VLLM_MAX_MODEL_LEN"
echo "vLLM max num seqs:   $VLLM_MAX_NUM_SEQS"
echo "vLLM GPU memory:     $VLLM_GPU_MEMORY_UTILIZATION"
echo
echo "Output:"
echo "  $OUTPUT_DIR"
echo "============================================================"
echo


# ============================================================
# GPU
# ============================================================

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo

nvidia-smi

echo


# ============================================================
# Environment information
# ============================================================

echo "Python:"
which python
python --version
echo


python - <<'PY'
import decord
import torch
import swift
import trl

print("decord:", decord.__version__)
print("swift:", swift.__version__)
print("trl:", trl.__version__)
print("torch:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo


# ============================================================
# Run each split
# ============================================================

for SPLIT in "${SPLITS[@]}"; do

    GOLD_DATA="data/${DATASET}/processed/zero_shot_${SPLIT}.jsonl"

    PRED_PATH="${OUTPUT_DIR}/${SPLIT}_predictions.jsonl"

    METRICS_PATH="${OUTPUT_DIR}/${SPLIT}_metrics.json"

    SCORED_PATH="${OUTPUT_DIR}/${SPLIT}_scored.jsonl"


    echo
    echo "============================================================"
    echo "Evaluating checkpoint"
    echo "============================================================"
    echo "Dataset:     $DATASET"
    echo "Family:      $FAMILY"
    echo "Method:      $METHOD"
    echo "Checkpoint:  checkpoint-${STEP}"
    echo "Split:       $SPLIT"
    echo
    echo "Output:"
    echo "  $OUTPUT_DIR"
    echo "============================================================"
    echo


    # ========================================================
    # Validate dataset
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
    # Existing output protection
    #
    # ms-swift may append to result_path.
    #
    # Therefore never reuse an existing prediction file
    # unless OVERWRITE=1 is explicitly supplied.
    #
    # Example:
    #
    # OVERWRITE=1 bash scripts/evaluation/eval_checkpoint.sh ...
    # ========================================================

    if [ -f "$PRED_PATH" ]; then

        if [ "${OVERWRITE:-0}" = "1" ]; then

            echo "OVERWRITE=1"
            echo "Removing old evaluation files..."

            rm -f \
                "$PRED_PATH" \
                "$METRICS_PATH" \
                "$SCORED_PATH"

            echo

        else

            echo "ERROR: prediction file already exists:"
            echo "  $PRED_PATH"
            echo
            echo "To intentionally rerun:"
            echo
            echo "OVERWRITE=1 bash $0 \\"
            echo "  $DATASET \\"
            echo "  $VARIANT \\"
            echo "  $ADAPTER \\"
            echo "  $SPLIT"
            echo

            exit 1
        fi

    fi


    # ========================================================
    # vLLM inference
    #
    # Safe wrapper:
    #
    #   1. forces Decord
    #   2. uses num_threads=1
    #   3. imports Swift after video patching
    #
    # Formal evaluation:
    #
    #   - full original split
    #   - text + audio + video
    #   - LoRA loaded directly
    #   - deterministic decoding
    # ========================================================

    python "$SAFE_INFER" \
        --model "$MODEL" \
        --adapters "$ADAPTER" \
        \
        --val_dataset "$GOLD_DATA" \
        \
        --infer_backend vllm \
        --stream false \
        \
        --torch_dtype bfloat16 \
        \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        \
        --vllm_tensor_parallel_size "$VLLM_TP" \
        \
        --vllm_gpu_memory_utilization \
            "$VLLM_GPU_MEMORY_UTILIZATION" \
        \
        --vllm_max_model_len \
            "$VLLM_MAX_MODEL_LEN" \
        \
        --vllm_max_num_seqs \
            "$VLLM_MAX_NUM_SEQS" \
        \
        --vllm_max_lora_rank \
            "$VLLM_MAX_LORA_RANK" \
        \
        --vllm_limit_mm_per_prompt \
            '{"audio": 1, "video": 1}' \
        \
        --load_from_cache_file false \
        \
        --dataset_num_proc 1 \
        \
        --seed "$SEED" \
        --data_seed "$SEED" \
        \
        --result_path "$PRED_PATH"


    # ========================================================
    # Validate output existence
    # ========================================================

    if [ ! -f "$PRED_PATH" ]; then

        echo
        echo "ERROR: prediction file was not created:"
        echo "  $PRED_PATH"

        exit 1
    fi


    # ========================================================
    # Validate number of predictions
    # ========================================================

    PRED_COUNT=$(wc -l < "$PRED_PATH")


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
    #
    # - Accuracy
    # - Macro Precision
    # - Macro Recall
    # - Macro F1
    # - class-specific metrics
    # - confusion matrix
    # ========================================================

    python "$METRIC_SCRIPT" \
        --gold "$GOLD_DATA" \
        --pred "$PRED_PATH" \
        --metrics "$METRICS_PATH" \
        --scored "$SCORED_PATH"


    echo
    echo "============================================================"
    echo "Finished"
    echo "============================================================"
    echo "Dataset:     $DATASET"
    echo "Family:      $FAMILY"
    echo "Method:      $METHOD"
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
    echo "============================================================"
    echo

done


echo
echo "============================================================"
echo "All requested splits finished"
echo "============================================================"
echo "Dataset:     $DATASET"
echo "Family:      $FAMILY"
echo "Method:      $METHOD"
echo "Checkpoint:  checkpoint-${STEP}"
echo
echo "Results:"
echo "  $OUTPUT_DIR"
echo "============================================================"
