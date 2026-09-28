#!/bin/bash

set -e
set -o pipefail


# ============================================================
# GenRM checkpoint evaluation
#
# Supports:
#   - MUStARD++
#   - MCSD
#
# Usage:
#
#   MUStARD++:
#     bash scripts/evaluation/eval_genrm_checkpoint.sh \
#         --dataset mustard \
#         --adapter results/mustard/genrm/qwen25_omni_3b/v1-20260926-001310/checkpoint-1000
#
#   MCSD:
#     bash scripts/evaluation/eval_genrm_checkpoint.sh \
#         --dataset mcsd \
#         --adapter results/mcsd/genrm/qwen25_omni_3b/<run>/checkpoint-XXXX
#
#   Rerun:
#     OVERWRITE=1 bash scripts/evaluation/eval_genrm_checkpoint.sh \
#         --dataset mcsd \
#         --adapter results/mcsd/genrm/qwen25_omni_3b/<run>/checkpoint-XXXX
#
# Optional:
#   --output-dir PATH
# ============================================================


DATASET=""
ADAPTER=""
OUTPUT_DIR=""

MODEL="Qwen/Qwen2.5-Omni-3B"

SAFE_INFER="src/evaluation/swift_infer_safe_video.py"
EVALUATOR="src/evaluation/evaluate_genrm_predictions.py"


# ============================================================
# Arguments
# ============================================================

while [[ $# -gt 0 ]]; do

    case "$1" in

        --dataset)
            DATASET="$2"
            shift 2
            ;;

        --adapter)
            ADAPTER="$2"
            shift 2
            ;;

        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;

        -h|--help)
            echo "Usage:"
            echo
            echo "  bash scripts/evaluation/eval_genrm_checkpoint.sh \\"
            echo "      --dataset {mustard|mcsd} \\"
            echo "      --adapter PATH_TO_CHECKPOINT \\"
            echo "      [--output-dir OUTPUT_DIR]"
            echo
            echo "Rerun existing evaluation:"
            echo
            echo "  OVERWRITE=1 bash scripts/evaluation/eval_genrm_checkpoint.sh ..."
            exit 0
            ;;

        *)
            echo "ERROR: unknown argument: $1"
            exit 1
            ;;
    esac

done


# ============================================================
# Dataset configuration
# ============================================================

if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required."
    echo "Choose: mustard or mcsd"
    exit 1
fi

case "$DATASET" in

    mustard)

        DATASET_NAME="MUStARD++"
        GOLD_DATA="data/mustard/processed/genrm/sft/genrm_valid.jsonl"
        EXPECTED_VALID_SIZE=1373
        ;;

    mcsd)

        DATASET_NAME="MCSD"
        GOLD_DATA="data/mcsd/processed/genrm/sft/genrm_valid.jsonl"
        EXPECTED_VALID_SIZE=3192
        ;;

    *)

        echo "ERROR: unsupported dataset: $DATASET"
        echo "Choose: mustard or mcsd"
        exit 1
        ;;

esac


# ============================================================
# Checkpoint
# ============================================================

if [[ -z "$ADAPTER" ]]; then
    echo "ERROR: --adapter is required."
    exit 1
fi

CHECKPOINT_NAME="$(basename "$ADAPTER")"


# ============================================================
# Output
# ============================================================

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="results/${DATASET}/genrm/evaluation/${CHECKPOINT_NAME}"
fi

PRED_PATH="${OUTPUT_DIR}/valid_predictions.jsonl"
METRICS_PATH="${OUTPUT_DIR}/valid_metrics.json"
SCORED_PATH="${OUTPUT_DIR}/valid_scored.jsonl"

mkdir -p "$OUTPUT_DIR"


# ============================================================
# Generation
# ============================================================

TEMPERATURE=0
MAX_NEW_TOKENS=64
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
# Checks
# ============================================================

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: checkpoint not found:"
    echo "  $ADAPTER"
    exit 1
fi

if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found:"
    echo "  $ADAPTER/adapter_config.json"
    exit 1
fi

if [ ! -f "$GOLD_DATA" ]; then
    echo "ERROR: validation data not found:"
    echo "  $GOLD_DATA"
    exit 1
fi

if [ ! -f "$SAFE_INFER" ]; then
    echo "ERROR: inference wrapper not found:"
    echo "  $SAFE_INFER"
    exit 1
fi

if [ ! -f "$EVALUATOR" ]; then
    echo "ERROR: evaluator not found:"
    echo "  $EVALUATOR"
    exit 1
fi


GOLD_COUNT=$(wc -l < "$GOLD_DATA")

if [ "$GOLD_COUNT" -ne "$EXPECTED_VALID_SIZE" ]; then
    echo "ERROR: unexpected validation size."
    echo "Dataset:  $DATASET"
    echo "Expected: $EXPECTED_VALID_SIZE"
    echo "Actual:   $GOLD_COUNT"
    exit 1
fi


# ============================================================
# Existing prediction protection
# ============================================================

if [ -f "$PRED_PATH" ]; then

    if [ "${OVERWRITE:-0}" = "1" ]; then

        echo "OVERWRITE=1: removing old evaluation files."

        rm -f \
            "$PRED_PATH" \
            "$METRICS_PATH" \
            "$SCORED_PATH"

    else

        echo "ERROR: prediction file already exists:"
        echo "  $PRED_PATH"
        echo
        echo "To rerun:"
        echo
        echo "  OVERWRITE=1 bash scripts/evaluation/eval_genrm_checkpoint.sh \\"
        echo "      --dataset $DATASET \\"
        echo "      --adapter $ADAPTER"
        exit 1

    fi
fi


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "${DATASET_NAME} GenRM validation"
echo "============================================================"
echo "Dataset:        $DATASET"
echo "Model:          $MODEL"
echo "Adapter:        $ADAPTER"
echo "Gold data:      $GOLD_DATA"
echo "Examples:       $GOLD_COUNT"
echo "Temperature:    $TEMPERATURE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output:         $OUTPUT_DIR"
echo "============================================================"
echo

echo "FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-not-set}"
echo

nvidia-smi
echo


# ============================================================
# Inference
# ============================================================

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
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK" \
    \
    --vllm_limit_mm_per_prompt '{"audio": 1, "video": 1}' \
    \
    --load_from_cache_file false \
    --dataset_num_proc 1 \
    \
    --seed "$SEED" \
    --data_seed "$SEED" \
    \
    --result_path "$PRED_PATH"


# ============================================================
# Prediction count
# ============================================================

if [ ! -f "$PRED_PATH" ]; then
    echo "ERROR: prediction file was not created."
    exit 1
fi

PRED_COUNT=$(wc -l < "$PRED_PATH")

echo
echo "Gold count:       $GOLD_COUNT"
echo "Prediction count: $PRED_COUNT"
echo

if [ "$GOLD_COUNT" -ne "$PRED_COUNT" ]; then
    echo "ERROR: prediction count mismatch."
    exit 1
fi


# ============================================================
# GenRM metrics
# ============================================================

python "$EVALUATOR" \
    --gold "$GOLD_DATA" \
    --pred "$PRED_PATH" \
    --metrics "$METRICS_PATH" \
    --scored "$SCORED_PATH"


echo
echo "============================================================"
echo "GenRM validation finished"
echo "============================================================"
echo "Dataset:     $DATASET"
echo "Checkpoint:  $CHECKPOINT_NAME"
echo "Predictions: $PRED_PATH"
echo "Metrics:     $METRICS_PATH"
echo "Scored:      $SCORED_PATH"
echo "============================================================"
