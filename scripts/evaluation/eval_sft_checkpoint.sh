#!/bin/bash

set -e
set -o pipefail


# ============================================================
# Generic multimodal SFT evaluation with vLLM
#
# Safe video decoding:
#   - Force Qwen2.5-Omni to use decord
#   - Force decord.VideoReader(num_threads=1)
#
# Usage:
#
# bash scripts/evaluation/eval_sft_checkpoint.sh \
#     <dataset> \
#     <variant> \
#     <adapter_path> \
#     [split ...]
#
# Example:
#
# bash scripts/evaluation/eval_sft_checkpoint.sh \
#     mustard \
#     greedy \
#     results/mustard/sft/greedy/v0-20260923-221644/checkpoint-105 \
#     valid test
#
# MCSD example:
#
# bash scripts/evaluation/eval_sft_checkpoint.sh \
#     mcsd \
#     sft_greedy \
#     results/mcsd/sft/greedy/v6-20260924-101848/checkpoint-228 \
#     test
#
# If split is omitted:
#     valid test
# ============================================================


if [ "$#" -lt 3 ]; then

    echo "Usage:"
    echo
    echo "bash $0 <dataset> <variant> <adapter_path> [split ...]"
    echo

    exit 1
fi


# ============================================================
# Arguments
# ============================================================

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
# Base model
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"


# ============================================================
# Safe inference wrapper
# ============================================================

SAFE_INFER="src/evaluation/swift_infer_safe_video.py"


# ============================================================
# Generation
# ============================================================

# Formal evaluation always uses deterministic greedy decoding.

TEMPERATURE=0

MAX_NEW_TOKENS=4096

SEED=42


# ============================================================
# vLLM
# ============================================================

# One H100.
VLLM_TP=1

# Qwen2.5-Omni 7B fits on one H100.
VLLM_GPU_MEMORY_UTILIZATION=0.90

# Same context budget used elsewhere in the project.
VLLM_MAX_MODEL_LEN=16384

# Conservative multimodal concurrency.
VLLM_MAX_NUM_SEQS=4

# LoRA r=8, so vLLM default max rank 16 is sufficient.
VLLM_MAX_LORA_RANK=16


# ============================================================
# Multimodal environment
# ============================================================

# We only generate textual reasoning.
export ENABLE_AUDIO_OUTPUT=0

# Audio is provided separately from the video.
export USE_AUDIO_IN_VIDEO=False

# Keep preprocessing aligned with teacher / SFT / GRPO.
export FPS_MAX_FRAMES=12

export VIDEO_MAX_PIXELS=50176
export MAX_PIXELS=1003520

# IMPORTANT:
# Force Qwen2.5-Omni video preprocessing to use Decord
# rather than torchvision.
export FORCE_QWENVL_VIDEO_READER=decord

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR="results/${DATASET}/evaluation/${VARIANT}"

mkdir -p "$OUTPUT_DIR"


# ============================================================
# Sanity checks
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

    echo "ERROR: safe inference wrapper missing:"
    echo "  $SAFE_INFER"

    exit 1
fi


if [ ! -f \
    "src/evaluation/evaluate_sarcasm_predictions.py" ]; then

    echo "ERROR: metric script missing:"
    echo "  src/evaluation/evaluate_sarcasm_predictions.py"

    exit 1
fi


# ============================================================
# Validate safe wrapper syntax
# ============================================================

python -m py_compile "$SAFE_INFER"


# ============================================================
# Configuration summary
# ============================================================

echo
echo "============================================================"
echo "Multimodal Sarcasm Evaluation"
echo "============================================================"
echo "Dataset:             $DATASET"
echo "Variant:             $VARIANT"
echo "Model:               $MODEL"
echo "Adapter:             $ADAPTER"
echo "Splits:              ${SPLITS[*]}"
echo
echo "Inference backend:   vLLM"
echo "Safe video wrapper:  $SAFE_INFER"
echo "Video backend:       decord"
echo "Decord threads:      1 (patched in wrapper)"
echo
echo "Temperature:         $TEMPERATURE"
echo "Max new tokens:      $MAX_NEW_TOKENS"
echo
echo "vLLM TP:             $VLLM_TP"
echo "vLLM max model len:  $VLLM_MAX_MODEL_LEN"
echo "vLLM max num seqs:   $VLLM_MAX_NUM_SEQS"
echo "vLLM GPU memory:     $VLLM_GPU_MEMORY_UTILIZATION"
echo
echo "Output:              $OUTPUT_DIR"
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
    echo "Evaluating: $DATASET / $VARIANT / $SPLIT"
    echo "============================================================"
    echo


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ms-swift appends to an existing result_path.
    #
    # Do not accidentally duplicate predictions.
    #
    # To intentionally rerun:
    #
    # OVERWRITE=1 bash ...
    # --------------------------------------------------------

    if [ -f "$PRED_PATH" ]; then

        if [ "${OVERWRITE:-0}" = "1" ]; then

            echo "OVERWRITE=1"
            echo "Removing old evaluation files..."

            rm -f \
                "$PRED_PATH" \
                "$METRICS_PATH" \
                "$SCORED_PATH"

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
    # vLLM inference with safe Decord wrapper
    #
    # IMPORTANT:
    #
    # Instead of:
    #
    #   swift infer ...
    #
    # use:
    #
    #   python src/evaluation/swift_infer_safe_video.py ...
    #
    # The wrapper:
    #
    #   1. sets FORCE_QWENVL_VIDEO_READER=decord
    #   2. patches decord.VideoReader(num_threads=1)
    #   3. imports Swift only AFTER the patch
    #   4. calls swift.pipelines.infer_main()
    #
    # This avoids the MCSD video decoding failure that can
    # otherwise fall back to torchvision and produce:
    #
    #   KeyError: 'video_fps'
    #
    # Evaluation protocol itself remains unchanged:
    #
    #   - Full original split
    #   - LoRA adapter loaded directly
    #   - vLLM backend
    #   - temperature=0
    #   - text + audio + video
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
        --vllm_gpu_memory_utilization \
            "$VLLM_GPU_MEMORY_UTILIZATION" \
        --vllm_max_model_len \
            "$VLLM_MAX_MODEL_LEN" \
        --vllm_max_num_seqs \
            "$VLLM_MAX_NUM_SEQS" \
        --vllm_max_lora_rank \
            "$VLLM_MAX_LORA_RANK" \
        \
        --vllm_limit_mm_per_prompt \
            '{"audio": 1, "video": 1}' \
        \
        --load_from_cache_file false \
        --dataset_num_proc 1 \
        \
        --seed "$SEED" \
        --data_seed "$SEED" \
        \
        --result_path "$PRED_PATH"


    # ========================================================
    # Validate result count
    # ========================================================

    if [ ! -f "$PRED_PATH" ]; then

        echo
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
        echo
        echo "Gold:        $GOLD_COUNT"
        echo "Predictions: $PRED_COUNT"

        exit 1
    fi


    # ========================================================
    # Accuracy / Macro-P / Macro-R / Macro-F1
    # ========================================================

    python \
        src/evaluation/evaluate_sarcasm_predictions.py \
        --gold "$GOLD_DATA" \
        --pred "$PRED_PATH" \
        --metrics "$METRICS_PATH" \
        --scored "$SCORED_PATH"


    echo
    echo "============================================================"
    echo "Finished: $SPLIT"
    echo "============================================================"
    echo

done


echo
echo "============================================================"
echo "All evaluation finished"
echo "============================================================"
echo "Dataset:  $DATASET"
echo "Variant:  $VARIANT"
echo "Results:  $OUTPUT_DIR"
echo "============================================================"