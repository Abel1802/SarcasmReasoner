#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MUStARD++ GenRM SFT
#
# Model:
#   Qwen/Qwen2.5-Omni-3B
#
# Usage:
#
#   Smoke test:
#     bash configs/mustard/genrm/qwen25_omni_3b_1gpu.sh smoke
#
#   Full training:
#     bash configs/mustard/genrm/qwen25_omni_3b_1gpu.sh full
#
# Hardware:
#   1 x H100
# ============================================================


MODE="${1:-smoke}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "ERROR: mode must be 'smoke' or 'full'"
    exit 1
fi


# ============================================================
# Model
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-3B"


# ============================================================
# Data
# ============================================================

TRAIN_DATA="data/mustard/processed/genrm/sft/genrm_train.jsonl"
VALID_DATA="data/mustard/processed/genrm/sft/genrm_valid.jsonl"

EXPECTED_TRAIN_SIZE=6448
EXPECTED_VALID_SIZE=1373


# ============================================================
# Training configuration
# ============================================================

PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1

# 1 GPU x batch 1 x grad_acc 16 = effective batch size 16
GRAD_ACC=16

LEARNING_RATE=2e-5

LORA_RANK=8
LORA_ALPHA=32

MAX_LENGTH=16384

NUM_EPOCHS=3

SEED=42


# ============================================================
# Smoke / full mode
# ============================================================

if [[ "$MODE" == "smoke" ]]; then

    OUTPUT_DIR="results/mustard/genrm/qwen25_omni_3b_smoke"

    # Only verify that:
    #   - model loads
    #   - audio/video preprocessing works
    #   - forward/backward pass works
    #   - LoRA training works
    #   - loss is finite
    #
    # No validation during smoke test.
    MAX_STEPS=20

else

    OUTPUT_DIR="results/mustard/genrm/qwen25_omni_3b"

fi


# ============================================================
# Multimodal environment
# ============================================================

export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=False

export FPS_MAX_FRAMES=12
export VIDEO_MAX_PIXELS=50176
export MAX_PIXELS=1003520

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


# ============================================================
# File checks
# ============================================================

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: training dataset not found:"
    echo "  $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VALID_DATA" ]; then
    echo "ERROR: validation dataset not found:"
    echo "  $VALID_DATA"
    exit 1
fi


TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
VALID_SIZE=$(wc -l < "$VALID_DATA")

if [ "$TRAIN_SIZE" -ne "$EXPECTED_TRAIN_SIZE" ]; then
    echo "ERROR: unexpected training size."
    echo "Expected: $EXPECTED_TRAIN_SIZE"
    echo "Actual:   $TRAIN_SIZE"
    exit 1
fi

if [ "$VALID_SIZE" -ne "$EXPECTED_VALID_SIZE" ]; then
    echo "ERROR: unexpected validation size."
    echo "Expected: $EXPECTED_VALID_SIZE"
    echo "Actual:   $VALID_SIZE"
    exit 1
fi


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ Qwen2.5-Omni-3B GenRM SFT"
echo "============================================================"
echo "Mode:                   $MODE"
echo "Model:                  $MODEL"
echo
echo "Training data:          $TRAIN_DATA"
echo "Training examples:      $TRAIN_SIZE"
echo
echo "Validation data:        $VALID_DATA"
echo "Validation examples:    $VALID_SIZE"
echo
echo "GPU count:              1"
echo "Batch size / GPU:       $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation:  $GRAD_ACC"
echo "Effective batch size:   $((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC))"
echo
echo "Learning rate:          $LEARNING_RATE"
echo "LoRA rank:              $LORA_RANK"
echo "LoRA alpha:             $LORA_ALPHA"
echo "Max sequence length:    $MAX_LENGTH"
echo "Seed:                   $SEED"

if [[ "$MODE" == "smoke" ]]; then
    echo "Max steps:              $MAX_STEPS"
else
    echo "Epochs:                 $NUM_EPOCHS"
fi

echo
echo "Output directory:       $OUTPUT_DIR"
echo "============================================================"
echo


# ============================================================
# Environment information
# ============================================================

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo

nvidia-smi
echo

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

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo


mkdir -p "$OUTPUT_DIR"


# ============================================================
# Smoke training
# ============================================================

if [[ "$MODE" == "smoke" ]]; then

    swift sft \
        --model "$MODEL" \
        \
        --dataset "$TRAIN_DATA" \
        --val_dataset "$VALID_DATA" \
        --split_dataset_ratio 0 \
        \
        --load_from_cache_file true \
        --dataset_num_proc 1 \
        --strict true \
        \
        --tuner_type lora \
        --torch_dtype bfloat16 \
        \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --target_modules all-linear \
        \
        --freeze_vit true \
        --freeze_aligner true \
        \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACC" \
        \
        --learning_rate "$LEARNING_RATE" \
        --warmup_ratio 0.05 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        \
        --max_length "$MAX_LENGTH" \
        --truncation_strategy delete \
        \
        --gradient_checkpointing true \
        \
        --max_steps "$MAX_STEPS" \
        \
        --logging_steps 1 \
        \
        --eval_strategy no \
        --save_strategy no \
        \
        --seed "$SEED" \
        --data_seed "$SEED" \
        \
        --output_dir "$OUTPUT_DIR" \
        --report_to tensorboard \
        \
        --dataloader_num_workers 4


# ============================================================
# Full training
# ============================================================

else

    swift sft \
        --model "$MODEL" \
        \
        --dataset "$TRAIN_DATA" \
        --val_dataset "$VALID_DATA" \
        --split_dataset_ratio 0 \
        \
        --load_from_cache_file true \
        --dataset_num_proc 1 \
        --strict true \
        \
        --tuner_type lora \
        --torch_dtype bfloat16 \
        \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --target_modules all-linear \
        \
        --freeze_vit true \
        --freeze_aligner true \
        \
        --num_train_epochs "$NUM_EPOCHS" \
        \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACC" \
        \
        --learning_rate "$LEARNING_RATE" \
        --warmup_ratio 0.05 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        \
        --max_length "$MAX_LENGTH" \
        --truncation_strategy delete \
        \
        --gradient_checkpointing true \
        \
        --logging_steps 5 \
        \
        --eval_strategy steps \
        --eval_steps 100 \
        \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 3 \
        \
        --seed "$SEED" \
        --data_seed "$SEED" \
        \
        --output_dir "$OUTPUT_DIR" \
        --report_to tensorboard \
        \
        --dataloader_num_workers 4

fi


STATUS=$?


echo
echo "============================================================"
echo "GenRM SFT finished"
echo "============================================================"
echo "Mode:             $MODE"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
