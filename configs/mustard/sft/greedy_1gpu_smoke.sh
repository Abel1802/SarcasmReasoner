#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MUStARD++ Greedy-SFT - 1 GPU smoke test
#
# Student:
#   Qwen2.5-Omni-7B
#
# Purpose:
#   Verify that multimodal SFT works correctly on one H100
#   before launching the full training.
#
# Data:
#   train = 550
#   valid = 116
#
# Effective batch size:
#   1 GPU x batch 1 x grad_acc 16 = 16
# ============================================================


MODEL="Qwen/Qwen2.5-Omni-7B"

TRAIN_DATA="data/mustard/processed/sft/greedy_train.jsonl"
VALID_DATA="data/mustard/processed/sft/greedy_valid.jsonl"

OUTPUT_DIR="results/mustard/sft/greedy_1gpu_smoke"


# ============================================================
# Multimodal settings
# ============================================================

# We only need text output; disable the speech-generation part.
export ENABLE_AUDIO_OUTPUT=0

# Audio is provided separately from video.
export USE_AUDIO_IN_VIDEO=False

# Keep visual preprocessing consistent with teacher generation.
export FPS_MAX_FRAMES=12

export VIDEO_MAX_PIXELS=50176
export MAX_PIXELS=1003520

export TOKENIZERS_PARALLELISM=false

export PYTHONUNBUFFERED=1


# ============================================================
# Sanity checks
# ============================================================

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: training file not found:"
    echo "  $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VALID_DATA" ]; then
    echo "ERROR: validation file not found:"
    echo "  $VALID_DATA"
    exit 1
fi


echo "============================================================"
echo "MUStARD++ Greedy-SFT - 1 GPU Smoke Test"
echo "============================================================"
echo "Model:            $MODEL"
echo "Train:            $TRAIN_DATA"
echo "Valid:            $VALID_DATA"
echo "GPU count:        1"
echo "Batch / GPU:      1"
echo "Grad accumulation:16"
echo "Effective batch:  16"
echo "Max steps:        10"
echo "Output:           $OUTPUT_DIR"
echo "============================================================"
echo


# ============================================================
# Train
# ============================================================

swift sft \
    --model "$MODEL" \
    \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VALID_DATA" \
    --split_dataset_ratio 0 \
    \
    --load_from_cache_file false \
    --dataset_num_proc 1 \
    --strict true \
    \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    \
    --freeze_vit true \
    --freeze_aligner true \
    \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    \
    --max_length 16384 \
    --truncation_strategy delete \
    \
    --gradient_checkpointing true \
    \
    --max_steps 10 \
    \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    \
    --dataloader_num_workers 4

