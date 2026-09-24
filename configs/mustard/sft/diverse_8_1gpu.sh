#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MUStARD++ Diverse-8 SFT
#
# Student:
#   Qwen/Qwen2.5-Omni-7B
#
# Teacher trajectories:
#   Qwen3-Omni-30B
#   N=8 stochastic sampling
#
# Selection:
#   Keep all structurally valid, correct, and textually
#   distinct trajectories for each source instance.
#
# SFT data:
#   train = 4447 examples
#   valid = 980 examples
#
# Source coverage:
#   train = 757 / 841
#   valid = 163 / 180
#
# Hardware:
#   1 x H100
#
# Effective batch size:
#   1 GPU x batch_size 1 x grad_acc 16 = 16
#
# IMPORTANT:
#   Training hyperparameters are intentionally identical to
#   Greedy-SFT and Best-of-8 SFT.
#
#   This is the FULL Diverse-8 setting, so it contains many
#   more SFT trajectories than Greedy / Best-of-8.
# ============================================================


# ============================================================
# Model
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"


# ============================================================
# Data
# ============================================================

TRAIN_DATA="data/mustard/processed/sft/sft_diverse_8_train.jsonl"
VALID_DATA="data/mustard/processed/sft/sft_diverse_8_valid.jsonl"


# ============================================================
# Output
# ============================================================

OUTPUT_DIR="results/mustard/sft/diverse_8"


# ============================================================
# Training configuration
# ============================================================

NUM_EPOCHS=3

PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1

GRAD_ACC=16

LEARNING_RATE=1e-4

LORA_RANK=8
LORA_ALPHA=32

MAX_LENGTH=16384

SEED=42


# ============================================================
# Multimodal environment
# ============================================================

# We only generate textual reasoning / classification output.
export ENABLE_AUDIO_OUTPUT=0

# Audio is supplied explicitly as a separate modality.
export USE_AUDIO_IN_VIDEO=False

# Keep video preprocessing consistent with teacher generation.
export FPS_MAX_FRAMES=12

# Qwen Omni multimodal preprocessing limits.
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


# ============================================================
# Dataset statistics
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
VALID_SIZE=$(wc -l < "$VALID_DATA")

echo
echo "============================================================"
echo "MUStARD++ Diverse-8 SFT"
echo "============================================================"
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
echo "Effective batch size:   $GRAD_ACC"
echo
echo "Epochs:                 $NUM_EPOCHS"
echo "Learning rate:          $LEARNING_RATE"
echo "LoRA rank:              $LORA_RANK"
echo "LoRA alpha:             $LORA_ALPHA"
echo "Max sequence length:    $MAX_LENGTH"
echo "Seed:                   $SEED"
echo
echo "Output directory:       $OUTPUT_DIR"
echo "============================================================"
echo


# ============================================================
# Expected dataset sizes
# ============================================================

EXPECTED_TRAIN_SIZE=4447
EXPECTED_VALID_SIZE=980

if [ "$TRAIN_SIZE" -ne "$EXPECTED_TRAIN_SIZE" ]; then
    echo "ERROR: unexpected Diverse-8 training size."
    echo "Expected: $EXPECTED_TRAIN_SIZE"
    echo "Actual:   $TRAIN_SIZE"
    exit 1
fi

if [ "$VALID_SIZE" -ne "$EXPECTED_VALID_SIZE" ]; then
    echo "ERROR: unexpected Diverse-8 validation size."
    echo "Expected: $EXPECTED_VALID_SIZE"
    echo "Actual:   $VALID_SIZE"
    exit 1
fi

echo "Dataset size checks passed."
echo


# ============================================================
# GPU information
# ============================================================

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo

nvidia-smi

echo


# ============================================================
# Python / environment information
# ============================================================

echo "Python:"
which python
python --version

echo

echo "Swift:"
python - <<'PY'
import swift
print(swift.__version__)
PY

echo


# ============================================================
# Create output directory
# ============================================================

mkdir -p "$OUTPUT_DIR"


# ============================================================
# SFT training
#
# Notes:
#
# - Same SFT recipe as Greedy-SFT and Best-of-8 SFT.
#
# - LoRA is applied to all linear layers in the language model.
#
# - Vision/audio towers and multimodal aligners are frozen.
#
# - Teacher <think> content has already been removed.
#
# - Diverse-8 retains all correct, complete, unique teacher
#   trajectories rather than only one trajectory per source.
#
# - Only the structured assistant response contributes to SFT
#   loss under ms-swift's default loss configuration.
#
# - max_length=16384 matches teacher generation.
#
# - truncation_strategy=delete avoids silently truncating
#   multimodal examples.
# ============================================================

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
    --eval_steps 20 \
    \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 2 \
    \
    --seed "$SEED" \
    --data_seed "$SEED" \
    \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard \
    \
    --dataloader_num_workers 4


STATUS=$?


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "MUStARD++ Diverse-8 SFT finished"
echo "============================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"