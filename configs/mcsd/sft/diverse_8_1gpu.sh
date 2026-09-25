#!/bin/bash

set -e
set -o pipefail

MODEL="Qwen/Qwen2.5-Omni-7B"

TRAIN_DATA="data/mcsd/processed/sft/sft_diverse_8_train.jsonl"
VALID_DATA="data/mcsd/processed/sft/sft_diverse_8_valid.jsonl"

OUTPUT_DIR="results/mcsd/sft/diverse_8"

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


# ============================================================
# Dataset statistics
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
VALID_SIZE=$(wc -l < "$VALID_DATA")

echo
echo "============================================================"
echo "MCSD Diverse-8 SFT"
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
# GPU information
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

echo "Swift:"
swift --version || true

echo


# ============================================================
# Output directory
# ============================================================

mkdir -p "$OUTPUT_DIR"


# ============================================================
# SFT training
# ============================================================

python src/training/swift_sft_safe_video.py \
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

echo
echo "============================================================"
echo "MCSD Diverse-8 SFT finished"
echo "============================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
