#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MCSD Diverse-8-SFT -> GRPO
#
# Base model:
#   Qwen/Qwen2.5-Omni-7B
#
# Initialization:
#   Diverse-8-SFT LoRA checkpoint-700
#
# Train:
#   1893 source instances
#
# Validation:
#   406 source instances
#
# GRPO:
#   1 epoch
#   G = 8
#
# Reward:
#   accuracy + 0.2 * format
#
# Hardware:
#   1 x H100
# ============================================================


# ============================================================
# Model / SFT initialization
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"

SFT_CKPT="results/mcsd/sft/diverse_8/v0-20260924-213700/checkpoint-700"


# ============================================================
# Data
# ============================================================

TRAIN_DATA="data/mcsd/processed/zero_shot_train.jsonl"
VALID_DATA="data/mcsd/processed/zero_shot_valid.jsonl"


# ============================================================
# Reward plugin
# ============================================================

PLUGIN="src/plugins/sarcasm_grpo_reward.py"


# ============================================================
# Output
# ============================================================

OUTPUT_DIR="results/mcsd/grpo/diverse_8"


# ============================================================
# GRPO configuration
# ============================================================

NUM_EPOCHS=1


# ------------------------------------------------------------
# Training batch
# ------------------------------------------------------------

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRAD_ACC=8
NUM_GENERATIONS=8


# ------------------------------------------------------------
# Validation batch
# ------------------------------------------------------------

PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_GENERATIONS_EVAL=8

EVAL_STEPS=1000


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

LEARNING_RATE=1e-5
WARMUP_RATIO=0.05

BETA=0.04


# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_RANK=8
LORA_ALPHA=32


# ------------------------------------------------------------
# Length
# ------------------------------------------------------------

MAX_LENGTH=16384
MAX_COMPLETION_LENGTH=1024


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

TEMPERATURE=1.0
TOP_P=0.95


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

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
# Required file / directory checks
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


if [ ! -f "$PLUGIN" ]; then
    echo "ERROR: reward plugin not found:"
    echo "  $PLUGIN"
    exit 1
fi


if [ ! -d "$SFT_CKPT" ]; then
    echo "ERROR: SFT checkpoint directory not found:"
    echo "  $SFT_CKPT"
    exit 1
fi


if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found:"
    echo "  $SFT_CKPT/adapter_config.json"
    exit 1
fi


# ============================================================
# Dataset statistics
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
VALID_SIZE=$(wc -l < "$VALID_DATA")


echo
echo "============================================================"
echo "MCSD Diverse-8-SFT -> GRPO"
echo "============================================================"
echo

echo "Base model:                 $MODEL"
echo "SFT checkpoint:             $SFT_CKPT"
echo

echo "Training data:              $TRAIN_DATA"
echo "Training sources:           $TRAIN_SIZE"
echo

echo "Validation data:            $VALID_DATA"
echo "Validation sources:         $VALID_SIZE"
echo

echo "GPU count:                  1"
echo

echo "Train batch / GPU:          $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation:      $GRAD_ACC"
echo "Generation batch size:      $((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC))"
echo "Train generations/source:   $NUM_GENERATIONS"
echo

echo "Eval completion batch:      $PER_DEVICE_EVAL_BATCH_SIZE"
echo "Eval generations/source:    $NUM_GENERATIONS_EVAL"
echo "Eval every steps:           $EVAL_STEPS"
echo

echo "Epochs:                     $NUM_EPOCHS"
echo "Expected optimizer steps:   $TRAIN_SIZE"
echo

echo "Learning rate:              $LEARNING_RATE"
echo "Warmup ratio:               $WARMUP_RATIO"
echo "Beta:                       $BETA"
echo

echo "LoRA rank:                  $LORA_RANK"
echo "LoRA alpha:                 $LORA_ALPHA"
echo

echo "Max sequence length:        $MAX_LENGTH"
echo "Max completion length:      $MAX_COMPLETION_LENGTH"
echo

echo "Temperature:                $TEMPERATURE"
echo "Top-p:                      $TOP_P"
echo

echo "Reward:"
echo "  accuracy                  1.0"
echo "  format                    0.2"
echo

echo "Seed:                       $SEED"
echo "Output directory:           $OUTPUT_DIR"
echo

echo "============================================================"
echo


# ============================================================
# Dataset size sanity checks
# ============================================================

EXPECTED_TRAIN_SIZE=1893
EXPECTED_VALID_SIZE=406


if [ "$TRAIN_SIZE" -ne "$EXPECTED_TRAIN_SIZE" ]; then
    echo "ERROR: unexpected MCSD train size."
    echo "Expected: $EXPECTED_TRAIN_SIZE"
    echo "Actual:   $TRAIN_SIZE"
    exit 1
fi


if [ "$VALID_SIZE" -ne "$EXPECTED_VALID_SIZE" ]; then
    echo "ERROR: unexpected MCSD validation size."
    echo "Expected: $EXPECTED_VALID_SIZE"
    echo "Actual:   $VALID_SIZE"
    exit 1
fi


echo "Dataset size checks passed."
echo


# ============================================================
# GRPO batch sanity checks
# ============================================================

TRAIN_GENERATION_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC))


if [ $((TRAIN_GENERATION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
    echo "ERROR:"
    echo "Training generation batch size must be divisible"
    echo "by NUM_GENERATIONS."
    echo
    echo "Generation batch: $TRAIN_GENERATION_BATCH"
    echo "Num generations:  $NUM_GENERATIONS"
    exit 1
fi


if [ $((PER_DEVICE_EVAL_BATCH_SIZE % NUM_GENERATIONS_EVAL)) -ne 0 ]; then
    echo "ERROR:"
    echo "Evaluation completion batch size must be divisible"
    echo "by NUM_GENERATIONS_EVAL."
    echo
    echo "Eval batch:       $PER_DEVICE_EVAL_BATCH_SIZE"
    echo "Eval generations: $NUM_GENERATIONS_EVAL"
    exit 1
fi


echo "GRPO batch checks passed."
echo


# ============================================================
# GPU / environment information
# ============================================================

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo

nvidia-smi

echo


echo "Python:"
which python
python --version

echo


echo "Swift / TRL / Torch:"

python - <<'PY'
import swift
import trl
import torch

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
# SFT checkpoint information
# ============================================================

echo "Diverse-8-SFT checkpoint:"
echo "  $SFT_CKPT"
echo

echo "Checkpoint files:"
ls -lh "$SFT_CKPT"
echo


# ============================================================
# Output directory
# ============================================================

mkdir -p "$OUTPUT_DIR"


# ============================================================
# GRPO training
#
# Policy at step 0:
#
#   Qwen/Qwen2.5-Omni-7B
#          +
#   Diverse-8-SFT LoRA checkpoint-700
#
# Reference policy:
#
#   Qwen/Qwen2.5-Omni-7B
#          +
#   same Diverse-8-SFT LoRA
#
# GRPO optimizer / scheduler start fresh.
#
# This is NOT resume_from_checkpoint.
#
# Train:
#   1893 sources
#   G=8
#   1 epoch
#
# Validation:
#   406 sources
#   G_eval=8
#   every 1000 optimizer steps
#
# Test:
#   NOT used here.
# ============================================================


swift rlhf \
    --rlhf_type grpo \
    \
    --model "$MODEL" \
    --adapters "$SFT_CKPT" \
    --ref_adapters "$SFT_CKPT" \
    \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VALID_DATA" \
    --split_dataset_ratio 0 \
    \
    --external_plugins "$PLUGIN" \
    --reward_funcs sarcasm_accuracy sarcasm_format \
    --reward_weights 1.0 0.2 \
    \
    --tuner_type lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    \
    --freeze_vit true \
    --freeze_aligner true \
    \
    --torch_dtype bfloat16 \
    \
    --num_train_epochs "$NUM_EPOCHS" \
    \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    \
    --num_generations "$NUM_GENERATIONS" \
    \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --num_generations_eval "$NUM_GENERATIONS_EVAL" \
    \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    \
    --max_length "$MAX_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --beta "$BETA" \
    \
    --gradient_checkpointing true \
    \
    --logging_steps 5 \
    \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 5 \
    \
    --seed "$SEED" \
    --data_seed "$SEED" \
    \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard \
    \
    --log_completions true \
    \
    --load_from_cache_file true \
    --dataset_num_proc 1 \
    --dataloader_num_workers 4


STATUS=$?


echo
echo "============================================================"
echo "MCSD Diverse-8-SFT -> GRPO finished"
echo "============================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
