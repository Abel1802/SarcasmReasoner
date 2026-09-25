#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MCSD Base -> GRPO
#
# Initialization:
#   Qwen/Qwen2.5-Omni-7B
#
# SFT:
#   None
#
# Training data:
#   train = 1893 source instances
#
# Validation data:
#   valid = 406 source instances
#
# Test data:
#   NOT used during GRPO training/model selection
#
# GRPO:
#   1 epoch
#   8 generations per training source
#
# Validation:
#   stochastic GRPO reward evaluation every 1000 steps
#   8 generations per validation source
#
# Reward:
#   classification accuracy + 0.2 * structured format
#
# Hardware:
#   1 x H100
#
# Expected optimizer steps:
#   ~1893
# ============================================================


# ============================================================
# Model
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"


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

OUTPUT_DIR="results/mcsd/grpo/base"


# ============================================================
# GRPO configuration
# ============================================================

NUM_EPOCHS=1


# ------------------------------------------------------------
# Training batch
# ------------------------------------------------------------

PER_DEVICE_TRAIN_BATCH_SIZE=1

# One GPU:
#
# generation_batch_size
#   = per_device_train_batch_size
#     x gradient_accumulation_steps
#     x world_size
#
#   = 1 x 8 x 1
#   = 8
#
# This matches num_generations=8.
GRAD_ACC=8

NUM_GENERATIONS=8


# ------------------------------------------------------------
# Validation batch
# ------------------------------------------------------------

# In GRPO, per_device_eval_batch_size refers to completions,
# not source prompts.
#
# With:
#   num_generations_eval = 8
#   world_size           = 1
#
# eval batch size must be divisible by 8.
#
# Therefore:
#
#   8 completions
#   = 1 validation source x 8 generations
#
PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_GENERATIONS_EVAL=8

EVAL_STEPS=1000


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

LEARNING_RATE=1e-5

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

# We only generate textual reasoning/classification output.
export ENABLE_AUDIO_OUTPUT=0

# Audio is supplied explicitly as a separate modality.
export USE_AUDIO_IN_VIDEO=False

# Keep multimodal preprocessing consistent with
# teacher generation and SFT.
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

if [ ! -f "$PLUGIN" ]; then
    echo "ERROR: reward plugin not found:"
    echo "  $PLUGIN"
    exit 1
fi


# ============================================================
# Dataset statistics
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
VALID_SIZE=$(wc -l < "$VALID_DATA")


echo
echo "============================================================"
echo "MCSD Base -> GRPO"
echo "============================================================"

echo "Model:                     $MODEL"
echo

echo "Training data:             $TRAIN_DATA"
echo "Training sources:          $TRAIN_SIZE"
echo

echo "Validation data:           $VALID_DATA"
echo "Validation sources:        $VALID_SIZE"
echo

echo "SFT initialization:        None"
echo

echo "GPU count:                 1"
echo

echo "Train batch / GPU:         $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation:     $GRAD_ACC"
echo "Generation batch size:     $GRAD_ACC"
echo "Train generations/source:  $NUM_GENERATIONS"
echo

echo "Eval completion batch:     $PER_DEVICE_EVAL_BATCH_SIZE"
echo "Eval generations/source:   $NUM_GENERATIONS_EVAL"
echo "Eval every steps:          $EVAL_STEPS"
echo

echo "Epochs:                    $NUM_EPOCHS"
echo "Expected optimizer steps:  $TRAIN_SIZE"
echo

echo "Learning rate:             $LEARNING_RATE"
echo "LoRA rank:                 $LORA_RANK"
echo "LoRA alpha:                $LORA_ALPHA"
echo

echo "Max sequence length:       $MAX_LENGTH"
echo "Max completion length:     $MAX_COMPLETION_LENGTH"
echo

echo "Temperature:               $TEMPERATURE"
echo "Top-p:                     $TOP_P"
echo

echo "Reward:"
echo "  accuracy                 1.0"
echo "  format                   0.2"
echo

echo "Seed:                      $SEED"
echo "Output directory:          $OUTPUT_DIR"

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
    echo "ERROR: training generation batch size must be divisible"
    echo "by NUM_GENERATIONS."
    echo
    echo "Generation batch: $TRAIN_GENERATION_BATCH"
    echo "Num generations:  $NUM_GENERATIONS"
    exit 1
fi


if [ $((PER_DEVICE_EVAL_BATCH_SIZE % NUM_GENERATIONS_EVAL)) -ne 0 ]; then
    echo "ERROR: evaluation completion batch size must be"
    echo "divisible by NUM_GENERATIONS_EVAL."
    echo
    echo "Eval batch:            $PER_DEVICE_EVAL_BATCH_SIZE"
    echo "Eval num generations:  $NUM_GENERATIONS_EVAL"
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
# Output directory
# ============================================================

mkdir -p "$OUTPUT_DIR"


# ============================================================
# GRPO training
#
# Experimental protocol:
#
# Train:
#   - 1893 original MCSD train sources
#   - G=8 stochastic rollouts
#   - 1 epoch
#
# Validation:
#   - 406 held-out validation sources
#   - no parameter updates
#   - G_eval=8 stochastic rollouts
#   - reward evaluation every 1000 optimizer steps
#
# Checkpoints:
#   - 1000
#   - final (1893)
#
# Test:
#   - NOT used here
#   - reserved for final deterministic evaluation
#
# Offline evaluation:
#   Saved checkpoints will additionally be evaluated using
#   deterministic inference (temperature=0) on the complete
#   validation split for Accuracy / Macro-F1 / parse rate.
# ============================================================


python src/training/swift_rlhf_safe_video.py \
    --rlhf_type grpo \
    \
    --model "$MODEL" \
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
    --warmup_ratio 0.05 \
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


# ============================================================
# Finish
# ============================================================

echo
echo "============================================================"
echo "MCSD Base -> GRPO finished"
echo "============================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
