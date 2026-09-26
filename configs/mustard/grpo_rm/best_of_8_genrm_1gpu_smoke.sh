#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MUStARD++ Best-of-8-SFT -> GRPO + GenRM
# 1-GPU smoke test
#
# IMPORTANT:
# Best-of-8 is used ONLY for this engineering smoke test.
#
# This does NOT determine the SFT initialization used for the
# final GRPO + GenRM experiments.
#
# Before formal full training, we must decide whether the
# GenRM-guided GRPO comparison should start from:
#
#   - Greedy SFT
#   - Best-of-8 SFT
#   - Diverse-8 SFT
#   - or all three matched initializations
#
# This choice must be fixed before the formal experiments.
# ============================================================


# ============================================================
# Policy model / initialization
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"

SFT_CKPT="results/mustard/sft/best_of_8/v0-20260924-103030/checkpoint-140"


# ============================================================
# Training data
# ============================================================

TRAIN_DATA="data/mustard/processed/zero_shot_train.jsonl"


# ============================================================
# Reward plugin
# ============================================================

PLUGIN="src/plugins/sarcasm_grpo_reward.py"

GENRM_PROMPT="src/prompts/genrm/grounding_genrm_system.txt"


# ============================================================
# GenRM
# ============================================================

GENRM_MODEL="Qwen/Qwen2.5-Omni-3B"

GENRM_CKPT="results/mustard/genrm/qwen25_omni_3b/v1-20260926-001310/checkpoint-1000"


# ============================================================
# Output
# ============================================================

OUTPUT_DIR="results/mustard/grpo_rm/best_of_8_genrm_1gpu_smoke"


# ============================================================
# Smoke configuration
# ============================================================

MAX_STEPS=20

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRAD_ACC=8
NUM_GENERATIONS=8

LEARNING_RATE=1e-5
WARMUP_RATIO=0.05
BETA=0.04

MAX_LENGTH=16384
MAX_COMPLETION_LENGTH=1024

TEMPERATURE=1.0
TOP_P=0.95

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

export PYTORCH_ALLOC_CONF=expandable_segments:True


# ============================================================
# Required file checks
# ============================================================

for FILE in \
    "$TRAIN_DATA" \
    "$PLUGIN" \
    "$GENRM_PROMPT"
do
    if [ ! -f "$FILE" ]; then
        echo "ERROR: required file not found:"
        echo "  $FILE"
        exit 1
    fi
done


for DIR in \
    "$SFT_CKPT" \
    "$GENRM_CKPT"
do
    if [ ! -d "$DIR" ]; then
        echo "ERROR: checkpoint directory not found:"
        echo "  $DIR"
        exit 1
    fi
done


if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    echo "ERROR: policy SFT adapter_config.json missing:"
    echo "  $SFT_CKPT/adapter_config.json"
    exit 1
fi


if [ ! -f "$GENRM_CKPT/adapter_config.json" ]; then
    echo "ERROR: GenRM adapter_config.json missing:"
    echo "  $GENRM_CKPT/adapter_config.json"
    exit 1
fi


# ============================================================
# Plugin syntax
# ============================================================

python -m py_compile "$PLUGIN"

echo "Reward plugin syntax OK."
echo


# ============================================================
# Verify GenRM prompt source
# ============================================================

python - <<'PY'
from pathlib import Path
import importlib.util

plugin_path = Path(
    "src/plugins/sarcasm_grpo_reward.py"
).resolve()

spec = importlib.util.spec_from_file_location(
    "sarcasm_grpo_reward",
    plugin_path,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

expected_path = Path(
    "src/prompts/genrm/grounding_genrm_system.txt"
).resolve()

expected = expected_path.read_text(
    encoding="utf-8"
).strip()

assert (
    module.GENRM_SYSTEM_PROMPT_PATH.resolve()
    == expected_path
), "GenRM prompt path mismatch"

assert (
    module.GENRM_SYSTEM_PROMPT
    == expected
), "GenRM prompt content mismatch"

print("GenRM prompt path/content OK.")
PY

echo


# ============================================================
# Dataset sanity check
# ============================================================

python - <<'PY'
import json

path = "data/mustard/processed/zero_shot_train.jsonl"

with open(path, encoding="utf-8") as f:
    row = json.loads(next(f))

print("First training-row keys:")
print(sorted(row.keys()))

print()
print("label:", row.get("label"))
print("audios:", row.get("audios"))
print("videos:", row.get("videos"))

assert "label" in row, "Missing label"
assert row.get("audios"), "Missing audios"
assert row.get("videos"), "Missing videos"

print()
print("Training-row metadata OK.")
PY

echo


# ============================================================
# GRPO batch sanity check
# ============================================================

GENERATION_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC))

if [ $((GENERATION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
    echo "ERROR:"
    echo "Generation batch must be divisible by num_generations."
    echo
    echo "Generation batch: $GENERATION_BATCH"
    echo "Num generations:  $NUM_GENERATIONS"
    exit 1
fi

echo "GRPO batch check OK."
echo


# ============================================================
# Environment summary
# ============================================================

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "MUStARD++ Best-of-8 + GenRM GRPO smoke"
echo "============================================================"
echo
echo "Policy model:             $MODEL"
echo "Policy SFT adapter:       $SFT_CKPT"
echo
echo "GenRM base model:         $GENRM_MODEL"
echo "GenRM adapter:            $GENRM_CKPT"
echo "GenRM prompt:             $GENRM_PROMPT"
echo
echo "Training data:            $TRAIN_DATA"
echo "Training examples:        $(wc -l < "$TRAIN_DATA")"
echo
echo "GPU count:                1"
echo
echo "Train batch / GPU:        $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation:    $GRAD_ACC"
echo "Generation batch:         $GENERATION_BATCH"
echo "Generations/source:       $NUM_GENERATIONS"
echo
echo "Max steps:                $MAX_STEPS"
echo
echo "Reward:"
echo "  accuracy                1.0"
echo "  format                  0.2"
echo "  grounding               0.2"
echo
echo "Grounding reward:"
echo "  correctness-gated"
echo "  (text + audio + visual) / 3"
echo "  integration excluded"
echo
echo "Output:                   $OUTPUT_DIR"
echo
echo "============================================================"
echo


# ============================================================
# GPU / software
# ============================================================

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo

nvidia-smi

echo

python - <<'PY'
import torch
import swift
import trl

print("swift:", swift.__version__)
print("trl:", trl.__version__)
print("torch:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "CUDA is not available"

assert torch.cuda.device_count() == 1, (
    f"Expected exactly 1 visible GPU, "
    f"found {torch.cuda.device_count()}"
)

print("GPU:", torch.cuda.get_device_name(0))
PY

echo


# ============================================================
# GRPO + GenRM smoke
#
# Reward order:
#
#   1. sarcasm_accuracy   weight 1.0
#   2. sarcasm_format     weight 0.2
#   3. GenRM grounding    weight 0.2
#
# GenRM itself returns:
#
#   (text + audio + visual) / 3
#
# only for classification-correct completions.
# ============================================================

swift rlhf \
    --rlhf_type grpo \
    \
    --model "$MODEL" \
    --adapters "$SFT_CKPT" \
    --ref_adapters "$SFT_CKPT" \
    \
    --dataset "$TRAIN_DATA" \
    --split_dataset_ratio 0 \
    \
    --external_plugins "$PLUGIN" \
    \
    --reward_funcs \
        sarcasm_accuracy \
        sarcasm_format \
    \
    --reward_model "$GENRM_MODEL" \
    --reward_adapters "$GENRM_CKPT" \
    --reward_model_plugin sarcasm_grounding \
    \
    --reward_weights \
        1.0 \
        0.2 \
        0.2 \
    \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    \
    --freeze_vit true \
    --freeze_aligner true \
    \
    --torch_dtype bfloat16 \
    \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    \
    --num_generations "$NUM_GENERATIONS" \
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
    --report_to none \
    \
    --log_completions true \
    \
    --load_from_cache_file true \
    --dataset_num_proc 1 \
    --dataloader_num_workers 2


STATUS=$?


echo
echo "============================================================"
echo "GenRM GRPO smoke finished"
echo "============================================================"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "Output:      $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
