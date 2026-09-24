#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MUStARD++ Base -> GRPO
# 1-GPU smoke test
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"

TRAIN_DATA="data/mustard/processed/grpo/smoke_train.jsonl"

OUTPUT_DIR="results/mustard/grpo/base_1gpu_smoke"

PLUGIN="src/plugins/sarcasm_grpo_reward.py"


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
# Checks
# ============================================================

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: dataset not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$PLUGIN" ]; then
    echo "ERROR: reward plugin not found: $PLUGIN"
    exit 1
fi


mkdir -p "$OUTPUT_DIR"


echo "============================================================"
echo "MUStARD++ Base GRPO - 1 GPU smoke test"
echo "============================================================"
echo "Model:       $MODEL"
echo "Dataset:     $TRAIN_DATA"
echo "Examples:    $(wc -l < "$TRAIN_DATA")"
echo "GPU:         1"
echo "Generations: 8"
echo "Max steps:   1"
echo "Output:      $OUTPUT_DIR"
echo "============================================================"


# ============================================================
# GRPO
# ============================================================

swift rlhf \
    --rlhf_type grpo \
    \
    --model "$MODEL" \
    \
    --dataset "$TRAIN_DATA" \
    --split_dataset_ratio 0 \
    \
    --external_plugins "$PLUGIN" \
    --reward_funcs sarcasm_accuracy sarcasm_format \
    --reward_weights 1.0 0.2 \
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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    \
    --num_generations 8 \
    \
    --temperature 1.0 \
    --top_p 0.95 \
    \
    --max_length 16384 \
    --max_completion_length 1024 \
    \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    \
    --gradient_checkpointing true \
    \
    --max_steps 10 \
    \
    --logging_steps 1 \
    --save_strategy no \
    \
    --seed 42 \
    --data_seed 42 \
    \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    \
    --log_completions true \
    \
    --dataset_num_proc 1 \
    --dataloader_num_workers 2


STATUS=$?

echo
echo "============================================================"
echo "Smoke test finished"
echo "Exit status: $STATUS"
echo "End time:    $(date)"
echo "============================================================"

exit "$STATUS"