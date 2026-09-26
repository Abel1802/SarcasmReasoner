#!/bin/bash

set -euo pipefail

MODEL="Qwen/Qwen2.5-Omni-7B"
SFT_CKPT="results/mustard/sft/diverse_8/v0-20260924-103353/checkpoint-560"

TRAIN_DATA="data/mustard/processed/zero_shot_train.jsonl"
PLUGIN="src/plugins/sarcasm_grpo_reward.py"
GENRM_PROMPT="src/prompts/genrm/grounding_genrm_system.txt"

GENRM_MODEL="Qwen/Qwen2.5-Omni-3B"
GENRM_CKPT="results/mustard/genrm/qwen25_omni_3b/v1-20260926-001310/checkpoint-1000"

OUTPUT_DIR="results/mustard/grpo_rm/diverse_8"

NUM_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRAD_ACC=8
NUM_GENERATIONS=8

LEARNING_RATE=1e-5
WARMUP_RATIO=0.05
BETA=0.04

LORA_RANK=8
LORA_ALPHA=32

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
# Required paths
# ============================================================

for FILE in \
    "$TRAIN_DATA" \
    "$PLUGIN" \
    "$GENRM_PROMPT"
do
    if [ ! -f "$FILE" ]; then
        echo "ERROR: required file not found:" >&2
        echo "  $FILE" >&2
        exit 1
    fi
done


for DIR in \
    "$SFT_CKPT" \
    "$GENRM_CKPT"
do
    if [ ! -d "$DIR" ]; then
        echo "ERROR: checkpoint directory not found:" >&2
        echo "  $DIR" >&2
        exit 1
    fi
done


if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    echo "ERROR: SFT adapter_config.json missing:" >&2
    echo "  $SFT_CKPT/adapter_config.json" >&2
    exit 1
fi


if [ ! -f "$GENRM_CKPT/adapter_config.json" ]; then
    echo "ERROR: GenRM adapter_config.json missing:" >&2
    echo "  $GENRM_CKPT/adapter_config.json" >&2
    exit 1
fi


python -m py_compile "$PLUGIN"

echo "Reward plugin syntax OK."
echo


# ============================================================
# Dataset size
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")

EXPECTED_TRAIN_SIZE=841


if [ "$TRAIN_SIZE" -ne "$EXPECTED_TRAIN_SIZE" ]; then
    echo "ERROR: unexpected train size."
    echo "Expected: $EXPECTED_TRAIN_SIZE"
    echo "Actual:   $TRAIN_SIZE"
    exit 1
fi


echo "Dataset size checks passed."
echo


# ============================================================
# GenRM-required dataset fields
# ============================================================

TRAIN_DATA="$TRAIN_DATA" python - <<'PY'
import json
import os

files = [os.environ["TRAIN_DATA"]]

required = [
    "label",
    "transcript",
    "audios",
    "videos",
    "messages",
]

for path in files:

    total = 0

    with open(path, encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            row = json.loads(line)
            total += 1

            for key in required:
                if key not in row:
                    raise RuntimeError(
                        f"{path}:{line_no}: "
                        f"missing {key!r}"
                    )

            if not (
                isinstance(row["transcript"], str)
                and row["transcript"].strip()
            ):
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    "empty transcript"
                )

            if not row["audios"]:
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    "missing audio"
                )

            if not row["videos"]:
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    "missing video"
                )

            label = int(row["label"])

            if label not in (0, 1):
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    f"invalid label={label}"
                )

    print(f"{path}: {total} rows OK")
PY

echo


# ============================================================
# Verify GenRM prompt source
# ============================================================

PLUGIN="$PLUGIN" GENRM_PROMPT="$GENRM_PROMPT" python - <<'PY'
import importlib.util
import os
from pathlib import Path

plugin_path = Path(os.environ["PLUGIN"]).resolve()

spec = importlib.util.spec_from_file_location(
    "sarcasm_grpo_reward",
    plugin_path,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

expected_path = Path(os.environ["GENRM_PROMPT"]).resolve()

expected = (
    expected_path
    .read_text(encoding="utf-8")
    .strip()
)

assert (
    module.GENRM_SYSTEM_PROMPT_PATH.resolve()
    == expected_path
)

assert (
    module.GENRM_SYSTEM_PROMPT
    == expected
)

print("GenRM prompt path/content OK.")
PY

echo


# ============================================================
# GRPO batch sanity checks
# ============================================================

TRAIN_GENERATION_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC))


if [ $((TRAIN_GENERATION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
    echo "ERROR:"
    echo "Training generation batch must be divisible"
    echo "by NUM_GENERATIONS."
    exit 1
fi



echo "GRPO batch checks passed."
echo


# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "MUStARD++ Diverse-8-SFT -> GRPO + GenRM"
echo "============================================================"
echo
echo "Policy model:               $MODEL"
echo "SFT checkpoint:             $SFT_CKPT"
echo
echo "GenRM model:                $GENRM_MODEL"
echo "GenRM checkpoint:           $GENRM_CKPT"
echo
echo "Training sources:           $TRAIN_SIZE"
echo
echo "GPU count:                  1"
echo
echo "Train batch / GPU:          $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation:      $GRAD_ACC"
echo "Generation batch:           $TRAIN_GENERATION_BATCH"
echo "Generations/source:         $NUM_GENERATIONS"
echo
echo
echo "Epochs:                     $NUM_EPOCHS"
echo "Expected optimizer steps:   $TRAIN_SIZE"
echo
echo "Learning rate:              $LEARNING_RATE"
echo "Warmup ratio:               $WARMUP_RATIO"
echo "Beta:                       $BETA"
echo
echo "Reward:"
echo "  accuracy                  1.0"
echo "  format                    0.2"
echo "  grounding                 0.2"
echo
echo "Grounding:"
echo "  correctness-gated"
echo "  (text + audio + visual) / 3"
echo "  integration excluded"
echo
echo "Seed:                       $SEED"
echo "Output:                     $OUTPUT_DIR"
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

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1

print("GPU:", torch.cuda.get_device_name(0))
PY

echo


# ============================================================
# Output
# ============================================================

mkdir -p "$OUTPUT_DIR"


# ============================================================
# Formal GRPO + GenRM training
#
# Policy initialization:
#
#   Qwen/Qwen2.5-Omni-7B
#          +
#   Diverse-8-SFT checkpoint-105
#
# Reference policy:
#
#   same base model
#          +
#   same Diverse-8-SFT checkpoint-105
#
# Reward:
#
#   accuracy      1.0
#   format        0.2
#   GenRM         0.2
#
# Train:
#   841 sources
#   G=8
#   1 epoch
#
# Online validation:
#   disabled during training
#
# Checkpoints:
#   every 200 steps
#
# Validation is performed offline after training.
#
# Primary matched test comparison:
#   checkpoint-800
# ============================================================

set +e
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
    --eval_strategy no \
    \
    --save_strategy steps \
    --save_steps 200 \
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
set -e


echo
echo "============================================================"
echo "MUStARD++ Diverse-8-SFT -> GRPO + GenRM finished"
echo "============================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
