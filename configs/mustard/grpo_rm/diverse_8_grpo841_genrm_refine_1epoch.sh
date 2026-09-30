#!/bin/bash

set -euo pipefail


# ============================================================
# Core paths
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"

# Strong Mustard plain-GRPO checkpoint
POLICY_CKPT="results/mustard/grpo/diverse_8/seed42/v0-20260928-114059/checkpoint-841"

# Keep reference fixed at the same strong plain-GRPO checkpoint.
REF_CKPT="$POLICY_CKPT"

TRAIN_DATA="data/mustard/processed/zero_shot_train.jsonl"

PLUGIN="src/plugins/sarcasm_grpo_reward_linear.py"
GENRM_PROMPT="src/prompts/genrm/grounding_genrm_system.txt"

GENRM_MODEL="Qwen/Qwen2.5-Omni-3B"
GENRM_CKPT="results/mustard/genrm/qwen25_omni_3b/v1-20260926-001310/checkpoint-1000"

OUTPUT_DIR="results/mustard/grpo_rm/diverse_8_grpo841_genrm_refine_1epoch"


# ============================================================
# One-epoch refinement setup
# ============================================================
#
# Mustard train set:
#   expected 841 source prompts
#
# Therefore:
#
#   MAX_STEPS=841
#
# corresponds to approximately one complete pass over the
# training prompt set under the current GRPO configuration.
# ============================================================

MAX_STEPS=841

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRAD_ACC=8
NUM_GENERATIONS=8

# Conservative refinement learning rate
LEARNING_RATE=2e-6
WARMUP_RATIO=0.05

# KL regularization against original plain-GRPO policy
BETA=0.04

LORA_RANK=8
LORA_ALPHA=32

MAX_LENGTH=16384
MAX_COMPLETION_LENGTH=1024

TEMPERATURE=1.0
TOP_P=0.95

SEED=42


# ============================================================
# Reward weights
# ============================================================

ACCURACY_WEIGHT=1.0
FORMAT_WEIGHT=0.2

# Conservative GenRM contribution
GROUNDING_WEIGHT=0.05


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
# GenRM refinement schedule
# ============================================================
#
# Use the same RELATIVE schedule as the original linear design:
#
#   0%--10%:
#       GenRM OFF
#
#   10%--30%:
#       linear ramp 0 -> 1
#
#   30%--100%:
#       full GenRM
#
# For MAX_STEPS=841:
#
#   ~ step 0--84:
#       OFF
#
#   ~ step 84--252:
#       ramp
#
#   ~ step 252--841:
#       full
#
# ============================================================

export GENRM_ZERO_RATIO=0.10
export GENRM_FULL_RATIO=0.30

export GENRM_SCHEDULE_LOG_STEPS=25


# ============================================================
# Required path checks
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
    "$POLICY_CKPT" \
    "$REF_CKPT" \
    "$GENRM_CKPT"
do
    if [ ! -d "$DIR" ]; then
        echo "ERROR: checkpoint directory not found:" >&2
        echo "  $DIR" >&2
        exit 1
    fi
done


if [ ! -f "$POLICY_CKPT/adapter_config.json" ]; then
    echo "ERROR: policy adapter_config.json missing:" >&2
    echo "  $POLICY_CKPT/adapter_config.json" >&2
    exit 1
fi


if [ ! -f "$REF_CKPT/adapter_config.json" ]; then
    echo "ERROR: reference adapter_config.json missing:" >&2
    echo "  $REF_CKPT/adapter_config.json" >&2
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
# Dataset checks
# ============================================================

TRAIN_SIZE=$(wc -l < "$TRAIN_DATA")
EXPECTED_TRAIN_SIZE=841

if [ "$TRAIN_SIZE" -ne "$EXPECTED_TRAIN_SIZE" ]; then
    echo "ERROR: unexpected Mustard train size." >&2
    echo "Expected: $EXPECTED_TRAIN_SIZE" >&2
    echo "Actual:   $TRAIN_SIZE" >&2
    exit 1
fi


if [ "$MAX_STEPS" -ne "$TRAIN_SIZE" ]; then
    echo "ERROR: one-epoch configuration mismatch." >&2
    echo "MAX_STEPS: $MAX_STEPS" >&2
    echo "TRAIN_SIZE: $TRAIN_SIZE" >&2
    exit 1
fi


echo "Dataset size checks passed."
echo "$TRAIN_DATA: $TRAIN_SIZE rows OK"
echo
echo "One-epoch refinement budget:"
echo "  $MAX_STEPS steps"
echo


# ============================================================
# Required multimodal fields
# ============================================================

TRAIN_DATA="$TRAIN_DATA" python - <<'PY'
import json
import os

path = os.environ["TRAIN_DATA"]

required = [
    "label",
    "transcript",
    "audios",
    "videos",
    "messages",
]

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
                    f"{path}:{line_no}: missing {key!r}"
                )

        if not (
            isinstance(row["transcript"], str)
            and row["transcript"].strip()
        ):
            raise RuntimeError(
                f"{path}:{line_no}: empty transcript"
            )

        if not row["audios"]:
            raise RuntimeError(
                f"{path}:{line_no}: missing audio"
            )

        if not row["videos"]:
            raise RuntimeError(
                f"{path}:{line_no}: missing video"
            )

        label = int(row["label"])

        if label not in (0, 1):
            raise RuntimeError(
                f"{path}:{line_no}: invalid label={label}"
            )

print(f"Multimodal dataset check passed: {total} rows")
PY

echo


# ============================================================
# Verify GenRM plugin / prompt / schedule
# ============================================================

PLUGIN="$PLUGIN" \
GENRM_PROMPT="$GENRM_PROMPT" \
MAX_STEPS="$MAX_STEPS" \
GENRM_ZERO_RATIO="$GENRM_ZERO_RATIO" \
GENRM_FULL_RATIO="$GENRM_FULL_RATIO" \
python - <<'PY'
import importlib.util
import os
from pathlib import Path


plugin_path = Path(
    os.environ["PLUGIN"]
).resolve()

spec = importlib.util.spec_from_file_location(
    "sarcasm_grpo_reward_linear",
    plugin_path,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# ------------------------------------------------------------
# Prompt check
# ------------------------------------------------------------

expected_path = Path(
    os.environ["GENRM_PROMPT"]
).resolve()

expected_prompt = expected_path.read_text(
    encoding="utf-8"
).strip()

assert (
    module.GENRM_SYSTEM_PROMPT_PATH.resolve()
    == expected_path
)

assert (
    module.GENRM_SYSTEM_PROMPT
    == expected_prompt
)


# ------------------------------------------------------------
# Plugin registration check
# ------------------------------------------------------------

assert (
    module.rm_plugins["sarcasm_grounding_linear"]
    is module.SarcasmGroundingLinearRMPlugin
)


# ------------------------------------------------------------
# Schedule check
# ------------------------------------------------------------

max_steps = int(
    os.environ["MAX_STEPS"]
)

zero_ratio = float(
    os.environ["GENRM_ZERO_RATIO"]
)

full_ratio = float(
    os.environ["GENRM_FULL_RATIO"]
)


def factor(step):
    return module.linear_grounding_factor(
        step,
        max_steps,
        zero_ratio=zero_ratio,
        full_ratio=full_ratio,
    )


# Start: OFF
assert abs(factor(0) - 0.0) < 1e-8

# Clearly inside OFF phase
assert abs(factor(50) - 0.0) < 1e-8

# Around 10% boundary
zero_step = int(round(max_steps * zero_ratio))

# Around 30% boundary
full_step = int(round(max_steps * full_ratio))

# End must be full
assert abs(factor(max_steps) - 1.0) < 1e-8


print("GenRM prompt path/content OK.")
print("GenRM plugin registration OK.")
print()

print(
    f"Expected zero boundary: ~step {zero_step}"
)

print(
    f"Expected full boundary: ~step {full_step}"
)

print()

print("GenRM schedule:")

for step in [
    0,
    25,
    50,
    75,
    84,
    100,
    125,
    150,
    200,
    252,
    300,
    400,
    500,
    600,
    700,
    800,
    841,
]:
    print(
        f"  step {step:4d}: "
        f"factor={factor(step):.6f}"
    )

print()
print("GenRM schedule verification passed.")
PY

echo


# ============================================================
# GRPO batch sanity check
# ============================================================

TRAIN_GENERATION_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC))

if [ $((TRAIN_GENERATION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
    echo "ERROR:" >&2
    echo \
        "Training generation batch must be divisible by NUM_GENERATIONS." \
        >&2
    exit 1
fi


echo "GRPO batch checks passed."
echo


# ============================================================
# Training summary
# ============================================================

echo "================================================================"
echo "Mustard plain-GRPO -> 1-epoch GenRM refinement"
echo "================================================================"
echo

echo "Policy model:"
echo "  $MODEL"
echo

echo "Policy initialization:"
echo "  $POLICY_CKPT"
echo

echo "Reference initialization:"
echo "  $REF_CKPT"
echo

echo "GenRM:"
echo "  model                     $GENRM_MODEL"
echo "  checkpoint                $GENRM_CKPT"
echo

echo "Dataset:"
echo "  training prompts          $TRAIN_SIZE"
echo

echo "Training:"
echo "  max steps                 $MAX_STEPS"
echo "  approx dataset epochs     1.0"
echo "  train batch / GPU         $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  gradient accumulation     $GRAD_ACC"
echo "  generation batch          $TRAIN_GENERATION_BATCH"
echo "  generations / prompt      $NUM_GENERATIONS"
echo

echo "Optimizer:"
echo "  learning rate             $LEARNING_RATE"
echo "  warmup ratio              $WARMUP_RATIO"
echo "  KL beta                   $BETA"
echo

echo "Reward:"
echo "  accuracy                  $ACCURACY_WEIGHT"
echo "  format                    $FORMAT_WEIGHT"
echo "  grounding                 $GROUNDING_WEIGHT"
echo

echo "GenRM schedule:"
echo "  OFF                       approx step 0--84"
echo "  linear ramp               approx step 84--252"
echo "  full                      approx step 252--841"
echo

echo "Checkpointing:"
echo "  save every                50 steps"
echo "  keep up to                20 checkpoints"
echo

echo "Expected checkpoints:"
echo "  50 / 100 / 150 / 200 / 250"
echo "  300 / 350 / 400 / 450 / 500"
echo "  550 / 600 / 650 / 700 / 750"
echo "  800 / final"
echo

echo "Seed:"
echo "  $SEED"
echo

echo "Output:"
echo "  $OUTPUT_DIR"
echo

echo "IMPORTANT:"
echo "  Policy initialization = Mustard plain-GRPO checkpoint-841"
echo "  Reference policy      = Mustard plain-GRPO checkpoint-841"
echo "  GenRM schedule        = 10% OFF / 10--30% ramp / 30%+ full"
echo "================================================================"
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

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "GPU count:",
    torch.cuda.device_count()
)

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)
PY

echo


# ============================================================
# Output
# ============================================================

mkdir -p "$OUTPUT_DIR"


# ============================================================
# One-epoch GRPO + GenRM refinement
# ============================================================

set +e

swift rlhf \
    --rlhf_type grpo \
    \
    --model "$MODEL" \
    --adapters "$POLICY_CKPT" \
    --ref_adapters "$REF_CKPT" \
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
    --reward_model_plugin sarcasm_grounding_linear \
    \
    --reward_weights \
        "$ACCURACY_WEIGHT" \
        "$FORMAT_WEIGHT" \
        "$GROUNDING_WEIGHT" \
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
    --max_steps "$MAX_STEPS" \
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
    --save_steps 50 \
    --save_total_limit 20 \
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


# ============================================================
# Finish
# ============================================================

echo
echo "================================================================"
echo "Mustard 1-epoch GenRM refinement finished"
echo "================================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================"

exit "$STATUS"
