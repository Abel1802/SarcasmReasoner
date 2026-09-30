#!/bin/bash

set -euo pipefail


# ============================================================
# Core paths
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"

# Strong plain-GRPO starting point
POLICY_CKPT="results/mcsd/grpo/diverse_8/v1-20260927-103912/checkpoint-1200"

# Keep the reference fixed at the original strong plain-GRPO policy.
REF_CKPT="$POLICY_CKPT"

TRAIN_DATA="data/mcsd/processed/zero_shot_train.jsonl"

PLUGIN="src/plugins/sarcasm_grpo_reward_linear.py"
GENRM_PROMPT="src/prompts/genrm/grounding_genrm_system.txt"

GENRM_MODEL="Qwen/Qwen2.5-Omni-3B"
GENRM_CKPT="results/mcsd/genrm/qwen25_omni_3b/v0-20260927-193102/checkpoint-1866"

OUTPUT_DIR="results/mcsd/grpo_rm/diverse_8_grpo1200_genrm_refine_1epoch"


# ============================================================
# One-epoch refinement setup
# ============================================================
#
# MCSD train set:
#   1893 source prompts
#
# Previous runs show:
#
#   epoch ~= global_step / 1893
#
# Therefore MAX_STEPS=1893 corresponds to approximately
# one complete pass over the training prompt set.
# ============================================================

MAX_STEPS=1893

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRAD_ACC=8
NUM_GENERATIONS=8

LEARNING_RATE=2e-6
WARMUP_RATIO=0.05

# KL regularization against the original plain-GRPO policy.
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

# Conservative GenRM weight.
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
# Preserve the ABSOLUTE GenRM schedule used in the 500-step run:
#
#   step   0--50:
#       grounding OFF
#
#   step  50--150:
#       linear ramp 0 -> 1
#
#   step 150--1893:
#       full GenRM
#
# The plugin uses relative progress, so:
#
#   ZERO_RATIO =  50 / 1893
#   FULL_RATIO = 150 / 1893
#
# ============================================================

GENRM_ZERO_STEPS=50
GENRM_FULL_STEPS=150

export GENRM_ZERO_RATIO="0.026413101954569466"
export GENRM_FULL_RATIO="0.07923930269413629"

export GENRM_SCHEDULE_LOG_STEPS=25


# ============================================================
# Check basic paths
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
EXPECTED_TRAIN_SIZE=1893

if [ "$TRAIN_SIZE" -ne "$EXPECTED_TRAIN_SIZE" ]; then
    echo "ERROR: unexpected MCSD train size." >&2
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


# Beginning: OFF
assert abs(factor(0) - 0.0) < 1e-8

# Step 50: boundary of OFF phase
assert abs(factor(50) - 0.0) < 1e-6

# Step 100: midpoint of 50--150 ramp
assert 0.49 <= factor(100) <= 0.51, factor(100)

# Step 150: full GenRM
assert abs(factor(150) - 1.0) < 1e-6

# End: still full
assert abs(factor(max_steps) - 1.0) < 1e-8


print("GenRM prompt path/content OK.")
print("GenRM plugin registration OK.")
print()

print("GenRM schedule:")
for step in [
    0,
    25,
    50,
    75,
    100,
    125,
    150,
    250,
    500,
    750,
    1000,
    1250,
    1500,
    1750,
    1893,
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
echo "MCSD plain-GRPO -> 1-epoch GenRM refinement"
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
echo "  OFF                       step 0--50"
echo "  linear ramp               step 50--150"
echo "  full                      step 150--1893"
echo

echo "Checkpointing:"
echo "  save every                50 steps"
echo "  keep up to                40 checkpoints"
echo

echo "Key checkpoints:"
echo "  50 / 100 / 150 / 200 / 250 / 300 / 350 / 400 / 450 / 500"
echo "  550 / 600 / ... / 1800 / 1850"
echo "  final training step       1893"
echo

echo "Seed:"
echo "  $SEED"
echo

echo "Output:"
echo "  $OUTPUT_DIR"
echo

echo "IMPORTANT:"
echo "  Policy initialization = plain-GRPO checkpoint-1200"
echo "  Reference policy      = plain-GRPO checkpoint-1200"
echo "  GenRM begins at step 50 and reaches full weight at step 150"
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
    --save_steps 100 \
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
echo "MCSD 1-epoch GenRM refinement finished"
echo "================================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================"

exit "$STATUS"
