#!/bin/bash

set -euo pipefail


# ============================================================
# Core paths
# ============================================================

MODEL="Qwen/Qwen2.5-Omni-7B"

# Strong plain-GRPO starting point
POLICY_CKPT="results/mcsd/grpo/diverse_8/v1-20260927-103912/checkpoint-1200"

# IMPORTANT:
# Keep the reference at exactly the same strong plain-GRPO checkpoint.
REF_CKPT="$POLICY_CKPT"

TRAIN_DATA="data/mcsd/processed/zero_shot_train.jsonl"

PLUGIN="src/plugins/sarcasm_grpo_reward_linear.py"
GENRM_PROMPT="src/prompts/genrm/grounding_genrm_system.txt"

GENRM_MODEL="Qwen/Qwen2.5-Omni-3B"
GENRM_CKPT="results/mcsd/genrm/qwen25_omni_3b/v0-20260927-193102/checkpoint-1866"

OUTPUT_DIR="results/mcsd/grpo_rm/diverse_8_grpo1200_genrm_refine_500"


# ============================================================
# Refinement setup
# ============================================================

MAX_STEPS=500

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRAD_ACC=8
NUM_GENERATIONS=8

# Conservative LR for refinement.
LEARNING_RATE=2e-6
WARMUP_RATIO=0.05

# Stay close to the strong plain-GRPO policy.
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

# Much smaller than previous 0.2.
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
# max_steps = 250
#
# 0%--10%:
#     grounding OFF
#     ~ step 0--25
#
# 10%--30%:
#     linear 0 -> 1
#     ~ step 25--75
#
# 30%--100%:
#     full factor
#     ~ step 75--250
#
# IMPORTANT:
# Maximum actual grounding coefficient is still controlled by
# --reward_weights = 0.05.
#
# Plugin must NOT multiply by 0.05 internally.
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
EXPECTED_TRAIN_SIZE=1893

if [ "$TRAIN_SIZE" -ne "$EXPECTED_TRAIN_SIZE" ]; then
    echo "ERROR: unexpected MCSD train size."
    echo "Expected: $EXPECTED_TRAIN_SIZE"
    echo "Actual:   $TRAIN_SIZE"
    exit 1
fi

echo "Dataset size checks passed."
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

print(f"{path}: {total} rows OK")
PY

echo


# ============================================================
# Verify GenRM plugin / prompt / schedule
# ============================================================

PLUGIN="$PLUGIN" \
GENRM_PROMPT="$GENRM_PROMPT" \
MAX_STEPS="$MAX_STEPS" \
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

assert (
    module.rm_plugins["sarcasm_grounding_linear"]
    is module.SarcasmGroundingLinearRMPlugin
)

max_steps = int(
    os.environ["MAX_STEPS"]
)

assert (
    module.linear_grounding_factor(
        0,
        max_steps,
        zero_ratio=0.10,
        full_ratio=0.30,
    )
    == 0.0
)

assert (
    module.linear_grounding_factor(
        max_steps,
        max_steps,
        zero_ratio=0.10,
        full_ratio=0.30,
    )
    == 1.0
)

print(
    "GenRM prompt, plugin registration, "
    "and refinement schedule OK."
)
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
# Summary
# ============================================================

echo "============================================================"
echo "MCSD plain-GRPO -> short GenRM refinement"
echo "============================================================"
echo
echo "Policy model:               $MODEL"
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
echo "Training sources:           $TRAIN_SIZE"
echo
echo "Max refinement steps:       $MAX_STEPS"
echo "Train batch / GPU:          $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation:      $GRAD_ACC"
echo "Generation batch:           $TRAIN_GENERATION_BATCH"
echo "Generations/source:         $NUM_GENERATIONS"
echo
echo "Learning rate:              $LEARNING_RATE"
echo "Warmup ratio:               $WARMUP_RATIO"
echo "KL beta:                    $BETA"
echo
echo "Reward:"
echo "  accuracy                  $ACCURACY_WEIGHT"
echo "  format                    $FORMAT_WEIGHT"
echo "  grounding                 $GROUNDING_WEIGHT"
echo
echo "Grounding schedule:"
echo "  OFF until                 10%"
echo "  linear ramp               10% -> 30%"
echo "  full after                30%"
echo
echo "Approximate boundaries:"
echo "  OFF                       step 0--25"
echo "  ramp                      step 25--75"
echo "  full                      step 75--250"
echo
echo "Save checkpoints:"
echo "  50 / 100 / 150 / 200 / 250"
echo
echo "Seed:                       $SEED"
echo "Output:"
echo "  $OUTPUT_DIR"
echo
echo "IMPORTANT:"
echo "  policy init == reference init"
echo "  both are plain-GRPO ckpt1200"
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
# Short GRPO + GenRM refinement
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
    --save_total_limit 10 \
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
echo "MCSD short GenRM refinement finished"
echo "============================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
