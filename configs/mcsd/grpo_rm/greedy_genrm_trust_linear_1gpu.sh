#!/bin/bash

set -euo pipefail

MODEL="Qwen/Qwen2.5-Omni-7B"
SFT_CKPT="results/mcsd/sft/greedy/v6-20260924-101848/checkpoint-228"

TRAIN_DATA="data/mcsd/processed/zero_shot_train.jsonl"
PLUGIN="src/plugins/sarcasm_grpo_reward_trust_linear_v2.py"
GENRM_PROMPT="src/prompts/genrm/grounding_genrm_system.txt"

GENRM_MODEL="Qwen/Qwen2.5-Omni-3B"
GENRM_CKPT="results/mcsd/genrm/qwen25_omni_3b/v0-20260927-193102/checkpoint-1866"

OUTPUT_DIR="results/mcsd/grpo_rm/greedy_genrm_trust_linear"

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
# Trust-aware grounding schedule
# ============================================================

# Must match --num_generations below.
export GENRM_NUM_GENERATIONS="$NUM_GENERATIONS"

# Keep exactly the same schedule as the previous linear run:
#   0%--10%: GenRM OFF
#   10%--40%: linear ramp 0 -> 1
#   40%--100%: full scheduled factor
#
# The final --reward_weights value still sets the maximum
# grounding coefficient (0.2); the plugin must not multiply
# by 0.2 internally.
export GENRM_ZERO_RATIO=0.10
export GENRM_FULL_RATIO=0.40

# Hard trust gate for mixed groups:
# trusted iff
#   mean(raw GenRM | correct) - mean(raw GenRM | wrong)
#       > GENRM_TRUST_MARGIN
#
# With margin 0.0, equality is NOT trusted.
export GENRM_TRUST_MARGIN=0.0

# Log group-level diagnostics every N optimizer steps.
export GENRM_GROUP_LOG_STEPS=25

# Current diagnostic implementation assumes each reward call
# sees complete contiguous GRPO groups on one process.
export GENRM_REQUIRE_SINGLE_PROCESS=1


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

EXPECTED_TRAIN_SIZE=1893

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
# Verify GenRM prompt and trust-aware plugin registration
# ============================================================

PLUGIN="$PLUGIN" GENRM_PROMPT="$GENRM_PROMPT" python - <<'PY'
import importlib.util
import os
from pathlib import Path

plugin_path = Path(os.environ["PLUGIN"]).resolve()

spec = importlib.util.spec_from_file_location(
    "sarcasm_grpo_reward_trust_linear_v2",
    plugin_path,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

expected_path = Path(os.environ["GENRM_PROMPT"]).resolve()
expected = expected_path.read_text(encoding="utf-8").strip()

assert module.GENRM_SYSTEM_PROMPT_PATH.resolve() == expected_path
assert module.GENRM_SYSTEM_PROMPT == expected

assert (
    module.rm_plugins["sarcasm_grounding_trust_linear"]
    is module.SarcasmGroundingTrustLinearRMPlugin
)

assert module.linear_grounding_factor(0, 1893) == 0.0
assert module.linear_grounding_factor(1893, 1893) == 1.0

print(
    "GenRM prompt, trust-aware plugin registration, "
    "and schedule endpoints OK."
)
PY

echo


# ============================================================
# GRPO batch / group sanity checks
# ============================================================

TRAIN_GENERATION_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC))

if [ $((TRAIN_GENERATION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
    echo "ERROR:" >&2
    echo "Training generation batch must be divisible" >&2
    echo "by NUM_GENERATIONS." >&2
    exit 1
fi

if [ "$GENRM_NUM_GENERATIONS" -ne "$NUM_GENERATIONS" ]; then
    echo "ERROR:" >&2
    echo "GENRM_NUM_GENERATIONS must match NUM_GENERATIONS." >&2
    echo "GENRM_NUM_GENERATIONS=$GENRM_NUM_GENERATIONS" >&2
    echo "NUM_GENERATIONS=$NUM_GENERATIONS" >&2
    exit 1
fi

echo "GRPO batch/group checks passed."
echo


# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "MCSD Greedy-SFT -> GRPO + trust-aware GenRM"
echo "============================================================"
echo
echo "Policy model:               $MODEL"
echo "SFT checkpoint:             $SFT_CKPT"
echo
echo "GenRM model:                $GENRM_MODEL"
echo "GenRM checkpoint:           $GENRM_CKPT"
echo
echo "Training sources:           $TRAIN_SIZE"
echo "GPU count:                  1"
echo "Train batch / GPU:          $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation:      $GRAD_ACC"
echo "Generation batch:           $TRAIN_GENERATION_BATCH"
echo "Generations/source:         $NUM_GENERATIONS"
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
echo "  grounding                 0.2 * linear_schedule * trust_gate"
echo "  zero until progress       $GENRM_ZERO_RATIO"
echo "  full from progress        $GENRM_FULL_RATIO"
echo
echo "Trust gate:"
echo "  group size                $GENRM_NUM_GENERATIONS"
echo "  trust margin              $GENRM_TRUST_MARGIN"
echo "  all-wrong group           grounding OFF"
echo "  mixed group               compare raw GenRM(correct) vs raw GenRM(wrong)"
echo "  trusted mixed group       reward correct responses only"
echo "  untrusted mixed group     grounding OFF"
echo "  all-correct group         grounding OFF (diagnostic V2)"
echo
echo "Grounding scalar:"
echo "  (text + audio + visual) / 3"
echo "  integration excluded"
echo
echo "Seed:                       $SEED"
echo "Output:                     $OUTPUT_DIR"
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
# GRPO + trust-aware GenRM with linear grounding schedule
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
    --reward_model_plugin sarcasm_grounding_trust_linear \
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
    --attn_impl sdpa \
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
echo "MCSD Greedy-SFT -> GRPO + trust-aware GenRM finished"
echo "============================================================"
echo "Exit status:      $STATUS"
echo "End time:         $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit "$STATUS"
