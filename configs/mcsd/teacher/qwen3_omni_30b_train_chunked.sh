#!/bin/bash
set -euo pipefail

MODEL="Qwen/Qwen3-Omni-30B-A3B-Thinking"

SOURCE_DATA="data/mcsd/processed/zero_shot_train.jsonl"
CHUNK_DIR="data/mcsd/processed/train_chunks"
OUTPUT_DIR="results/mcsd/teacher/train_chunks"
LOG_DIR="results/mcsd/teacher/logs/train_chunks"

FINAL_OUTPUT="results/mcsd/teacher/qwen3_omni_30b_train.jsonl"

CHUNK_SIZE=300

mkdir -p "$CHUNK_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ============================================================
# Model / multimodal settings
# ============================================================

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/tmp/modelscope}"

export USE_AUDIO_IN_VIDEO=False
export FORCE_QWENVL_VIDEO_READER=decord
export FPS_MAX_FRAMES=12


# ============================================================
# Step 1: Split train into chunks
# ============================================================

python - <<'PY'
from pathlib import Path

source = Path("data/mcsd/processed/zero_shot_train.jsonl")
out_dir = Path("data/mcsd/processed/train_chunks")
chunk_size = 300

out_dir.mkdir(parents=True, exist_ok=True)

# Remove old chunks
for p in out_dir.glob("chunk_*.jsonl"):
    p.unlink()

with source.open("r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total train samples: {len(lines)}")

for i in range(0, len(lines), chunk_size):
    chunk = lines[i:i + chunk_size]
    chunk_id = i // chunk_size

    path = out_dir / f"chunk_{chunk_id:03d}.jsonl"

    with path.open("w", encoding="utf-8") as f:
        f.writelines(chunk)

    print(f"{path}: {len(chunk)} samples")
PY


# ============================================================
# Step 2: Run each chunk separately
# ============================================================

for DATA in "$CHUNK_DIR"/chunk_*.jsonl; do

    NAME=$(basename "$DATA" .jsonl)

    OUTPUT="$OUTPUT_DIR/${NAME}.jsonl"
    LOG="$LOG_DIR/${NAME}.log"

    EXPECTED=$(wc -l < "$DATA")

    echo
    echo "============================================================"
    echo "Running:  $NAME"
    echo "Input:    $DATA"
    echo "Samples:  $EXPECTED"
    echo "Output:   $OUTPUT"
    echo "============================================================"

    # Resume support:
    # if this chunk already finished successfully, skip it
    if [ -f "$OUTPUT" ]; then
        ACTUAL=$(wc -l < "$OUTPUT")

        if [ "$ACTUAL" -eq "$EXPECTED" ]; then
            echo "Already complete: $ACTUAL / $EXPECTED"
            echo "Skipping $NAME"
            continue
        fi

        echo "Incomplete previous output: $ACTUAL / $EXPECTED"
        echo "Removing and rerunning..."
        rm -f "$OUTPUT"
    fi

    set +e

    python src/generation/swift_infer_local_video.py \
        --model "$MODEL" \
        --infer_backend vllm \
        --vllm_tensor_parallel_size 4 \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_max_model_len 12288 \
        --vllm_max_num_seqs 4 \
        --vllm_limit_mm_per_prompt '{"audio": 1, "video": 1}' \
        --val_dataset "$DATA" \
        --temperature 0 \
        --max_new_tokens 4096 \
        --remove_unused_columns false \
        --result_path "$OUTPUT" \
        2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    set -e

    if [ "$STATUS" -ne 0 ]; then
        echo
        echo "ERROR: $NAME failed with exit code $STATUS"
        echo "Stop here. Completed chunks are preserved."
        exit "$STATUS"
    fi

    ACTUAL=$(wc -l < "$OUTPUT")

    echo
    echo "Finished $NAME: $ACTUAL / $EXPECTED"

    if [ "$ACTUAL" -ne "$EXPECTED" ]; then
        echo "ERROR: Output count does not match input count."
        exit 1
    fi

done


# ============================================================
# Step 3: Merge all completed chunks
# ============================================================

echo
echo "============================================================"
echo "Merging train results"
echo "============================================================"

cat "$OUTPUT_DIR"/chunk_*.jsonl > "$FINAL_OUTPUT"

FINAL_COUNT=$(wc -l < "$FINAL_OUTPUT")
EXPECTED_TOTAL=$(wc -l < "$SOURCE_DATA")

echo "Final output: $FINAL_OUTPUT"
echo "Samples: $FINAL_COUNT / $EXPECTED_TOTAL"

if [ "$FINAL_COUNT" -ne "$EXPECTED_TOTAL" ]; then
    echo "ERROR: Final merged count is incorrect."
    exit 1
fi

echo
echo "Train inference completed successfully."
