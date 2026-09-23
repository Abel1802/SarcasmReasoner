#!/bin/bash
set -e
set -o pipefail

MODEL="Qwen/Qwen3-Omni-30B-A3B-Thinking"

# ============================================================
# Local model cache on the compute node
# ============================================================

if [ -z "$TMPDIR" ]; then
    echo "ERROR: TMPDIR is not set. Run this inside a compute job."
    exit 1
fi

export MODELSCOPE_CACHE="$TMPDIR/modelscope"
mkdir -p "$MODELSCOPE_CACHE"

echo "TMPDIR:           $TMPDIR"
echo "MODELSCOPE_CACHE: $MODELSCOPE_CACHE"
echo

df -h "$TMPDIR"

# ============================================================
# Multimodal settings
# ============================================================

export USE_AUDIO_IN_VIDEO=False
export FORCE_QWENVL_VIDEO_READER=decord
export FPS_MAX_FRAMES=12

mkdir -p results/mustard/teacher/logs


run_split () {
    SPLIT=$1

    DATA="data/mustard/processed/zero_shot_${SPLIT}.jsonl"
    OUTPUT="results/mustard/teacher/qwen3_omni_30b_${SPLIT}.jsonl"
    LOG="results/mustard/teacher/logs/qwen3_omni_30b_${SPLIT}.log"

    echo "============================================================"
    echo "Running split: ${SPLIT}"
    echo "Input:  ${DATA}"
    echo "Output: ${OUTPUT}"
    echo "============================================================"

    rm -f "$OUTPUT"

    python src/generation/swift_infer_local_video.py \
        --model "$MODEL" \
        --infer_backend vllm \
        --vllm_tensor_parallel_size 4 \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_max_model_len 12288 \
        --vllm_max_num_seqs 1 \
        --vllm_limit_mm_per_prompt '{"audio": 1, "video": 1}' \
        --val_dataset "$DATA" \
        --temperature 0 \
        --max_new_tokens 4096 \
        --remove_unused_columns false \
        --result_path "$OUTPUT" \
        2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "ERROR: ${SPLIT} failed with exit code ${STATUS}"
        exit "$STATUS"
    fi

    echo
    echo "Finished: ${SPLIT}"
    wc -l "$OUTPUT"
    echo
}


run_split valid
run_split test
run_split train