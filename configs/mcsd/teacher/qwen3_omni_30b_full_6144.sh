#!/bin/bash
set -e

MODEL="Qwen/Qwen3-Omni-30B-A3B-Thinking"

export USE_AUDIO_IN_VIDEO=False
export FORCE_QWENVL_VIDEO_READER=decord
export FPS_MAX_FRAMES=12

mkdir -p results/mcsd/teacher
mkdir -p results/mcsd/teacher/logs


run_split () {
    SPLIT=$1

    DATA="data/mcsd/processed/zero_shot_${SPLIT}.jsonl"
    OUTPUT="results/mcsd/teacher/qwen3_omni_30b_${SPLIT}.jsonl"
    LOG="results/mcsd/teacher/logs/qwen3_omni_30b_${SPLIT}.log"

    echo "============================================================"
    echo "Running split: ${SPLIT}"
    echo "Input:  ${DATA}"
    echo "Output: ${OUTPUT}"
    echo "============================================================"

    # ms-swift appends to an existing result file,
    # so remove old output before a fresh run.
    rm -f "$OUTPUT"

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

    echo
    echo "Finished: ${SPLIT}"
    wc -l "$OUTPUT"
    echo
}


run_split train
run_split valid
run_split test
