#!/bin/bash
set -e

MODEL="Qwen/Qwen3-Omni-30B-A3B-Thinking"
DATA="data/mustard/processed/zero_shot_train_smoke.jsonl"
OUTPUT="results/mustard/teacher/qwen3_omni_30b_smoke.jsonl"

mkdir -p results/mustard/teacher

# ms-swift appends to an existing result file.
rm -f "$OUTPUT"

# Audio is provided separately from the video.
export USE_AUDIO_IN_VIDEO=False

# Use decord and limit the number of video frames.
export FORCE_QWENVL_VIDEO_READER=decord
export FPS_MAX_FRAMES=12

python src/generation/swift_infer_local_video.py \
    --model "$MODEL" \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --vllm_limit_mm_per_prompt '{"audio": 1, "video": 1}' \
    --val_dataset "$DATA" \
    --temperature 0 \
    --max_new_tokens 3072 \
    --remove_unused_columns false \
    --result_path "$OUTPUT"
