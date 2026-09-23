#!/bin/bash
set -e
set -o pipefail

MODEL="Qwen/Qwen2.5-Omni-7B"

DATA="data/mcsd/processed/zero_shot_test.jsonl"
OUTPUT="results/mcsd/inference/qwen2_5_omni_7b_test.jsonl"
LOG="results/mcsd/inference/logs/qwen2_5_omni_7b_test.log"

mkdir -p results/mcsd/inference/logs
rm -f "$OUTPUT"

export USE_AUDIO_IN_VIDEO=False
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
    --max_new_tokens 4096 \
    --remove_unused_columns false \
    --result_path "$OUTPUT"

STATUS=${PIPESTATUS[0]}

echo
echo "Inference exit code: $STATUS"

exit "$STATUS"
