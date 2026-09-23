python src/generation/swift_sample_local_video.py \
    --model Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --dataset data/mcsd/processed/zero_shot_test.jsonl \
    --sampler_type sample \
    --sampler_engine vllm \
    --num_return_sequences 8 \
    --num_sampling_batch_size 1 \
    --num_sampling_batches 2 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_new_tokens 4096 \
    --dataset_shuffle false \
    --output_dir results/mcsd/teacher/logprob_smoke \
    --output_file mcsd_test_smoke2_n8.jsonl \
    --override_exist_file true \
    --engine_kwargs '{
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 16384,
        "max_num_seqs": 4,
        "limit_mm_per_prompt": {
            "audio": 1,
            "video": 1
        }
    }'