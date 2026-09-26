# Post-training Multimodal LLMs for Text-Audio-Visual Sarcasm Reasoning

## Step 1. Prepare data

#### Original CSV(train / valid / test.csv) to base_train.jsonl, base_valid.jsonl, base_test.jsonl
```bash
python src/data/prepare_mcsd.py
python src/data/prepare_mustard.py
```


#### Prepare zero_shot data for Both teacher and student models(zero_shot_train.jsonl, zero_shot_valid.jsonl, zero_shot_test.jsonl)
```bash
python src/data/prepare_zero_shot.py
```
Note: the zero_shot data is based on designed prompts(src/prompts/sarcasm_reasoning.txt)


## Step 2. Teacher reasoning generation

#### Greedy decoding
```bash
bash configs/mustard/teacher/qwen3_omni_30b_full_4096.sh
```

#### Diverse sampling
```bash
bash configs/mustard/teacher/qwen3_omni_30b_sample_n8.sh
```

#### SFT traning data generation (greedy, best-of-8, diverse-8)
```bash
python src/data/build_sft_from_greedy.py --input results/mustard/teacher/qwen3_omni_30b_train.jsonl --output data/mustard/processed/sft/greedy_train.jsonl
python src/data/build_sft_from_greedy.py --input results/mustard/teacher/qwen3_omni_30b_valid.jsonl --output data/mustard/processed/sft/greedy_valid.jsonl
```
```bash
python src/data/build_sft_from_samples.py \
  --input results/mustard/teacher/sample_n8/train/qwen3_omni_30b_train_n8.jsonl \
  --output-dir data/mustard/processed/sft
python src/data/build_sft_from_samples.py \
  --input results/mustard/teacher/sample_n8/valid/qwen3_omni_30b_valid_n8.jsonl \
  --output-dir data/mustard/processed/sft
```


## Step 3. SFT stage

#### Traning
```bash
bash configs/mustard/sft/greedy_1gpu.sh
bash configs/mustard/sft/best_of_8_1gpu.sh
bash configs/mustard/sft/diverse_8_1gpu.sh
```

#### Inference
```bash
bash scripts/evaluation/eval_sft_checkpoint.sh \
    mustard \
    sft_best_of_8 \
    results/mustard/sft/best_of_8/v0-20260924-103030/checkpoint-140 \
    test
```


## Step 4. GRPO stage

#### Base (W/O SFT)
```bash
bash configs/mustard/grpo/base_1gpu.sh
```

#### greedy (W/ Greedy SFT)
```bash
bash configs/mustard/sft/greedy_1gpu.sh
```

#### best-8 (W/ Best-of-8 SFT)
```bash
bash configs/mustard/sft/best_of_8_1gpu.sh
```

#### diverse-8 (W/ Diverse-8 SFT)
```bash
bash configs/mustard/sft/diverse_8_1gpu.sh
```


## Step 5. Reward Model data building

```bash
python src/data/build_grounding_judge_pool.py
```

```bash
python src/evaluation/run_text_integration_grounding_full.py
python src/evaluation/run_audio_grounding_full.py
python src/evaluation/run_visual_grounding_vllm.py
```

```bash
python src/data/merge_grounding_labels.py
```

```bash
python src/data/build_genrm_sft_data.py
```