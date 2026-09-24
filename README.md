# Post-training Multimodal LLMs for Text-Audio-Visual Sarcasm Reasoning

## Step 1. Prepare data

### Original CSV(train / valid / test.csv) to base_train.jsonl, base_valid.jsonl, base_test.jsonl
```bash
python src/data/prepare_mcsd.py
python src/data/prepare_mustard.py
```


### Prepare zero_shot data for Both teacher and student models(zero_shot_train.jsonl, zero_shot_valid.jsonl, zero_shot_test.jsonl)
```bash
python src/data/prepare_zero_shot.py
```
Note: the zero_shot data is based on designed prompts(src/prompts/sarcasm_reasoning.txt)


## Step 2. Teacher reasoning generation

### Greedy decoding
```bash
bash configs/mustard/teacher/qwen3_omni_30b_full_4096.sh
```

### Diverse sampling
```bash
bash configs/mustard/teacher/qwen3_omni_30b_sample_n8.sh
```

### SFT traning data generation
```bash
python src/data/build_sft_from_greedy.py --input results/mustard/teacher/qwen3_omni_30b_train.jsonl --output data/mustard/processed/sft/greedy_train.jsonl
```


## Step 3. SFT stage

### Traning
```bash
bash configs/mustard/sft/greedy_1gpu.sh
```

### Inference
```bash
bash scripts/evaluation/eval_sft_checkpoint.sh \
    mustard \
    greedy \
    results/mustard/sft/greedy/v0-20260923-221644/checkpoint-105 \
    test
```