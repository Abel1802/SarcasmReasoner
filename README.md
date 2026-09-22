# Post-training Multimodal LLMs for Text-Audio-Visual Sarcasm Reasoning

## Step. 1. Prepare data

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


## Step. 2. Teacher reasoning generation
```bash
bash configs/mcsd/teacher/qwen3_omni_30b_smoke.sh
```