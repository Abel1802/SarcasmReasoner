# Data Format

## Datasets

### MCSD

| Split | Samples | Sarcasm | Non-sarcasm |
|---|---:|---:|---:|
| Train | 1893 | 936 | 957 |
| Valid | 406 | 221 | 185 |
| Test | 406 | 187 | 219 |
| Total | 2705 | 1344 | 1361 |

Language: Chinese

### MUStARD++

| Split | Samples | Sarcasm | Non-sarcasm |
|---|---:|---:|---:|
| Train | 841 | 415 | 426 |
| Valid | 180 | 97 | 83 |
| Test | 181 | 89 | 92 |
| Total | 1202 | 601 | 601 |

Language: English

## Directory Structure

    data/
    ├── mcsd/
    │   ├── raw/
    │   │   ├── audios/
    │   │   └── videos/
    │   ├── splits/
    │   │   ├── train.csv
    │   │   ├── valid.csv
    │   │   └── test.csv
    │   └── processed/
    │       ├── base_train.jsonl
    │       ├── base_valid.jsonl
    │       └── base_test.jsonl
    │
    └── mustard/
        ├── raw/
        │   ├── audios/
        │   └── videos/
        ├── splits/
        │   ├── train.csv
        │   ├── valid.csv
        │   └── test.csv
        └── processed/
            ├── base_train.jsonl
            ├── base_valid.jsonl
            └── base_test.jsonl

Raw audio and video files are stored externally and linked locally.

## Canonical Format

All downstream stages use the same JSONL schema.

Example:

    {
      "id": "1892",
      "text": "...",
      "label": 1,
      "label_text": "sarcasm",
      "audio": "data/mcsd/raw/audios/1892.wav",
      "video": "data/mcsd/raw/videos/1892.mp4",
      "dataset": "mcsd",
      "language": "zh",
      "split": "train"
    }

Labels are standardized as:

- 1: sarcasm
- 0: non-sarcasm
