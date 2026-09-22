import json
from pathlib import Path

import pandas as pd


DATASET = "mustard"
LANGUAGE = "en"

SPLIT_DIR = Path("data/mustard/splits")
AUDIO_DIR = Path("data/mustard/raw/audios")
VIDEO_DIR = Path("data/mustard/raw/videos")
OUTPUT_DIR = Path("data/mustard/processed")


def convert_split(split):
    input_path = SPLIT_DIR / f"{split}.csv"
    output_path = OUTPUT_DIR / f"base_{split}.jsonl"

    df = pd.read_csv(input_path)

    records = []

    for _, row in df.iterrows():
        sample_id = str(row["KEY"])

        label = int(row["Sarcasm"])

        if label not in {0, 1}:
            raise ValueError(
                f"Unknown label '{label}' for sample {sample_id}"
            )

        audio_path = AUDIO_DIR / f"{sample_id}.wav"
        video_path = VIDEO_DIR / f"{sample_id}.mp4"

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        record = {
            "id": sample_id,
            "text": str(row["SENTENCE"]),
            "label": label,
            "label_text": "sarcasm" if label == 1 else "non_sarcasm",
            "audio": str(audio_path),
            "video": str(video_path),
            "dataset": DATASET,
            "language": LANGUAGE,
            "split": split,
        }

        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{split}: {len(records)} samples -> {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        convert_split(split)


if __name__ == "__main__":
    main()