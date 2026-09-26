import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = "You are an expert in multimodal sarcasm reasoning."

PROMPT_PATH = Path("src/prompts/sarcasm_reasoning.txt")


def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_user_content(sample, task_prompt):
    """
    Place multimodal tokens first, followed by the textual instruction.

    The gold label is intentionally NOT included in the prompt.
    """
    transcript = sample["text"]

    return (
        "<audio><video>\n"
        f"Transcript:\n{transcript}\n\n"
        f"{task_prompt}"
    )


def convert_file(dataset, split):
    input_path = Path(
        f"data/{dataset}/processed/base_{split}.jsonl"
    )

    output_path = Path(
        f"data/{dataset}/processed/zero_shot_{split}.jsonl"
    )

    task_prompt = load_prompt()

    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            record = {
                "id": sample["id"],
                "transcript": sample["text"],

                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_user_content(
                            sample,
                            task_prompt,
                        ),
                    },
                ],

                "audios": [sample["audio"]],
                "videos": [sample["video"]],

                # Kept only for evaluation / filtering.
                # These fields are NOT exposed in the prompt.
                "label": sample["label"],
                "label_text": sample["label_text"],

                "dataset": sample["dataset"],
                "language": sample["language"],
                "split": sample["split"],
            }

            records.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(
        f"{dataset}/{split}: "
        f"{len(records)} samples -> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["mcsd", "mustard", "all"],
        default="all",
    )

    parser.add_argument(
        "--split",
        choices=["train", "valid", "test", "all"],
        default="all",
    )

    args = parser.parse_args()

    datasets = (
        ["mcsd", "mustard"]
        if args.dataset == "all"
        else [args.dataset]
    )

    splits = (
        ["train", "valid", "test"]
        if args.split == "all"
        else [args.split]
    )

    for dataset in datasets:
        for split in splits:
            convert_file(dataset, split)


if __name__ == "__main__":
    main()