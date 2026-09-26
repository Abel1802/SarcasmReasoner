#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


# ============================================================
# Human calibration labels
# ============================================================

CALIBRATION = {
    907: {
        "text": 2,
        "audio": 2,
        "visual": 1,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "NONE",
        "visual_issue": "OVERINTERPRETATION",
        "integration_issue": "NONE",
    },
    3297: {
        "text": 2,
        "audio": 2,
        "visual": 2,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "NONE",
        "visual_issue": "NONE",
        "integration_issue": "NONE",
    },
    144: {
        "text": 2,
        "audio": 1,
        "visual": 1,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "OVERINTERPRETATION",
        "visual_issue": "OVERINTERPRETATION",
        "integration_issue": "NONE",
    },
    333: {
        "text": 2,
        "audio": 1,
        "visual": 1,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "OVERINTERPRETATION",
        "visual_issue": "OVERINTERPRETATION",
        "integration_issue": "NONE",
    },
    5735: {
        "text": 2,
        "audio": 1,
        "visual": 0,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "OVERINTERPRETATION",
        "visual_issue": "ENTITY_BINDING_ERROR",
        "integration_issue": "NONE",
    },
    64: {
        "text": 2,
        "audio": 2,
        "visual": 2,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "NONE",
        "visual_issue": "NONE",
        "integration_issue": "NONE",
    },
    1417: {
        "text": 2,
        "audio": 2,
        "visual": 1,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "NONE",
        "visual_issue": "OVERINTERPRETATION",
        "integration_issue": "NONE",
    },
    4242: {
        "text": 2,
        "audio": 2,
        "visual": 2,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "NONE",
        "visual_issue": "NONE",
        "integration_issue": "NONE",
    },
    5790: {
        "text": 1,
        "audio": 2,
        "visual": 2,
        "integration": 2,
        "text_issue": "OVERINTERPRETATION",
        "audio_issue": "NONE",
        "visual_issue": "NONE",
        "integration_issue": "NONE",
    },
    5918: {
        "text": 2,
        "audio": 2,
        "visual": 1,
        "integration": 2,
        "text_issue": "NONE",
        "audio_issue": "NONE",
        "visual_issue": "OVERINTERPRETATION",
        "integration_issue": "NONE",
    },
}


ORDER = [
    907,
    3297,
    144,
    333,
    5735,
    64,
    1417,
    4242,
    5790,
    5918,
]


# ============================================================
# Helpers
# ============================================================

def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_user_message(row):
    messages = row.get("messages", [])

    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text")

                    if isinstance(text, str):
                        parts.append(text)

            return "\n".join(parts)

    return ""


def extract_transcript(user_message):
    patterns = [
        (
            r"Transcript:\s*\n(.*?)"
            r"\n\s*\nAnalyze the provided utterance"
        ),
        (
            r"Transcript:\s*\n(.*?)"
            r"\n\s*\nAnalyze the provided"
        ),
        (
            r"Transcript:\s*\n(.*?)(?:\n\s*\n|\Z)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            user_message,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


def replace_required(text, key, value):
    token = "{{" + key + "}}"

    if token not in text:
        raise RuntimeError(
            f"Missing placeholder {token}"
        )

    return text.replace(
        token,
        value,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        default=(
            "data/mustard/processed/genrm/"
            "judge_pool_smoke100.jsonl"
        ),
    )

    parser.add_argument(
        "--raw",
        default=(
            "results/mustard/teacher/"
            "sample_n8/train/"
            "qwen3_omni_30b_train_n8.jsonl"
        ),
    )

    parser.add_argument(
        "--prompt-dir",
        default=(
            "src/prompts/grounding_judge"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/mustard/processed/genrm/"
            "judge_calibration10"
        ),
    )

    args = parser.parse_args()

    smoke_rows = read_jsonl(args.smoke)
    raw_rows = read_jsonl(args.raw)

    smoke_by_id = {
        int(row["judge_id"]): row
        for row in smoke_rows
    }

    missing = [
        judge_id
        for judge_id in ORDER
        if judge_id not in smoke_by_id
    ]

    if missing:
        raise RuntimeError(
            f"Missing judge IDs: {missing}"
        )

    prompt_dir = Path(args.prompt_dir)

    text_template = read_text(
        prompt_dir / "text_grounding_v1.txt"
    )

    audio_template = read_text(
        prompt_dir / "audio_grounding_v3.txt"
    )

    visual_template = read_text(
        prompt_dir / "visual_grounding_v1.txt"
    )

    integration_template = read_text(
        prompt_dir
        / "integration_faithfulness_v1.txt"
    )

    output_dir = Path(args.output_dir)

    text_rows = []
    audio_rows = []
    visual_rows = []
    integration_rows = []
    gold_rows = []

    for judge_id in ORDER:
        row = smoke_by_id[judge_id]

        sections = row["sections"]

        raw_index = int(row["raw_index"])

        if not (
            0 <= raw_index < len(raw_rows)
        ):
            raise RuntimeError(
                f"Invalid raw_index={raw_index} "
                f"for judge_id={judge_id}"
            )

        raw_row = raw_rows[raw_index]

        # ----------------------------------------------------
        # Verify traceability
        # ----------------------------------------------------

        if (
            str(raw_row.get("source_id"))
            != str(row["source_id"])
        ):
            raise RuntimeError(
                "source_id mismatch for "
                f"judge_id={judge_id}"
            )

        if (
            int(raw_row.get("sample_idx"))
            != int(row["sample_idx"])
        ):
            raise RuntimeError(
                "sample_idx mismatch for "
                f"judge_id={judge_id}"
            )

        # ----------------------------------------------------
        # Recover transcript
        # ----------------------------------------------------

        transcript = extract_transcript(
            get_user_message(raw_row)
        )

        if not transcript:
            raise RuntimeError(
                "Could not recover transcript for "
                f"judge_id={judge_id}"
            )

        # ----------------------------------------------------
        # Text prompt
        # ----------------------------------------------------

        text_prompt = replace_required(
            text_template,
            "TRANSCRIPT",
            transcript,
        )

        text_prompt = replace_required(
            text_prompt,
            "TEXT_EVIDENCE",
            sections["text_evidence"],
        )

        text_rows.append({
            "id": str(judge_id),
            "messages": [
                {
                    "role": "user",
                    "content": text_prompt,
                }
            ],
        })

        # ----------------------------------------------------
        # Audio prompt
        # ----------------------------------------------------

        audio_prompt = replace_required(
            audio_template,
            "AUDIO_EVIDENCE",
            sections["audio_evidence"],
        )

        audios = row.get("audio")

        if not audios:
            raise RuntimeError(
                f"No audio for judge_id={judge_id}"
            )

        audio_rows.append({
            "id": str(judge_id),
            "messages": [
                {
                    "role": "user",
                    "content": audio_prompt,
                }
            ],
            "audios": audios,
        })

        # ----------------------------------------------------
        # Visual prompt
        # ----------------------------------------------------

        visual_prompt = replace_required(
            visual_template,
            "VISUAL_EVIDENCE",
            sections["visual_evidence"],
        )

        videos = row.get("video")

        if not videos:
            raise RuntimeError(
                f"No video for judge_id={judge_id}"
            )

        visual_rows.append({
            "id": str(judge_id),
            "messages": [
                {
                    "role": "user",
                    "content": visual_prompt,
                }
            ],
            "videos": videos,
        })

        # ----------------------------------------------------
        # Integration prompt
        # ----------------------------------------------------

        integration_prompt = replace_required(
            integration_template,
            "TEXT_EVIDENCE",
            sections["text_evidence"],
        )

        integration_prompt = replace_required(
            integration_prompt,
            "AUDIO_EVIDENCE",
            sections["audio_evidence"],
        )

        integration_prompt = replace_required(
            integration_prompt,
            "VISUAL_EVIDENCE",
            sections["visual_evidence"],
        )

        integration_prompt = replace_required(
            integration_prompt,
            "INTEGRATION",
            sections["integration"],
        )

        integration_rows.append({
            "id": str(judge_id),
            "messages": [
                {
                    "role": "user",
                    "content": integration_prompt,
                }
            ],
        })

        # ----------------------------------------------------
        # Human gold / metadata
        # ----------------------------------------------------

        gold = CALIBRATION[judge_id]

        gold_rows.append({
            "judge_id": judge_id,
            "source_id": row["source_id"],
            "sample_idx": row["sample_idx"],

            "human_text": gold["text"],
            "human_audio": gold["audio"],
            "human_visual": gold["visual"],
            "human_integration":
                gold["integration"],

            "human_text_issue":
                gold["text_issue"],
            "human_audio_issue":
                gold["audio_issue"],
            "human_visual_issue":
                gold["visual_issue"],
            "human_integration_issue":
                gold["integration_issue"],

            # Analysis only.
            # These fields are NOT included in judge prompts.
            "gold_label_text":
                row["gold_label_text"],
            "pred_label_text":
                row["pred_label_text"],
            "is_correct":
                row["is_correct"],
        })

    # ========================================================
    # Write
    # ========================================================

    write_jsonl(
        output_dir / "text_input.jsonl",
        text_rows,
    )

    write_jsonl(
        output_dir / "audio_input.jsonl",
        audio_rows,
    )

    write_jsonl(
        output_dir / "visual_input.jsonl",
        visual_rows,
    )

    write_jsonl(
        output_dir
        / "integration_input.jsonl",
        integration_rows,
    )

    write_jsonl(
        output_dir / "human_gold.jsonl",
        gold_rows,
    )

    # ========================================================
    # Report
    # ========================================================

    print("=" * 72)
    print("Grounding judge calibration10")
    print("=" * 72)

    print(f"Examples:     {len(ORDER)}")
    print()

    print("Human scores:")
    print()

    print(
        "judge_id   text   audio   visual   integration"
    )

    for judge_id in ORDER:
        g = CALIBRATION[judge_id]

        print(
            f"{judge_id:>7}   "
            f"{g['text']:>4}   "
            f"{g['audio']:>5}   "
            f"{g['visual']:>6}   "
            f"{g['integration']:>11}"
        )

    print()
    print("Outputs:")
    print(
        f"  {output_dir / 'text_input.jsonl'}"
    )
    print(
        f"  {output_dir / 'audio_input.jsonl'}"
    )
    print(
        f"  {output_dir / 'visual_input.jsonl'}"
    )
    print(
        f"  {output_dir / 'integration_input.jsonl'}"
    )
    print(
        f"  {output_dir / 'human_gold.jsonl'}"
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
