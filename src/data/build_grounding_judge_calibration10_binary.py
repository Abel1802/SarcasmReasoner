#!/usr/bin/env python3

import json
import re
from pathlib import Path


IDS = [
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
# Human calibration labels
# ============================================================

HUMAN_AUDIO = {
    907: 1,
    3297: 1,
    144: 0,
    333: 0,
    5735: 0,
    64: 1,
    1417: 1,
    4242: 1,
    5790: 1,
    5918: 1,
}


# Revised after re-checking the actual videos using the
# finalized visual-grounding criterion.
HUMAN_VISUAL = {
    907: 1,
    3297: 1,
    144: 0,
    333: 1,
    5735: 1,
    64: 1,
    1417: 0,
    4242: 1,
    5790: 1,
    5918: 0,
}


# ============================================================
# Paths
# ============================================================

INPUT = Path(
    "data/mustard/processed/genrm/"
    "judge_pool_smoke100.jsonl"
)

# Used only to recover the original transcript by source_id.
TEXT_SOURCE = Path(
    "data/mustard/processed/"
    "zero_shot_train.jsonl"
)

PROMPT_DIR = Path(
    "src/prompts/grounding_judge"
)

OUTPUT_DIR = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary"
)


# ============================================================
# Helpers
# ============================================================

def read_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def message_content_to_text(content):
    """
    Convert either a normal string message or multimodal
    list-style message content into plain text.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        pieces = []

        for item in content:

            if isinstance(item, str):
                pieces.append(item)

            elif isinstance(item, dict):

                if (
                    item.get("type") == "text"
                    and isinstance(
                        item.get("text"),
                        str,
                    )
                ):
                    pieces.append(
                        item["text"]
                    )

        return "\n".join(pieces)

    return str(content)


def extract_transcript(row):
    """
    Recover transcript from a zero-shot source record.

    We intentionally recover it from the original input rather
    than from teacher reasoning.
    """

    # Best case: explicit field exists.
    for key in [
        "text",
        "transcript",
        "utterance",
    ]:
        value = row.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    messages = row.get(
        "messages",
        [],
    )

    user_texts = []

    for message in messages:

        if (
            isinstance(message, dict)
            and message.get("role") == "user"
        ):

            text = message_content_to_text(
                message.get(
                    "content",
                    "",
                )
            )

            if text.strip():
                user_texts.append(
                    text.strip()
                )

    content = "\n\n".join(
        user_texts
    )

    # --------------------------------------------------------
    # Try common transcript formats.
    # --------------------------------------------------------

    patterns = [

        # <transcript> ... </transcript>
        r"<transcript>\s*(.*?)\s*</transcript>",

        # Transcript: "..."
        r'''Transcript\s*:\s*["“](.*?)["”]''',

        # Transcript:
        # actual text
        r"Transcript\s*:\s*\n\s*(.+?)(?=\n\s*\n|\Z)",

        # Transcript: actual text
        r"Transcript\s*:\s*(.+?)(?=\n\s*\n|\Z)",

        # Utterance: "..."
        r'''Utterance\s*:\s*["“](.*?)["”]''',

        # Utterance:
        # actual text
        r"Utterance\s*:\s*\n\s*(.+?)(?=\n\s*\n|\Z)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            content,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if match:

            transcript = (
                match.group(1)
                .strip()
            )

            if transcript:
                return transcript

    raise RuntimeError(
        "Could not recover transcript from "
        "zero-shot source.\n\n"
        "Source keys:\n"
        f"{sorted(row.keys())}\n\n"
        "User message preview:\n"
        f"{content[:1500]}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load selected teacher trajectories
    # --------------------------------------------------------

    rows = read_jsonl(
        INPUT
    )

    by_id = {
        int(x["judge_id"]): x
        for x in rows
    }

    missing = [
        x
        for x in IDS
        if x not in by_id
    ]

    if missing:
        raise RuntimeError(
            f"Missing judge IDs: {missing}"
        )

    # --------------------------------------------------------
    # Load original zero-shot source examples
    # --------------------------------------------------------

    source_rows = read_jsonl(
        TEXT_SOURCE
    )

    source_by_id = {}

    for row in source_rows:

        source_id = row.get(
            "source_id",
            row.get("id"),
        )

        if source_id is not None:
            source_by_id[
                str(source_id)
            ] = row

    # --------------------------------------------------------
    # Load four final prompts
    # --------------------------------------------------------

    text_template = (
        PROMPT_DIR
        / "text_grounding_binary_final.txt"
    ).read_text(
        encoding="utf-8"
    )

    audio_template = (
        PROMPT_DIR
        / "audio_grounding_binary_v3_final.txt"
    ).read_text(
        encoding="utf-8"
    )

    visual_template = (
        PROMPT_DIR
        / "visual_grounding_binary_v4_final.txt"
    ).read_text(
        encoding="utf-8"
    )

    integration_template = (
        PROMPT_DIR
        / "integration_grounding_binary_final.txt"
    ).read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    text_rows = []
    audio_rows = []
    visual_rows = []
    integration_rows = []
    gold_rows = []

    # --------------------------------------------------------
    # Build 10 examples
    # --------------------------------------------------------

    for judge_id in IDS:

        row = by_id[judge_id]

        sections = row[
            "sections"
        ]

        source_id = str(
            row["source_id"]
        )

        # ====================================================
        # Recover original transcript
        # ====================================================

        if source_id not in source_by_id:
            raise RuntimeError(
                "Missing source in zero-shot train: "
                f"{source_id}"
            )

        transcript = (
            extract_transcript(
                source_by_id[
                    source_id
                ]
            )
        )

        # ====================================================
        # TEXT
        # ====================================================

        text_prompt = (
            text_template
            .replace(
                "{{TRANSCRIPT}}",
                transcript,
            )
            .replace(
                "{{TEXT_EVIDENCE}}",
                sections[
                    "text_evidence"
                ],
            )
        )

        text_rows.append({
            "id":
                str(judge_id),

            "source_id":
                source_id,

            "messages": [
                {
                    "role": "user",
                    "content":
                        text_prompt,
                }
            ],
        })

        # ====================================================
        # AUDIO
        # ====================================================

        audio_prompt = (
            audio_template
            .replace(
                "{{AUDIO_EVIDENCE}}",
                sections[
                    "audio_evidence"
                ],
            )
        )

        audio_rows.append({
            "id":
                str(judge_id),

            "source_id":
                source_id,

            "messages": [
                {
                    "role": "user",
                    "content":
                        audio_prompt,
                }
            ],

            "audios":
                row["audio"],
        })

        # ====================================================
        # VISUAL
        # ====================================================

        visual_prompt = (
            visual_template
            .replace(
                "{{VISUAL_EVIDENCE}}",
                sections[
                    "visual_evidence"
                ],
            )
        )

        visual_rows.append({
            "id":
                str(judge_id),

            "source_id":
                source_id,

            "messages": [
                {
                    "role": "user",
                    "content":
                        visual_prompt,
                }
            ],

            "videos":
                row["video"],
        })

        # ====================================================
        # INTEGRATION
        # ====================================================

        integration_prompt = (
            integration_template
            .replace(
                "{{TEXT_EVIDENCE}}",
                sections[
                    "text_evidence"
                ],
            )
            .replace(
                "{{AUDIO_EVIDENCE}}",
                sections[
                    "audio_evidence"
                ],
            )
            .replace(
                "{{VISUAL_EVIDENCE}}",
                sections[
                    "visual_evidence"
                ],
            )
            .replace(
                "{{INTEGRATION}}",
                sections[
                    "integration"
                ],
            )
        )

        integration_rows.append({
            "id":
                str(judge_id),

            "source_id":
                source_id,

            "messages": [
                {
                    "role": "user",
                    "content":
                        integration_prompt,
                }
            ],
        })

        # ====================================================
        # Existing human gold
        #
        # Text / Integration are intentionally NOT added yet.
        # We first run the judges and then calibrate them
        # manually on these 10 examples.
        # ====================================================

        gold_rows.append({
            "judge_id":
                judge_id,

            "source_id":
                source_id,

            "sample_idx":
                row["sample_idx"],

            "human_audio_grounded":
                HUMAN_AUDIO[
                    judge_id
                ],

            "human_visual_grounded":
                HUMAN_VISUAL[
                    judge_id
                ],
        })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_jsonl(
        OUTPUT_DIR
        / "text_input.jsonl",
        text_rows,
    )

    write_jsonl(
        OUTPUT_DIR
        / "audio_input.jsonl",
        audio_rows,
    )

    write_jsonl(
        OUTPUT_DIR
        / "visual_input.jsonl",
        visual_rows,
    )

    write_jsonl(
        OUTPUT_DIR
        / "integration_input.jsonl",
        integration_rows,
    )

    write_jsonl(
        OUTPUT_DIR
        / "human_gold.jsonl",
        gold_rows,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 80)
    print(
        "Binary grounding calibration10"
    )
    print("=" * 80)

    print()
    print(
        "judge_id   audio   visual   "
        "transcript"
    )

    for judge_id in IDS:

        row = by_id[
            judge_id
        ]

        source_id = str(
            row["source_id"]
        )

        transcript = extract_transcript(
            source_by_id[
                source_id
            ]
        )

        preview = (
            transcript
            .replace("\n", " ")
        )

        if len(preview) > 70:
            preview = (
                preview[:67]
                + "..."
            )

        print(
            f"{judge_id:>7}   "
            f"{HUMAN_AUDIO[judge_id]:>5}   "
            f"{HUMAN_VISUAL[judge_id]:>6}   "
            f"{preview}"
        )

    print()
    print("Output files:")

    for name in [
        "text_input.jsonl",
        "audio_input.jsonl",
        "visual_input.jsonl",
        "integration_input.jsonl",
        "human_gold.jsonl",
    ]:

        path = (
            OUTPUT_DIR
            / name
        )

        print(
            f"  {name:<28}"
            f"{sum(1 for _ in open(path, encoding='utf-8'))}"
        )


if __name__ == "__main__":
    main()
