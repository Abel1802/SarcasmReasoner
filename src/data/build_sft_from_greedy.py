#!/usr/bin/env python3

"""
Build SFT training data from greedy teacher generations.

Pipeline
--------
greedy teacher trajectory
    -> require complete structured output
    -> parse final answer
    -> require prediction == gold label
    -> remove <think>...</think>
    -> keep only the five explicit reasoning sections
    -> write clean multimodal SFT JSONL

Designed for both:
    - MCSD
    - MUStARD++
"""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path


REQUIRED_TAGS = [
    "text_evidence",
    "audio_evidence",
    "visual_evidence",
    "integration",
    "answer",
]


# ============================================================
# Answer parsing
# ============================================================

def normalize_answer(text):
    """
    Convert textual answer to binary label.

    Returns
    -------
    1 : sarcasm
    0 : non-sarcasm
    None : unparseable
    """

    x = re.sub(
        r"[\s_-]+",
        "",
        str(text).strip().lower(),
    )

    if x in {
        "sarcasm",
        "sarcastic",
    }:
        return 1

    if x in {
        "nonsarcasm",
        "nonsarcastic",
    }:
        return 0

    return None


# ============================================================
# Structured reasoning parsing
# ============================================================

def extract_sections(content):
    """
    Extract the five required reasoning sections.

    Each required tag must occur exactly once and must be non-empty.

    Returns
    -------
    dict or None
    """

    sections = {}

    for tag in REQUIRED_TAGS:

        matches = re.findall(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if len(matches) != 1:
            return None

        value = matches[0].strip()

        if not value:
            return None

        sections[tag] = value

    return sections


# ============================================================
# Rebuild clean SFT target
# ============================================================

def build_canonical_target(sections):
    """
    Rebuild the teacher reasoning without <think>.

    Only the explicitly requested five reasoning sections are
    retained as the student training target.
    """

    prediction = normalize_answer(
        sections["answer"]
    )

    if prediction == 1:
        answer = "Sarcasm"

    elif prediction == 0:
        answer = "Non-Sarcasm"

    else:
        raise ValueError(
            f"Invalid answer: {sections['answer']!r}"
        )

    return (
        "<text_evidence>\n"
        f"{sections['text_evidence']}\n"
        "</text_evidence>\n\n"

        "<audio_evidence>\n"
        f"{sections['audio_evidence']}\n"
        "</audio_evidence>\n\n"

        "<visual_evidence>\n"
        f"{sections['visual_evidence']}\n"
        "</visual_evidence>\n\n"

        "<integration>\n"
        f"{sections['integration']}\n"
        "</integration>\n\n"

        "<answer>\n"
        f"{answer}\n"
        "</answer>"
    )


# ============================================================
# Obtain teacher response
# ============================================================

def get_teacher_response(row):
    """
    Prefer the top-level `response` field when available.

    Fall back to the final assistant message.
    """

    response = row.get("response")

    if isinstance(response, str) and response.strip():
        return response

    messages = row.get("messages", [])

    if (
        messages
        and messages[-1].get("role") == "assistant"
    ):
        return messages[-1].get(
            "content",
            "",
        )

    return ""


# ============================================================
# Validate one greedy trajectory
# ============================================================

def validate_greedy_row(row):
    """
    A greedy teacher trajectory is usable iff:

    1. teacher response exists
    2. all five required sections are complete
    3. final answer is parseable
    4. final prediction matches the gold label

    Returns
    -------
    target, None
        if usable

    None, rejection_reason
        otherwise
    """

    content = get_teacher_response(row)

    if not content.strip():
        return None, "missing_response"

    sections = extract_sections(content)

    if sections is None:
        return None, "malformed_structure"

    prediction = normalize_answer(
        sections["answer"]
    )

    if prediction is None:
        return None, "unparseable_answer"

    try:
        gold = int(row["label"])

    except Exception:
        return None, "invalid_gold_label"

    if prediction != gold:
        return None, "wrong_prediction"

    target = build_canonical_target(
        sections
    )

    return target, None


# ============================================================
# Build one SFT record
# ============================================================

def make_sft_row(row, target):
    """
    Convert the greedy teacher example into a clean multimodal
    SFT example.
    """

    messages = deepcopy(
        row.get("messages", [])
    )

    if not messages:
        raise ValueError(
            f"No messages for id={row.get('id')}"
        )

    # --------------------------------------------------------
    # Existing greedy outputs normally already end in assistant.
    # Replace that raw teacher response with the cleaned target.
    # --------------------------------------------------------

    if messages[-1].get("role") == "assistant":

        messages[-1]["content"] = target

    else:

        messages.append({
            "role": "assistant",
            "content": target,
        })

    output = {
        "messages": messages,
        "source_id": str(
            row.get(
                "source_id",
                row.get("id", ""),
            )
        ),
        "label": int(row["label"]),
        "sft_variant": "greedy",
    }

    # --------------------------------------------------------
    # Preserve multimodal inputs
    # --------------------------------------------------------

    if row.get("audios"):
        output["audios"] = deepcopy(
            row["audios"]
        )

    if row.get("videos"):
        output["videos"] = deepcopy(
            row["videos"]
        )

    # --------------------------------------------------------
    # Optional audit metadata
    # --------------------------------------------------------

    if "label_text" in row:
        output["label_text"] = row["label_text"]

    if "dataset" in row:
        output["dataset"] = row["dataset"]

    if "language" in row:
        output["language"] = row["language"]

    if "split" in row:
        output["split"] = row["split"]

    return output


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Greedy teacher JSONL file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output SFT JSONL file.",
    )

    parser.add_argument(
        "--stats",
        default=None,
        help=(
            "Optional path for JSON statistics. "
            "Defaults to <output>.stats.json"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.stats:
        stats_path = Path(args.stats)
    else:
        stats_path = Path(
            str(output_path) + ".stats.json"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = 0
    kept = 0

    rejection_stats = {
        "missing_response": 0,
        "malformed_structure": 0,
        "unparseable_answer": 0,
        "invalid_gold_label": 0,
        "wrong_prediction": 0,
    }

    kept_source_ids = []
    rejected_source_ids = {}

    # --------------------------------------------------------
    # Process file
    # --------------------------------------------------------

    with input_path.open(
        encoding="utf-8"
    ) as fin, output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for line_number, line in enumerate(
            fin,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            total += 1

            row = json.loads(line)

            source_id = str(
                row.get(
                    "source_id",
                    row.get("id", ""),
                )
            )

            target, reason = validate_greedy_row(
                row
            )

            if target is None:

                rejection_stats[reason] += 1

                rejected_source_ids[
                    source_id
                ] = reason

                continue

            sft_row = make_sft_row(
                row,
                target,
            )

            fout.write(
                json.dumps(
                    sft_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            kept += 1

            kept_source_ids.append(
                source_id
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_greedy_examples": total,
        "kept_sft_examples": kept,
        "rejected_examples": total - kept,
        "retention_rate": (
            kept / total
            if total
            else 0.0
        ),
        "rejections": rejection_stats,
        "kept_source_ids": kept_source_ids,
        "rejected_source_ids": (
            rejected_source_ids
        ),
    }

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("GREEDY SFT DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Total greedy examples: {total}"
    )

    print(
        f"Kept SFT examples:     {kept}"
    )

    print(
        f"Rejected examples:     {total - kept}"
    )

    if total:

        print(
            "Retention rate:        "
            f"{kept / total:.2%}"
        )

    print()
    print("Rejections:")

    for reason, count in (
        rejection_stats.items()
    ):
        print(
            f"  {reason:24s} {count}"
        )

    print()
    print("Saved:")
    print(f"  SFT data: {output_path}")
    print(f"  Stats:    {stats_path}")
    print()


if __name__ == "__main__":
    main()
