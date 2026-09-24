#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict
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
# Parsing
# ============================================================

def normalize_answer(text):
    """Map model answer text to binary label."""

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


def extract_sections(content):
    """
    Extract exactly one non-empty occurrence of every required tag.

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

        # Require exactly one section.
        if len(matches) != 1:
            return None

        value = matches[0].strip()

        if not value:
            return None

        sections[tag] = value

    return sections


def build_canonical_target(sections):
    """
    Rebuild only the explicit structured reasoning trajectory.

    This intentionally removes:
        <think> ... </think>
        free text before/after the requested structure
    """

    answer_label = normalize_answer(
        sections["answer"]
    )

    if answer_label == 1:
        answer = "Sarcasm"

    elif answer_label == 0:
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
# Deduplication
# ============================================================

def normalize_for_dedup(text):
    """
    Conservative exact-text normalization.

    We collapse whitespace but do NOT use semantic similarity.
    """

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# Candidate validation
# ============================================================

def validate_candidate(row):
    """
    Candidate is usable for SFT iff:

    1. generation stopped normally
    2. all required structured sections exist
    3. answer can be parsed
    4. answer matches gold label
    """

    if row.get("finish_reason") != "stop":
        return None, "finish_reason"

    messages = row.get("messages")

    if not messages:
        return None, "missing_messages"

    if messages[-1].get("role") != "assistant":
        return None, "missing_assistant"

    content = messages[-1].get(
        "content",
        "",
    )

    sections = extract_sections(content)

    if sections is None:
        return None, "malformed_structure"

    prediction = normalize_answer(
        sections["answer"]
    )

    if prediction is None:
        return None, "unparseable_answer"

    gold = row.get("label")

    try:
        gold = int(gold)
    except Exception:
        return None, "invalid_gold"

    if prediction != gold:
        return None, "wrong_prediction"

    target = build_canonical_target(
        sections
    )

    return target, None


# ============================================================
# Convert raw teacher row into SFT row
# ============================================================

def make_sft_row(row, target, variant):
    """
    Build an ms-swift conversational SFT example.
    """

    output = {}

    # Preserve multimodal conversation.
    messages = deepcopy(
        row["messages"]
    )

    # Replace teacher raw response, including <think>,
    # with clean structured reasoning target.
    messages[-1]["content"] = target

    output["messages"] = messages

    if row.get("videos"):
        output["videos"] = deepcopy(
            row["videos"]
        )

    if row.get("audios"):
        output["audios"] = deepcopy(
            row["audios"]
        )

    # Metadata retained for auditing / reproducibility.
    output["source_id"] = str(
        row["source_id"]
    )

    output["label"] = int(
        row["label"]
    )

    output["teacher_sample_idx"] = int(
        row["sample_idx"]
    )

    output["teacher_mean_token_logprob"] = (
        row.get("mean_token_logprob")
    )

    output["sft_variant"] = variant

    return output


# ============================================================
# Main builder
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Raw N=8 teacher sampling JSONL.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default=None,
        help=(
            "Dataset split used in output filenames. "
            "If omitted, infer from the input filename/path."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # --------------------------------------------------------
    # Determine split
    #
    # Explicit --split has priority. Otherwise infer from the
    # input path so existing commands keep working, e.g.:
    #
    #   .../train/qwen3_omni_30b_train_n8.jsonl -> train
    #   .../valid/qwen3_omni_30b_valid_n8.jsonl -> valid
    #   .../test/qwen3_omni_30b_test_n8.jsonl   -> test
    # --------------------------------------------------------

    split = args.split

    if split is None:
        input_text = str(input_path).lower()

        matches = [
            name
            for name in ("train", "valid", "test")
            if (
                f"/{name}/" in input_text
                or f"_{name}_" in input_path.name.lower()
                or input_path.name.lower().startswith(f"{name}_")
                or input_path.name.lower().endswith(f"_{name}.jsonl")
            )
        ]

        if len(matches) != 1:
            raise ValueError(
                "Could not uniquely infer dataset split from input path. "
                "Please pass --split train|valid|test explicitly.\n"
                f"input={input_path}"
            )

        split = matches[0]

    print(f"Detected split: {split}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read raw trajectories
    # --------------------------------------------------------

    rows = []

    with input_path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    print(
        f"Raw trajectories: {len(rows)}"
    )

    # --------------------------------------------------------
    # Group by source input
    # --------------------------------------------------------

    groups = defaultdict(list)

    for row in rows:

        source_id = str(
            row.get("source_id", "")
        ).strip()

        if not source_id:
            raise ValueError(
                "Encountered empty source_id."
            )

        groups[source_id].append(
            row
        )

    print(
        f"Source inputs: {len(groups)}"
    )

    # --------------------------------------------------------
    # Validate + filter
    # --------------------------------------------------------

    rejection_stats = defaultdict(int)

    usable_groups = {}

    sources_without_candidate = []

    for source_id, candidates in groups.items():

        usable = []

        for row in candidates:

            target, reason = (
                validate_candidate(row)
            )

            if target is None:

                rejection_stats[reason] += 1
                continue

            usable.append(
                {
                    "row": row,
                    "target": target,
                }
            )

        if not usable:

            sources_without_candidate.append(
                source_id
            )

        else:

            usable_groups[source_id] = (
                usable
            )

    # --------------------------------------------------------
    # Best-of-N
    # --------------------------------------------------------

    best_rows = []

    for source_id, candidates in (
        usable_groups.items()
    ):

        candidates_with_score = [
            x
            for x in candidates
            if x["row"].get(
                "mean_token_logprob"
            ) is not None
        ]

        if not candidates_with_score:
            raise ValueError(
                "No candidate with logprob for "
                f"source_id={source_id}"
            )

        best = max(
            candidates_with_score,
            key=lambda x: x["row"][
                "mean_token_logprob"
            ],
        )

        best_rows.append(
            make_sft_row(
                best["row"],
                best["target"],
                variant="best_of_8",
            )
        )

    # --------------------------------------------------------
    # Diverse-N
    #
    # Keep every correct, complete, unique trajectory.
    # Deduplicate only normalized exact text.
    # --------------------------------------------------------

    diverse_rows = []

    for source_id, candidates in (
        usable_groups.items()
    ):

        seen = set()

        # Keep deterministic order.
        candidates = sorted(
            candidates,
            key=lambda x: x["row"][
                "sample_idx"
            ],
        )

        for candidate in candidates:

            key = normalize_for_dedup(
                candidate["target"]
            )

            if key in seen:
                continue

            seen.add(key)

            diverse_rows.append(
                make_sft_row(
                    candidate["row"],
                    candidate["target"],
                    variant="diverse_8",
                )
            )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    best_path = (
        output_dir
        / f"sft_best_of_8_{split}.jsonl"
    )

    diverse_path = (
        output_dir
        / f"sft_diverse_8_{split}.jsonl"
    )

    with best_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in best_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with diverse_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in diverse_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "split": split,
        "raw_trajectories": len(rows),
        "source_inputs": len(groups),
        "sources_with_usable_candidate": (
            len(usable_groups)
        ),
        "sources_without_usable_candidate": (
            len(sources_without_candidate)
        ),
        "best_of_8_examples": len(best_rows),
        "diverse_8_examples": len(diverse_rows),
        "rejections": dict(
            sorted(rejection_stats.items())
        ),
        "sources_without_candidate": (
            sources_without_candidate
        ),
    }

    stats_path = (
        output_dir
        / f"sft_sampling_stats_{split}.json"
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

    print()
    print("=" * 60)
    print("SFT DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(
        "Raw trajectories:",
        len(rows),
    )

    print(
        "Source inputs:",
        len(groups),
    )

    print(
        "Sources with usable candidate:",
        len(usable_groups),
    )

    print(
        "Sources without usable candidate:",
        len(sources_without_candidate),
    )

    print(
        "Best-of-8 examples:",
        len(best_rows),
    )

    print(
        "Diverse-8 examples:",
        len(diverse_rows),
    )

    print()

    print("Rejected trajectories:")

    for reason, count in sorted(
        rejection_stats.items()
    ):
        print(
            f"  {reason}: {count}"
        )

    print()
    print("Saved:")
    print(f"  {best_path}")
    print(f"  {diverse_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
