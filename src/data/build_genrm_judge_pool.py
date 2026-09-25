#!/usr/bin/env python3

"""
Build the raw GenRM judge pool from the complete N=8 teacher
trajectory pool.

This script does NOT assign consistency labels.

It only:

1. reads the complete teacher trajectory pool;
2. removes unfinished generations;
3. removes malformed / unparseable responses;
4. keeps BOTH correct and incorrect predictions;
5. extracts metadata needed for later consistency labeling;
6. writes dataset statistics.

For MUStARD++ train:

    raw teacher pool:
        results/mustard/teacher/sample_n8/train/
        qwen3_omni_30b_train_n8.jsonl

    output:
        data/mustard/processed/genrm/judge_pool_raw.jsonl

Expected from the previously observed N=8 statistics:

    raw trajectories:        6728
    source inputs:            841

    rejected finish_reason:    27
    rejected malformed:       253

    valid correct:            4447
    valid incorrect:          2001

    total judge candidates:   6448

The exact counts are checked optionally with --strict-expected.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_TAGS = [
    "text_evidence",
    "audio_evidence",
    "visual_evidence",
    "integration",
    "answer",
]


# ============================================================
# JSONL helpers
# ============================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_no}, "
                    f"got {type(row).__name__}"
                )

            row["_raw_line_no"] = line_no
            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


# ============================================================
# Generic field extraction
# ============================================================

def first_present(
    row: Dict[str, Any],
    keys: List[str],
) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]

    return None


def extract_source_id(row: Dict[str, Any]) -> str:
    """
    Prefer an explicit source ID.

    We deliberately do NOT invent an ID by stripping suffixes from
    arbitrary trajectory IDs, because source grouping is critical
    for later leakage-free GenRM splitting.
    """

    value = first_present(
        row,
        [
            "source_id",
            "id",
            "example_id",
            "utterance_id",
            "uid",
        ],
    )

    if value is None:
        raise KeyError(
            "Could not identify source ID. "
            "Expected one of: source_id, id, example_id, "
            "utterance_id, uid."
        )

    return str(value)


def extract_sample_idx(row: Dict[str, Any]) -> Optional[Any]:
    return first_present(
        row,
        [
            "sample_idx",
            "generation_idx",
            "candidate_idx",
            "trajectory_idx",
            "sample_id",
        ],
    )


def extract_gold_label(row: Dict[str, Any]) -> int:
    value = first_present(
        row,
        [
            "label",
            "gold_label",
            "target",
            "y",
        ],
    )

    if value is None:
        raise KeyError(
            "Could not identify gold label. "
            "Expected one of: label, gold_label, target, y."
        )

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in (0, 1):
            return value

    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)

    text = str(value).strip().lower()

    if text in {
        "1",
        "sarcasm",
        "sarcastic",
    }:
        return 1

    if text in {
        "0",
        "non-sarcasm",
        "non_sarcasm",
        "nonsarcasm",
        "non-sarcastic",
        "non sarcastic",
    }:
        return 0

    raise ValueError(f"Unsupported gold label: {value!r}")


def label_to_text(label: int) -> str:
    if label == 1:
        return "Sarcasm"

    if label == 0:
        return "Non-Sarcasm"

    raise ValueError(f"Unexpected binary label: {label}")


# ============================================================
# Teacher response extraction
# ============================================================

def _message_content_to_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content

    # Support OpenAI / multimodal style message content:
    # [{"type": "text", "text": "..."}]
    if isinstance(content, list):
        chunks = []

        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue

            if isinstance(part, dict):
                text = part.get("text")

                if isinstance(text, str):
                    chunks.append(text)

        if chunks:
            return "\n".join(chunks)

    return None


def extract_trajectory(row: Dict[str, Any]) -> Optional[str]:
    """
    Try the common output fields first, then fall back to the final
    assistant message.
    """

    for key in [
        "response",
        "generated_text",
        "completion",
        "output",
        "prediction_text",
        "assistant_response",
    ]:
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, dict):
            # Some generation formats wrap text in a dict.
            for subkey in ["content", "text", "response"]:
                subvalue = value.get(subkey)

                if isinstance(subvalue, str) and subvalue.strip():
                    return subvalue.strip()

    messages = row.get("messages")

    if isinstance(messages, list):
        assistant_messages = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            if message.get("role") != "assistant":
                continue

            text = _message_content_to_text(
                message.get("content")
            )

            if text and text.strip():
                assistant_messages.append(text.strip())

        if assistant_messages:
            return assistant_messages[-1]

    # Optional OpenAI-like fallback.
    choices = row.get("choices")

    if isinstance(choices, list) and choices:
        choice = choices[0]

        if isinstance(choice, dict):
            message = choice.get("message")

            if isinstance(message, dict):
                text = _message_content_to_text(
                    message.get("content")
                )

                if text and text.strip():
                    return text.strip()

            text = choice.get("text")

            if isinstance(text, str) and text.strip():
                return text.strip()

    return None


# ============================================================
# Finish reason
# ============================================================

def extract_finish_reason(
    row: Dict[str, Any],
) -> Optional[str]:

    value = first_present(
        row,
        [
            "finish_reason",
            "stop_reason",
        ],
    )

    if value is not None:
        return str(value).strip().lower()

    choices = row.get("choices")

    if isinstance(choices, list) and choices:
        choice = choices[0]

        if isinstance(choice, dict):
            value = choice.get("finish_reason")

            if value is not None:
                return str(value).strip().lower()

    return None


def finish_reason_is_valid(
    finish_reason: Optional[str],
) -> bool:
    """
    Missing finish_reason is not automatically rejected because some
    generation formats do not store it.

    If it is present, only normal completion reasons are accepted.
    """

    if finish_reason is None:
        return True

    return finish_reason in {
        "stop",
        "eos",
        "eos_token",
    }


# ============================================================
# Structure parsing
# ============================================================

def extract_single_tag(
    text: str,
    tag: str,
) -> Tuple[bool, Optional[str], str]:
    """
    Require exactly one opening tag and exactly one closing tag.

    Returns:
        valid, content, error_reason
    """

    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    open_count = text.count(open_tag)
    close_count = text.count(close_tag)

    if open_count != 1:
        return (
            False,
            None,
            f"{tag}_open_count_{open_count}",
        )

    if close_count != 1:
        return (
            False,
            None,
            f"{tag}_close_count_{close_count}",
        )

    pattern = re.compile(
        rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>",
        flags=re.DOTALL,
    )

    match = pattern.search(text)

    if match is None:
        return (
            False,
            None,
            f"{tag}_not_parseable",
        )

    content = match.group(1).strip()

    if not content:
        return (
            False,
            None,
            f"{tag}_empty",
        )

    return True, content, ""


def parse_required_structure(
    trajectory: str,
) -> Tuple[bool, Dict[str, str], str]:

    sections: Dict[str, str] = {}

    for tag in REQUIRED_TAGS:
        valid, content, reason = extract_single_tag(
            trajectory,
            tag,
        )

        if not valid:
            return False, {}, reason

        assert content is not None
        sections[tag] = content

    return True, sections, ""


def parse_answer(answer: str) -> Optional[int]:
    normalized = " ".join(
        answer.strip().split()
    ).lower()

    if normalized == "sarcasm":
        return 1

    if normalized in {
        "non-sarcasm",
        "non sarcasm",
    }:
        return 0

    return None


# ============================================================
# Optional metadata
# ============================================================

def extract_mean_logprob(
    row: Dict[str, Any],
) -> Optional[float]:

    value = first_present(
        row,
        [
            "mean_token_logprob",
            "mean_logprob",
            "mean_logp",
            "avg_logprob",
            "average_logprob",
            "response_mean_logprob",
        ],
    )

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_optional_path(
    row: Dict[str, Any],
    singular: str,
    plural: str,
) -> Any:
    if singular in row:
        return row[singular]

    if plural in row:
        return row[plural]

    return None


# ============================================================
# Main pool construction
# ============================================================

def build_pool(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    output_rows: List[Dict[str, Any]] = []

    rejection_reasons = Counter()
    finish_reason_counts = Counter()

    valid_by_gold = Counter()
    valid_by_pred = Counter()
    source_candidate_counts = Counter()

    missing_finish_reason = 0

    observed_source_ids = set()

    for raw_idx, row in enumerate(rows):

        line_no = row.get("_raw_line_no")

        # ----------------------------------------------------
        # Source / gold metadata
        # ----------------------------------------------------

        try:
            source_id = extract_source_id(row)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to extract source ID at "
                f"input line {line_no}: {exc}"
            ) from exc

        observed_source_ids.add(source_id)

        try:
            gold_label = extract_gold_label(row)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to extract gold label at "
                f"input line {line_no}: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Finish reason
        # ----------------------------------------------------

        finish_reason = extract_finish_reason(row)

        if finish_reason is None:
            missing_finish_reason += 1
        else:
            finish_reason_counts[finish_reason] += 1

        if not finish_reason_is_valid(finish_reason):
            rejection_reasons["finish_reason"] += 1
            continue

        # ----------------------------------------------------
        # Generated trajectory
        # ----------------------------------------------------

        trajectory = extract_trajectory(row)

        if trajectory is None:
            rejection_reasons["missing_trajectory"] += 1
            continue

        # ----------------------------------------------------
        # Required structure
        # ----------------------------------------------------

        structure_valid, sections, structure_error = (
            parse_required_structure(trajectory)
        )

        if not structure_valid:
            rejection_reasons["malformed_structure"] += 1
            rejection_reasons[
                f"malformed::{structure_error}"
            ] += 1
            continue

        # ----------------------------------------------------
        # Parse final answer
        # ----------------------------------------------------

        pred_label = parse_answer(
            sections["answer"]
        )

        if pred_label is None:
            rejection_reasons["invalid_answer"] += 1
            continue

        is_correct = int(
            pred_label == gold_label
        )

        # ----------------------------------------------------
        # Keep BOTH correct and incorrect.
        # ----------------------------------------------------

        sample_idx = extract_sample_idx(row)
        mean_logprob = extract_mean_logprob(row)

        candidate = {
            "judge_id": len(output_rows),
            "source_id": source_id,
            "sample_idx": sample_idx,
            "dataset": "mustard",
            "split": "train",
            "origin": "teacher_n8",
            "teacher_model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",

            "gold_label": gold_label,
            "gold_label_text": label_to_text(
                gold_label
            ),

            "pred_label": pred_label,
            "pred_label_text": label_to_text(
                pred_label
            ),

            "is_correct": is_correct,

            # This only means structural validity.
            # It is NOT the GenRM consistency label.
            "format_valid": True,

            "finish_reason": finish_reason,
            "teacher_mean_logprob": mean_logprob,

            "trajectory": trajectory,

            # Parsed sections are useful later for inspection
            # and consistency labeling.
            "sections": {
                "text_evidence": sections[
                    "text_evidence"
                ],
                "audio_evidence": sections[
                    "audio_evidence"
                ],
                "visual_evidence": sections[
                    "visual_evidence"
                ],
                "integration": sections[
                    "integration"
                ],
                "answer": sections[
                    "answer"
                ],
            },

            # Traceability back to the original JSONL.
            "raw_line_no": line_no,
            "raw_index": raw_idx,
        }

        # Preserve paths when available, but these are NOT used
        # by the first text-only consistency judge.
        audio = extract_optional_path(
            row,
            "audio",
            "audios",
        )
        video = extract_optional_path(
            row,
            "video",
            "videos",
        )

        if audio is not None:
            candidate["audio"] = audio

        if video is not None:
            candidate["video"] = video

        output_rows.append(candidate)

        source_candidate_counts[source_id] += 1

        valid_by_gold[
            label_to_text(gold_label)
        ] += 1

        valid_by_pred[
            label_to_text(pred_label)
        ] += 1

    # ========================================================
    # Statistics
    # ========================================================

    correct = sum(
        row["is_correct"]
        for row in output_rows
    )

    incorrect = len(output_rows) - correct

    candidates_per_source = list(
        source_candidate_counts.values()
    )

    source_count_after_filter = len(
        source_candidate_counts
    )

    stats: Dict[str, Any] = {
        "raw_trajectories": len(rows),
        "raw_unique_sources": len(
            observed_source_ids
        ),

        "kept_format_valid": len(
            output_rows
        ),

        "kept_correct": correct,
        "kept_incorrect": incorrect,

        "kept_correct_ratio": (
            correct / len(output_rows)
            if output_rows
            else None
        ),

        "sources_with_judge_candidate":
            source_count_after_filter,

        "sources_without_judge_candidate":
            len(observed_source_ids)
            - source_count_after_filter,

        "candidates_per_retained_source": {
            "mean": (
                sum(candidates_per_source)
                / len(candidates_per_source)
                if candidates_per_source
                else None
            ),
            "min": (
                min(candidates_per_source)
                if candidates_per_source
                else None
            ),
            "max": (
                max(candidates_per_source)
                if candidates_per_source
                else None
            ),
        },

        "valid_gold_label_distribution":
            dict(valid_by_gold),

        "valid_prediction_distribution":
            dict(valid_by_pred),

        "finish_reason_distribution":
            dict(finish_reason_counts),

        "missing_finish_reason":
            missing_finish_reason,

        "rejections": dict(
            rejection_reasons
        ),
    }

    return output_rows, stats


# ============================================================
# Expected-count validation
# ============================================================

def check_expected(
    stats: Dict[str, Any],
    strict: bool,
) -> None:

    expected = {
        "raw_trajectories": 6728,
        "raw_unique_sources": 841,
        "kept_format_valid": 6448,
        "kept_correct": 4447,
        "kept_incorrect": 2001,
    }

    mismatches = []

    for key, expected_value in expected.items():
        actual = stats.get(key)

        if actual != expected_value:
            mismatches.append(
                (
                    key,
                    expected_value,
                    actual,
                )
            )

    if not mismatches:
        print()
        print(
            "Expected MUStARD++ N=8 counts: PASSED"
        )
        return

    print()
    print(
        "WARNING: counts differ from the previously "
        "observed MUStARD++ N=8 statistics."
    )

    for key, expected_value, actual in mismatches:
        print(
            f"  {key}: "
            f"expected={expected_value}, "
            f"actual={actual}"
        )

    if strict:
        raise RuntimeError(
            "Strict expected-count check failed. "
            "Do not continue to consistency labeling until "
            "the discrepancy is understood."
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/mustard/teacher/"
            "sample_n8/train/"
            "qwen3_omni_30b_train_n8.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/mustard/processed/genrm/"
            "judge_pool_raw.jsonl"
        ),
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=Path(
            "data/mustard/processed/genrm/"
            "judge_pool_raw.stats.json"
        ),
    )

    parser.add_argument(
        "--strict-expected",
        action="store_true",
        help=(
            "Fail if the generated counts do not match "
            "the previously observed MUStARD++ N=8 counts."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    print("=" * 68)
    print("Build GenRM raw judge pool")
    print("=" * 68)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Stats:  {args.stats}")
    print("=" * 68)

    rows = read_jsonl(args.input)

    if not rows:
        raise RuntimeError(
            f"No records found in {args.input}"
        )

    print()
    print(
        f"Loaded raw trajectories: {len(rows)}"
    )

    print()
    print(
        "First-record keys:"
    )
    print(
        "  "
        + ", ".join(
            sorted(
                k
                for k in rows[0].keys()
                if not k.startswith("_")
            )
        )
    )

    pool, stats = build_pool(rows)

    check_expected(
        stats,
        strict=args.strict_expected,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        args.output,
        pool,
    )

    with args.stats.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print()
    print("=" * 68)
    print("GenRM raw judge pool statistics")
    print("=" * 68)

    print(
        f"Raw trajectories:            "
        f"{stats['raw_trajectories']}"
    )

    print(
        f"Raw unique sources:          "
        f"{stats['raw_unique_sources']}"
    )

    print()

    print(
        f"Kept format-valid:           "
        f"{stats['kept_format_valid']}"
    )

    print(
        f"  correct prediction:        "
        f"{stats['kept_correct']}"
    )

    print(
        f"  incorrect prediction:      "
        f"{stats['kept_incorrect']}"
    )

    print()

    print(
        f"Sources with candidates:     "
        f"{stats['sources_with_judge_candidate']}"
    )

    print(
        f"Sources without candidates:  "
        f"{stats['sources_without_judge_candidate']}"
    )

    cps = stats[
        "candidates_per_retained_source"
    ]

    print()
    print(
        "Candidates / retained source:"
    )
    print(
        f"  mean: {cps['mean']:.3f}"
        if cps["mean"] is not None
        else "  mean: N/A"
    )
    print(
        f"  min:  {cps['min']}"
    )
    print(
        f"  max:  {cps['max']}"
    )

    print()

    print("Rejections:")

    for key, value in sorted(
        stats["rejections"].items()
    ):
        # Avoid printing every fine-grained malformed subtype
        # twice in the compact summary.
        if key.startswith("malformed::"):
            continue

        print(
            f"  {key:<28} {value}"
        )

    print()

    print(
        "Gold-label distribution among "
        "judge candidates:"
    )

    for key, value in sorted(
        stats[
            "valid_gold_label_distribution"
        ].items()
    ):
        print(
            f"  {key:<16} {value}"
        )

    print()

    print(
        "Prediction distribution among "
        "judge candidates:"
    )

    for key, value in sorted(
        stats[
            "valid_prediction_distribution"
        ].items()
    ):
        print(
            f"  {key:<16} {value}"
        )

    print()
    print("=" * 68)
    print("Finished")
    print("=" * 68)
    print(f"Judge pool: {args.output}")
    print(f"Stats:      {args.stats}")
    print()

    print(
        "IMPORTANT: no consistency labels have "
        "been assigned yet."
    )
    print(
        "The next stage should judge only the "
        "reasoning consistency of each trajectory."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
