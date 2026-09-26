#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from collections import Counter


REQUIRED_SECTIONS = [
    "text_evidence",
    "audio_evidence",
    "visual_evidence",
    "integration",
    "answer",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build compact grounding-judge pools "
            "from N=8 teacher trajectories."
        )
    )

    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        required=True,
        help="Dataset split to process.",
    )

    return parser.parse_args()


def get_paths(split):

    teacher_file = Path(
        "results/mustard/teacher/sample_n8/"
        f"{split}/qwen3_omni_30b_{split}_n8.jsonl"
    )

    source_file = Path(
        "data/mustard/processed/"
        f"zero_shot_{split}.jsonl"
    )

    output_file = Path(
        "data/mustard/processed/genrm/"
        f"grounding_judge_pool_{split}.jsonl"
    )

    stats_file = Path(
        "data/mustard/processed/genrm/"
        f"grounding_judge_pool_{split}.stats.json"
    )

    return (
        teacher_file,
        source_file,
        output_file,
        stats_file,
    )


def read_jsonl(path):

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                yield json.loads(line)


def get_assistant_text(row):

    messages = row.get(
        "messages",
        [],
    )

    for message in reversed(messages):

        if (
            message.get("role")
            == "assistant"
        ):

            content = message.get(
                "content",
                "",
            )

            if isinstance(
                content,
                str,
            ):

                return content

    return None


def parse_sections(text):

    if not text:
        return None

    result = {}

    for section in REQUIRED_SECTIONS:

        pattern = (
            rf"<{section}>"
            rf"(.*?)"
            rf"</{section}>"
        )

        matches = re.findall(
            pattern,
            text,
            flags=(
                re.DOTALL
                | re.IGNORECASE
            ),
        )

        if len(matches) != 1:
            return None

        value = matches[0].strip()

        if not value:
            return None

        result[section] = value

    return result


def parse_answer(answer):

    x = (
        answer
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    x = re.sub(
        r"\s+",
        " ",
        x,
    )

    if x == "sarcasm":
        return 1

    if x in {
        "non sarcasm",
        "nonsarcasm",
    }:
        return 0

    return None


def extract_transcript_from_source(row):

    messages = row.get(
        "messages",
        [],
    )

    user_text = None

    for message in messages:

        if (
            message.get("role")
            == "user"
        ):

            content = message.get(
                "content",
                "",
            )

            if isinstance(
                content,
                str,
            ):

                user_text = content

    if not user_text:
        return None

    patterns = [
        r"(?is)"
        r"(?:transcript|utterance|text)"
        r"\s*:\s*"
        r"[\"']?"
        r"(.+?)"
        r"[\"']?"
        r"(?:\n|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            user_text,
        )

        if match:

            value = (
                match.group(1)
                .strip()
            )

            if value:
                return value

    return None


def main():

    args = parse_args()
    split = args.split

    (
        teacher_file,
        source_file,
        output_file,
        stats_file,
    ) = get_paths(split)

    print("=" * 80)
    print(
        f"BUILDING GROUNDING JUDGE POOL: {split}"
    )
    print("=" * 80)

    print(
        "Teacher:",
        teacher_file,
    )

    print(
        "Source:",
        source_file,
    )

    print(
        "Output:",
        output_file,
    )

    print()

    if not teacher_file.exists():

        raise FileNotFoundError(
            f"Teacher file not found: {teacher_file}"
        )

    if not source_file.exists():

        raise FileNotFoundError(
            f"Source file not found: {source_file}"
        )

    print(
        "Loading zero-shot source data..."
    )

    source_rows = list(
        read_jsonl(
            source_file
        )
    )

    source_by_id = {
        str(row["id"]): row
        for row in source_rows
    }

    print(
        "Sources:",
        len(source_by_id),
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counters = Counter()

    source_candidate_counts = Counter()

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as fout:

        for raw_index, row in enumerate(
            read_jsonl(
                teacher_file
            )
        ):

            counters[
                "raw_trajectories"
            ] += 1

            # --------------------------------
            # 1. completion must be finished
            # --------------------------------

            if (
                row.get(
                    "finish_reason"
                )
                != "stop"
            ):

                counters[
                    "rejected_finish_reason"
                ] += 1

                continue

            # --------------------------------
            # 2. recover assistant trajectory
            # --------------------------------

            assistant_text = (
                get_assistant_text(
                    row
                )
            )

            sections = parse_sections(
                assistant_text
            )

            if sections is None:

                counters[
                    "rejected_malformed"
                ] += 1

                continue

            # --------------------------------
            # 3. parse teacher prediction
            # --------------------------------

            pred = parse_answer(
                sections["answer"]
            )

            if pred is None:

                counters[
                    "rejected_bad_answer"
                ] += 1

                continue

            # --------------------------------
            # 4. match original source
            # --------------------------------

            source_id = str(
                row["source_id"]
            )

            source = source_by_id.get(
                source_id
            )

            if source is None:

                counters[
                    "rejected_missing_source"
                ] += 1

                continue

            # --------------------------------
            # 5. transcript
            # --------------------------------

            transcript = (
                extract_transcript_from_source(
                    source
                )
            )

            if transcript is None:

                counters[
                    "rejected_missing_transcript"
                ] += 1

                continue

            # --------------------------------
            # 6. media
            # --------------------------------

            audios = source.get(
                "audios",
                [],
            )

            videos = source.get(
                "videos",
                [],
            )

            # fallback to teacher row
            if not audios:

                audios = row.get(
                    "audios",
                    [],
                )

            if not videos:

                videos = row.get(
                    "videos",
                    [],
                )

            audio = (
                audios[0]
                if audios
                else None
            )

            video = (
                videos[0]
                if videos
                else None
            )

            if not audio:

                counters[
                    "rejected_missing_audio"
                ] += 1

                continue

            if not video:

                counters[
                    "rejected_missing_video"
                ] += 1

                continue

            # --------------------------------
            # 7. gold label
            # --------------------------------

            gold = int(
                source["label"]
            )

            # --------------------------------
            # 8. compact unique judge id
            # --------------------------------

            sample_idx = row.get(
                "sample_idx"
            )

            if sample_idx is None:
                sample_idx = raw_index

            judge_id = (
                f"{split}:"
                f"{source_id}:"
                f"{sample_idx}"
            )

            output = {

                "judge_id":
                    str(judge_id),

                "teacher_id":
                    row.get("id"),

                "source_id":
                    source_id,

                "sample_idx":
                    sample_idx,

                "split":
                    split,

                "text":
                    transcript,

                "audio":
                    audio,

                "video":
                    video,

                "label":
                    gold,

                "prediction":
                    pred,

                "prediction_correct":
                    pred == gold,

                "sections": {

                    "text_evidence":
                        sections[
                            "text_evidence"
                        ],

                    "audio_evidence":
                        sections[
                            "audio_evidence"
                        ],

                    "visual_evidence":
                        sections[
                            "visual_evidence"
                        ],

                    "integration":
                        sections[
                            "integration"
                        ],
                },

                "teacher_mean_logprob":
                    row.get(
                        "mean_token_logprob"
                    ),

                "teacher_num_completion_tokens":
                    row.get(
                        "num_completion_tokens"
                    ),
            }

            fout.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                )
                + "\n"
            )

            counters[
                "kept"
            ] += 1

            if pred == gold:

                counters[
                    "prediction_correct"
                ] += 1

            else:

                counters[
                    "prediction_incorrect"
                ] += 1

            source_candidate_counts[
                source_id
            ] += 1

    counts = list(
        source_candidate_counts.values()
    )

    stats = {
        "split": split,
        **dict(counters),
    }

    stats[
        "num_source_examples"
    ] = len(
        source_by_id
    )

    stats[
        "sources_with_candidates"
    ] = len(
        source_candidate_counts
    )

    stats[
        "sources_without_candidates"
    ] = (
        len(source_by_id)
        -
        len(source_candidate_counts)
    )

    if counts:

        stats[
            "candidates_per_source_mean"
        ] = (
            sum(counts)
            /
            len(counts)
        )

        stats[
            "candidates_per_source_min"
        ] = min(counts)

        stats[
            "candidates_per_source_max"
        ] = max(counts)

    with open(
        stats_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print(
        f"GROUNDING JUDGE POOL: {split}"
    )
    print("=" * 80)

    for key, value in stats.items():

        print(
            f"{key}: {value}"
        )

    print()
    print(
        "Output:",
        output_file,
    )

    print(
        "Stats:",
        stats_file,
    )


if __name__ == "__main__":
    main()
