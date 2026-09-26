#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT_PATH = Path(
    "src/prompts/genrm/"
    "grounding_genrm_system.txt"
)

INPUT_TEMPLATE = """<audio><video>

Transcript:
{transcript}

Candidate reasoning:

<text_evidence>
{text_evidence}
</text_evidence>

<audio_evidence>
{audio_evidence}
</audio_evidence>

<visual_evidence>
{visual_evidence}
</visual_evidence>

<integration>
{integration}
</integration>

Evaluate the grounding of the four candidate reasoning sections.
Return only the required JSON object."""


COMPONENTS = [
    "text",
    "audio",
    "visual",
    "integration",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def read_jsonl(path):

    rows = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


def build_target(labels):

    target = {
        "text":
            int(labels["text"]),

        "audio":
            int(labels["audio"]),

        "visual":
            int(labels["visual"]),

        "integration":
            int(labels["integration"]),
    }

    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_row(row):

    required = [
        "judge_id",
        "source_id",
        "sample_idx",
        "text",
        "audio",
        "video",
        "sections",
        "grounding_labels",
    ]

    for key in required:

        if key not in row:

            raise KeyError(
                f"{row.get('judge_id')} "
                f"missing key: {key}"
            )

    sections = row["sections"]

    for key in [
        "text_evidence",
        "audio_evidence",
        "visual_evidence",
        "integration",
    ]:

        if key not in sections:

            raise KeyError(
                f"{row['judge_id']} "
                f"missing section: {key}"
            )

    labels = row[
        "grounding_labels"
    ]

    for component in COMPONENTS:

        if labels.get(
            component
        ) not in {
            0,
            1,
        }:

            raise ValueError(
                f"{row['judge_id']} "
                f"invalid {component} label: "
                f"{labels.get(component)}"
            )


def main():

    args = parse_args()

    split = args.split

    input_path = (
        args.input
        if args.input is not None
        else Path(
            "data/mustard/processed/genrm/"
            f"grounding_judge_labeled_{split}.jsonl"
        )
    )

    output_path = (
        args.output
        if args.output is not None
        else Path(
            "data/mustard/processed/genrm/sft/"
            f"genrm_{split}.jsonl"
        )
    )

    stats_path = (
        output_path.with_suffix(
            ".stats.json"
        )
    )

    print("=" * 80)
    print(
        f"BUILD GenRM SFT DATA: {split}"
    )
    print("=" * 80)

    print(
        "Input:",
        input_path,
    )

    print(
        "Output:",
        output_path,
    )

    rows = read_jsonl(
        input_path
    )

    system_prompt = (
        SYSTEM_PROMPT_PATH
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_ids = set()

    component_counts = {
        x: Counter()
        for x in COMPONENTS
    }

    target_pattern_counts = (
        Counter()
    )

    prediction_correct_counts = (
        Counter()
    )

    source_ids = set()

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as fout:

        for row in rows:

            validate_row(
                row
            )

            judge_id = str(
                row["judge_id"]
            )

            if judge_id in seen_ids:

                raise RuntimeError(
                    f"Duplicate judge_id: "
                    f"{judge_id}"
                )

            seen_ids.add(
                judge_id
            )

            source_ids.add(
                str(
                    row["source_id"]
                )
            )

            sections = row[
                "sections"
            ]

            labels = row[
                "grounding_labels"
            ]

            user_content = (
                INPUT_TEMPLATE.format(
                    transcript=(
                        row["text"]
                    ),

                    text_evidence=(
                        sections[
                            "text_evidence"
                        ]
                    ),

                    audio_evidence=(
                        sections[
                            "audio_evidence"
                        ]
                    ),

                    visual_evidence=(
                        sections[
                            "visual_evidence"
                        ]
                    ),

                    integration=(
                        sections[
                            "integration"
                        ]
                    ),
                )
            )

            target = build_target(
                labels
            )

            # Standard ms-swift multimodal
            # SFT structure.
            #
            # IMPORTANT:
            # gold sarcasm label is NOT
            # included in the model input.

            example = {
                "messages": [
                    {
                        "role":
                            "system",

                        "content":
                            system_prompt,
                    },
                    {
                        "role":
                            "user",

                        "content":
                            user_content,
                    },
                    {
                        "role":
                            "assistant",

                        "content":
                            target,
                    },
                ],

                "audios": [
                    row["audio"]
                ],

                "videos": [
                    row["video"]
                ],

                # Audit metadata.
                # These fields are not part
                # of the conversation input.
                "judge_id":
                    judge_id,

                "source_id":
                    str(
                        row["source_id"]
                    ),

                "sample_idx":
                    row[
                        "sample_idx"
                    ],
            }

            fout.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

            for component in COMPONENTS:

                component_counts[
                    component
                ][
                    int(
                        labels[
                            component
                        ]
                    )
                ] += 1

            pattern = (
                f"{labels['text']}"
                f"{labels['audio']}"
                f"{labels['visual']}"
                f"{labels['integration']}"
            )

            target_pattern_counts[
                pattern
            ] += 1

            prediction_correct_counts[
                bool(
                    row.get(
                        "prediction_correct",
                        False,
                    )
                )
            ] += 1

    stats = {
        "split":
            split,

        "total":
            len(rows),

        "unique_judge_ids":
            len(
                seen_ids
            ),

        "unique_sources":
            len(
                source_ids
            ),

        "component_counts": {
            component: {
                "supported":
                    component_counts[
                        component
                    ][1],

                "unsupported":
                    component_counts[
                        component
                    ][0],

                "supported_rate":
                    round(
                        component_counts[
                            component
                        ][1]
                        / len(rows),
                        6,
                    ),
            }
            for component
            in COMPONENTS
        },

        "target_pattern_counts": {
            key:
                value
            for key, value
            in target_pattern_counts
            .most_common()
        },

        "prediction_correct": {
            "true":
                prediction_correct_counts[
                    True
                ],

            "false":
                prediction_correct_counts[
                    False
                ],
        },
    }

    with open(
        stats_path,
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
    print(
        "Rows:",
        len(rows),
    )

    print(
        "Unique IDs:",
        len(seen_ids),
    )

    print(
        "Unique sources:",
        len(source_ids),
    )

    print()

    for component in COMPONENTS:

        pos = (
            component_counts[
                component
            ][1]
        )

        neg = (
            component_counts[
                component
            ][0]
        )

        print(
            f"{component:12s} "
            f"SUPPORTED={pos:5d} "
            f"UNSUPPORTED={neg:5d} "
            f"rate={100*pos/len(rows):6.2f}%"
        )

    print()
    print(
        "Top target patterns:"
    )

    for (
        pattern,
        count,
    ) in target_pattern_counts.most_common(
        10
    ):

        print(
            f"  {pattern}: "
            f"{count} "
            f"({100*count/len(rows):.2f}%)"
        )

    print()
    print(
        "Stats:",
        stats_path,
    )


if __name__ == "__main__":
    main()
