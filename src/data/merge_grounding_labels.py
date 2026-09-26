#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


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


def load_labels(path):

    rows = read_jsonl(path)

    result = {}

    duplicate_ids = []

    for row in rows:

        judge_id = str(
            row["judge_id"]
        )

        if judge_id in result:
            duplicate_ids.append(
                judge_id
            )

        result[judge_id] = row

    if duplicate_ids:

        raise RuntimeError(
            f"Duplicate IDs in {path}: "
            f"{duplicate_ids[:10]}"
        )

    return result


def main():

    args = parse_args()
    split = args.split

    pool_path = Path(
        "data/mustard/processed/genrm/"
        f"grounding_judge_pool_{split}.jsonl"
    )

    label_dir = Path(
        "results/mustard/genrm/"
        "grounding_judging"
    ) / split

    output_path = Path(
        "data/mustard/processed/genrm/"
        f"grounding_judge_labeled_{split}.jsonl"
    )

    stats_path = Path(
        "data/mustard/processed/genrm/"
        f"grounding_judge_labeled_{split}.stats.json"
    )

    print("=" * 80)
    print(
        f"MERGING GROUNDING LABELS: {split}"
    )
    print("=" * 80)

    pool = read_jsonl(
        pool_path
    )

    print(
        "Pool rows:",
        len(pool),
    )

    label_maps = {}

    for component in COMPONENTS:

        path = (
            label_dir
            / f"{component}_labels.jsonl"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing: {path}"
            )

        label_maps[
            component
        ] = load_labels(
            path
        )

        print(
            component,
            "labels:",
            len(
                label_maps[
                    component
                ]
            ),
        )

    pool_ids = {
        str(row["judge_id"])
        for row in pool
    }

    # --------------------------------
    # Coverage validation
    # --------------------------------

    for component in COMPONENTS:

        label_ids = set(
            label_maps[
                component
            ].keys()
        )

        missing = (
            pool_ids
            - label_ids
        )

        extra = (
            label_ids
            - pool_ids
        )

        print()
        print(
            component.upper()
        )

        print(
            "Missing:",
            len(missing),
        )

        print(
            "Extra:",
            len(extra),
        )

        if missing:

            print(
                "Example missing:",
                list(missing)[:5],
            )

        if extra:

            print(
                "Example extra:",
                list(extra)[:5],
            )

        if missing or extra:

            raise RuntimeError(
                f"Coverage mismatch "
                f"for {component}"
            )

    # --------------------------------
    # Merge
    # --------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    component_counts = {
        component:
            Counter()
        for component
        in COMPONENTS
    }

    sum_counts = Counter()
    all_grounded_counts = Counter()

    prediction_correct_counts = Counter()

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as fout:

        for row in pool:

            judge_id = str(
                row["judge_id"]
            )

            grounding_labels = {}

            grounding_judgments = {}

            for component in COMPONENTS:

                label_row = (
                    label_maps[
                        component
                    ][
                        judge_id
                    ]
                )

                grounded = (
                    label_row[
                        "grounded"
                    ]
                )

                if grounded not in {
                    0,
                    1,
                }:

                    raise RuntimeError(
                        f"Invalid {component} "
                        f"label for {judge_id}: "
                        f"{grounded}"
                    )

                grounding_labels[
                    component
                ] = grounded

                # Preserve judge explanation
                # separately for later audit.
                grounding_judgments[
                    component
                ] = {
                    "verdict":
                        label_row.get(
                            "verdict"
                        ),

                    "evidence":
                        label_row.get(
                            "evidence"
                        ),
                }

                component_counts[
                    component
                ][
                    grounded
                ] += 1

            grounding_sum = sum(
                grounding_labels.values()
            )

            all_grounded = (
                grounding_sum
                == len(COMPONENTS)
            )

            sum_counts[
                grounding_sum
            ] += 1

            all_grounded_counts[
                all_grounded
            ] += 1

            prediction_correct_counts[
                bool(
                    row[
                        "prediction_correct"
                    ]
                )
            ] += 1

            merged = dict(row)

            merged[
                "grounding_labels"
            ] = grounding_labels

            merged[
                "grounding_sum"
            ] = grounding_sum

            merged[
                "all_grounded"
            ] = all_grounded

            merged[
                "grounding_judgments"
            ] = grounding_judgments

            fout.write(
                json.dumps(
                    merged,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------
    # Stats
    # --------------------------------

    stats = {
        "split":
            split,

        "total":
            len(pool),

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
            }
            for component
            in COMPONENTS
        },

        "grounding_sum_counts": {
            str(k):
                v
            for k, v
            in sorted(
                sum_counts.items()
            )
        },

        "all_grounded": {
            "true":
                all_grounded_counts[
                    True
                ],

            "false":
                all_grounded_counts[
                    False
                ],
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
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)

    print(
        "Output:",
        output_path,
    )

    print(
        "Stats:",
        stats_path,
    )

    print()
    print(
        "Grounding sum distribution:"
    )

    for k in range(5):

        print(
            f"  {k}:",
            sum_counts[k],
        )

    print()
    print(
        "All grounded:",
        all_grounded_counts[
            True
        ],
    )

    print(
        "Not all grounded:",
        all_grounded_counts[
            False
        ],
    )


if __name__ == "__main__":
    main()
