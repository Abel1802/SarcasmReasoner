#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/mustard/processed/genrm/"
            "judge_pool_raw.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/mustard/processed/genrm/"
            "judge_pool_smoke100.jsonl"
        ),
    )

    parser.add_argument(
        "--n-correct",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--n-incorrect",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    rows = read_jsonl(args.input)

    # --------------------------------------------------------
    # Group trajectories by source.
    # --------------------------------------------------------

    by_source = defaultdict(
        lambda: {
            "correct": [],
            "incorrect": [],
        }
    )

    for row in rows:
        sid = str(row["source_id"])

        if int(row["is_correct"]) == 1:
            by_source[sid]["correct"].append(row)
        else:
            by_source[sid]["incorrect"].append(row)

    correct_sources = [
        sid
        for sid, groups in by_source.items()
        if groups["correct"]
    ]

    incorrect_sources = [
        sid
        for sid, groups in by_source.items()
        if groups["incorrect"]
    ]

    rng.shuffle(correct_sources)
    rng.shuffle(incorrect_sources)

    # --------------------------------------------------------
    # First choose correct examples.
    # --------------------------------------------------------

    if len(correct_sources) < args.n_correct:
        raise RuntimeError(
            f"Need {args.n_correct} sources with correct "
            f"trajectories, found {len(correct_sources)}."
        )

    selected_correct_sources = set(
        correct_sources[:args.n_correct]
    )

    selected = []

    for sid in selected_correct_sources:
        candidate = rng.choice(
            by_source[sid]["correct"]
        )
        selected.append(candidate)

    # --------------------------------------------------------
    # Choose incorrect examples from DIFFERENT sources.
    # --------------------------------------------------------

    remaining_incorrect_sources = [
        sid
        for sid in incorrect_sources
        if sid not in selected_correct_sources
    ]

    if (
        len(remaining_incorrect_sources)
        < args.n_incorrect
    ):
        raise RuntimeError(
            f"Need {args.n_incorrect} additional sources "
            f"with incorrect trajectories, found "
            f"{len(remaining_incorrect_sources)}."
        )

    selected_incorrect_sources = set(
        remaining_incorrect_sources[
            :args.n_incorrect
        ]
    )

    for sid in selected_incorrect_sources:
        candidate = rng.choice(
            by_source[sid]["incorrect"]
        )
        selected.append(candidate)

    rng.shuffle(selected)

    output = Path(args.output)

    write_jsonl(
        output,
        selected,
    )

    # --------------------------------------------------------
    # Sanity checks.
    # --------------------------------------------------------

    n = len(selected)

    n_correct = sum(
        int(x["is_correct"]) == 1
        for x in selected
    )

    n_incorrect = sum(
        int(x["is_correct"]) == 0
        for x in selected
    )

    unique_sources = len({
        str(x["source_id"])
        for x in selected
    })

    assert n == (
        args.n_correct
        + args.n_incorrect
    )

    assert n_correct == args.n_correct
    assert n_incorrect == args.n_incorrect
    assert unique_sources == n

    print("=" * 68)
    print("GenRM consistency smoke set")
    print("=" * 68)
    print(f"Input:            {args.input}")
    print(f"Output:           {output}")
    print()
    print(f"Total:            {n}")
    print(f"Correct:          {n_correct}")
    print(f"Incorrect:        {n_incorrect}")
    print(f"Unique sources:   {unique_sources}")
    print(f"Seed:             {args.seed}")
    print("=" * 68)


if __name__ == "__main__":
    main()
