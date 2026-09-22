import argparse
import json
import random
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    sarcasm = [x for x in rows if x["label"] == 1]
    non_sarcasm = [x for x in rows if x["label"] == 0]

    if len(sarcasm) < args.n_per_class:
        raise ValueError(
            f"Not enough sarcasm samples: "
            f"{len(sarcasm)} < {args.n_per_class}"
        )

    if len(non_sarcasm) < args.n_per_class:
        raise ValueError(
            f"Not enough non-sarcasm samples: "
            f"{len(non_sarcasm)} < {args.n_per_class}"
        )

    rng = random.Random(args.seed)

    selected = (
        rng.sample(sarcasm, args.n_per_class)
        + rng.sample(non_sarcasm, args.n_per_class)
    )

    rng.shuffle(selected)

    with open(args.output, "w", encoding="utf-8") as f:
        for x in selected:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    counts = Counter(x["label"] for x in selected)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Seed:   {args.seed}")
    print(f"Total:  {len(selected)}")
    print(f"Sarcasm:     {counts[1]}")
    print(f"Non-Sarcasm: {counts[0]}")
    print()
    print("First 10 samples:")
    for x in selected[:10]:
        print(
            f'{x["id"]:20s} '
            f'label={x["label"]} '
            f'{x["label_text"]}'
        )


if __name__ == "__main__":
    main()
