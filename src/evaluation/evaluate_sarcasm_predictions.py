#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def get_source_id(row):
    """
    Recover a stable source ID.

    Priority:
    1. source_id
    2. id
    3. video filename stem
    4. audio filename stem
    """

    # --------------------------------------------------------
    # Explicit metadata
    # --------------------------------------------------------

    for key in ("source_id", "id"):
        value = row.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    # --------------------------------------------------------
    # Recover from video path
    #
    # Example:
    # data/mustard/raw/videos/2_381_u.mp4
    # -> 2_381_u
    # --------------------------------------------------------

    videos = row.get("videos")

    if isinstance(videos, list) and videos:
        path = str(videos[0]).strip()

        if path:
            return Path(path).stem

    # --------------------------------------------------------
    # Recover from audio path
    # --------------------------------------------------------

    audios = row.get("audios")

    if isinstance(audios, list) and audios:
        path = str(audios[0]).strip()

        if path:
            return Path(path).stem

    return None


def normalize_answer(text):
    """
    0 = Non-Sarcasm
    1 = Sarcasm
    None = invalid / unparseable
    """

    value = re.sub(
        r"[\s_-]+",
        "",
        str(text).strip().lower(),
    )

    if value in {"sarcasm", "sarcastic"}:
        return 1

    if value in {"nonsarcasm", "nonsarcastic"}:
        return 0

    return None


def parse_prediction(response):
    """
    Strict parsing:
    exactly one <answer>...</answer> block is required.
    """

    if not isinstance(response, str) or not response.strip():
        return None, "missing_response"

    matches = ANSWER_RE.findall(response)

    if len(matches) == 0:
        return None, "missing_answer_tag"

    if len(matches) > 1:
        return None, "multiple_answer_tags"

    pred = normalize_answer(matches[0])

    if pred is None:
        return None, "invalid_answer"

    return pred, None


def safe_div(a, b):
    return a / b if b else 0.0


def class_metrics(y_true, y_pred, label):
    """
    Invalid predictions count as false negatives for their
    corresponding gold class.

    This means malformed outputs are penalized rather than
    silently excluded from evaluation.
    """

    tp = sum(
        t == label and p == label
        for t, p in zip(y_true, y_pred)
    )

    fp = sum(
        t != label and p == label
        for t, p in zip(y_true, y_pred)
    )

    fn = sum(
        t == label and p != label
        for t, p in zip(y_true, y_pred)
    )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": sum(t == label for t in y_true),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold",
        required=True,
    )

    parser.add_argument(
        "--pred",
        required=True,
    )

    parser.add_argument(
        "--metrics",
        required=True,
    )

    parser.add_argument(
        "--scored",
        required=True,
    )

    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)

    print("=" * 60)
    print("Sarcasm evaluation")
    print("=" * 60)
    print("Gold examples:", len(gold_rows))
    print("Predictions:  ", len(pred_rows))
    print()

    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            "Prediction count does not match gold count: "
            f"{len(pred_rows)} != {len(gold_rows)}"
        )

    # ========================================================
    # Alignment
    # ========================================================

    gold_keys = [
        get_source_id(row)
        for row in gold_rows
    ]

    pred_keys = [
        get_source_id(row)
        for row in pred_rows
    ]

    can_join_by_id = (
        all(x is not None for x in gold_keys)
        and all(x is not None for x in pred_keys)
        and len(set(gold_keys)) == len(gold_keys)
        and len(set(pred_keys)) == len(pred_keys)
        and set(gold_keys) == set(pred_keys)
    )

    if can_join_by_id:
        print("Alignment: source_id / id")

        pred_map = {
            get_source_id(row): row
            for row in pred_rows
        }

        aligned = [
            (
                gold,
                pred_map[get_source_id(gold)],
            )
            for gold in gold_rows
        ]

    else:
        print(
            "WARNING: IDs could not be matched exactly; "
            "falling back to file order."
        )

        aligned = list(
            zip(gold_rows, pred_rows)
        )

    print()

    # ========================================================
    # Parse predictions
    # ========================================================

    y_true = []
    y_pred = []

    scored_rows = []

    parse_errors = {}

    for index, (gold_row, pred_row) in enumerate(aligned):

        gold = int(gold_row["label"])

        response = pred_row.get(
            "response",
            "",
        )

        pred, error = parse_prediction(
            response
        )

        y_true.append(gold)
        y_pred.append(pred)

        if error:
            parse_errors[error] = (
                parse_errors.get(error, 0) + 1
            )

        source_id = (
            get_source_id(gold_row)
            or str(index)
        )

        scored_rows.append({
            "source_id": source_id,

            "gold": gold,

            "gold_text": (
                "Sarcasm"
                if gold == 1
                else "Non-Sarcasm"
            ),

            "prediction": pred,

            "prediction_text": (
                "Sarcasm"
                if pred == 1
                else (
                    "Non-Sarcasm"
                    if pred == 0
                    else None
                )
            ),

            "correct": pred == gold,

            "parse_error": error,

            "response": response,
        })

    # ========================================================
    # Metrics
    # ========================================================

    total = len(y_true)

    parsed = sum(
        p is not None
        for p in y_pred
    )

    correct = sum(
        p == t
        for p, t in zip(y_pred, y_true)
    )

    accuracy = safe_div(
        correct,
        total,
    )

    non_sarcasm_metrics = class_metrics(
        y_true,
        y_pred,
        0,
    )

    sarcasm_metrics = class_metrics(
        y_true,
        y_pred,
        1,
    )

    macro_precision = (
        non_sarcasm_metrics["precision"]
        + sarcasm_metrics["precision"]
    ) / 2

    macro_recall = (
        non_sarcasm_metrics["recall"]
        + sarcasm_metrics["recall"]
    ) / 2

    macro_f1 = (
        non_sarcasm_metrics["f1"]
        + sarcasm_metrics["f1"]
    ) / 2

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn = sum(
        t == 0 and p == 0
        for t, p in zip(y_true, y_pred)
    )

    fp = sum(
        t == 0 and p == 1
        for t, p in zip(y_true, y_pred)
    )

    fn = sum(
        t == 1 and p == 0
        for t, p in zip(y_true, y_pred)
    )

    tp = sum(
        t == 1 and p == 1
        for t, p in zip(y_true, y_pred)
    )

    invalid_non_sarcasm = sum(
        t == 0 and p is None
        for t, p in zip(y_true, y_pred)
    )

    invalid_sarcasm = sum(
        t == 1 and p is None
        for t, p in zip(y_true, y_pred)
    )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics = {
        "total": total,
        "correct": correct,

        "accuracy": accuracy,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "parsed": parsed,
        "invalid": total - parsed,
        "parse_rate": safe_div(
            parsed,
            total,
        ),

        "non_sarcasm": non_sarcasm_metrics,
        "sarcasm": sarcasm_metrics,

        "confusion": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,

            "invalid_non_sarcasm":
                invalid_non_sarcasm,

            "invalid_sarcasm":
                invalid_sarcasm,
        },

        "parse_errors": parse_errors,
    }

    # ========================================================
    # Save per-example scores
    # ========================================================

    scored_path = Path(args.scored)

    scored_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with scored_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in scored_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # Save summary
    # ========================================================

    metrics_path = Path(args.metrics)

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Console output
    # ========================================================

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Total:       {total}")
    print(f"Parsed:      {parsed}")

    print(
        f"Parse rate:  "
        f"{safe_div(parsed, total):.4f}"
    )

    print()

    print(
        f"Accuracy:    "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f})"
    )

    print(
        f"Macro-P:     "
        f"{macro_precision:.4f} "
        f"({macro_precision * 100:.2f})"
    )

    print(
        f"Macro-R:     "
        f"{macro_recall:.4f} "
        f"({macro_recall * 100:.2f})"
    )

    print(
        f"Macro-F1:    "
        f"{macro_f1:.4f} "
        f"({macro_f1 * 100:.2f})"
    )

    print()

    print("Per-class metrics:")

    print(
        "  Non-Sarcasm:"
        f"  P={non_sarcasm_metrics['precision']:.4f}"
        f"  R={non_sarcasm_metrics['recall']:.4f}"
        f"  F1={non_sarcasm_metrics['f1']:.4f}"
        f"  N={non_sarcasm_metrics['support']}"
    )

    print(
        "  Sarcasm:    "
        f"  P={sarcasm_metrics['precision']:.4f}"
        f"  R={sarcasm_metrics['recall']:.4f}"
        f"  F1={sarcasm_metrics['f1']:.4f}"
        f"  N={sarcasm_metrics['support']}"
    )

    print()

    print("Confusion matrix:")
    print()
    print(
        "                 Pred NS   Pred S   Invalid"
    )

    print(
        f"Gold NS          "
        f"{tn:7d}   "
        f"{fp:6d}   "
        f"{invalid_non_sarcasm:7d}"
    )

    print(
        f"Gold S           "
        f"{fn:7d}   "
        f"{tp:6d}   "
        f"{invalid_sarcasm:7d}"
    )

    print()

    if parse_errors:

        print("Parse errors:")

        for key, value in sorted(
            parse_errors.items()
        ):
            print(
                f"  {key}: {value}"
            )

        print()

    print("Saved:")
    print("  Metrics:", metrics_path)
    print("  Scored: ", scored_path)


if __name__ == "__main__":
    main()
