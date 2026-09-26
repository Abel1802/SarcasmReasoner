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


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {e}"
                )

    return rows


# ============================================================
# GenRM parser
# ============================================================

def parse_genrm_json(value):
    """
    Parse:
      {"text":1,"audio":0,"visual":1,"integration":1}

    value may already be a dict or may be a JSON string.
    """

    if isinstance(value, dict):
        obj = value

    elif isinstance(value, str):
        value = value.strip()

        if not value:
            return None, "missing"

        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            return None, "invalid_json"

    else:
        return None, "invalid_type"

    if not isinstance(obj, dict):
        return None, "not_json_object"

    if set(obj.keys()) != set(COMPONENTS):
        return None, "wrong_fields"

    parsed = {}

    for component in COMPONENTS:
        v = obj[component]

        # Reject True/False even though bool subclasses int.
        if isinstance(v, bool):
            return None, f"invalid_{component}_value"

        if not isinstance(v, int) or v not in (0, 1):
            return None, f"invalid_{component}_value"

        parsed[component] = v

    return parsed, None


def pattern(labels):
    if labels is None:
        return None

    return "".join(
        str(labels[c])
        for c in COMPONENTS
    )


# ============================================================
# Metrics
# ============================================================

def safe_div(a, b):
    return a / b if b else 0.0


def class_metrics(y_true, y_pred, label):
    """
    Invalid predictions count as false negatives for the
    corresponding gold class.
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


def component_metrics(y_true, y_pred):
    total = len(y_true)

    correct = sum(
        t == p
        for t, p in zip(y_true, y_pred)
    )

    accuracy = safe_div(correct, total)

    unsupported = class_metrics(
        y_true,
        y_pred,
        0,
    )

    supported = class_metrics(
        y_true,
        y_pred,
        1,
    )

    macro_precision = (
        unsupported["precision"]
        + supported["precision"]
    ) / 2

    macro_recall = (
        unsupported["recall"]
        + supported["recall"]
    ) / 2

    macro_f1 = (
        unsupported["f1"]
        + supported["f1"]
    ) / 2

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

    invalid_gold_0 = sum(
        t == 0 and p is None
        for t, p in zip(y_true, y_pred)
    )

    invalid_gold_1 = sum(
        t == 1 and p is None
        for t, p in zip(y_true, y_pred)
    )

    return {
        "accuracy": accuracy,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "unsupported_recall": unsupported["recall"],
        "supported_recall": supported["recall"],

        "unsupported": unsupported,
        "supported": supported,

        "confusion": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "invalid_gold_0": invalid_gold_0,
            "invalid_gold_1": invalid_gold_1,
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate four-component GenRM predictions. "
            "Gold labels are read directly from each prediction "
            "row's `labels` field."
        )
    )

    parser.add_argument(
        "--pred",
        required=True,
        help="Prediction JSONL produced by ms-swift.",
    )

    parser.add_argument(
        "--metrics",
        required=True,
        help="Output metrics JSON.",
    )

    parser.add_argument(
        "--scored",
        required=True,
        help="Output per-example scored JSONL.",
    )

    args = parser.parse_args()

    rows = read_jsonl(args.pred)

    print("=" * 70)
    print("GenRM evaluation")
    print("=" * 70)
    print("Examples:", len(rows))
    print("Gold source: prediction row['labels']")
    print()

    gold_by_component = {
        c: []
        for c in COMPONENTS
    }

    pred_by_component = {
        c: []
        for c in COMPONENTS
    }

    gold_errors = Counter()
    pred_errors = Counter()

    gold_pattern_counts = Counter()
    pred_pattern_counts = Counter()

    parsed_count = 0
    exact_correct = 0

    scored_rows = []

    for index, row in enumerate(rows, 1):

        # ====================================================
        # Gold labels
        # ====================================================

        if "labels" not in row:
            raise ValueError(
                f"Missing `labels` field at prediction row {index}"
            )

        gold, gold_error = parse_genrm_json(
            row["labels"]
        )

        if gold_error is not None:
            gold_errors[gold_error] += 1

            raise ValueError(
                f"Invalid gold labels at row {index}: "
                f"{gold_error}: {row['labels']!r}"
            )

        # ====================================================
        # Prediction
        # ====================================================

        response = row.get("response")

        pred, pred_error = parse_genrm_json(
            response
        )

        if pred_error is None:
            parsed_count += 1
        else:
            pred_errors[pred_error] += 1

        # ====================================================
        # Pattern
        # ====================================================

        gold_pat = pattern(gold)
        pred_pat = pattern(pred)

        gold_pattern_counts[gold_pat] += 1

        if pred_pat is not None:
            pred_pattern_counts[pred_pat] += 1

        exact_match = (
            pred is not None
            and all(
                pred[c] == gold[c]
                for c in COMPONENTS
            )
        )

        if exact_match:
            exact_correct += 1

        # ====================================================
        # Component values
        # ====================================================

        component_correct = {}

        for component in COMPONENTS:
            gold_value = gold[component]

            pred_value = (
                pred[component]
                if pred is not None
                else None
            )

            gold_by_component[component].append(
                gold_value
            )

            pred_by_component[component].append(
                pred_value
            )

            component_correct[component] = (
                pred_value == gold_value
            )

        # Recover metadata when present.
        source_id = row.get("source_id")
        sample_idx = row.get("sample_idx")
        judge_id = row.get("judge_id")

        scored_rows.append({
            "index": index,

            "judge_id": judge_id,
            "source_id": source_id,
            "sample_idx": sample_idx,

            "gold": gold,
            "prediction": pred,

            "gold_pattern": gold_pat,
            "prediction_pattern": pred_pat,

            "exact_match": exact_match,

            "component_correct":
                component_correct,

            "parse_error":
                pred_error,

            "response":
                response,
        })

    # ========================================================
    # Metrics
    # ========================================================

    metrics_by_component = {}

    for component in COMPONENTS:
        metrics_by_component[component] = (
            component_metrics(
                gold_by_component[component],
                pred_by_component[component],
            )
        )

    total = len(rows)

    parse_rate = safe_div(
        parsed_count,
        total,
    )

    exact_pattern_accuracy = safe_div(
        exact_correct,
        total,
    )

    average_component_accuracy = (
        sum(
            metrics_by_component[c]["accuracy"]
            for c in COMPONENTS
        )
        / len(COMPONENTS)
    )

    average_component_macro_f1 = (
        sum(
            metrics_by_component[c]["macro_f1"]
            for c in COMPONENTS
        )
        / len(COMPONENTS)
    )

    # Integration is extremely imbalanced.
    # Use T/A/V Macro-F1 as the main grounding summary.
    tav_macro_f1 = (
        metrics_by_component["text"]["macro_f1"]
        + metrics_by_component["audio"]["macro_f1"]
        + metrics_by_component["visual"]["macro_f1"]
    ) / 3

    metrics = {
        "total": total,

        "gold_source":
            "prediction.labels",

        "parsed": parsed_count,

        "invalid":
            total - parsed_count,

        "parse_rate":
            parse_rate,

        "parse_errors":
            dict(pred_errors),

        "exact_pattern_correct":
            exact_correct,

        "exact_pattern_accuracy":
            exact_pattern_accuracy,

        "average_component_accuracy":
            average_component_accuracy,

        "average_component_macro_f1":
            average_component_macro_f1,

        "text_audio_visual_macro_f1":
            tav_macro_f1,

        "components":
            metrics_by_component,

        "gold_pattern_counts":
            dict(gold_pattern_counts),

        "prediction_pattern_counts":
            dict(pred_pattern_counts),
    }

    # ========================================================
    # Save scored
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
    # Save metrics
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

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Total:                  "
        f"{total}"
    )

    print(
        f"Parsed:                 "
        f"{parsed_count}"
    )

    print(
        f"Parse rate:             "
        f"{parse_rate:.4f} "
        f"({parse_rate * 100:.2f}%)"
    )

    print(
        f"Exact-pattern accuracy: "
        f"{exact_pattern_accuracy:.4f} "
        f"({exact_pattern_accuracy * 100:.2f}%)"
    )

    print(
        f"Avg component accuracy: "
        f"{average_component_accuracy:.4f} "
        f"({average_component_accuracy * 100:.2f}%)"
    )

    print(
        f"Avg component Macro-F1: "
        f"{average_component_macro_f1:.4f} "
        f"({average_component_macro_f1 * 100:.2f}%)"
    )

    print(
        f"T/A/V Macro-F1:         "
        f"{tav_macro_f1:.4f} "
        f"({tav_macro_f1 * 100:.2f}%)"
    )

    print()

    print(
        "Component        Acc      Macro-F1  "
        "R(unsupported)  R(supported)"
    )

    print("-" * 70)

    for component in COMPONENTS:
        m = metrics_by_component[component]

        print(
            f"{component:<14}"
            f"{m['accuracy']:>8.4f}"
            f"{m['macro_f1']:>12.4f}"
            f"{m['unsupported_recall']:>16.4f}"
            f"{m['supported_recall']:>14.4f}"
        )

    print()

    if pred_errors:
        print("Prediction parse errors:")

        for key, value in sorted(
            pred_errors.items()
        ):
            print(
                f"  {key}: {value}"
            )

        print()

    print("Gold pattern counts:")

    for key, value in sorted(
        gold_pattern_counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"  {key}: {value}"
        )

    print()

    print("Prediction pattern counts:")

    for key, value in sorted(
        pred_pattern_counts.items(),
        key=lambda x: (-x[1], x[0]),
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
