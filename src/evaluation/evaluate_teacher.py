import argparse
import json
import re
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ANSWER_RE = re.compile(
    r"<answer>\s*(Sarcasm|Non-Sarcasm)\s*</answer>",
    flags=re.IGNORECASE,
)


def normalize_prediction(text):
    m = ANSWER_RE.search(text or "")
    if not m:
        return None

    answer = m.group(1).lower()

    if answer == "sarcasm":
        return 1
    if answer == "non-sarcasm":
        return 0

    return None


def is_complete(response):
    required_tags = [
        "</text_evidence>",
        "</audio_evidence>",
        "</visual_evidence>",
        "</integration>",
        "</answer>",
    ]
    return all(tag in response for tag in required_tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    rows = []

    with open(args.path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    y_true = []
    y_pred = []

    parse_errors = []
    incomplete = []

    prediction_counts = Counter()

    for x in rows:
        response = x.get("response", "")
        pred = normalize_prediction(response)

        if not is_complete(response):
            incomplete.append(x["id"])

        if pred is None:
            parse_errors.append(x["id"])
            continue

        gold = int(x["label"])

        y_true.append(gold)
        y_pred.append(pred)

        prediction_counts[pred] += 1

    print("=" * 60)
    print("Teacher evaluation")
    print("=" * 60)

    print(f"Total samples:     {len(rows)}")
    print(f"Parsed:            {len(y_pred)}")
    print(f"Parse errors:      {len(parse_errors)}")
    print(f"Incomplete:        {len(incomplete)}")
    print()

    if parse_errors:
        print("Parse-error IDs:")
        print(parse_errors)
        print()

    if incomplete:
        print("Incomplete IDs:")
        print(incomplete)
        print()

    if not y_pred:
        print("No parseable predictions.")
        return

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print()

    print("Prediction counts:")
    print(
        f"Non-Sarcasm: {prediction_counts[0]}"
    )
    print(
        f"Sarcasm:     {prediction_counts[1]}"
    )
    print()

    print("Confusion matrix")
    print("rows = gold, columns = prediction")
    print("labels = [Non-Sarcasm, Sarcasm]")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
    )
    print()

    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-Sarcasm",
                "Sarcasm",
            ],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
