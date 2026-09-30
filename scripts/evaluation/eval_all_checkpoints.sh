#!/bin/bash

set -e
set -o pipefail


# ============================================================
# Evaluate ALL checkpoints on VALIDATION set only.
#
# IMPORTANT:
#
# This script NEVER evaluates the test set.
#
# Use scripts/evaluation/eval_checkpoint.sh separately
# for final test evaluation of a selected checkpoint.
#
#
# Variant format:
#
#   sft_<method>
#   grpo_<method>
#   grporm_<method>
#
#
# New validation output structure:
#
# results/<dataset>/evaluation/<family>/<method>/
#   valid/
#     <method>_ckpt200/
#       valid_predictions.jsonl
#       valid_metrics.json
#       valid_scored.jsonl
#
#     <method>_ckpt400/
#       ...
#
#     checkpoint_metrics.csv
#
#     images/
#       accuracy_over_steps.png
#       macro_f1_over_steps.png
#       sarcasm_f1_over_steps.png
#       non_sarcasm_f1_over_steps.png
#       all_metrics_over_steps.png
#
#
# Existing COMPLETE validation results are skipped.
#
# A validation result is complete when all three exist:
#
#   valid_predictions.jsonl
#   valid_metrics.json
#   valid_scored.jsonl
#
#
# Partial results cause an error by default.
#
# To delete partial validation files and rerun:
#
#   OVERWRITE=1 bash scripts/evaluation/eval_all_checkpoints.sh ...
#
#
# NOTE:
#
# eval_checkpoint.sh currently writes validation results to:
#
# results/<dataset>/evaluation/<family>/<method>/
#     <method>_ckpt<step>/
#
# This script automatically moves the three VALIDATION files
# into the new valid/ directory after each evaluation.
#
# Test results are never moved.
# ============================================================


# ============================================================
# Arguments
# ============================================================

if [ "$#" -ne 3 ]; then

    echo "Usage:"
    echo
    echo "bash $0 <dataset> <variant> <run_dir>"
    echo
    echo "Variant format:"
    echo "  sft_<method>"
    echo "  grpo_<method>"
    echo "  grporm_<method>"
    echo

    exit 1
fi


DATASET="$1"

VARIANT="$2"

RUN_DIR="$3"

SPLIT="valid"


# ============================================================
# Parse variant
# ============================================================

case "$VARIANT" in

    sft_*)
        FAMILY="sft"
        METHOD="${VARIANT#sft_}"
        ;;

    grpo_*)
        FAMILY="grpo"
        METHOD="${VARIANT#grpo_}"
        ;;

    grporm_*)
        FAMILY="grpo_rm"
        METHOD="${VARIANT#grporm_}"
        ;;

    *)
        echo "ERROR: invalid variant:"
        echo "  $VARIANT"
        echo
        echo "Expected format:"
        echo "  sft_<method>"
        echo "  grpo_<method>"
        echo "  grporm_<method>"
        echo

        exit 1
        ;;

esac


if [ -z "$METHOD" ]; then

    echo "ERROR: method cannot be empty:"
    echo "  $VARIANT"

    exit 1
fi


# ============================================================
# Validate run directory
# ============================================================

if [ ! -d "$RUN_DIR" ]; then

    echo "ERROR: run directory not found:"
    echo "  $RUN_DIR"

    exit 1
fi


# ============================================================
# Validate single-checkpoint evaluator
# ============================================================

EVAL_SCRIPT="scripts/evaluation/eval_checkpoint.sh"


if [ ! -f "$EVAL_SCRIPT" ]; then

    echo "ERROR: evaluation script not found:"
    echo "  $EVAL_SCRIPT"

    exit 1
fi


bash -n "$EVAL_SCRIPT"


# ============================================================
# Output directories
# ============================================================

METHOD_ROOT="results/${DATASET}/evaluation/${FAMILY}/${METHOD}"

VALID_ROOT="${METHOD_ROOT}/valid"

IMAGES_DIR="${VALID_ROOT}/images"

SUMMARY_CSV="${VALID_ROOT}/checkpoint_metrics.csv"


mkdir -p "$VALID_ROOT"

mkdir -p "$IMAGES_DIR"


# ============================================================
# Discover checkpoints
# ============================================================

mapfile -t CHECKPOINTS < <(
    find "$RUN_DIR" \
        -maxdepth 1 \
        -type d \
        -name "checkpoint-*" \
        | sort -V
)


if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then

    echo "ERROR: no checkpoint-* directories found:"
    echo "  $RUN_DIR"

    exit 1
fi


# ============================================================
# Helper:
# move VALIDATION files from old eval_checkpoint location
# to the new valid/ location.
#
# IMPORTANT:
# Only validation files are moved.
# Any test files in the legacy directory are untouched.
# ============================================================

move_validation_files() {

    local LEGACY_DIR="$1"

    local NEW_DIR="$2"


    mkdir -p "$NEW_DIR"


    for NAME in \
        valid_predictions.jsonl \
        valid_metrics.json \
        valid_scored.jsonl
    do

        if [ -f "${LEGACY_DIR}/${NAME}" ]; then

            mv \
                "${LEGACY_DIR}/${NAME}" \
                "${NEW_DIR}/${NAME}"

        fi

    done


    # Remove legacy directory only if completely empty.
    if [ -d "$LEGACY_DIR" ]; then

        rmdir "$LEGACY_DIR" 2>/dev/null || true

    fi
}


# ============================================================
# Configuration summary
# ============================================================

echo
echo "============================================================"
echo "Validation evaluation for all checkpoints"
echo "============================================================"
echo

echo "Dataset:        $DATASET"
echo "Variant:        $VARIANT"
echo "Family:         $FAMILY"
echo "Method:         $METHOD"
echo "Split:          VALID ONLY"
echo

echo "Run directory:"
echo "  $RUN_DIR"
echo

echo "Validation root:"
echo "  $VALID_ROOT"
echo

echo "Images:"
echo "  $IMAGES_DIR"
echo

echo "Checkpoints:"
printf "  %s\n" "${CHECKPOINTS[@]}"

echo
echo "============================================================"
echo


# ============================================================
# Counters
# ============================================================

TOTAL="${#CHECKPOINTS[@]}"

EVALUATED=0

SKIPPED=0

MIGRATED=0


# ============================================================
# Evaluate checkpoints
# ============================================================

for CKPT in "${CHECKPOINTS[@]}"; do

    CKPT_NAME=$(basename "$CKPT")


    if [[ "$CKPT_NAME" =~ ^checkpoint-([0-9]+)$ ]]; then

        STEP="${BASH_REMATCH[1]}"

    else

        echo "ERROR: invalid checkpoint directory:"
        echo "  $CKPT"

        exit 1
    fi


    # --------------------------------------------------------
    # New output location
    # --------------------------------------------------------

    OUTPUT_DIR="${VALID_ROOT}/${METHOD}_ckpt${STEP}"

    PRED_PATH="${OUTPUT_DIR}/valid_predictions.jsonl"

    METRICS_PATH="${OUTPUT_DIR}/valid_metrics.json"

    SCORED_PATH="${OUTPUT_DIR}/valid_scored.jsonl"


    # --------------------------------------------------------
    # Legacy eval_checkpoint.sh location
    # --------------------------------------------------------

    LEGACY_OUTPUT_DIR="${METHOD_ROOT}/${METHOD}_ckpt${STEP}"

    LEGACY_PRED="${LEGACY_OUTPUT_DIR}/valid_predictions.jsonl"

    LEGACY_METRICS="${LEGACY_OUTPUT_DIR}/valid_metrics.json"

    LEGACY_SCORED="${LEGACY_OUTPUT_DIR}/valid_scored.jsonl"


    echo
    echo "############################################################"
    echo "# checkpoint-${STEP}"
    echo "############################################################"
    echo


    # ========================================================
    # Existing complete result in NEW location
    # ========================================================

    if \
        [ -f "$PRED_PATH" ] && \
        [ -f "$METRICS_PATH" ] && \
        [ -f "$SCORED_PATH" ]; then

        echo "SKIP: validation result already complete."
        echo
        echo "Output:"
        echo "  $OUTPUT_DIR"
        echo

        SKIPPED=$((SKIPPED + 1))

        continue
    fi


    # ========================================================
    # Existing complete result in OLD location
    #
    # Automatically migrate it.
    # ========================================================

    if \
        [ -f "$LEGACY_PRED" ] && \
        [ -f "$LEGACY_METRICS" ] && \
        [ -f "$LEGACY_SCORED" ]; then

        echo "Existing complete validation result found"
        echo "in legacy location."
        echo
        echo "Moving validation files to:"
        echo "  $OUTPUT_DIR"
        echo


        move_validation_files \
            "$LEGACY_OUTPUT_DIR" \
            "$OUTPUT_DIR"


        MIGRATED=$((MIGRATED + 1))

        continue
    fi


    # ========================================================
    # Detect partial result in NEW location
    # ========================================================

    if \
        [ -f "$PRED_PATH" ] || \
        [ -f "$METRICS_PATH" ] || \
        [ -f "$SCORED_PATH" ]; then

        if [ "${OVERWRITE:-0}" = "1" ]; then

            echo "Partial validation result detected"
            echo "in new location."
            echo
            echo "OVERWRITE=1 -> removing incomplete files."
            echo

            rm -f \
                "$PRED_PATH" \
                "$METRICS_PATH" \
                "$SCORED_PATH"

        else

            echo "ERROR: partial validation result exists."
            echo
            echo "Checkpoint:"
            echo "  checkpoint-${STEP}"
            echo
            echo "Output:"
            echo "  $OUTPUT_DIR"
            echo
            echo "Existing files:"
            echo "  predictions: $([ -f "$PRED_PATH" ] && echo YES || echo NO)"
            echo "  metrics:     $([ -f "$METRICS_PATH" ] && echo YES || echo NO)"
            echo "  scored:      $([ -f "$SCORED_PATH" ] && echo YES || echo NO)"
            echo
            echo "To remove partial output and rerun:"
            echo
            echo "OVERWRITE=1 bash $0 \\"
            echo "  $DATASET \\"
            echo "  $VARIANT \\"
            echo "  $RUN_DIR"
            echo

            exit 1
        fi

    fi


    # ========================================================
    # Detect partial result in LEGACY location
    # ========================================================

    if \
        [ -f "$LEGACY_PRED" ] || \
        [ -f "$LEGACY_METRICS" ] || \
        [ -f "$LEGACY_SCORED" ]; then

        if [ "${OVERWRITE:-0}" = "1" ]; then

            echo "Partial validation result detected"
            echo "in legacy location."
            echo
            echo "OVERWRITE=1 -> removing incomplete validation files."
            echo

            rm -f \
                "$LEGACY_PRED" \
                "$LEGACY_METRICS" \
                "$LEGACY_SCORED"

        else

            echo "ERROR: partial legacy validation result exists."
            echo
            echo "Checkpoint:"
            echo "  checkpoint-${STEP}"
            echo
            echo "Legacy output:"
            echo "  $LEGACY_OUTPUT_DIR"
            echo
            echo "Existing files:"
            echo "  predictions: $([ -f "$LEGACY_PRED" ] && echo YES || echo NO)"
            echo "  metrics:     $([ -f "$LEGACY_METRICS" ] && echo YES || echo NO)"
            echo "  scored:      $([ -f "$LEGACY_SCORED" ] && echo YES || echo NO)"
            echo
            echo "Use:"
            echo
            echo "OVERWRITE=1 bash $0 \\"
            echo "  $DATASET \\"
            echo "  $VARIANT \\"
            echo "  $RUN_DIR"
            echo

            exit 1
        fi

    fi


    # ========================================================
    # Validation evaluation
    # ========================================================

    echo "Running validation evaluation:"
    echo

    echo "Adapter:"
    echo "  $CKPT"
    echo

    echo "Final validation output:"
    echo "  $OUTPUT_DIR"
    echo


    bash "$EVAL_SCRIPT" \
        "$DATASET" \
        "$VARIANT" \
        "$CKPT" \
        valid


    # ========================================================
    # eval_checkpoint.sh currently writes to legacy location.
    #
    # Move ONLY validation files into valid/.
    # ========================================================

    if \
        [ -f "$LEGACY_PRED" ] && \
        [ -f "$LEGACY_METRICS" ] && \
        [ -f "$LEGACY_SCORED" ]; then

        move_validation_files \
            "$LEGACY_OUTPUT_DIR" \
            "$OUTPUT_DIR"

    fi


    # ========================================================
    # Final result validation
    # ========================================================

    if \
        [ ! -f "$PRED_PATH" ] || \
        [ ! -f "$METRICS_PATH" ] || \
        [ ! -f "$SCORED_PATH" ]; then

        echo
        echo "ERROR:"
        echo "Evaluation finished but complete validation"
        echo "output was not found."
        echo

        echo "Expected:"
        echo "  $PRED_PATH"
        echo "  $METRICS_PATH"
        echo "  $SCORED_PATH"
        echo

        exit 1
    fi


    EVALUATED=$((EVALUATED + 1))

done


# ============================================================
# Build checkpoint summary + plots
# ============================================================

echo
echo "============================================================"
echo "Building validation curves"
echo "============================================================"
echo


python - \
    "$VALID_ROOT" \
    "$METHOD" \
    "$SUMMARY_CSV" \
    "$IMAGES_DIR" <<'PY'

import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Arguments
# ============================================================

valid_root = Path(sys.argv[1])

method = sys.argv[2]

summary_csv = Path(sys.argv[3])

images_dir = Path(sys.argv[4])


images_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def as_float(value):

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def first_value(data, keys):

    for key in keys:

        if key in data:

            value = as_float(
                data[key]
            )

            if value is not None:
                return value

    return None


def nested_f1(data, keys):

    for key in keys:

        block = data.get(key)

        if isinstance(block, dict):

            value = first_value(
                block,
                [
                    "f1",
                    "f1_score",
                    "f1-score",
                ],
            )

            if value is not None:
                return value

    return None


# ============================================================
# Read checkpoint metrics
# ============================================================

rows = []


pattern = re.compile(
    rf"^{re.escape(method)}_ckpt(\d+)$"
)


for checkpoint_dir in valid_root.iterdir():

    if not checkpoint_dir.is_dir():
        continue

    match = pattern.match(
        checkpoint_dir.name
    )

    if not match:
        continue

    step = int(
        match.group(1)
    )

    metrics_path = (
        checkpoint_dir
        / "valid_metrics.json"
    )

    if not metrics_path.is_file():
        continue

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        metrics = json.load(f)


    # --------------------------------------------------------
    # Top-level classification metrics
    # --------------------------------------------------------

    accuracy = first_value(
        metrics,
        [
            "accuracy",
            "acc",
        ],
    )


    macro_f1 = first_value(
        metrics,
        [
            "macro_f1",
            "macro-f1",
            "macro_f1_score",
        ],
    )


    # --------------------------------------------------------
    # Per-class metrics
    #
    # Current evaluator uses nested structures such as:
    #
    #   "sarcasm": {
    #       "precision": ...,
    #       "recall": ...,
    #       "f1": ...
    #   }
    #
    #   "non_sarcasm": {
    #       ...
    #   }
    #
    # A few alternative names are also accepted.
    # --------------------------------------------------------

    sarcasm_f1 = nested_f1(
        metrics,
        [
            "sarcasm",
            "sarc",
        ],
    )


    if sarcasm_f1 is None:

        sarcasm_f1 = first_value(
            metrics,
            [
                "sarcasm_f1",
                "sarc_f1",
            ],
        )


    non_sarcasm_f1 = nested_f1(
        metrics,
        [
            "non_sarcasm",
            "non-sarcasm",
            "nonsarcasm",
            "non_sarc",
        ],
    )


    if non_sarcasm_f1 is None:

        non_sarcasm_f1 = first_value(
            metrics,
            [
                "non_sarcasm_f1",
                "non_sarc_f1",
            ],
        )


    rows.append(
        {
            "step": step,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "sarcasm_f1": sarcasm_f1,
            "non_sarcasm_f1": non_sarcasm_f1,
            "metrics_path": str(metrics_path),
        }
    )


rows.sort(
    key=lambda x: x["step"]
)


if not rows:

    raise RuntimeError(
        f"No validation metric files found under {valid_root}"
    )


# ============================================================
# CSV summary
# ============================================================

summary_csv.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with summary_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "step",
            "accuracy",
            "macro_f1",
            "sarcasm_f1",
            "non_sarcasm_f1",
            "metrics_path",
        ],
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print(
    "Saved summary:",
    summary_csv,
)


# ============================================================
# Plot helper
# ============================================================

def plot_metric(
    *,
    key,
    ylabel,
    title,
    filename,
):

    xs = []
    ys = []


    for row in rows:

        value = row.get(key)

        if value is None:
            continue

        xs.append(
            row["step"]
        )

        ys.append(
            value
        )


    if not xs:

        print(
            f"SKIP plot {key}: "
            "metric unavailable."
        )

        return


    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        xs,
        ys,
        marker="o",
    )

    plt.xlabel(
        "Checkpoint step"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.xticks(
        xs,
        rotation=45,
    )

    plt.tight_layout()


    output_path = (
        images_dir
        / filename
    )


    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


    print(
        "Saved:",
        output_path,
    )


# ============================================================
# Individual figures
# ============================================================

plot_metric(
    key="accuracy",
    ylabel="Accuracy",
    title="Validation Accuracy vs. Checkpoint Step",
    filename="accuracy_over_steps.png",
)


plot_metric(
    key="macro_f1",
    ylabel="Macro-F1",
    title="Validation Macro-F1 vs. Checkpoint Step",
    filename="macro_f1_over_steps.png",
)


plot_metric(
    key="sarcasm_f1",
    ylabel="Sarcasm F1",
    title="Validation Sarcasm F1 vs. Checkpoint Step",
    filename="sarcasm_f1_over_steps.png",
)


plot_metric(
    key="non_sarcasm_f1",
    ylabel="Non-Sarcasm F1",
    title="Validation Non-Sarcasm F1 vs. Checkpoint Step",
    filename="non_sarcasm_f1_over_steps.png",
)


# ============================================================
# Combined figure
# ============================================================

metric_specs = [
    (
        "accuracy",
        "Accuracy",
    ),
    (
        "macro_f1",
        "Macro-F1",
    ),
    (
        "sarcasm_f1",
        "Sarcasm F1",
    ),
    (
        "non_sarcasm_f1",
        "Non-Sarcasm F1",
    ),
]


plt.figure(
    figsize=(9, 5.5)
)


num_plotted = 0


for key, label in metric_specs:

    xs = []

    ys = []


    for row in rows:

        value = row.get(key)

        if value is None:
            continue

        xs.append(
            row["step"]
        )

        ys.append(
            value
        )


    if not xs:
        continue


    plt.plot(
        xs,
        ys,
        marker="o",
        label=label,
    )


    num_plotted += 1


if num_plotted > 0:

    all_steps = [
        row["step"]
        for row in rows
    ]


    plt.xlabel(
        "Checkpoint step"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Validation Metrics vs. Checkpoint Step"
    )

    plt.xticks(
        all_steps,
        rotation=45,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()


    combined_path = (
        images_dir
        / "all_metrics_over_steps.png"
    )


    plt.savefig(
        combined_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


    print(
        "Saved:",
        combined_path,
    )


# ============================================================
# Console table
# ============================================================

print()
print("=" * 88)

print(
    f"{'step':>8} "
    f"{'accuracy':>12} "
    f"{'macro_f1':>12} "
    f"{'sarc_f1':>12} "
    f"{'non_sarc_f1':>14}"
)

print("-" * 88)


def fmt(value):

    if value is None:
        return "NA"

    return f"{value:.4f}"


for row in rows:

    print(
        f"{row['step']:>8} "
        f"{fmt(row['accuracy']):>12} "
        f"{fmt(row['macro_f1']):>12} "
        f"{fmt(row['sarcasm_f1']):>12} "
        f"{fmt(row['non_sarcasm_f1']):>14}"
    )


print("=" * 88)
print()

PY


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "Validation evaluation finished"
echo "============================================================"
echo

echo "Dataset:            $DATASET"
echo "Variant:            $VARIANT"
echo "Split:              VALID ONLY"
echo

echo "Total checkpoints:  $TOTAL"
echo "Evaluated now:      $EVALUATED"
echo "Migrated old:       $MIGRATED"
echo "Skipped existing:   $SKIPPED"
echo

echo "Validation results:"
echo "  $VALID_ROOT"
echo

echo "Metrics summary:"
echo "  $SUMMARY_CSV"
echo

echo "Figures:"
echo "  $IMAGES_DIR"
echo

echo "============================================================"

