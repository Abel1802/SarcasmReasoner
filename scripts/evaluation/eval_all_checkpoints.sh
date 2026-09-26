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
# <method> may be any non-empty string.
#
#
# Examples:
#
# Ordinary GRPO:
#
# bash scripts/evaluation/eval_all_checkpoints.sh \
#     mustard \
#     grpo_greedy \
#     results/mustard/grpo/greedy/v0-20260924-122118
#
#
# GRPO + GenRM:
#
# bash scripts/evaluation/eval_all_checkpoints.sh \
#     mustard \
#     grporm_greedy \
#     results/mustard/grpo_rm/greedy/v0-20260926-125827
#
#
# Output example:
#
# results/mustard/evaluation/
#   grpo_rm/
#     greedy/
#       greedy_ckpt200/
#         valid_predictions.jsonl
#         valid_metrics.json
#         valid_scored.jsonl
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
# Partial results cause an error by default.
#
# To delete partial results and rerun:
#
#   OVERWRITE=1 bash scripts/evaluation/eval_all_checkpoints.sh ...
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
# Configuration summary
# ============================================================

echo
echo "============================================================"
echo "Validation evaluation for all checkpoints"
echo "============================================================"
echo "Dataset:        $DATASET"
echo "Variant:        $VARIANT"
echo "Family:         $FAMILY"
echo "Method:         $METHOD"
echo "Split:          VALID ONLY"
echo
echo "Run directory:"
echo "  $RUN_DIR"
echo
echo "Output root:"
echo "  results/${DATASET}/evaluation/${FAMILY}/${METHOD}"
echo
echo "Checkpoints:"
printf "  %s\n" "${CHECKPOINTS[@]}"
echo "============================================================"
echo


# ============================================================
# Counters
# ============================================================

TOTAL="${#CHECKPOINTS[@]}"

EVALUATED=0

SKIPPED=0


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


    OUTPUT_DIR="results/${DATASET}/evaluation/${FAMILY}/${METHOD}/${METHOD}_ckpt${STEP}"

    PRED_PATH="${OUTPUT_DIR}/valid_predictions.jsonl"

    METRICS_PATH="${OUTPUT_DIR}/valid_metrics.json"

    SCORED_PATH="${OUTPUT_DIR}/valid_scored.jsonl"


    echo
    echo "############################################################"
    echo "# checkpoint-${STEP}"
    echo "############################################################"
    echo


    # ========================================================
    # Existing complete validation result
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
    # Detect partial result
    # ========================================================

    if \
        [ -f "$PRED_PATH" ] || \
        [ -f "$METRICS_PATH" ] || \
        [ -f "$SCORED_PATH" ]; then

        if [ "${OVERWRITE:-0}" = "1" ]; then

            echo "Partial validation result detected."
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
    # Validation evaluation
    # ========================================================

    echo "Running validation evaluation:"
    echo
    echo "Adapter:"
    echo "  $CKPT"
    echo
    echo "Output:"
    echo "  $OUTPUT_DIR"
    echo


    bash "$EVAL_SCRIPT" \
        "$DATASET" \
        "$VARIANT" \
        "$CKPT" \
        valid


    EVALUATED=$((EVALUATED + 1))

done


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "Validation evaluation finished"
echo "============================================================"
echo "Dataset:            $DATASET"
echo "Variant:            $VARIANT"
echo "Split:              VALID ONLY"
echo
echo "Total checkpoints:  $TOTAL"
echo "Evaluated now:      $EVALUATED"
echo "Skipped existing:   $SKIPPED"
echo
echo "Results:"
echo "  results/${DATASET}/evaluation/${FAMILY}/${METHOD}"
echo "============================================================"
