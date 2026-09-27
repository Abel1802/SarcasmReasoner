#!/bin/bash

#SBATCH --job-name=mustard_vllm_10
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=180G
#SBATCH --time=03:00:00

#SBATCH --output=logs/mustard_vllm_final10_%j.out
#SBATCH --error=logs/mustard_vllm_final10_%j.err

#SBATCH --no-requeue

set -euo pipefail


# ============================================================
# Environment
# ============================================================

source activate ms2

cd /gpfs/work3/0/prjs1171/zhu_workspace/SarcasmReasoner


# ============================================================
# MUStARD++ checkpoints
# ============================================================

VARIANTS=(
    "sft_greedy"
    "sft_best_of_8"
    "sft_diverse_8"

    "grpo_base"
    "grpo_greedy"
    "grpo_best_of_8"
    "grpo_diverse_8"

    "grporm_greedy"
    "grporm_best_of_8"
    "grporm_diverse_8"
)


CHECKPOINTS=(
    "results/mustard/sft/greedy/v0-20260923-221644/checkpoint-105"
    "results/mustard/sft/best_of_8/v0-20260924-103030/checkpoint-144"
    "results/mustard/sft/diverse_8/v0-20260924-103353/checkpoint-834"

    "results/mustard/grpo/base/v2-20260924-114119/checkpoint-841"
    "results/mustard/grpo/greedy/v0-20260924-122118/checkpoint-841"
    "results/mustard/grpo/best_of_8/v0-20260924-165734/checkpoint-841"
    "results/mustard/grpo/diverse_8/v0-20260924-170016/checkpoint-841"

    "results/mustard/grpo_rm/greedy/v0-20260926-125827/checkpoint-841"
    "results/mustard/grpo_rm/best_of_8/v0-20260926-151027/checkpoint-841"
    "results/mustard/grpo_rm/diverse_8/v0-20260926-151025/checkpoint-841"
)


# ============================================================
# Sanity checks
# ============================================================

if [ ! -f scripts/evaluation/eval_checkpoint.sh ]; then
    echo "ERROR: evaluator not found:"
    echo "  scripts/evaluation/eval_checkpoint.sh"
    exit 1
fi


if [ "${#VARIANTS[@]}" -ne "${#CHECKPOINTS[@]}" ]; then
    echo "ERROR: VARIANTS and CHECKPOINTS length mismatch."
    exit 1
fi


echo
echo "============================================================"
echo "MUStARD++ vLLM 10-checkpoint TEST"
echo "============================================================"
echo "Host:        $(hostname)"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Split:       test"
echo "============================================================"
echo

nvidia-smi
echo


# ============================================================
# Sequential evaluation
# ============================================================

for i in "${!CHECKPOINTS[@]}"; do

    VARIANT="${VARIANTS[$i]}"
    CHECKPOINT="${CHECKPOINTS[$i]}"

    echo
    echo
    echo "################################################################"
    echo "# CHECKPOINT $((i + 1)) / ${#CHECKPOINTS[@]}"
    echo "################################################################"
    echo
    echo "Variant:"
    echo "  $VARIANT"
    echo
    echo "Checkpoint:"
    echo "  $CHECKPOINT"
    echo

    if [ ! -d "$CHECKPOINT" ]; then
        echo "ERROR: checkpoint directory missing:"
        echo "  $CHECKPOINT"
        exit 1
    fi

    if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
        echo "ERROR: adapter_config.json missing:"
        echo "  $CHECKPOINT/adapter_config.json"
        exit 1
    fi


    # --------------------------------------------------------
    # Formal vLLM test
    #
    # OVERWRITE=1 ensures this run regenerates predictions
    # using the current evaluation protocol.
    # --------------------------------------------------------

    OVERWRITE=1 \
    bash scripts/evaluation/eval_checkpoint.sh \
        mustard \
        "$VARIANT" \
        "$CHECKPOINT" \
        test


    echo
    echo "Finished:"
    echo "  $VARIANT"
    echo "  $CHECKPOINT"
    echo

done


echo
echo "============================================================"
echo "ALL 10 vLLM TESTS FINISHED"
echo "============================================================"
