#!/bin/bash

#SBATCH --job-name=tf_repro_rm_g800
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tf_repro_rm_g800_%j.out
#SBATCH --error=logs/tf_repro_rm_g800_%j.err
#SBATCH --no-requeue

set -euo pipefail

# ============================================================
# Environment
# ============================================================

source activate ms2

cd /gpfs/work3/0/prjs1171/zhu_workspace/SarcasmReasoner

MODEL="Qwen/Qwen2.5-Omni-7B"

ADAPTER="results/mustard/grpo_rm/greedy/v0-20260926-125827/checkpoint-800"

GOLD="data/mustard/processed/zero_shot_test.jsonl"

SAFE_INFER="src/evaluation/swift_infer_safe_video.py"

EVALUATOR="src/evaluation/evaluate_sarcasm_predictions.py"

BASE_OUT="results/mustard/reproducibility/transformers/grpo_rm_greedy_ckpt800"

mkdir -p "$BASE_OUT"


# ============================================================
# Deterministic environment
# ============================================================

export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=False

export FPS_MAX_FRAMES=12
export VIDEO_MAX_PIXELS=50176
export MAX_PIXELS=1003520

export FORCE_QWENVL_VIDEO_READER=decord
export TOKENIZERS_PARALLELISM=false

export PYTHONHASHSEED=42

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8


# ============================================================
# Environment summary
# ============================================================

echo "============================================================"
echo "Transformers reproducibility test"
echo "============================================================"
echo "Host:        $(hostname)"
echo "Model:       $MODEL"
echo "Adapter:     $ADAPTER"
echo "Gold:        $GOLD"
echo "Backend:     transformers"
echo "Temperature: 0"
echo "Seed:        42"
echo "Batch size:  1"
echo "============================================================"
echo

nvidia-smi

echo

python - <<'PY'
import torch
import swift
import transformers

print("torch:", torch.__version__)
print("swift:", swift.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo


# ============================================================
# Run function
# ============================================================

run_eval () {

    RUN="$1"

    OUT_DIR="${BASE_OUT}/run${RUN}"

    PRED="${OUT_DIR}/test_predictions.jsonl"
    METRICS="${OUT_DIR}/test_metrics.json"
    SCORED="${OUT_DIR}/test_scored.jsonl"

    mkdir -p "$OUT_DIR"

    # Never append to an old prediction file.
    rm -f \
        "$PRED" \
        "$METRICS" \
        "$SCORED"

    echo
    echo "============================================================"
    echo "TRANSFORMERS RUN ${RUN}"
    echo "============================================================"
    echo "Output:"
    echo "  $OUT_DIR"
    echo "============================================================"
    echo

    python "$SAFE_INFER" \
        --model "$MODEL" \
        --adapters "$ADAPTER" \
        --val_dataset "$GOLD" \
        --infer_backend transformers \
        --stream false \
        --torch_dtype bfloat16 \
        --temperature 0 \
        --max_new_tokens 4096 \
        --max_batch_size 1 \
        --load_from_cache_file false \
        --dataset_num_proc 1 \
        --seed 42 \
        --data_seed 42 \
        --result_path "$PRED"

    echo
    echo "Inference finished."
    echo

    GOLD_COUNT=$(wc -l < "$GOLD")
    PRED_COUNT=$(wc -l < "$PRED")

    echo "Gold count:       $GOLD_COUNT"
    echo "Prediction count: $PRED_COUNT"

    if [ "$GOLD_COUNT" -ne "$PRED_COUNT" ]; then
        echo "ERROR: prediction count mismatch."
        exit 1
    fi

    python "$EVALUATOR" \
        --gold "$GOLD" \
        --pred "$PRED" \
        --metrics "$METRICS" \
        --scored "$SCORED"

    echo
    echo "RUN ${RUN} SHA256:"
    sha256sum "$PRED"

    echo
    echo "============================================================"
    echo "RUN ${RUN} COMPLETE"
    echo "============================================================"
    echo
}


# ============================================================
# Same GPU / same node / same process environment:
# run twice sequentially
# ============================================================

run_eval 1
run_eval 2


# ============================================================
# Final reproducibility comparison
# ============================================================

RUN1="${BASE_OUT}/run1/test_predictions.jsonl"
RUN2="${BASE_OUT}/run2/test_predictions.jsonl"

echo
echo "============================================================"
echo "FINAL SHA256 COMPARISON"
echo "============================================================"

sha256sum "$RUN1" "$RUN2"

echo

if cmp -s "$RUN1" "$RUN2"; then
    echo "SUCCESS: Run 1 and Run 2 are byte-for-byte identical."
else
    echo "NOTICE: Run 1 and Run 2 differ."
fi

echo
echo "Results:"
echo "  ${BASE_OUT}/run1"
echo "  ${BASE_OUT}/run2"
echo "============================================================"
