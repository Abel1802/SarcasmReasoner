#!/bin/bash

set -e
set -o pipefail


# ============================================================
# MCSD Qwen3-Omni Teacher N=8 Sampling
#
# Order:
#   valid -> test -> train
#
# Sampling:
#   N = 8
#   temperature = 0.6
#   top_p = 0.95
#
# Batch sizes:
#   valid = 2
#   test  = 2
#   train = 3
# ============================================================


MODEL="Qwen/Qwen3-Omni-30B-A3B-Thinking"

N=8
TEMPERATURE=0.6
TOP_P=0.95

MAX_NEW_TOKENS=4096
MAX_MODEL_LEN=16384

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
MAX_NUM_SEQS=8

PROGRESS_INTERVAL=10
PROGRESS_BAR_WIDTH=40


# ============================================================
# Multimodal
# ============================================================

export USE_AUDIO_IN_VIDEO=False
export FORCE_QWENVL_VIDEO_READER=decord
export FPS_MAX_FRAMES=12


# ============================================================
# Paths
# ============================================================

WRAPPER="src/generation/swift_sample_local_video.py"
DATA_ROOT="data/mcsd/processed"
RESULT_ROOT="results/mcsd/teacher/sample_n8"


# ============================================================
# Dataset information
# ============================================================

expected_inputs() {

    case "$1" in
        valid)
            echo 406
            ;;
        test)
            echo 406
            ;;
        train)
            echo 1893
            ;;
        *)
            echo "Unknown split: $1" >&2
            exit 1
            ;;
    esac
}


sampling_batch_size() {

    case "$1" in
        valid)
            echo 2
            ;;
        test)
            echo 2
            ;;
        train)
            echo 3
            ;;
        *)
            echo "Unknown split: $1" >&2
            exit 1
            ;;
    esac
}


# ============================================================
# Time formatting
# ============================================================

format_time() {

    local total="$1"

    if [ "$total" -lt 0 ]; then
        echo "--:--:--"
        return
    fi

    local hours=$(( total / 3600 ))
    local minutes=$(( (total % 3600) / 60 ))
    local seconds=$(( total % 60 ))

    printf "%02d:%02d:%02d" \
        "$hours" "$minutes" "$seconds"
}


# ============================================================
# Read ms-swift checkpoint index
# ============================================================

read_ckpt_index() {

    local file="$1"

    if [ ! -f "$file" ]; then
        echo -1
        return
    fi

    python - "$file" <<'PY'
import json
import sys

path = sys.argv[1]

try:
    with open(path, encoding="utf-8") as f:
        state = json.load(f)

    print(int(state.get("index", -1)))

except Exception:
    print(-1)
PY
}


# ============================================================
# Progress bar
# ============================================================

show_progress() {

    local split="$1"
    local done="$2"
    local total="$3"
    local start_time="$4"
    local rate_start_time="$5"
    local rate_start_done="$6"

    if [ "$done" -lt 0 ]; then
        done=0
    fi

    if [ "$done" -gt "$total" ]; then
        done="$total"
    fi

    local now
    now=$(date +%s)

    local elapsed=$(( now - start_time ))
    local percent=$(( done * 100 / total ))

    local filled=$(( done * PROGRESS_BAR_WIDTH / total ))
    local empty=$(( PROGRESS_BAR_WIDTH - filled ))

    local bar=""

    if [ "$filled" -gt 0 ]; then
        bar=$(printf "%${filled}s" "" | tr ' ' '#')
    fi

    if [ "$empty" -gt 0 ]; then
        bar="${bar}$(printf "%${empty}s" "" | tr ' ' '-')"
    fi

    local eta_text="estimating..."

    if [ "$rate_start_time" -gt 0 ] && \
       [ "$done" -gt "$rate_start_done" ]; then

        local rate_elapsed=$(( now - rate_start_time ))
        local rate_done=$(( done - rate_start_done ))

        if [ "$rate_elapsed" -gt 0 ] && \
           [ "$rate_done" -gt 0 ]; then

            local remaining=$(( total - done ))
            local eta=$(( remaining * rate_elapsed / rate_done ))

            eta_text=$(format_time "$eta")
        fi
    fi

    printf \
        "\r[%s] [%s] %3d%%  %4d/%4d inputs  elapsed %s  ETA %s" \
        "$split" \
        "$bar" \
        "$percent" \
        "$done" \
        "$total" \
        "$(format_time "$elapsed")" \
        "$eta_text"
}


# ============================================================
# Basic checks
# ============================================================

if [ ! -f "$WRAPPER" ]; then
    echo "ERROR: wrapper not found:"
    echo "  $WRAPPER"
    exit 1
fi


for split in valid test train; do

    data="${DATA_ROOT}/zero_shot_${split}.jsonl"

    if [ ! -f "$data" ]; then
        echo "ERROR: dataset not found:"
        echo "  $data"
        exit 1
    fi

done


# ============================================================
# Run one split
# ============================================================

run_split() {

    local split="$1"

    local data="${DATA_ROOT}/zero_shot_${split}.jsonl"

    local output_dir="${RESULT_ROOT}/${split}"
    local output_file="qwen3_omni_30b_${split}_n8.jsonl"
    local output_path="${output_dir}/${output_file}"

    local log_file="${output_dir}/sampling.log"
    local status_file="${output_dir}/exit_status.txt"
    local ckpt_file="${output_dir}/ckpt_state.json"

    local num_inputs
    num_inputs=$(expected_inputs "$split")

    local batch_size
    batch_size=$(sampling_batch_size "$split")

    local expected_lines=$(( num_inputs * N ))
    local total_batches=$(( num_inputs / batch_size ))


    # --------------------------------------------------------
    # Protect against ms-swift 4.2.2 dropping remainder
    # --------------------------------------------------------

    if [ $(( num_inputs % batch_size )) -ne 0 ]; then

        echo "ERROR:"
        echo "$split has $num_inputs inputs but batch_size=$batch_size"
        echo "The split size must be divisible by batch size."

        exit 1
    fi


    mkdir -p "$output_dir"


    echo
    echo "============================================================"
    echo "MCSD Qwen3-Omni N=8 Sampling"
    echo "============================================================"
    echo "Split:                 $split"
    echo "Input:                 $data"
    echo "Inputs:                $num_inputs"
    echo "Sampling batch size:   $batch_size"
    echo "Sampling batches:      $total_batches"
    echo "N per input:           $N"
    echo "Expected trajectories: $expected_lines"
    echo
    echo "Temperature:           $TEMPERATURE"
    echo "Top-p:                 $TOP_P"
    echo "Max new tokens:        $MAX_NEW_TOKENS"
    echo "Max model length:      $MAX_MODEL_LEN"
    echo
    echo "Tensor parallel:       $TENSOR_PARALLEL_SIZE"
    echo "Max vLLM seqs:         $MAX_NUM_SEQS"
    echo
    echo "Output:"
    echo "  $output_path"
    echo
    echo "Log:"
    echo "  $log_file"
    echo "============================================================"
    echo


    # --------------------------------------------------------
    # Already complete
    # --------------------------------------------------------

    if [ -f "$output_path" ]; then

        local current_lines
        current_lines=$(wc -l < "$output_path")

        if [ "$current_lines" -eq "$expected_lines" ]; then

            echo "Already complete:"
            echo "  $current_lines / $expected_lines trajectories"
            echo

            return 0

        else

            echo "ERROR:"
            echo "Final output exists but has unexpected size."
            echo
            echo "Current:  $current_lines"
            echo "Expected: $expected_lines"
            echo
            echo "Not overwriting automatically."

            exit 1
        fi
    fi


    # --------------------------------------------------------
    # Resume information
    # --------------------------------------------------------

    local start_index
    start_index=$(read_ckpt_index "$ckpt_file")

    local start_completed_batches=$(( start_index + 1 ))

    if [ "$start_completed_batches" -lt 0 ]; then
        start_completed_batches=0
    fi

    local start_completed_inputs=$(( start_completed_batches * batch_size ))

    if [ "$start_completed_inputs" -gt "$num_inputs" ]; then
        start_completed_inputs="$num_inputs"
    fi


    if [ "$start_completed_inputs" -gt 0 ]; then

        echo "Resume detected:"
        echo "  completed batches: $start_completed_batches / $total_batches"
        echo "  completed inputs:  $start_completed_inputs / $num_inputs"
        echo
    fi


    rm -f "$status_file"


    # ========================================================
    # Launch sampling
    # ========================================================

    (
        set +e

        python "$WRAPPER" \
            --model "$MODEL" \
            --dataset "$data" \
            --sampler_type sample \
            --sampler_engine vllm \
            --num_return_sequences "$N" \
            --num_sampling_batch_size "$batch_size" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --dataset_shuffle false \
            --resume true \
            --output_dir "$output_dir" \
            --output_file "$output_file" \
            --engine_kwargs "{
                \"tensor_parallel_size\": ${TENSOR_PARALLEL_SIZE},
                \"gpu_memory_utilization\": ${GPU_MEMORY_UTILIZATION},
                \"max_model_len\": ${MAX_MODEL_LEN},
                \"max_num_seqs\": ${MAX_NUM_SEQS},
                \"limit_mm_per_prompt\": {
                    \"audio\": 1,
                    \"video\": 1
                }
            }" \
            >> "$log_file" 2>&1

        status=$?

        echo "$status" > "$status_file"

        exit "$status"

    ) &

    local sampling_pid=$!


    echo "Sampling PID: $sampling_pid"
    echo
    echo "Loading model / initializing vLLM..."
    echo "ETA will appear after generation starts."
    echo


    # ========================================================
    # Monitor progress
    # ========================================================

    local start_time
    start_time=$(date +%s)

    local rate_start_time=0
    local rate_start_done=0


    while kill -0 "$sampling_pid" 2>/dev/null; do

        local current_index
        current_index=$(read_ckpt_index "$ckpt_file")

        local completed_batches=$(( current_index + 1 ))

        if [ "$completed_batches" -lt 0 ]; then
            completed_batches=0
        fi

        local completed_inputs=$(( completed_batches * batch_size ))

        if [ "$completed_inputs" -gt "$num_inputs" ]; then
            completed_inputs="$num_inputs"
        fi


        # Begin throughput timing once the first new batch completes.
        if [ "$rate_start_time" -eq 0 ] && \
           [ "$completed_inputs" -gt "$start_completed_inputs" ]; then

            rate_start_time=$(date +%s)
            rate_start_done="$completed_inputs"
        fi


        show_progress \
            "$split" \
            "$completed_inputs" \
            "$num_inputs" \
            "$start_time" \
            "$rate_start_time" \
            "$rate_start_done"


        sleep "$PROGRESS_INTERVAL"
    done


    # ========================================================
    # Collect exit status
    # ========================================================

    set +e
    wait "$sampling_pid"
    local wait_status=$?
    set -e

    local status="$wait_status"

    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
    fi


    # --------------------------------------------------------
    # Final progress
    # --------------------------------------------------------

    local final_index
    final_index=$(read_ckpt_index "$ckpt_file")

    local final_completed_batches=$(( final_index + 1 ))

    if [ "$final_completed_batches" -lt 0 ]; then
        final_completed_batches=0
    fi

    local final_completed_inputs=$(( final_completed_batches * batch_size ))

    if [ "$final_completed_inputs" -gt "$num_inputs" ]; then
        final_completed_inputs="$num_inputs"
    fi


    show_progress \
        "$split" \
        "$final_completed_inputs" \
        "$num_inputs" \
        "$start_time" \
        "$rate_start_time" \
        "$rate_start_done"

    echo
    echo


    # ========================================================
    # Handle failure
    # ========================================================

    if [ "$status" -ne 0 ]; then

        echo "============================================================"
        echo "SAMPLING FAILED"
        echo "============================================================"
        echo
        echo "Split:     $split"
        echo "Exit code: $status"
        echo
        echo "Progress is preserved."
        echo "Run the same script again to resume."
        echo
        echo "Last 40 log lines:"
        echo "------------------------------------------------------------"

        tail -n 40 "$log_file" || true

        echo "------------------------------------------------------------"

        exit "$status"
    fi


    # ========================================================
    # Verify final output
    # ========================================================

    if [ ! -f "$output_path" ]; then

        echo "ERROR:"
        echo "Sampling exited successfully but output file is missing:"
        echo
        echo "  $output_path"

        exit 1
    fi


    local actual_lines
    actual_lines=$(wc -l < "$output_path")


    echo "============================================================"
    echo "FINISHED: $split"
    echo "============================================================"
    echo
    echo "Expected trajectories: $expected_lines"
    echo "Actual trajectories:   $actual_lines"
    echo


    if [ "$actual_lines" -ne "$expected_lines" ]; then

        echo "ERROR:"
        echo "Trajectory count does not match expectation."

        exit 1
    fi


    echo "OK: $split completed successfully."
    echo
}


# ============================================================
# Main
# ============================================================

run_split valid
run_split test
run_split train


echo
echo "============================================================"
echo "ALL MCSD N=8 SAMPLING COMPLETE"
echo "============================================================"
echo

wc -l \
    "${RESULT_ROOT}/valid/qwen3_omni_30b_valid_n8.jsonl" \
    "${RESULT_ROOT}/test/qwen3_omni_30b_test_n8.jsonl" \
    "${RESULT_ROOT}/train/qwen3_omni_30b_train_n8.jsonl"

echo
echo "Expected:"
echo "  valid:  3248"
echo "  test:   3248"
echo "  train: 15144"
echo "  total: 21640"
echo
