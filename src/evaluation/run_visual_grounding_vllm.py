#!/usr/bin/env python3

import argparse
import ast
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

PROMPT_FILE = Path(
    "src/prompts/grounding_judge/"
    "visual_grounding_binary_v4_final.txt"
)

DEFAULT_FPS = 4.0


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-VL visual grounding judge "
            "with vLLM offline inference."
        )
    )

    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge first N trajectories.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=768,
        help=(
            "Qwen3-VL video frame cap. "
            "768 matches the model-side maximum."
        ),
    )

    parser.add_argument(
        "--min_frames",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=16,
        help=(
            "Maximum concurrent sequences inside vLLM."
        ),
    )

    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=64,
        help=(
            "Approximate maximum trajectory requests "
            "submitted in one llm.generate() call."
        ),
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--mm_processor_cache_gb",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# JSONL
# ============================================================

def read_jsonl(path):

    rows = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


def load_completed(path):

    completed = set()

    if not path.exists():
        return completed

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:

                row = json.loads(line)

                if (
                    row.get("judge_id") is not None
                    and row.get(
                        "parse_ok",
                        False,
                    )
                ):

                    completed.add(
                        str(
                            row["judge_id"]
                        )
                    )

            except Exception:
                continue

    return completed


# ============================================================
# Judge parser
# ============================================================

def parse_judge_output(text):

    raw = text.strip()

    candidates = [raw]

    # Markdown fences.
    if "```" in raw:

        candidates.append(
            raw
            .replace(
                "```json",
                "",
            )
            .replace(
                "```JSON",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

    # Escaped JSON case:
    #
    # {\"verdict\": \"SUPPORTED\", ...}
    if '\\"' in raw:

        candidates.append(
            raw.replace(
                '\\"',
                '"',
            )
        )

    left = raw.find("{")
    right = raw.rfind("}")

    if (
        left != -1
        and right != -1
        and right > left
    ):

        candidates.append(
            raw[
                left:
                right + 1
            ]
        )

    # --------------------------------
    # Strict JSON / Python dict
    # --------------------------------

    for candidate in candidates:

        obj = None

        try:

            obj = json.loads(
                candidate
            )

        except Exception:

            try:

                value = (
                    ast.literal_eval(
                        candidate
                    )
                )

                if isinstance(
                    value,
                    dict,
                ):

                    obj = value

            except Exception:
                pass

        if isinstance(
            obj,
            dict,
        ):

            verdict = str(
                obj.get(
                    "verdict",
                    "",
                )
            ).strip().upper()

            issue = str(
                obj.get(
                    "issue",
                    "",
                )
            ).strip().upper()

            # Standard schema.
            if verdict in {
                "SUPPORTED",
                "UNSUPPORTED",
            }:

                return {
                    "parse_ok": True,
                    "verdict": verdict,
                    "grounded": (
                        1
                        if verdict == "SUPPORTED"
                        else 0
                    ),
                    "issue": (
                        issue
                        if issue
                        else None
                    ),
                    "evidence": obj.get(
                        "evidence"
                    ),
                }

            # Schema-drift fallback:
            # model occasionally puts the
            # issue type in the verdict field.
            unsupported_issue_labels = {
                "OVERINTERPRETATION",
                "HALLUCINATED_CUE",
                "CONTRADICTED_BY_INPUT",
                "WRONG_SPEAKER",
                "NOT_ASSESSABLE",
                "NO_VISUAL",
                "NO_VIDEO",
            }

            if (
                verdict in unsupported_issue_labels
                or (
                    issue
                    and issue != "NONE"
                )
            ):

                return {
                    "parse_ok": True,
                    "verdict": "UNSUPPORTED",
                    "grounded": 0,
                    "issue": (
                        issue
                        if issue
                        else verdict
                    ),
                    "evidence": obj.get(
                        "evidence"
                    ),
                }

            # If issue is explicitly NONE,
            # interpret it as supported.
            if issue == "NONE":

                return {
                    "parse_ok": True,
                    "verdict": "SUPPORTED",
                    "grounded": 1,
                    "issue": "NONE",
                    "evidence": obj.get(
                        "evidence"
                    ),
                }

    # --------------------------------
    # Robust verdict fallback
    # --------------------------------

    verdict_match = re.search(
        r"[\\\"']?verdict[\\\"']?"
        r"\s*:\s*"
        r"[\\\"']*"
        r"(SUPPORTED|UNSUPPORTED)"
        r"[\\\"']*",
        raw,
        flags=re.IGNORECASE,
    )

    if verdict_match:

        verdict = (
            verdict_match
            .group(1)
            .upper()
        )

        return {
            "parse_ok":
                True,

            "verdict":
                verdict,

            "grounded":
                (
                    1
                    if verdict
                    == "SUPPORTED"
                    else 0
                ),

            "evidence":
                None,
        }

    return {
        "parse_ok":
            False,

        "verdict":
            None,

        "grounded":
            None,

        "evidence":
            None,
    }


# ============================================================
# Prompt
# ============================================================

def build_judge_prompt(
    template,
    row,
):

    return template.replace(
        "{{VISUAL_EVIDENCE}}",
        row[
            "sections"
        ][
            "visual_evidence"
        ],
    )


def build_qwen3vl_prompt(
    judge_prompt,
):

    # Match Qwen3-VL chat format for a
    # user turn containing:
    #
    #   video
    #   text
    #
    # No explicit system message, matching
    # our previous HF calibration runner.

    return (
        "<|im_start|>user\n"
        "<|vision_start|>"
        "<|video_pad|>"
        "<|vision_end|>"
        f"{judge_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ============================================================
# Video sampling
# ============================================================

def load_video_at_fps(
    video_path,
    target_fps,
    min_frames,
    max_frames,
):

    """
    Decode a local video once using decord.

    Sampling mirrors the Qwen3-VL fps-based rule:

        num_frames =
            int(duration * target_fps)

    followed by:
        min_frames <= num_frames <= max_frames

    Frames are then uniformly sampled across
    the original video.

    Returns:
        frames:
            uint8 ndarray
            [T, H, W, 3]

        metadata:
            metadata required by Qwen3-VL
            in vLLM.
    """

    vr = VideoReader(
        str(video_path),
        ctx=cpu(0),
    )

    total_frames = len(vr)

    if total_frames <= 0:

        raise ValueError(
            f"No frames in video: "
            f"{video_path}"
        )

    original_fps = float(
        vr.get_avg_fps()
    )

    if (
        not math.isfinite(
            original_fps
        )
        or original_fps <= 0
    ):

        raise ValueError(
            f"Invalid FPS "
            f"{original_fps} "
            f"for {video_path}"
        )

    duration = (
        total_frames
        / original_fps
    )

    num_frames = int(
        duration
        * target_fps
    )

    num_frames = max(
        min_frames,
        num_frames,
    )

    num_frames = min(
        num_frames,
        max_frames,
        total_frames,
    )

    # Qwen3-VL uses uniform sampling.
    frame_indices = (
        np.linspace(
            0,
            total_frames - 1,
            num_frames,
        )
        .round()
        .astype(np.int64)
    )

    # Remove duplicate indices that can
    # occur for very short clips.
    frame_indices = np.unique(
        frame_indices
    )

    frames = (
        vr.get_batch(
            frame_indices.tolist()
        )
        .asnumpy()
    )

    if frames.ndim != 4:

        raise ValueError(
            f"Unexpected video shape "
            f"{frames.shape} "
            f"for {video_path}"
        )

    if frames.shape[-1] != 3:

        raise ValueError(
            f"Expected RGB video, got "
            f"{frames.shape} "
            f"for {video_path}"
        )

    frames = np.asarray(
        frames,
        dtype=np.uint8,
    )

    # IMPORTANT:
    #
    # frames_indices refer to locations in
    # the ORIGINAL video. This preserves
    # correct temporal positions/timestamps.
    #
    # do_sample_frames=False tells the
    # downstream processor that these frames
    # have already been sampled.

    metadata = {
        "total_num_frames":
            int(total_frames),

        "fps":
            float(original_fps),

        "duration":
            float(duration),

        "video_backend":
            "decord",

        "frames_indices":
            [
                int(x)
                for x
                in frame_indices
            ],

        "do_sample_frames":
            False,
    }

    return (
        frames,
        metadata,
    )


# ============================================================
# Group trajectories by source
# ============================================================

def group_by_source(rows):

    groups = OrderedDict()

    for row in rows:

        source_id = str(
            row["source_id"]
        )

        if source_id not in groups:

            groups[source_id] = []

        groups[
            source_id
        ].append(
            row
        )

    return list(
        groups.items()
    )


def make_source_batches(
    source_groups,
    max_requests,
):

    """
    Keep all trajectories from one source
    together so the video is decoded once.
    """

    current = []
    current_n = 0

    for (
        source_id,
        rows,
    ) in source_groups:

        group_n = len(rows)

        if (
            current
            and
            current_n
            + group_n
            > max_requests
        ):

            yield current

            current = []
            current_n = 0

        current.append(
            (
                source_id,
                rows,
            )
        )

        current_n += group_n

    if current:
        yield current


# ============================================================
# Build vLLM request batch
# ============================================================

def build_vllm_requests(
    *,
    source_batch,
    template,
    split,
    target_fps,
    min_frames,
    max_frames,
):

    """
    Decode each unique source video exactly
    once for this request batch.

    All trajectories from the same source
    share:
        - same video ndarray
        - same metadata
        - same multimodal UUID
    """

    request_items = []

    for (
        source_id,
        rows,
    ) in source_batch:

        if not rows:
            continue

        video_path = Path(
            rows[0]["video"]
        )

        frames, metadata = (
            load_video_at_fps(
                video_path,
                target_fps=target_fps,
                min_frames=min_frames,
                max_frames=max_frames,
            )
        )

        # Stable source-level UUID.
        #
        # Every trajectory from this source
        # refers to exactly the same video.
        video_uuid = (
            f"mustard:"
            f"{split}:"
            f"{source_id}"
        )

        # Qwen3-VL vLLM 0.13 expects
        # video + metadata.
        #
        # Official example uses:
        #
        # [(video_array, metadata)]
        #
        # for Qwen3-VL.
        video_data = [
            (
                frames,
                metadata,
            )
        ]

        for row in rows:

            judge_prompt = (
                build_judge_prompt(
                    template,
                    row,
                )
            )

            prompt = (
                build_qwen3vl_prompt(
                    judge_prompt
                )
            )

            request = {
                "prompt":
                    prompt,

                "multi_modal_data": {
                    "video":
                        video_data,
                },

                "multi_modal_uuids": {
                    "video":
                        video_uuid,
                },
            }

            request_items.append(
                (
                    row,
                    request,
                )
            )

    return request_items


# ============================================================
# Output
# ============================================================

def append_failure(
    path,
    row,
    reason,
    raw_output=None,
):

    item = {
        "judge_id":
            str(
                row["judge_id"]
            ),

        "source_id":
            str(
                row["source_id"]
            ),

        "sample_idx":
            row.get(
                "sample_idx"
            ),

        "split":
            row.get(
                "split"
            ),

        "reason":
            reason,

        "raw_output":
            raw_output,
    }

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


def write_success(
    fout,
    row,
    parsed,
    raw_output,
    split,
):

    result = {
        "judge_id":
            str(
                row["judge_id"]
            ),

        "source_id":
            str(
                row["source_id"]
            ),

        "sample_idx":
            row.get(
                "sample_idx"
            ),

        "split":
            split,

        "component":
            "visual",

        "grounded":
            parsed[
                "grounded"
            ],

        "verdict":
            parsed[
                "verdict"
            ],

        "parse_ok":
            True,

        "evidence":
            parsed[
                "evidence"
            ],

        "raw_output":
            raw_output.strip(),

        "engine":
            "vllm",
    }

    fout.write(
        json.dumps(
            result,
            ensure_ascii=False,
        )
        + "\n"
    )

    fout.flush()


# ============================================================
# Generate with fallback
# ============================================================

def run_request_items(
    *,
    llm,
    sampling_params,
    request_items,
):

    """
    Run one vLLM batch.

    If a large batch fails, recursively
    split it instead of losing everything.
    """

    if not request_items:

        return (
            [],
            [],
        )

    requests = [
        item[1]
        for item in request_items
    ]

    try:

        outputs = llm.generate(
            requests,
            sampling_params,
            use_tqdm=False,
        )

        if (
            len(outputs)
            != len(request_items)
        ):

            raise RuntimeError(
                "vLLM returned "
                f"{len(outputs)} outputs "
                "for "
                f"{len(request_items)} inputs."
            )

        successes = []

        for (
            (row, _),
            output,
        ) in zip(
            request_items,
            outputs,
        ):

            text = (
                output.outputs[0].text
            )

            successes.append(
                (
                    row,
                    text,
                )
            )

        return (
            successes,
            [],
        )

    except Exception as exc:

        # One request left:
        # report genuine generation failure.
        if len(
            request_items
        ) == 1:

            row = (
                request_items[0][0]
            )

            return (
                [],
                [
                    (
                        row,
                        repr(exc),
                    )
                ],
            )

        # Split recursively.
        midpoint = (
            len(request_items)
            // 2
        )

        left_ok, left_bad = (
            run_request_items(
                llm=llm,
                sampling_params=(
                    sampling_params
                ),
                request_items=(
                    request_items[
                        :midpoint
                    ]
                ),
            )
        )

        right_ok, right_bad = (
            run_request_items(
                llm=llm,
                sampling_params=(
                    sampling_params
                ),
                request_items=(
                    request_items[
                        midpoint:
                    ]
                ),
            )
        )

        return (
            left_ok
            + right_ok,

            left_bad
            + right_bad,
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    split = args.split

    input_path = (
        args.input
        if args.input is not None
        else Path(
            "data/mustard/processed/genrm/"
            f"grounding_judge_pool_{split}.jsonl"
        )
    )

    outdir = (
        args.outdir
        if args.outdir is not None
        else Path(
            "results/mustard/genrm/"
            "grounding_judging/"
            f"{split}"
        )
    )

    output_path = (
        outdir
        / "visual_labels.jsonl"
    )

    failure_path = (
        outdir
        / "visual_failures.jsonl"
    )

    print(
        "=" * 80
    )

    print(
        "VISUAL GROUNDING JUDGE — vLLM"
    )

    print(
        "=" * 80
    )

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Split:",
        split,
    )

    print(
        "Input:",
        input_path,
    )

    print(
        "Output:",
        output_path,
    )

    print(
        "FPS:",
        args.fps,
    )

    print(
        "max_num_seqs:",
        args.max_num_seqs,
    )

    print(
        "request_batch_size:",
        args.request_batch_size,
    )

    print(
        "max_model_len:",
        args.max_model_len,
    )

    # --------------------------------
    # Load data
    # --------------------------------

    rows = read_jsonl(
        input_path
    )

    if args.limit is not None:

        rows = rows[
            :args.limit
        ]

    completed = load_completed(
        output_path
    )

    pending = [
        row
        for row in rows
        if str(
            row["judge_id"]
        ) not in completed
    ]

    print()
    print(
        "Total rows:",
        len(rows),
    )

    print(
        "Already completed:",
        len(completed),
    )

    print(
        "Pending:",
        len(pending),
    )

    if not pending:

        print(
            "Nothing to do."
        )

        return

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = (
        PROMPT_FILE
        .read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------
    # Initialize vLLM
    # --------------------------------

    print()
    print(
        "Initializing vLLM..."
    )

    llm = LLM(
        model=MODEL_NAME,

        dtype="bfloat16",

        tensor_parallel_size=1,

        max_model_len=(
            args.max_model_len
        ),

        max_num_seqs=(
            args.max_num_seqs
        ),

        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),

        limit_mm_per_prompt={
            "video": 1,
        },

        mm_processor_kwargs={
            "fps":
                args.fps,
        },

        mm_processor_cache_gb=(
            args.mm_processor_cache_gb
        ),

        seed=args.seed,
    )

    sampling_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=(
                args.max_tokens
            ),
        )
    )

    print(
        "vLLM initialized."
    )

    # --------------------------------
    # Group by source
    # --------------------------------

    source_groups = (
        group_by_source(
            pending
        )
    )

    print(
        "Pending unique sources:",
        len(source_groups),
    )

    batches = list(
        make_source_batches(
            source_groups,
            max_requests=(
                args.request_batch_size
            ),
        )
    )

    print(
        "Generate batches:",
        len(batches),
    )

    # --------------------------------
    # Run
    # --------------------------------

    parsed_count = 0
    supported_count = 0
    unsupported_count = 0
    failure_count = 0

    processed_count = 0

    with open(
        output_path,
        "a",
        encoding="utf-8",
    ) as fout:

        for (
            batch_idx,
            source_batch,
        ) in enumerate(
            batches,
            start=1,
        ):

            try:

                request_items = (
                    build_vllm_requests(
                        source_batch=(
                            source_batch
                        ),
                        template=template,
                        split=split,
                        target_fps=(
                            args.fps
                        ),
                        min_frames=(
                            args.min_frames
                        ),
                        max_frames=(
                            args.max_frames
                        ),
                    )
                )

            except Exception as exc:

                print()
                print(
                    "Video decode/build failure "
                    f"in batch {batch_idx}:"
                )

                print(
                    repr(exc)
                )

                # Fall back source-by-source
                # so one broken file does not
                # destroy the whole batch.
                request_items = []

                for (
                    source_id,
                    source_rows,
                ) in source_batch:

                    try:

                        one_items = (
                            build_vllm_requests(
                                source_batch=[
                                    (
                                        source_id,
                                        source_rows,
                                    )
                                ],
                                template=template,
                                split=split,
                                target_fps=(
                                    args.fps
                                ),
                                min_frames=(
                                    args.min_frames
                                ),
                                max_frames=(
                                    args.max_frames
                                ),
                            )
                        )

                        request_items.extend(
                            one_items
                        )

                    except Exception as one_exc:

                        for row in source_rows:

                            failure_count += 1

                            append_failure(
                                failure_path,
                                row,
                                reason=(
                                    "video_decode_error: "
                                    + repr(
                                        one_exc
                                    )
                                ),
                            )

            successes, failures = (
                run_request_items(
                    llm=llm,
                    sampling_params=(
                        sampling_params
                    ),
                    request_items=(
                        request_items
                    ),
                )
            )

            for (
                row,
                error,
            ) in failures:

                failure_count += 1

                append_failure(
                    failure_path,
                    row,
                    reason=(
                        "generation_error: "
                        + error
                    ),
                )

            for (
                row,
                raw_output,
            ) in successes:

                parsed = (
                    parse_judge_output(
                        raw_output
                    )
                )

                if not parsed[
                    "parse_ok"
                ]:

                    failure_count += 1

                    append_failure(
                        failure_path,
                        row,
                        reason=(
                            "parse_error"
                        ),
                        raw_output=(
                            raw_output
                        ),
                    )

                    continue

                write_success(
                    fout=fout,
                    row=row,
                    parsed=parsed,
                    raw_output=(
                        raw_output
                    ),
                    split=split,
                )

                parsed_count += 1

                if (
                    parsed[
                        "grounded"
                    ]
                    == 1
                ):

                    supported_count += 1

                else:

                    unsupported_count += 1

            processed_count += sum(
                len(rows_)
                for _, rows_
                in source_batch
            )

            print(
                f"[visual/vllm] "
                f"batch "
                f"{batch_idx}/"
                f"{len(batches)} "
                f"| processed="
                f"{processed_count}/"
                f"{len(pending)} "
                f"| parsed="
                f"{parsed_count} "
                f"| supported="
                f"{supported_count} "
                f"| unsupported="
                f"{unsupported_count} "
                f"| failures="
                f"{failure_count}"
            )

    print()
    print(
        "=" * 80
    )

    print(
        "VISUAL vLLM FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        "New parsed:",
        parsed_count,
    )

    print(
        "SUPPORTED:",
        supported_count,
    )

    print(
        "UNSUPPORTED:",
        unsupported_count,
    )

    print(
        "Failures:",
        failure_count,
    )

    print(
        "Output:",
        output_path,
    )


if __name__ == "__main__":
    main()
