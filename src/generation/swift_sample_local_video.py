#!/usr/bin/env python3

"""
Sampling wrapper for multimodal sarcasm reasoning.

Features
--------
1. Keeps local video paths for decord.
2. Accepts the existing zero_shot_*.jsonl files directly.
3. Adds the gold assistant message internally because `swift sample`
   expects the final message to be an assistant ground truth.
   The assistant gold message is removed BEFORE teacher generation.
4. Requests token-level log-probabilities from the teacher.
5. Stores only the mean token log-probability and token count
   for each generated trajectory.
6. Recovers the original dataset source_id from the original ID
   or the audio/video filename.

Designed for:
    ms-swift 4.2.2
    vLLM 0.13.0
"""

import json
import math
from copy import deepcopy
from pathlib import Path


# ============================================================
# 1. Local-video patch
# ============================================================

from swift.model.models import qwen as qwen_module


_original_get_new_read_video_func = (
    qwen_module._get_new_read_video_func
)


def _get_new_read_video_func(read_video_func, read_backend):

    if read_backend != "decord":
        return _original_get_new_read_video_func(
            read_video_func,
            read_backend,
        )

    original_reader = _original_get_new_read_video_func(
        read_video_func,
        read_backend,
    )

    def _reader(ele):

        video = ele.get("video")

        # Keep local files as paths for decord.
        if (
            isinstance(video, str)
            and not video.startswith(
                ("http://", "https://", "data:")
            )
        ):
            return read_video_func(ele)

        return original_reader(ele)

    return _reader


qwen_module._get_new_read_video_func = (
    _get_new_read_video_func
)


# ============================================================
# 2. Sampling extensions
# ============================================================

from swift.infer_engine import RequestConfig
from swift.pipelines.sampling.vanilla_sampler import VanillaSampler
from swift.pipelines.sampling.utils import get_messages_md5


def recover_source_id(row):
    """
    Recover the original dataset sample ID.

    Priority
    --------
    1. Existing non-empty source_id.
    2. Original id, if it does not look like an ms-swift MD5 hash.
    3. Video filename stem.
    4. Audio filename stem.

    Examples
    --------
    MCSD:
        data/mcsd/raw/videos/174.mp4
        -> 174

        data/mcsd/raw/audios/174.wav
        -> 174

    MUStARD++:
        data/mustard/raw/videos/2_570_u.mp4
        -> 2_570_u
    """

    # --------------------------------------------------------
    # 1. Existing source_id
    # --------------------------------------------------------

    source_id = row.get("source_id")

    if source_id is not None:

        source_id = str(source_id).strip()

        if source_id:
            return source_id

    # --------------------------------------------------------
    # 2. Original dataset id
    #
    # ms-swift may replace the original id with a 32-character
    # MD5-like internal hash. Do not mistake that for source_id.
    # --------------------------------------------------------

    original_id = row.get("id")

    if original_id is not None:

        original_id = str(original_id).strip()

        if original_id:

            is_md5 = (
                len(original_id) == 32
                and all(
                    c in "0123456789abcdefABCDEF"
                    for c in original_id
                )
            )

            if not is_md5:
                return original_id

    # --------------------------------------------------------
    # 3/4. Recover from video/audio filename
    # --------------------------------------------------------

    for key in ("videos", "audios"):

        values = row.get(key)

        if not values:
            continue

        if not isinstance(values, (list, tuple)):
            values = [values]

        for value in values:

            # Media may occasionally be represented as:
            #
            # {"path": "..."}
            #
            # instead of a plain path string.
            if isinstance(value, dict):
                value = value.get("path")

            if not isinstance(value, str):
                continue

            value = value.strip()

            if not value:
                continue

            # We only recover IDs from local file paths.
            if value.startswith(
                (
                    "http://",
                    "https://",
                    "data:",
                )
            ):
                continue

            source_id = Path(value).stem

            if source_id:
                return source_id

    # --------------------------------------------------------
    # Nothing worked
    # --------------------------------------------------------

    raise ValueError(
        "Cannot recover source_id.\n"
        f"id={row.get('id')!r}\n"
        f"source_id={row.get('source_id')!r}\n"
        f"videos={row.get('videos')!r}\n"
        f"audios={row.get('audios')!r}"
    )


def get_gold_answer(row):
    """
    Convert the canonical binary sarcasm label to the answer
    expected by the sampling pipeline.
    """

    label = row.get("label")

    if label in (1, "1"):
        return "Sarcasm"

    if label in (0, "0"):
        return "Non-Sarcasm"

    label_text = str(
        row.get("label_text", "")
    ).strip().lower()

    if label_text in {
        "sarcasm",
        "sarcastic",
        "s",
    }:
        return "Sarcasm"

    if label_text in {
        "non-sarcasm",
        "non_sarcasm",
        "nonsarcasm",
        "non-sarcastic",
        "ns",
    }:
        return "Non-Sarcasm"

    raise ValueError(
        "Cannot determine gold label for "
        f"id={row.get('id')}: "
        f"label={label!r}, "
        f"label_text={row.get('label_text')!r}"
    )


def get_mean_token_logprob(logprobs):
    """
    Compute the mean log-probability of the generated tokens.

    ms-swift returns:
        {
            "content": [
                {
                    "token": ...,
                    "logprob": ...
                },
                ...
            ]
        }

    Returns
    -------
    mean_logprob : float or None
    n_tokens     : int
    """

    if not logprobs:
        return None, 0

    content = logprobs.get("content", [])

    values = []

    for token_info in content:

        value = token_info.get("logprob")

        if value is None:
            continue

        value = float(value)

        # This should normally never happen for sampled tokens.
        if not math.isfinite(value):
            return None, len(content)

        values.append(value)

    if not values:
        return None, 0

    return sum(values) / len(values), len(values)


class LogprobVanillaSampler(VanillaSampler):

    # ========================================================
    # Accept zero-shot files directly
    # ========================================================

    @staticmethod
    def convert_data_to_rows(data):

        # Use ms-swift's normal multimodal conversion first.
        rows = VanillaSampler.convert_data_to_rows(data)

        converted = []

        for row in rows:

            row = deepcopy(row)

            # ------------------------------------------------
            # Recover the original dataset ID.
            #
            # In practice, custom `id` fields may be removed
            # during ms-swift dataset preprocessing, while
            # videos/audios remain available.
            # ------------------------------------------------

            if not str(
                row.get("source_id", "")
            ).strip():

                row["source_id"] = recover_source_id(
                    row
                )

            messages = deepcopy(
                row["messages"]
            )

            # ------------------------------------------------
            # zero_shot_*.jsonl normally ends in USER.
            #
            # swift sample requires a final assistant ground
            # truth. This assistant message is removed before
            # actual teacher generation.
            # ------------------------------------------------

            if (
                not messages
                or messages[-1].get("role") != "assistant"
            ):
                messages.append({
                    "role": "assistant",
                    "content": get_gold_answer(row),
                })

            row["messages"] = messages

            converted.append(row)

        return converted

    # ========================================================
    # Generate N trajectories + teacher logprobs
    # ========================================================

    def generate(self, data):
        """
        Generate N trajectories per input using native vLLM `n=N`.

        Important:
        Instead of duplicating the same multimodal request N times,
        each source input is sent to vLLM only once:

            1 input request
                -> RequestConfig(n=N)
                -> N completion choices

        This allows vLLM to handle the N-way generation internally
        and avoids constructing N identical multimodal requests in
        Python/ms-swift.
        """

        resp_all = []
        infer_requests = []

        rows = self.convert_data_to_rows(data)

        # ====================================================
        # Construct ONE inference request per source input
        # ====================================================

        for row in rows:

            infer_row = deepcopy(row)

            messages = infer_row["messages"]

            if self.args.system:

                if messages[0]["role"] == "system":
                    messages[0]["content"] = self.args.system

                else:
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": self.args.system,
                        },
                    )

            # ------------------------------------------------
            # CRITICAL:
            #
            # The final assistant message contains the temporary
            # gold answer required by `swift sample`.
            #
            # Remove it BEFORE teacher generation so there is
            # absolutely no label leakage.
            # ------------------------------------------------

            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]

            infer_row["messages"] = messages

            # ------------------------------------------------
            # OLD:
            #
            # for _ in range(self.args.num_return_sequences):
            #     infer_requests.append(deepcopy(infer_row))
            #
            # NEW:
            #
            # Only ONE request per source input.
            # vLLM will create N completions internally.
            # ------------------------------------------------

            infer_requests.append(
                deepcopy(infer_row)
            )

        # ====================================================
        # Native N-way generation
        # ====================================================

        request_config = RequestConfig(
            max_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,

            # Return log-probability of each sampled token.
            logprobs=True,

            # Native vLLM N-way sampling.
            n=self.args.num_return_sequences,
        )

        resp_list = []

        if infer_requests:

            resp_list = self.infer_engine.infer(
                infer_requests,
                request_config=request_config,
            )

        # ====================================================
        # Sanity check
        #
        # We now expect:
        #
        # len(resp_list) == number of source inputs
        #
        # and each:
        #
        # len(response.choices) == N
        # ====================================================

        if len(resp_list) != len(rows):

            raise RuntimeError(
                "Unexpected number of vLLM responses: "
                f"got {len(resp_list)}, "
                f"expected {len(rows)}."
            )

        # ====================================================
        # Convert each response's N choices into our candidate
        # representation.
        # ====================================================

        for row, response in zip(
            rows,
            resp_list,
        ):

            if isinstance(response, Exception):

                source_id = recover_source_id(row)

                raise RuntimeError(
                    "vLLM generation failed for "
                    f"source_id={source_id}: "
                    f"{response}"
                ) from response

            result_row = deepcopy(row)

            choices = response.choices

            # ------------------------------------------------
            # We requested exactly N trajectories.
            #
            # Do not silently continue if vLLM returns fewer,
            # otherwise Best-of-8 would no longer actually be
            # Best-of-8.
            # ------------------------------------------------

            if len(choices) != self.args.num_return_sequences:

                source_id = recover_source_id(row)

                raise RuntimeError(
                    "Unexpected number of trajectories for "
                    f"source_id={source_id}: "
                    f"got {len(choices)}, "
                    f"expected "
                    f"{self.args.num_return_sequences}."
                )

            # vLLM choices normally carry index=0,...,N-1.
            # Sort explicitly so output order remains stable.
            choices = sorted(
                choices,
                key=lambda choice: choice.index,
            )

            candidates = []

            for choice in choices:

                mean_logprob, n_tokens = (
                    get_mean_token_logprob(
                        choice.logprobs
                    )
                )

                candidates.append({
                    "sample_idx": int(choice.index),
                    "content": choice.message.content,
                    "mean_token_logprob": (
                        mean_logprob
                    ),
                    "num_completion_tokens": (
                        n_tokens
                    ),
                    "finish_reason": (
                        choice.finish_reason
                    ),
                })

            result_row["candidates"] = candidates

            resp_all.append(
                result_row
            )

        return resp_all

    # ========================================================
    # Save one generated trajectory per JSONL line
    # ========================================================

    def do_sample(self, data):

        # ----------------------------------------------------
        # This wrapper is designed for RAW candidate generation.
        #
        # Best-of-N / Diverse filtering happens later.
        # ----------------------------------------------------

        if self.args.prm_model is not None:

            raise ValueError(
                "PRM filtering is not supported in this "
                "raw logprob sampling wrapper."
            )

        if self.args.orm_model is not None:

            raise ValueError(
                "ORM filtering is not supported in this "
                "raw logprob sampling wrapper."
            )

        generated = []

        resp_all = self.generate(data)

        for resps in resp_all:

            candidates = resps["candidates"]

            messages = deepcopy(
                resps["messages"]
            )

            assert (
                messages[-1]["role"]
                == "assistant"
            )

            # ------------------------------------------------
            # Recover the original dataset ID.
            #
            # This is intentionally performed again here as a
            # final safety check before writing the output.
            # ------------------------------------------------

            source_id = recover_source_id(
                resps
            )

            # ------------------------------------------------
            # Keep an ms-swift-style internal hash ID as well.
            # source_id is the ID we use for grouping the N
            # trajectories belonging to the same input.
            # ------------------------------------------------

            internal_id = get_messages_md5(
                resps
            )

            for candidate in candidates:

                output = deepcopy(
                    resps
                )

                output.pop(
                    "candidates",
                    None,
                )

                output["id"] = (
                    internal_id
                )

                output["source_id"] = (
                    source_id
                )

                output["sample_idx"] = (
                    candidate["sample_idx"]
                )

                output[
                    "mean_token_logprob"
                ] = candidate[
                    "mean_token_logprob"
                ]

                output[
                    "num_completion_tokens"
                ] = candidate[
                    "num_completion_tokens"
                ]

                output[
                    "finish_reason"
                ] = candidate[
                    "finish_reason"
                ]

                # ------------------------------------------------
                # Replace temporary gold assistant content with
                # the actual teacher-generated trajectory.
                # ------------------------------------------------

                output["messages"][-1][
                    "content"
                ] = candidate[
                    "content"
                ]

                generated.append(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return generated


# ============================================================
# 3. Replace ms-swift's default sampler with ours
# ============================================================

import swift.pipelines.sampling.sampling as sampling_module


sampling_module.VanillaSampler = (
    LogprobVanillaSampler
)


if __name__ == "__main__":

    sampling_module.sampling_main()
