"""Sarcasm rewards with an optional linear grounding schedule.

Install as src/plugins/sarcasm_grpo_reward_linear.py in SarcasmReasoner.
The original prompt path remains src/prompts/genrm/grounding_genrm_system.txt.
Load ONLY this file as the external plugin for this run; it includes the original
accuracy, format and constant-grounding registrations too.

Scheduled training:
  export GENRM_ZERO_RATIO=0.10
  export GENRM_FULL_RATIO=0.40
  export GENRM_SCHEDULE_LOG_STEPS=50
  --reward_funcs sarcasm_accuracy sarcasm_format
  --reward_model_plugin sarcasm_grounding_linear
  --reward_weights 1.0 0.2 0.2

The plugin multiplies grounding by a factor in [0, 1]. The last external
reward_weight is the maximum grounding coefficient (0.2), applied ONCE.
Do not multiply the factor by 0.2 again inside the plugin.
Requires trainer_state.global_step and trainer_state.max_steps in reward kwargs.
Schedule progress is captured when a reward batch is scored, not per call or
per microbatch. Reused rollouts retain the reward computed at scoring time.
Resuming requires the original trainer state, total budget and schedule settings.

The original sarcasm_grounding plugin remains unscheduled and can be used for
independent scoring. Scheduled evaluation inside Trainer uses current progress;
its total reward is NOT a time-invariant checkpoint-selection metric.
Select checkpoints with validation classification Macro-F1 instead.

Logs are rank-zero, first-local-batch snapshots at each logging step/phase change,
not distributed aggregates. raw_gated_mean includes zeros from correctness and
structure gates. It is null when GenRM was skipped during zero-weight warm-up.
GenRM is still loaded at startup: skipping calls saves inference work, not the
memory needed to load the judge. Group normalization behavior is unchanged.
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from swift.infer_engine import RequestConfig, TransformersEngine
from swift.rewards import ORM, orms, rm_plugins
from swift.rewards.rm_plugin import DefaultRMPlugin


# ============================================================
# Constants
# ============================================================

REQUIRED_TAGS = [
    "text_evidence",
    "audio_evidence",
    "visual_evidence",
    "integration",
    "answer",
]

GROUNDING_TAGS = [
    "text_evidence",
    "audio_evidence",
    "visual_evidence",
    "integration",
]


# ============================================================
# GenRM system prompt
#
# Single source of truth:
# exactly the same prompt used for GenRM training.
# ============================================================

GENRM_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "genrm"
    / "grounding_genrm_system.txt"
)

GENRM_SYSTEM_PROMPT = (
    GENRM_SYSTEM_PROMPT_PATH
    .read_text(encoding="utf-8")
    .strip()
)


# ============================================================
# Shared parsing
# ============================================================

def extract_tag(
    text: str,
    tag: str,
) -> Optional[str]:

    matches = re.findall(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if len(matches) != 1:
        return None

    content = matches[0].strip()

    if not content:
        return None

    return content


def extract_answer(
    text: str,
):
    content = extract_tag(
        text,
        "answer",
    )

    if content is None:
        return None

    answer = content.lower()

    if answer == "sarcasm":
        return 1

    if answer == "non-sarcasm":
        return 0

    return None


def has_valid_format(
    text: str,
) -> bool:

    for tag in REQUIRED_TAGS:
        if extract_tag(text, tag) is None:
            return False

    return extract_answer(text) is not None


def extract_reasoning_for_genrm(
    completion: str,
) -> Optional[str]:
    """
    Extract only the four reasoning sections.

    <answer> is intentionally excluded so that the GenRM
    judges grounding rather than final classification
    correctness.
    """

    sections = []

    for tag in GROUNDING_TAGS:

        content = extract_tag(
            completion,
            tag,
        )

        if content is None:
            return None

        sections.append(
            f"<{tag}>\n"
            f"{content}\n"
            f"</{tag}>"
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# Explicit dataset fields
# ============================================================

def get_transcript(
    infer_request,
) -> str:
    """
    Transcript must now be stored explicitly in the dataset.

    We deliberately do NOT recover it from messages.
    """

    transcript = infer_request.get(
        "transcript"
    )

    if not (
        isinstance(transcript, str)
        and transcript.strip()
    ):
        raise RuntimeError(
            "Missing or empty `transcript` field "
            "in GRPO reward input."
        )

    return transcript.strip()


def get_audio(
    infer_request,
):
    audios = infer_request.get(
        "audios"
    )

    if not audios:
        raise RuntimeError(
            "Missing `audios` field "
            "in GRPO reward input."
        )

    return audios


def get_video(
    infer_request,
):
    videos = infer_request.get(
        "videos"
    )

    if not videos:
        raise RuntimeError(
            "Missing `videos` field "
            "in GRPO reward input."
        )

    return videos


# ============================================================
# GenRM output parser
# ============================================================

def parse_genrm_output(
    text: str,
):
    """
    Expected exact format:

    {"text":1,"audio":0,"visual":1,"integration":1}
    """

    if not (
        isinstance(text, str)
        and text.strip()
    ):
        return None

    try:
        obj = json.loads(
            text.strip()
        )

    except Exception:
        return None

    if not isinstance(
        obj,
        dict,
    ):
        return None

    required = {
        "text",
        "audio",
        "visual",
        "integration",
    }

    if set(obj.keys()) != required:
        return None

    parsed = {}

    for key in required:

        value = obj[key]

        # bool is a subclass of int in Python.
        if isinstance(value, bool):
            return None

        if (
            not isinstance(value, int)
            or value not in (0, 1)
        ):
            return None

        parsed[key] = value

    return parsed


# ============================================================
# Accuracy reward
# ============================================================

class SarcasmAccuracyReward(ORM):

    def __call__(
        self,
        completions,
        label,
        **kwargs,
    ) -> List[float]:

        rewards = []

        for completion, gold in zip(
            completions,
            label,
        ):

            pred = extract_answer(
                completion
            )

            try:
                gold = int(gold)

            except Exception:
                rewards.append(
                    0.0
                )
                continue

            rewards.append(
                1.0
                if pred == gold
                else 0.0
            )

        return rewards


# ============================================================
# Format reward
# ============================================================

class SarcasmFormatReward(ORM):

    def __call__(
        self,
        completions,
        **kwargs,
    ) -> List[float]:

        return [
            1.0
            if has_valid_format(
                completion
            )
            else 0.0
            for completion in completions
        ]


# ============================================================
# Accuracy-gated multimodal GenRM
# ============================================================

class SarcasmGroundingRMPlugin(
    DefaultRMPlugin
):
    """
    Accuracy-first, grounding-second.

    Wrong classification:
        grounding reward = 0
        GenRM is not called.

    Correct classification:
        GenRM evaluates grounding.

    Scalar grounding reward:

        (text + audio + visual) / 3

    Integration is predicted by GenRM because this matches its
    training task, but is deliberately excluded from the scalar
    downstream reward because its negative class is extremely
    underrepresented.
    """

    def __init__(
        self,
        model,
        template,
    ):
        super().__init__(
            model,
            template,
        )

        self.engine = TransformersEngine(
            self.model,
            template=self.template,
            max_batch_size=0,
        )

        self.request_config = RequestConfig(
            max_tokens=64,
            temperature=0,
        )

    # --------------------------------------------------------
    # Construct GenRM input
    # --------------------------------------------------------

    @staticmethod
    def build_genrm_request(
        infer_request,
        transcript,
        reasoning,
    ):
        """
        Build the same input structure used when training
        the GenRM.

        Only required multimodal fields are preserved.
        """

        user_prompt = (
            "<audio><video>\n\n"
            "Transcript:\n"
            f"{transcript}\n\n"
            "Candidate reasoning:\n\n"
            f"{reasoning}\n\n"
            "Evaluate the grounding of the four "
            "candidate reasoning sections.\n"
            "Return only the required JSON object."
        )

        rm_request = {
            "messages": [
                {
                    "role": "system",
                    "content":
                        GENRM_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        user_prompt,
                },
            ],

            "audios": deepcopy(
                infer_request["audios"]
            ),

            "videos": deepcopy(
                infer_request["videos"]
            ),
        }

        return rm_request

    # --------------------------------------------------------
    # Main reward-model call
    # --------------------------------------------------------

    def __call__(
        self,
        inputs,
        **kwargs,
    ) -> List[float]:

        batch_size = len(
            inputs
        )

        rewards = [
            0.0
            for _ in range(batch_size)
        ]

        rm_inputs = []
        rm_indices = []

        # ====================================================
        # Build GenRM batch
        # ====================================================

        for idx, infer_request in enumerate(
            inputs
        ):

            # -----------------------------------------------
            # Validate explicit dataset fields.
            # -----------------------------------------------

            transcript = get_transcript(
                infer_request
            )

            get_audio(
                infer_request
            )

            get_video(
                infer_request
            )

            # -----------------------------------------------
            # Recover generated completion.
            # -----------------------------------------------

            messages = infer_request.get(
                "messages",
                [],
            )

            if not (
                isinstance(messages, list)
                and messages
            ):
                raise RuntimeError(
                    "Missing `messages` in "
                    "GRPO reward input."
                )

            last_message = messages[-1]

            if not (
                isinstance(last_message, dict)
                and last_message.get("role")
                == "assistant"
            ):
                raise RuntimeError(
                    "Expected the final GRPO message "
                    "to be an assistant completion."
                )

            completion = last_message.get(
                "content",
                "",
            )

            if not isinstance(
                completion,
                str,
            ):
                raise RuntimeError(
                    "GRPO assistant completion "
                    "is not a string."
                )

            # -----------------------------------------------
            # Gold classification.
            # -----------------------------------------------

            gold = infer_request.get(
                "label"
            )

            try:
                gold = int(gold)

            except Exception as e:
                raise RuntimeError(
                    "Invalid or missing `label` "
                    "in GRPO reward input."
                ) from e

            # -----------------------------------------------
            # Accuracy gate.
            # -----------------------------------------------

            pred = extract_answer(
                completion
            )

            if pred != gold:
                # Classification is wrong.
                #
                # Do not spend GenRM inference compute.
                rewards[idx] = 0.0
                continue

            # -----------------------------------------------
            # Reasoning structure gate.
            # -----------------------------------------------

            reasoning = (
                extract_reasoning_for_genrm(
                    completion
                )
            )

            if reasoning is None:
                rewards[idx] = 0.0
                continue

            # -----------------------------------------------
            # Construct GenRM request.
            # -----------------------------------------------

            rm_request = (
                self.build_genrm_request(
                    infer_request,
                    transcript,
                    reasoning,
                )
            )

            rm_inputs.append(
                rm_request
            )

            rm_indices.append(
                idx
            )

        # ====================================================
        # No correct completions in this reward batch.
        # ====================================================

        if not rm_inputs:
            return rewards

        # ====================================================
        # Batched GenRM inference.
        # ====================================================

        results = self.engine.infer(
            rm_inputs,
            self.request_config,
            use_tqdm=False,
        )

        if len(results) != len(
            rm_indices
        ):
            raise RuntimeError(
                "GenRM returned an unexpected "
                "number of outputs: "
                f"{len(results)} vs "
                f"{len(rm_indices)}"
            )

        # ====================================================
        # Convert four GenRM labels into scalar grounding.
        # ====================================================

        for original_idx, result in zip(
            rm_indices,
            results,
        ):

            try:
                response = (
                    result
                    .choices[0]
                    .message
                    .content
                )

            except Exception as e:
                raise RuntimeError(
                    "Could not read GenRM "
                    "generation output."
                ) from e

            judgment = parse_genrm_output(
                response
            )

            if judgment is None:
                raise RuntimeError(
                    "Invalid GenRM output. "
                    "Expected exact four-field JSON, got:\n"
                    f"{response!r}"
                )

            # -----------------------------------------------
            # Scalar grounding reward.
            #
            # Integration intentionally excluded.
            # -----------------------------------------------

            grounding_score = (
                judgment["text"]
                + judgment["audio"]
                + judgment["visual"]
            ) / 3.0

            rewards[
                original_idx
            ] = float(
                grounding_score
            )

        return rewards


# ============================================================
# Registration
# ============================================================

orms[
    "sarcasm_accuracy"
] = SarcasmAccuracyReward

orms[
    "sarcasm_format"
] = SarcasmFormatReward

rm_plugins[
    "sarcasm_grounding"
] = SarcasmGroundingRMPlugin


# ============================================================
# Optional linear grounding schedule
# ============================================================

import math
import os

from swift.utils import get_logger

_schedule_logger = get_logger()


def linear_grounding_factor(step, total_steps, zero_ratio=0.10, full_ratio=0.40):
    """Return a 0..1 multiplier; the external reward weight sets the cap."""
    if not (math.isfinite(zero_ratio) and math.isfinite(full_ratio)
            and 0.0 <= zero_ratio < full_ratio <= 1.0):
        raise ValueError("Require 0 <= GENRM_ZERO_RATIO < GENRM_FULL_RATIO <= 1.")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("trainer_state.global_step must be a nonnegative integer.")
    if (isinstance(total_steps, bool) or not isinstance(total_steps, int)
            or total_steps <= 0):
        raise ValueError("trainer_state.max_steps must be a positive integer.")
    progress = step / total_steps
    return min(1.0, max(0.0, (progress - zero_ratio) / (full_ratio - zero_ratio)))


class SarcasmGroundingLinearRMPlugin(SarcasmGroundingRMPlugin):
    """Apply a schedule around the unchanged, correctness-gated judge."""

    def __init__(self, model, template):
        self.zero_ratio = float(os.environ.get("GENRM_ZERO_RATIO", "0.10"))
        self.full_ratio = float(os.environ.get("GENRM_FULL_RATIO", "0.40"))
        self.log_steps = int(os.environ.get("GENRM_SCHEDULE_LOG_STEPS", "50"))
        linear_grounding_factor(0, 1, self.zero_ratio, self.full_ratio)
        if self.log_steps <= 0:
            raise ValueError("GENRM_SCHEDULE_LOG_STEPS must be positive.")
        self._last_logged_step = None
        self._last_logged_phase = None
        super().__init__(model, template)

    def __call__(self, inputs, **kwargs) -> List[float]:
        state = kwargs.get("trainer_state")
        if state is None:
            raise RuntimeError(
                "sarcasm_grounding_linear requires trainer_state in reward kwargs. "
                "Check the installed ms-swift reward-model call path. "
                "Do not substitute a local call counter. For standalone raw "
                "reasoning evaluation use the original sarcasm_grounding plugin "
                "or your separate ungated reasoning evaluator."
            )
        # Capture once: every completion in this call receives the same factor.
        step = getattr(state, "global_step", None)
        total_steps = getattr(state, "max_steps", None)
        factor = linear_grounding_factor(
            step, total_steps, self.zero_ratio, self.full_ratio
        )
        progress = step / total_steps
        phase = "zero" if factor == 0.0 else ("full" if factor == 1.0 else "ramp")

        if factor == 0.0:
            # No judge inference. No raw grounding measurement is available.
            raw_rewards = None
            rewards = [0.0] * len(inputs)
        else:
            # Preserves all original validation, gates, prompting and parsing.
            raw_rewards = super().__call__(inputs, **kwargs)
            rewards = [factor * score for score in raw_rewards]

        is_main = getattr(state, "is_world_process_zero", None)
        if is_main is None:
            is_main = os.environ.get("RANK", "0") == "0"
        should_log = (
            self._last_logged_step is None
            or phase != self._last_logged_phase
            or (step % self.log_steps == 0 and step != self._last_logged_step)
        )
        if is_main and should_log:
            payload = {
                "step": step,
                "total_steps": total_steps,
                "progress": progress,
                "phase": phase,
                "factor": factor,
                "zero_ratio": self.zero_ratio,
                "full_ratio": self.full_ratio,
                "local_batch_size": len(inputs),
                "skipped_for_zero_weight": factor == 0.0,
                "raw_gated_mean": (
                    sum(raw_rewards) / len(raw_rewards)
                    if raw_rewards else None
                ),
                "scheduled_gated_mean": (
                    sum(rewards) / len(rewards) if rewards else None
                ),
            }
            _schedule_logger.info("[genrm_schedule] " + json.dumps(payload))
            self._last_logged_step = step
            self._last_logged_phase = phase
        return rewards


rm_plugins["sarcasm_grounding_linear"] = SarcasmGroundingLinearRMPlugin