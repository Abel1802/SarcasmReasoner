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
# Trust-aware + linear-scheduled multimodal GenRM
# ============================================================
#
# Immediate diagnostic experiment:
#   - keep the SAME linear schedule as the previous run;
#   - keep the SAME external max grounding weight (normally 0.2);
#   - add a hard group-level trust gate;
#   - disable grounding for all-correct groups so GenRM cannot become
#     the sole within-group advantage signal.
#
# Environment:
#   export GENRM_NUM_GENERATIONS=8
#   export GENRM_ZERO_RATIO=0.10
#   export GENRM_FULL_RATIO=0.40
#   export GENRM_TRUST_MARGIN=0.0
#   export GENRM_GROUP_LOG_STEPS=25
#   export GENRM_REQUIRE_SINGLE_PROCESS=1
#
# Training:
#   --reward_funcs sarcasm_accuracy sarcasm_format
#   --reward_model_plugin sarcasm_grounding_trust_linear
#   --reward_weights 1.0 0.2 0.2
#
# Keep every other GRPO hyperparameter identical to the existing linear run.
# ============================================================

import math
import os

from swift.utils import get_logger

_trust_logger = get_logger()


def linear_grounding_factor(step, total_steps, zero_ratio=0.10, full_ratio=0.40):
    """Return a 0..1 multiplier; the external reward weight sets the cap."""
    if not (
        math.isfinite(zero_ratio)
        and math.isfinite(full_ratio)
        and 0.0 <= zero_ratio < full_ratio <= 1.0
    ):
        raise ValueError("Require 0 <= GENRM_ZERO_RATIO < GENRM_FULL_RATIO <= 1.")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("trainer_state.global_step must be a nonnegative integer.")
    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps <= 0
    ):
        raise ValueError("trainer_state.max_steps must be a positive integer.")
    progress = step / total_steps
    return min(1.0, max(0.0, (progress - zero_ratio) / (full_ratio - zero_ratio)))


def _get_completion_and_gold(infer_request):
    """Return (completion, gold, parsed_prediction)."""
    messages = infer_request.get("messages", [])
    if not (isinstance(messages, list) and messages):
        raise RuntimeError("Missing `messages` in GRPO reward input.")

    last_message = messages[-1]
    if not (
        isinstance(last_message, dict)
        and last_message.get("role") == "assistant"
    ):
        raise RuntimeError(
            "Expected the final GRPO message to be an assistant completion."
        )

    completion = last_message.get("content", "")
    if not isinstance(completion, str):
        raise RuntimeError("GRPO assistant completion is not a string.")

    gold = infer_request.get("label")
    try:
        gold = int(gold)
    except Exception as e:
        raise RuntimeError("Invalid or missing `label` in GRPO reward input.") from e

    if gold not in (0, 1):
        raise RuntimeError(f"`label` must be 0/1, got {gold!r}.")

    return completion, gold, extract_answer(completion)


def _group_signature(infer_request):
    """Lightweight check that a contiguous block is one repeated GRPO prompt."""
    transcript = get_transcript(infer_request)
    gold = infer_request.get("label")
    try:
        gold = int(gold)
    except Exception as e:
        raise RuntimeError("Invalid `label` while checking GRPO group structure.") from e
    return transcript, gold


class SarcasmGroundingTrustLinearRMPlugin(SarcasmGroundingRMPlugin):
    """Linear curriculum + hard group-level trust gate.

    For each GRPO group of size G:

      k == 0 (all wrong):
          grounding OFF.

      0 < k < G (mixed):
          Run GenRM on structurally valid correct AND wrong completions.
          Let delta = mean(raw GenRM | correct) - mean(raw GenRM | wrong).
          The group is trusted iff delta > GENRM_TRUST_MARGIN.
          If trusted, only correct completions receive their raw GenRM score;
          wrong completions still receive zero grounding reward.
          If untrusted, grounding is zero for the whole group.

      k == G (all correct):
          grounding OFF in this diagnostic version. This deliberately prevents
          GenRM from becoming the only source of within-group advantage when
          outcome correctness has zero variance.

    The previous linear factor is then applied. The external Trainer weight
    (normally 0.2) is applied exactly once outside this plugin.
    """

    def __init__(self, model, template):
        self.num_generations = int(os.environ.get("GENRM_NUM_GENERATIONS", "8"))
        self.zero_ratio = float(os.environ.get("GENRM_ZERO_RATIO", "0.10"))
        self.full_ratio = float(os.environ.get("GENRM_FULL_RATIO", "0.40"))
        self.trust_margin = float(os.environ.get("GENRM_TRUST_MARGIN", "0.0"))
        self.log_steps = int(os.environ.get("GENRM_GROUP_LOG_STEPS", "25"))
        self.require_single_process = os.environ.get(
            "GENRM_REQUIRE_SINGLE_PROCESS", "1"
        ) not in ("0", "false", "False")

        if self.num_generations <= 1:
            raise ValueError("GENRM_NUM_GENERATIONS must be > 1.")
        if not math.isfinite(self.trust_margin):
            raise ValueError("GENRM_TRUST_MARGIN must be finite.")
        if self.log_steps <= 0:
            raise ValueError("GENRM_GROUP_LOG_STEPS must be positive.")

        linear_grounding_factor(0, 1, self.zero_ratio, self.full_ratio)

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.require_single_process and world_size != 1:
            raise RuntimeError(
                "sarcasm_grounding_trust_linear currently requires WORLD_SIZE=1 "
                "so each reward-model call sees complete GRPO groups. Use the "
                "existing 1-GPU diagnostic configuration, or implement a trainer-level "
                "cross-rank gather before multi-GPU use."
            )

        self._last_logged_step = None
        self._last_logged_phase = None
        super().__init__(model, template)

    def _score_raw_genrm(self, prepared):
        """prepared: [(original_idx, rm_request), ...] -> {idx: raw_score}."""
        if not prepared:
            return {}

        rm_indices = [idx for idx, _ in prepared]
        rm_inputs = [request for _, request in prepared]
        results = self.engine.infer(
            rm_inputs,
            self.request_config,
            use_tqdm=False,
        )

        if len(results) != len(rm_indices):
            raise RuntimeError(
                "GenRM returned an unexpected number of outputs: "
                f"{len(results)} vs {len(rm_indices)}"
            )

        raw_scores = {}
        for original_idx, result in zip(rm_indices, results):
            try:
                response = result.choices[0].message.content
            except Exception as e:
                raise RuntimeError("Could not read GenRM generation output.") from e

            judgment = parse_genrm_output(response)
            if judgment is None:
                raise RuntimeError(
                    "Invalid GenRM output. Expected exact four-field JSON, got:\n"
                    f"{response!r}"
                )

            # Keep exactly the same scalar definition as previous experiments.
            # Integration remains excluded.
            grounding_score = (
                judgment["text"] + judgment["audio"] + judgment["visual"]
            ) / 3.0
            raw_scores[original_idx] = float(grounding_score)

        return raw_scores

    def __call__(self, inputs, **kwargs) -> List[float]:
        state = kwargs.get("trainer_state")
        if state is None:
            raise RuntimeError(
                "sarcasm_grounding_trust_linear requires trainer_state in reward kwargs."
            )

        step = getattr(state, "global_step", None)
        total_steps = getattr(state, "max_steps", None)
        factor = linear_grounding_factor(
            step,
            total_steps,
            self.zero_ratio,
            self.full_ratio,
        )
        progress = step / total_steps
        phase = "zero" if factor == 0.0 else ("full" if factor == 1.0 else "ramp")

        n = len(inputs)
        rewards = [0.0] * n

        # Preserve the previous linear run exactly during warm-up: no GenRM call.
        if factor == 0.0:
            self._maybe_log(
                state=state,
                step=step,
                total_steps=total_steps,
                progress=progress,
                phase=phase,
                factor=factor,
                inputs=inputs,
                rewards=rewards,
                group_stats=None,
                skipped_for_zero_weight=True,
            )
            return rewards

        if n == 0:
            return rewards

        G = self.num_generations
        if n % G != 0:
            raise RuntimeError(
                "Trust-aware GenRM requires complete contiguous GRPO groups, but "
                f"reward batch size {n} is not divisible by GENRM_NUM_GENERATIONS={G}."
            )

        parsed_items = []
        for idx, infer_request in enumerate(inputs):
            transcript = get_transcript(infer_request)
            get_audio(infer_request)
            get_video(infer_request)

            completion, gold, pred = _get_completion_and_gold(infer_request)
            reasoning = extract_reasoning_for_genrm(completion)
            parsed_items.append(
                {
                    "idx": idx,
                    "infer_request": infer_request,
                    "transcript": transcript,
                    "gold": gold,
                    "pred": pred,
                    "correct": pred == gold,
                    "reasoning": reasoning,
                }
            )

        groups = []
        prepared_for_rm = []
        group_stats = {
            "num_groups": 0,
            "all_wrong_groups": 0,
            "mixed_groups": 0,
            "all_correct_groups": 0,
            "mixed_trusted_groups": 0,
            "mixed_untrusted_groups": 0,
            "mixed_unscorable_groups": 0,
            "valid_raw_correct": 0,
            "valid_raw_wrong": 0,
            "deltas": [],
            "raw_correct_scores": [],
            "raw_wrong_scores": [],
        }

        # First pass: identify group type and collect mixed-group responses for GenRM.
        for start in range(0, n, G):
            stop = start + G
            group = parsed_items[start:stop]

            signatures = {
                _group_signature(item["infer_request"])
                for item in group
            }
            if len(signatures) != 1:
                raise RuntimeError(
                    "The reward batch is not arranged as contiguous GRPO groups: "
                    "items inside one inferred group have different (transcript, label) "
                    "signatures. Do not use index-based group gating with this layout."
                )

            correct_count = sum(1 for item in group if item["correct"])
            group_stats["num_groups"] += 1

            if correct_count == 0:
                group_type = "all_wrong"
                group_stats["all_wrong_groups"] += 1
            elif correct_count == G:
                group_type = "all_correct"
                group_stats["all_correct_groups"] += 1
            else:
                group_type = "mixed"
                group_stats["mixed_groups"] += 1

                # IMPORTANT: raw trust estimation sees both correct and wrong reasoning.
                for item in group:
                    reasoning = item["reasoning"]
                    if reasoning is None:
                        continue
                    rm_request = self.build_genrm_request(
                        item["infer_request"],
                        item["transcript"],
                        reasoning,
                    )
                    prepared_for_rm.append((item["idx"], rm_request))

            groups.append(
                {
                    "start": start,
                    "stop": stop,
                    "type": group_type,
                    "correct_count": correct_count,
                }
            )

        # One GenRM batch for every scorable response in mixed groups.
        raw_scores = self._score_raw_genrm(prepared_for_rm)

        # Second pass: compute trust and apply downstream correctness gate.
        for meta in groups:
            if meta["type"] != "mixed":
                # Both all-wrong and all-correct groups are deliberately zero here.
                continue

            group = parsed_items[meta["start"]:meta["stop"]]
            correct_scores = []
            wrong_scores = []

            for item in group:
                score = raw_scores.get(item["idx"])
                if score is None:
                    continue
                if item["correct"]:
                    correct_scores.append(score)
                else:
                    wrong_scores.append(score)

            # Cannot estimate alignment if either side has no valid GenRM score.
            if not correct_scores or not wrong_scores:
                group_stats["mixed_unscorable_groups"] += 1
                continue

            mean_correct = sum(correct_scores) / len(correct_scores)
            mean_wrong = sum(wrong_scores) / len(wrong_scores)
            delta = mean_correct - mean_wrong

            group_stats["valid_raw_correct"] += len(correct_scores)
            group_stats["valid_raw_wrong"] += len(wrong_scores)
            group_stats["raw_correct_scores"].extend(correct_scores)
            group_stats["raw_wrong_scores"].extend(wrong_scores)
            group_stats["deltas"].append(delta)

            # Hard gate is intentional. A positive scalar trust multiplier can be
            # largely canceled by GRPO group normalization; OFF vs ON cannot.
            trusted = delta > self.trust_margin
            if not trusted:
                group_stats["mixed_untrusted_groups"] += 1
                continue

            group_stats["mixed_trusted_groups"] += 1

            # Correctness gate is applied AFTER raw trust estimation.
            for item in group:
                if not item["correct"]:
                    continue
                score = raw_scores.get(item["idx"])
                if score is None:
                    continue
                rewards[item["idx"]] = float(factor * score)

        self._maybe_log(
            state=state,
            step=step,
            total_steps=total_steps,
            progress=progress,
            phase=phase,
            factor=factor,
            inputs=inputs,
            rewards=rewards,
            group_stats=group_stats,
            skipped_for_zero_weight=False,
        )
        return rewards

    def _maybe_log(
        self,
        *,
        state,
        step,
        total_steps,
        progress,
        phase,
        factor,
        inputs,
        rewards,
        group_stats,
        skipped_for_zero_weight,
    ):
        is_main = getattr(state, "is_world_process_zero", None)
        if is_main is None:
            is_main = os.environ.get("RANK", "0") == "0"

        should_log = (
            self._last_logged_step is None
            or phase != self._last_logged_phase
            or (
                step % self.log_steps == 0
                and step != self._last_logged_step
            )
        )
        if not (is_main and should_log):
            return

        payload = {
            "step": step,
            "total_steps": total_steps,
            "progress": progress,
            "phase": phase,
            "factor": factor,
            "zero_ratio": self.zero_ratio,
            "full_ratio": self.full_ratio,
            "trust_margin": self.trust_margin,
            "num_generations": self.num_generations,
            "local_batch_size": len(inputs),
            "skipped_for_zero_weight": skipped_for_zero_weight,
            "scheduled_grounding_mean": (
                sum(rewards) / len(rewards) if rewards else None
            ),
        }

        if group_stats is not None:
            deltas = group_stats["deltas"]
            c_scores = group_stats["raw_correct_scores"]
            w_scores = group_stats["raw_wrong_scores"]
            scorable_mixed = (
                group_stats["mixed_trusted_groups"]
                + group_stats["mixed_untrusted_groups"]
            )
            payload.update(
                {
                    "num_groups": group_stats["num_groups"],
                    "all_wrong_groups": group_stats["all_wrong_groups"],
                    "mixed_groups": group_stats["mixed_groups"],
                    "all_correct_groups": group_stats["all_correct_groups"],
                    "mixed_trusted_groups": group_stats["mixed_trusted_groups"],
                    "mixed_untrusted_groups": group_stats["mixed_untrusted_groups"],
                    "mixed_unscorable_groups": group_stats["mixed_unscorable_groups"],
                    "trust_active_rate_over_scorable_mixed": (
                        group_stats["mixed_trusted_groups"] / scorable_mixed
                        if scorable_mixed
                        else None
                    ),
                    "mean_delta_correct_minus_wrong": (
                        sum(deltas) / len(deltas) if deltas else None
                    ),
                    "min_delta_correct_minus_wrong": min(deltas) if deltas else None,
                    "max_delta_correct_minus_wrong": max(deltas) if deltas else None,
                    "raw_correct_mean": sum(c_scores) / len(c_scores) if c_scores else None,
                    "raw_wrong_mean": sum(w_scores) / len(w_scores) if w_scores else None,
                    "valid_raw_correct": group_stats["valid_raw_correct"],
                    "valid_raw_wrong": group_stats["valid_raw_wrong"],
                }
            )

        _trust_logger.info(
            "[genrm_trust_linear] "
            + json.dumps(payload, ensure_ascii=False)
        )
        self._last_logged_step = step
        self._last_logged_phase = phase


rm_plugins["sarcasm_grounding_trust_linear"] = SarcasmGroundingTrustLinearRMPlugin