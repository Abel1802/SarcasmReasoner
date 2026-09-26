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
