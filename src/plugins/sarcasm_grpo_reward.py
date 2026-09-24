import re
from typing import List

from swift.rewards import ORM, orms


REQUIRED_TAGS = [
    "text_evidence",
    "audio_evidence",
    "visual_evidence",
    "integration",
    "answer",
]


def extract_answer(text: str):
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if len(matches) != 1:
        return None

    answer = matches[0].strip().lower()

    if answer == "sarcasm":
        return 1

    if answer == "non-sarcasm":
        return 0

    return None


class SarcasmAccuracyReward(ORM):

    def __call__(
        self,
        completions,
        label,
        **kwargs,
    ) -> List[float]:

        rewards = []

        for completion, gold in zip(completions, label):

            pred = extract_answer(completion)

            try:
                gold = int(gold)
            except Exception:
                rewards.append(0.0)
                continue

            reward = 1.0 if pred == gold else 0.0

            rewards.append(reward)

        return rewards


class SarcasmFormatReward(ORM):

    def __call__(
        self,
        completions,
        **kwargs,
    ) -> List[float]:

        rewards = []

        for completion in completions:

            valid = True

            for tag in REQUIRED_TAGS:

                matches = re.findall(
                    rf"<{tag}>\s*(.*?)\s*</{tag}>",
                    completion,
                    flags=re.IGNORECASE | re.DOTALL,
                )

                if len(matches) != 1:
                    valid = False
                    break

                if not matches[0].strip():
                    valid = False
                    break

            if valid:
                pred = extract_answer(completion)
                if pred is None:
                    valid = False

            rewards.append(1.0 if valid else 0.0)

        return rewards


orms["sarcasm_accuracy"] = SarcasmAccuracyReward
orms["sarcasm_format"] = SarcasmFormatReward