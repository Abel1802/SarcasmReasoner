#!/usr/bin/env python3

import json
from pathlib import Path


INPUT = Path(
    "data/mustard/processed/genrm/"
    "judge_pool_smoke100.jsonl"
)

PROMPT = Path(
    "src/prompts/grounding_judge/"
    "integration_grounding_binary_final.txt"
)

OUTPUT_DIR = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary"
)

OUTPUT = (
    OUTPUT_DIR
    / "integration_hardneg5_input.jsonl"
)

GOLD_OUTPUT = (
    OUTPUT_DIR
    / "integration_hardneg5_gold.jsonl"
)


# ------------------------------------------------------------
# These integrations are intentionally wrong.
#
# They either:
# - invent evidence,
# - contradict the provided modality evidence,
# - or materially misrepresent a modality.
#
# Therefore ALL gold labels are UNSUPPORTED = 0.
# ------------------------------------------------------------

HARD_NEGATIVES = {

    907: {
        "corruption_type": "INVENTED_EVIDENCE",
        "integration": (
            "The text contains clear ironic framing, while the audio uses "
            "an exaggerated mocking intonation and the video shows a visible "
            "smirk. These cues jointly provide strong evidence of sarcasm."
        ),
    },

    144: {
        "corruption_type": "EVIDENCE_CONTRADICTION",
        "integration": (
            "All three modalities are neutral and literal. The text contains "
            "no evaluative contrast, the audio contains no mocking or "
            "insincere delivery, and the visual evidence shows no non-sincere "
            "expression."
        ),
    },

    333: {
        "corruption_type": "MODALITY_MISREPRESENTATION",
        "integration": (
            "The audio explicitly contains no laughter or chuckling, and the "
            "speaker maintains a neutral facial expression without smiling. "
            "The modalities therefore provide no expressive cues relevant to "
            "the interpretation."
        ),
    },

    5735: {
        "corruption_type": "INVENTED_EVIDENCE",
        "integration": (
            "The speaker uses an excited, highly emotional vocal delivery "
            "with strong pitch variation, while the visual evidence shows a "
            "broad grin and exaggerated gestures. These expressive cues "
            "reinforce the interpretation."
        ),
    },

    5918: {
        "corruption_type": "EVIDENCE_CONTRADICTION",
        "integration": (
            "The text explicitly states an opposite meaning, the audio is "
            "delivered in a flat and deadpan manner, and the visual evidence "
            "contains no relevant facial expression. All three modalities "
            "therefore directly support the same interpretation."
        ),
    },
}


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


def write_jsonl(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    rows = read_jsonl(
        INPUT
    )

    by_id = {
        int(x["judge_id"]): x
        for x in rows
    }

    template = PROMPT.read_text(
        encoding="utf-8"
    )

    output_rows = []
    gold_rows = []

    for judge_id, corruption in HARD_NEGATIVES.items():

        if judge_id not in by_id:

            raise RuntimeError(
                f"Missing judge_id={judge_id}"
            )

        row = by_id[
            judge_id
        ]

        sections = row[
            "sections"
        ]

        prompt = (
            template
            .replace(
                "{{TEXT_EVIDENCE}}",
                sections[
                    "text_evidence"
                ],
            )
            .replace(
                "{{AUDIO_EVIDENCE}}",
                sections[
                    "audio_evidence"
                ],
            )
            .replace(
                "{{VISUAL_EVIDENCE}}",
                sections[
                    "visual_evidence"
                ],
            )
            .replace(
                "{{INTEGRATION}}",
                corruption[
                    "integration"
                ],
            )
        )

        output_rows.append({

            "id":
                str(judge_id),

            "source_id":
                row["source_id"],

            "corruption_type":
                corruption[
                    "corruption_type"
                ],

            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            "original_integration":
                sections[
                    "integration"
                ],

            "corrupted_integration":
                corruption[
                    "integration"
                ],
        })

        gold_rows.append({

            "judge_id":
                judge_id,

            "source_id":
                row["source_id"],

            "gold_grounded":
                0,

            "gold_verdict":
                "UNSUPPORTED",

            "corruption_type":
                corruption[
                    "corruption_type"
                ],
        })

    write_jsonl(
        OUTPUT,
        output_rows,
    )

    write_jsonl(
        GOLD_OUTPUT,
        gold_rows,
    )

    print("=" * 80)
    print(
        "Integration hard-negative sanity set"
    )
    print("=" * 80)

    for row in output_rows:

        print()
        print(
            "judge_id:",
            row["id"],
        )

        print(
            "corruption:",
            row[
                "corruption_type"
            ],
        )

        print(
            "candidate integration:"
        )

        print(
            row[
                "corrupted_integration"
            ]
        )

    print()
    print(
        "Saved:",
        OUTPUT,
    )

    print(
        "Saved:",
        GOLD_OUTPUT,
    )


if __name__ == "__main__":
    main()
