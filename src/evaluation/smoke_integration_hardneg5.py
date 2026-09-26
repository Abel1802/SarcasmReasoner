#!/usr/bin/env python3

import ast
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL = "Qwen/Qwen2.5-7B-Instruct"

BASE = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary"
)

INPUT = (
    BASE
    / "integration_hardneg5_input.jsonl"
)

GOLD = (
    BASE
    / "integration_hardneg5_gold.jsonl"
)

OUTPUT = Path(
    "results/mustard/genrm/"
    "judge_calibration10_binary/"
    "integration_hardneg5_qwen2_5_7b_pred.jsonl"
)


VALID_ISSUES = {
    "NONE",
    "OVERINTERPRETATION",
    "UNSUPPORTED_INFERENCE",
    "EVIDENCE_CONTRADICTION",
    "INVENTED_EVIDENCE",
    "MODALITY_MISREPRESENTATION",
    "NOT_ASSESSABLE",
}


def read_jsonl(path):

    with open(
        path,
        encoding="utf-8",
    ) as f:

        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


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


def parse_output(raw):

    raw = raw.strip()

    obj = None

    try:

        obj = json.loads(
            raw
        )

    except Exception:

        try:

            obj = ast.literal_eval(
                raw
            )

        except Exception:

            pass

    if obj is None:

        match = re.search(
            r"""['"]verdict['"]\s*:\s*['"](SUPPORTED|UNSUPPORTED)['"]""",
            raw,
            flags=re.IGNORECASE,
        )

        issue_match = re.search(
            r"""['"]issue['"]\s*:\s*['"]([^'"]+)['"]""",
            raw,
            flags=re.IGNORECASE,
        )

        if not match:

            return {
                "parse_ok": False,
                "grounded": None,
                "verdict": None,
                "issue": None,
                "rationale": None,
            }

        verdict = (
            match
            .group(1)
            .upper()
        )

        issue = (
            issue_match
            .group(1)
            .upper()
            if issue_match
            else "NONE"
        )

        return {
            "parse_ok": True,

            "grounded":
                1
                if verdict
                == "SUPPORTED"
                else 0,

            "verdict":
                verdict,

            "issue":
                issue,

            "rationale":
                "",
        }

    verdict = str(
        obj.get(
            "verdict",
            "",
        )
    ).upper()

    issue = str(
        obj.get(
            "issue",
            "NONE",
        )
    ).upper()

    return {
        "parse_ok":
            verdict
            in {
                "SUPPORTED",
                "UNSUPPORTED",
            },

        "grounded":
            1
            if verdict
            == "SUPPORTED"
            else (
                0
                if verdict
                == "UNSUPPORTED"
                else None
            ),

        "verdict":
            verdict,

        "issue":
            issue,

        "rationale":
            obj.get(
                "evidence",
                "",
            ),
    }


def main():

    rows = read_jsonl(
        INPUT
    )

    gold_rows = read_jsonl(
        GOLD
    )

    gold_by_id = {
        str(x["judge_id"]): x
        for x in gold_rows
    }

    print("=" * 90)
    print(
        "Loading Qwen2.5-7B-Instruct"
    )
    print("=" * 90)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    results = []

    for i, row in enumerate(
        rows,
        start=1,
    ):

        judge_id = row["id"]

        gold = gold_by_id[
            judge_id
        ]

        prompt = (
            row["messages"][0][
                "content"
            ]
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(
            model.device
        )

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=(
                    tokenizer
                    .eos_token_id
                ),
            )

        new_tokens = output[
            :,
            inputs.input_ids.shape[1]:
        ]

        raw = tokenizer.decode(
            new_tokens[0],
            skip_special_tokens=True,
        ).strip()

        parsed = parse_output(
            raw
        )

        correct = (
            parsed[
                "grounded"
            ]
            == gold[
                "gold_grounded"
            ]
        )

        result = {

            "judge_id":
                int(judge_id),

            "corruption_type":
                row[
                    "corruption_type"
                ],

            "gold_grounded":
                gold[
                    "gold_grounded"
                ],

            "pred_grounded":
                parsed[
                    "grounded"
                ],

            "pred_verdict":
                parsed[
                    "verdict"
                ],

            "pred_issue":
                parsed[
                    "issue"
                ],

            "issue_valid":
                parsed[
                    "issue"
                ]
                in VALID_ISSUES,

            "parse_ok":
                parsed[
                    "parse_ok"
                ],

            "correct":
                correct,

            "rationale":
                parsed[
                    "rationale"
                ],

            "raw_output":
                raw,

            "judge_model":
                MODEL,
        }

        results.append(
            result
        )

        print()
        print(
            "=" * 90
        )

        print(
            f"[{i}/5] "
            f"id={judge_id} "
            f"type="
            f"{row['corruption_type']}"
        )

        print(
            "=" * 90
        )

        print(raw)

        print()
        print(
            "gold:",
            gold[
                "gold_grounded"
            ],
            "pred:",
            parsed[
                "grounded"
            ],
            "correct:",
            correct,
        )

        write_jsonl(
            OUTPUT,
            results,
        )

    correct_n = sum(
        x["correct"]
        for x in results
    )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"Correct: {correct_n}/5"
    )

    print(
        "UNSUPPORTED detected:",
        sum(
            x[
                "pred_grounded"
            ]
            == 0
            for x in results
        ),
        "/5",
    )

    print(
        "Parse success:",
        sum(
            x[
                "parse_ok"
            ]
            for x in results
        ),
        "/5",
    )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
