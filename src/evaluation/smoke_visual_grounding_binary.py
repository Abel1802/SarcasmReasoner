#!/usr/bin/env python3

import ast
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


MODEL = "Qwen/Qwen3-VL-8B-Instruct"

INPUT = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/"
    "visual_input.jsonl"
)

GOLD = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/"
    "human_gold.jsonl"
)

OUTPUT = Path(
    "results/mustard/genrm/"
    "judge_calibration10_binary/"
    "visual_qwen3_vl_8b_binary_v4_final_pred.jsonl"
)


VALID_ISSUES = {
    "NONE",
    "OVERINTERPRETATION",
    "HALLUCINATED_CUE",
    "CONTRADICTED_BY_INPUT",
    "ENTITY_BINDING_ERROR",
    "NOT_ASSESSABLE",
}


def read_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def parse_output(text):

    raw = text.strip()

    candidates = [raw]

    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.DOTALL,
    )

    if match:
        candidates.append(
            match.group(0)
        )

    if '\\"' in raw:
        candidates.append(
            raw.replace('\\"', '"')
        )

    obj = None

    for candidate in candidates:

        try:
            value = json.loads(
                candidate
            )

            if isinstance(value, dict):
                obj = value
                break

        except Exception:
            pass

        try:
            value = ast.literal_eval(
                candidate
            )

            if isinstance(value, dict):
                obj = value
                break

        except Exception:
            pass

    if obj is None:
        return {
            "parse_ok": False,
            "grounded": None,
            "verdict": None,
            "issue": None,
            "issue_valid": False,
            "rationale": None,
        }

    verdict = str(
        obj.get(
            "verdict",
            "",
        )
    ).strip().upper()

    mapping = {
        "SUPPORTED": 1,
        "UNSUPPORTED": 0,
    }

    grounded = mapping.get(
        verdict
    )

    issue = obj.get(
        "issue"
    )

    if issue is None:
        issue = "NONE"
    else:
        issue = (
            str(issue)
            .strip()
            .upper()
        )

    aliases = {
        "OVERINTERPRETED":
            "OVERINTERPRETATION",
        "HALLUCINATION":
            "HALLUCINATED_CUE",
        "ENTITY_BINDING_ERROR":
            "WRONG_PERSON",
        "MISATTRIBUTION":
            "WRONG_PERSON",
    }

    issue = aliases.get(
        issue,
        issue,
    )

    valid_issues = {
        "NONE",
        "OVERINTERPRETATION",
        "HALLUCINATED_CUE",
        "WRONG_PERSON",
        "NOT_ASSESSABLE",
    }

    rationale = obj.get(
        "evidence",
        obj.get(
            "rationale",
            "",
        ),
    )

    return {
        "parse_ok":
            grounded is not None,

        "grounded":
            grounded,

        "verdict":
            verdict,

        "issue":
            issue,

        "issue_valid":
            issue in valid_issues,

        "rationale":
            rationale,
    }

def main():

    rows = read_jsonl(INPUT)
    gold_rows = read_jsonl(GOLD)

    if len(rows) != 10:
        raise RuntimeError(
            f"Expected 10 rows, got {len(rows)}"
        )

    gold_by_id = {
        str(x["judge_id"]): x
        for x in gold_rows
    }

    print("=" * 96)
    print(
        "Qwen3-VL visual grounding judge: "
        "binary calibration10"
    )
    print("=" * 96)
    print("Model:", MODEL)
    print()

    # --------------------------------------------------------
    # Load once
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL
    )

    model = (
        AutoModelForImageTextToText
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

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # --------------------------------------------------------
    # Run 10
    # --------------------------------------------------------

    for idx, row in enumerate(
        rows,
        start=1,
    ):

        judge_id = str(row["id"])

        human = int(
            gold_by_id[judge_id][
                "human_visual_grounded"
            ]
        )

        video_path = row["videos"][0]

        prompt = row[
            "messages"
        ][0]["content"]

        # The actual video is inserted separately.
        # Remove the textual placeholder.
        prompt = prompt.replace(
            "<video>",
            "",
        ).strip()

        abs_video = str(
            Path(video_path).resolve()
        )

        print()
        print("=" * 96)
        print(
            f"[{idx}/10] "
            f"id={judge_id} "
            f"human={human}"
        )
        print("=" * 96)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": abs_video,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # fps=2 because the clips are short and
        # subtle facial/gestural cues matter.
        inputs = (
            processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                fps=2,
            )
        )

        inputs = inputs.to(
            model.device
        )

        with torch.no_grad():

            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_trimmed = [
            out_ids[
                len(in_ids):
            ]
            for in_ids, out_ids
            in zip(
                inputs.input_ids,
                generated,
            )
        ]

        raw = (
            processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
        )

        parsed = parse_output(
            raw
        )

        result = {
            "judge_id":
                int(judge_id),

            "video":
                video_path,

            "human_grounded":
                human,

            "pred_grounded":
                parsed["grounded"],

            "pred_issue":
                parsed["issue"],

            "issue_valid":
                parsed["issue_valid"],

            "parse_ok":
                parsed["parse_ok"],

            "rationale":
                parsed["rationale"],

            "raw_output":
                raw,

            "judge_model":
                MODEL,

            "fps":
                4,
        }

        results.append(result)

        print("raw:")
        print(raw)

        print()
        print(
            "human:",
            human,
            "pred:",
            parsed["grounded"],
            "match:",
            human
            == parsed["grounded"],
        )

        # Incremental save.
        with open(
            OUTPUT,
            "w",
            encoding="utf-8",
        ) as f:

            for x in results:
                f.write(
                    json.dumps(
                        x,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    valid = [
        x
        for x in results
        if x["pred_grounded"]
        is not None
    ]

    print()
    print("=" * 96)
    print("FINAL COMPARISON")
    print("=" * 96)

    print(
        f"{'ID':>7}  "
        f"{'Human':>5}  "
        f"{'Pred':>4}  "
        f"{'Issue':<25}  "
        f"{'Match':>5}"
    )

    print("-" * 96)

    for x in results:

        match = (
            x["human_grounded"]
            == x["pred_grounded"]
        )

        print(
            f"{x['judge_id']:>7}  "
            f"{x['human_grounded']:>5}  "
            f"{str(x['pred_grounded']):>4}  "
            f"{str(x['pred_issue']):<25}  "
            f"{str(match):>5}"
        )

    if not valid:
        print("\nNo valid predictions.")
        return

    human = [
        x["human_grounded"]
        for x in valid
    ]

    pred = [
        x["pred_grounded"]
        for x in valid
    ]

    agreement = sum(
        h == p
        for h, p in zip(
            human,
            pred,
        )
    ) / len(valid)

    print()
    print(
        f"Parsed:     "
        f"{len(valid)}/{len(results)}"
    )

    print(
        f"Agreement:  "
        f"{agreement:.3f}"
    )

    try:
        from sklearn.metrics import (
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
        )

        kappa = cohen_kappa_score(
            human,
            pred,
        )

        macro_f1 = f1_score(
            human,
            pred,
            average="macro",
        )

        cm = confusion_matrix(
            human,
            pred,
            labels=[0, 1],
        )

        print(
            f"Macro-F1:  "
            f"{macro_f1:.3f}"
        )

        print(
            f"Cohen κ:   "
            f"{kappa:.3f}"
        )

        print()
        print(
            "Confusion matrix "
            "(rows=human, cols=pred):"
        )

        print(cm)

    except Exception as exc:
        print(
            "sklearn metrics unavailable:",
            exc,
        )

    print()
    print("Output:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
