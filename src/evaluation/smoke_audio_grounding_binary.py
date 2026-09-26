#!/usr/bin/env python3

import ast
import json
import re
from pathlib import Path

import librosa
import torch

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)


MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

INPUT = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/"
    "audio_input.jsonl"
)

GOLD = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/"
    "human_gold.jsonl"
)

OUTPUT = Path(
    "results/mustard/genrm/"
    "judge_calibration10_binary/"
    "audio_qwen2_audio_7b_binary_v3_final_pred.jsonl"
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
            if line.strip():
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

    # --------------------------------------------------------
    # 1. Normal JSON / Python-dict parsing
    # --------------------------------------------------------

    for candidate in candidates:

        try:
            value = json.loads(candidate)

            if isinstance(value, dict):
                obj = value
                break

        except Exception:
            pass

        try:
            value = ast.literal_eval(candidate)

            if isinstance(value, dict):
                obj = value
                break

        except Exception:
            pass

    # --------------------------------------------------------
    # 2. Regex fallback
    #
    # Qwen2-Audio sometimes emits Python-style dictionaries
    # containing apostrophes inside the evidence string:
    #
    # {'verdict': 'SUPPORTED',
    #  'evidence': 'The speaker's tone ...'}
    #
    # Such output can break ast.literal_eval(), even though
    # verdict and issue are still perfectly recoverable.
    # --------------------------------------------------------

    if obj is None:

        verdict_match = re.search(
            r"""['"]verdict['"]\s*:\s*['"](SUPPORTED|UNSUPPORTED)['"]""",
            raw,
            flags=re.IGNORECASE,
        )

        issue_match = re.search(
            r"""['"]issue['"]\s*:\s*['"]([^'"]+)['"]""",
            raw,
            flags=re.IGNORECASE,
        )

        if verdict_match:

            verdict = (
                verdict_match
                .group(1)
                .strip()
                .upper()
            )

            grounded = {
                "SUPPORTED": 1,
                "UNSUPPORTED": 0,
            }.get(verdict)

            issue = (
                issue_match
                .group(1)
                .strip()
                .upper()
                if issue_match
                else "NONE"
            )

            aliases = {
                "OVERINTERPRETED":
                    "OVERINTERPRETATION",
                "HALLUCINATION":
                    "HALLUCINATED_CUE",
            }

            issue = aliases.get(
                issue,
                issue,
            )

            valid_issues = {
                "NONE",
                "OVERINTERPRETATION",
                "HALLUCINATED_CUE",
                "CONTRADICTED_BY_INPUT",
                "NOT_ASSESSABLE",
            }

            # Evidence extraction is intentionally optional.
            # Verdict recovery is more important than perfectly
            # parsing arbitrary apostrophes in free text.
            evidence = ""

            evidence_match = re.search(
                r"""['"]evidence['"]\s*:\s*['"](.+?)['"]\s*\}?\s*$""",
                raw,
                flags=re.DOTALL,
            )

            if evidence_match:
                evidence = (
                    evidence_match
                    .group(1)
                    .strip()
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
                    evidence,
            }

        return {
            "parse_ok": False,
            "grounded": None,
            "verdict": None,
            "issue": None,
            "issue_valid": False,
            "rationale": None,
        }

    # --------------------------------------------------------
    # 3. Parsed object
    # --------------------------------------------------------

    verdict = str(
        obj.get("verdict", "")
    ).strip().upper()

    verdict_map = {
        "SUPPORTED": 1,
        "UNSUPPORTED": 0,
    }

    grounded = verdict_map.get(
        verdict
    )

    issue = obj.get("issue")

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
    }

    issue = aliases.get(
        issue,
        issue,
    )

    valid_issues = {
        "NONE",
        "OVERINTERPRETATION",
        "HALLUCINATED_CUE",
        "CONTRADICTED_BY_INPUT",
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

    gold_by_id = {
        str(x["judge_id"]): x
        for x in gold_rows
    }

    processor = (
        AutoProcessor.from_pretrained(
            MODEL
        )
    )

    model = (
        Qwen2AudioForConditionalGeneration
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

    for idx, row in enumerate(
        rows,
        start=1,
    ):

        judge_id = str(row["id"])

        human = int(
            gold_by_id[judge_id][
                "human_audio_grounded"
            ]
        )

        audio_path = row["audios"][0]
        prompt = row["messages"][0][
            "content"
        ]

        print()
        print("=" * 88)
        print(
            f"[{idx}/10] "
            f"id={judge_id} "
            f"human={human}"
        )
        print("=" * 88)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url":
                            audio_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = (
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        target_sr = (
            processor
            .feature_extractor
            .sampling_rate
        )

        audio, sr = librosa.load(
            audio_path,
            sr=target_sr,
            mono=True,
        )

        inputs = processor(
            text=text,
            audios=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            k: (
                v.to(model.device)
                if torch.is_tensor(v)
                else v
            )
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated = generated[
            :,
            inputs["input_ids"].shape[1]:
        ]

        raw = (
            processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
        )

        parsed = parse_output(raw)

        result = {
            "judge_id":
                int(judge_id),

            "audio":
                audio_path,

            "human_grounded":
                human,

            "pred_grounded":
                parsed["grounded"],

            "pred_issue":
                parsed["issue"],

            "issue_valid":
                parsed.get(
                    "issue_valid",
                    False,
                ),

            "parse_ok":
                parsed["parse_ok"],

            "rationale":
                parsed["rationale"],

            "raw_output":
                raw,

            "judge_model":
                MODEL,
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
            human == parsed["grounded"],
        )

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

    valid = [
        x
        for x in results
        if x["pred_grounded"]
        is not None
    ]

    print()
    print("=" * 88)
    print("FINAL COMPARISON")
    print("=" * 88)

    print(
        f"{'ID':>7}  "
        f"{'Human':>5}  "
        f"{'Pred':>4}  "
        f"{'Issue':<24}  "
        f"{'Match':>5}"
    )

    for x in results:

        match = (
            x["human_grounded"]
            == x["pred_grounded"]
        )

        print(
            f"{x['judge_id']:>7}  "
            f"{x['human_grounded']:>5}  "
            f"{str(x['pred_grounded']):>4}  "
            f"{str(x['pred_issue']):<24}  "
            f"{str(match):>5}"
        )

    if not valid:
        return

    human = [
        x["human_grounded"]
        for x in valid
    ]

    pred = [
        x["pred_grounded"]
        for x in valid
    ]

    accuracy = sum(
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
        f"{accuracy:.3f}"
    )

    try:
        from sklearn.metrics import (
            cohen_kappa_score,
            confusion_matrix,
        )

        kappa = cohen_kappa_score(
            human,
            pred,
        )

        cm = confusion_matrix(
            human,
            pred,
            labels=[0, 1],
        )

        print(
            f"Cohen kappa: "
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
