#!/usr/bin/env python3

import ast
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-7B-Instruct"

BASE = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary"
)

TEXT_INPUT = BASE / "text_input.jsonl"
INTEGRATION_INPUT = BASE / "integration_input.jsonl"

OUTDIR = Path(
    "results/mustard/genrm/"
    "judge_calibration10_binary"
)

TEXT_OUTPUT = (
    OUTDIR
    / "text_qwen2_5_7b_final_pred.jsonl"
)

INTEGRATION_OUTPUT = (
    OUTDIR
    / "integration_qwen2_5_7b_final_pred.jsonl"
)


TEXT_ISSUES = {
    "NONE",
    "OVERINTERPRETATION",
    "HALLUCINATED_CLAIM",
    "CONTRADICTED_BY_INPUT",
    "OUTSIDE_KNOWLEDGE",
    "NOT_ASSESSABLE",
}

INTEGRATION_ISSUES = {
    "NONE",
    "OVERINTERPRETATION",
    "UNSUPPORTED_INFERENCE",
    "EVIDENCE_CONTRADICTION",
    "INVENTED_EVIDENCE",
    "MODALITY_MISREPRESENTATION",
    "NOT_ASSESSABLE",
}


def read_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

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


def parse_output(raw, valid_issues):

    raw = raw.strip()

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

    # ----------------------------------------------------
    # Regex fallback
    # ----------------------------------------------------

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

        if not verdict_match:
            return {
                "parse_ok": False,
                "grounded": None,
                "verdict": None,
                "issue": None,
                "issue_valid": False,
                "rationale": None,
            }

        verdict = (
            verdict_match
            .group(1)
            .upper()
        )

        grounded = {
            "SUPPORTED": 1,
            "UNSUPPORTED": 0,
        }[verdict]

        issue = (
            issue_match.group(1)
            .strip()
            .upper()
            if issue_match
            else "NONE"
        )

        return {
            "parse_ok": True,
            "grounded": grounded,
            "verdict": verdict,
            "issue": issue,
            "issue_valid":
                issue in valid_issues,
            "rationale": "",
        }

    verdict = str(
        obj.get("verdict", "")
    ).strip().upper()

    grounded = {
        "SUPPORTED": 1,
        "UNSUPPORTED": 0,
    }.get(verdict)

    issue = str(
        obj.get("issue", "NONE")
    ).strip().upper()

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


def run_one(
    model,
    tokenizer,
    prompt,
):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[
        :,
        inputs.input_ids.shape[1]:
    ]

    return tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True,
    ).strip()


def run_group(
    name,
    rows,
    output_path,
    valid_issues,
    model,
    tokenizer,
):

    results = []

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    for idx, row in enumerate(
        rows,
        start=1,
    ):

        judge_id = row["id"]

        prompt = (
            row["messages"][0]["content"]
        )

        print()
        print(
            f"[{idx}/10] id={judge_id}"
        )

        raw = run_one(
            model,
            tokenizer,
            prompt,
        )

        parsed = parse_output(
            raw,
            valid_issues,
        )

        result = {
            "judge_id":
                int(judge_id),

            "source_id":
                row.get("source_id"),

            "pred_grounded":
                parsed["grounded"],

            "pred_verdict":
                parsed["verdict"],

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
        }

        results.append(result)

        print("raw:")
        print(raw)

        print(
            "pred:",
            parsed["grounded"],
            "issue:",
            parsed["issue"],
        )

        # Save incrementally
        write_jsonl(
            output_path,
            results,
        )

    print()
    print("-" * 100)
    print(name, "SUMMARY")
    print("-" * 100)

    for x in results:
        print(
            f"id={x['judge_id']:>4}   "
            f"pred={x['pred_grounded']}   "
            f"issue={x['pred_issue']}   "
            f"parse={x['parse_ok']}"
        )

    print()
    print(
        "SUPPORTED:",
        sum(
            x["pred_grounded"] == 1
            for x in results
        ),
    )

    print(
        "UNSUPPORTED:",
        sum(
            x["pred_grounded"] == 0
            for x in results
        ),
    )

    print(
        "PARSE FAIL:",
        sum(
            x["pred_grounded"] is None
            for x in results
        ),
    )

    return results


def main():

    text_rows = read_jsonl(
        TEXT_INPUT
    )

    integration_rows = read_jsonl(
        INTEGRATION_INPUT
    )

    if len(text_rows) != 10:
        raise RuntimeError(
            f"Expected 10 text rows, "
            f"got {len(text_rows)}"
        )

    if len(integration_rows) != 10:
        raise RuntimeError(
            f"Expected 10 integration rows, "
            f"got {len(integration_rows)}"
        )

    print("=" * 100)
    print(
        "Loading Qwen2.5-7B-Instruct"
    )
    print("=" * 100)

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

    print("Model loaded.")

    run_group(
        name="TEXT GROUNDING",
        rows=text_rows,
        output_path=TEXT_OUTPUT,
        valid_issues=TEXT_ISSUES,
        model=model,
        tokenizer=tokenizer,
    )

    run_group(
        name="INTEGRATION GROUNDING",
        rows=integration_rows,
        output_path=INTEGRATION_OUTPUT,
        valid_issues=INTEGRATION_ISSUES,
        model=model,
        tokenizer=tokenizer,
    )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)
    print(TEXT_OUTPUT)
    print(INTEGRATION_OUTPUT)


if __name__ == "__main__":
    main()
