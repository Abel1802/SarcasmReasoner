#!/usr/bin/env python3

import ast
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-7B-Instruct"

INPUT = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/"
    "text_hardneg5.jsonl"
)

PROMPT = Path(
    "src/prompts/grounding_judge/"
    "text_grounding_binary_final.txt"
)

OUTPUT = Path(
    "results/mustard/genrm/"
    "judge_calibration10_binary/"
    "text_hardneg5_qwen2_5_7b_pred.jsonl"
)


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

    if obj is None:
        return {
            "parse_ok": False,
            "pred": None,
            "verdict": None,
            "evidence": None,
        }

    verdict = str(
        obj.get("verdict", "")
    ).strip().upper()

    verdict_map = {
        "SUPPORTED": 1,
        "UNSUPPORTED": 0,
    }

    pred = verdict_map.get(
        verdict
    )

    evidence = obj.get(
        "evidence"
    )

    return {
        "parse_ok": pred is not None,
        "pred": pred,
        "verdict": verdict,
        "evidence": evidence,
    }


def main():

    rows = read_jsonl(INPUT)

    prompt_template = PROMPT.read_text(
        encoding="utf-8"
    )

    print("=" * 90)
    print("Loading text grounding judge")
    print("=" * 90)
    print(MODEL)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    correct = 0
    parse_success = 0
    unsupported_detected = 0

    for i, row in enumerate(rows, 1):

        prompt = (
            prompt_template
            .replace(
                "{{TRANSCRIPT}}",
                row["transcript"],
            )
            .replace(
                "{{TEXT_EVIDENCE}}",
                row["text_evidence"],
            )
        )

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

        with torch.inference_mode():

            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        new_tokens = generated[
            :,
            inputs["input_ids"].shape[1]:
        ]

        raw_output = tokenizer.decode(
            new_tokens[0],
            skip_special_tokens=True,
        ).strip()

        parsed = parse_output(
            raw_output
        )

        pred = parsed["pred"]
        gold = int(row["gold"])

        is_correct = (
            pred == gold
        )

        if parsed["parse_ok"]:
            parse_success += 1

        if is_correct:
            correct += 1

        if (
            gold == 0
            and pred == 0
        ):
            unsupported_detected += 1

        result = {
            "id": row["id"],
            "type": row["type"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "parse_ok":
                parsed["parse_ok"],
            "verdict":
                parsed["verdict"],
            "evidence":
                parsed["evidence"],
            "raw_output":
                raw_output,
            "judge_model":
                MODEL,
        }

        results.append(result)

        print()
        print("=" * 90)
        print(
            f"[{i}/{len(rows)}] "
            f"id={row['id']} "
            f"type={row['type']}"
        )
        print("=" * 90)

        print(raw_output)

        print()
        print(
            "gold:",
            gold,
            "pred:",
            pred,
            "correct:",
            is_correct,
        )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"Correct: "
        f"{correct}/{len(rows)}"
    )

    print(
        "UNSUPPORTED detected:",
        unsupported_detected,
        "/4",
    )

    print(
        "Parse success:",
        parse_success,
        f"/{len(rows)}",
    )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
