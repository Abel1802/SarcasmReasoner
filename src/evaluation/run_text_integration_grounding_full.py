#!/usr/bin/env python3

import argparse
import ast
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TEXT_PROMPT_FILE = Path(
    "src/prompts/grounding_judge/"
    "text_grounding_binary_final.txt"
)

INTEGRATION_PROMPT_FILE = Path(
    "src/prompts/grounding_judge/"
    "integration_grounding_binary_final.txt"
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only use the first N pool rows. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    return parser.parse_args()


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


def load_completed(path):

    completed = set()

    if not path.exists():
        return completed

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:

                row = json.loads(line)

                judge_id = row.get(
                    "judge_id"
                )

                parse_ok = row.get(
                    "parse_ok",
                    False,
                )

                if (
                    judge_id is not None
                    and parse_ok
                ):

                    completed.add(
                        str(judge_id)
                    )

            except Exception:
                continue

    return completed


def load_prompt(path):

    return path.read_text(
        encoding="utf-8"
    )


def parse_judge_output(text):

    import re

    raw = text.strip()

    candidates = [raw]

    # Remove markdown code fences if present.
    if "```" in raw:

        stripped = (
            raw
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

        candidates.append(
            stripped
        )

    # Extract apparent {...} object.
    left = raw.find("{")
    right = raw.rfind("}")

    if (
        left != -1
        and right != -1
        and right > left
    ):

        candidates.append(
            raw[left:right + 1]
        )

    # --------------------------------
    # 1. Strict JSON / Python dict
    # --------------------------------

    for candidate in candidates:

        obj = None

        try:

            obj = json.loads(
                candidate
            )

        except Exception:

            try:

                value = ast.literal_eval(
                    candidate
                )

                if isinstance(
                    value,
                    dict,
                ):

                    obj = value

            except Exception:

                pass

        if isinstance(
            obj,
            dict,
        ):

            verdict = str(
                obj.get(
                    "verdict",
                    "",
                )
            ).strip().upper()

            if verdict in {
                "SUPPORTED",
                "UNSUPPORTED",
            }:

                return {
                    "parse_ok": True,
                    "verdict": verdict,
                    "grounded": (
                        1
                        if verdict == "SUPPORTED"
                        else 0
                    ),
                    "evidence": obj.get(
                        "evidence"
                    ),
                }

    # --------------------------------
    # 2. Robust verdict fallback
    #
    # Handles malformed JSON such as:
    #
    # {"verdict": "SUPPORTED",
    #  "evidence": "The text "Noo.." ..."}
    # --------------------------------

    verdict_match = re.search(
        r"[\"']?verdict[\"']?\s*:\s*[\"']?"
        r"(SUPPORTED|UNSUPPORTED)"
        r"[\"']?",
        raw,
        flags=re.IGNORECASE,
    )

    if verdict_match:

        verdict = (
            verdict_match
            .group(1)
            .upper()
        )

        return {
            "parse_ok": True,
            "verdict": verdict,
            "grounded": (
                1
                if verdict == "SUPPORTED"
                else 0
            ),
            "evidence": None,
        }

    return {
        "parse_ok": False,
        "verdict": None,
        "grounded": None,
        "evidence": None,
    }


def build_text_prompt(
    template,
    row,
):

    transcript = row["text"]

    text_evidence = (
        row["sections"][
            "text_evidence"
        ]
    )

    prompt = template.replace(
        "{{TRANSCRIPT}}",
        transcript,
    )

    prompt = prompt.replace(
        "{{TEXT_EVIDENCE}}",
        text_evidence,
    )

    return prompt


def build_integration_prompt(
    template,
    row,
):

    sections = row["sections"]

    prompt = template.replace(
        "{{TEXT_EVIDENCE}}",
        sections[
            "text_evidence"
        ],
    )

    prompt = prompt.replace(
        "{{AUDIO_EVIDENCE}}",
        sections[
            "audio_evidence"
        ],
    )

    prompt = prompt.replace(
        "{{VISUAL_EVIDENCE}}",
        sections[
            "visual_evidence"
        ],
    )

    prompt = prompt.replace(
        "{{INTEGRATION}}",
        sections[
            "integration"
        ],
    )

    return prompt


def make_chat_prompt(
    tokenizer,
    prompt,
):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
):

    chat_prompts = [
        make_chat_prompt(
            tokenizer,
            prompt,
        )
        for prompt in prompts
    ]

    inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    inputs = {
        key: value.to(
            model.device
        )
        for key, value in inputs.items()
    }

    input_width = (
        inputs["input_ids"].shape[1]
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    generated = outputs[
        :,
        input_width:
    ]

    texts = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )

    return texts


def write_result(
    fout,
    row,
    component,
    raw_output,
):

    parsed = parse_judge_output(
        raw_output
    )

    result = {
        "judge_id":
            str(row["judge_id"]),

        "source_id":
            str(row["source_id"]),

        "sample_idx":
            row.get(
                "sample_idx"
            ),

        "split":
            row.get(
                "split"
            ),

        "component":
            component,

        "grounded":
            parsed["grounded"],

        "verdict":
            parsed["verdict"],

        "parse_ok":
            parsed["parse_ok"],

        "evidence":
            parsed["evidence"],

        "raw_output":
            raw_output.strip(),
    }

    fout.write(
        json.dumps(
            result,
            ensure_ascii=False,
        )
        + "\n"
    )

    fout.flush()

    return parsed


def run_component(
    *,
    component,
    rows,
    output_path,
    prompt_builder,
    template,
    model,
    tokenizer,
    batch_size,
    max_new_tokens,
):

    completed = load_completed(
        output_path
    )

    pending = [
        row
        for row in rows
        if str(
            row["judge_id"]
        ) not in completed
    ]

    print()
    print("=" * 80)
    print(
        component.upper()
    )
    print("=" * 80)

    print(
        "Total rows:",
        len(rows),
    )

    print(
        "Already completed:",
        len(completed),
    )

    print(
        "Pending:",
        len(pending),
    )

    if not pending:

        print(
            "Nothing to do."
        )

        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parsed_count = 0
    supported_count = 0
    unsupported_count = 0

    with open(
        output_path,
        "a",
        encoding="utf-8",
    ) as fout:

        for start in range(
            0,
            len(pending),
            batch_size,
        ):

            batch = pending[
                start:
                start + batch_size
            ]

            prompts = [
                prompt_builder(
                    template,
                    row,
                )
                for row in batch
            ]

            try:

                outputs = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                )

            except RuntimeError as exc:

                print()
                print(
                    "Batch generation failed."
                )

                print(
                    "Falling back to batch_size=1."
                )

                print(
                    "Error:",
                    str(exc)[:500],
                )

                outputs = []

                for prompt in prompts:

                    one_output = (
                        generate_batch(
                            model=model,
                            tokenizer=tokenizer,
                            prompts=[prompt],
                            max_new_tokens=max_new_tokens,
                        )[0]
                    )

                    outputs.append(
                        one_output
                    )

            for row, raw_output in zip(
                batch,
                outputs,
            ):

                parsed = write_result(
                    fout=fout,
                    row=row,
                    component=component,
                    raw_output=raw_output,
                )

                if parsed["parse_ok"]:

                    parsed_count += 1

                if (
                    parsed["grounded"]
                    == 1
                ):

                    supported_count += 1

                elif (
                    parsed["grounded"]
                    == 0
                ):

                    unsupported_count += 1

            done = min(
                start + len(batch),
                len(pending),
            )

            if (
                done % 50 == 0
                or done == len(pending)
            ):

                print(
                    f"[{component}] "
                    f"{done}/{len(pending)} "
                    f"| parse={parsed_count} "
                    f"| supported={supported_count} "
                    f"| unsupported={unsupported_count}"
                )

    print()
    print(
        f"{component} finished."
    )

    print(
        "Output:",
        output_path,
    )


def main():

    args = parse_args()

    split = args.split

    input_path = (
        args.input
        if args.input is not None
        else Path(
            "data/mustard/processed/genrm/"
            f"grounding_judge_pool_{split}.jsonl"
        )
    )

    outdir = (
        args.outdir
        if args.outdir is not None
        else Path(
            "results/mustard/genrm/"
            "grounding_judging/"
            f"{split}"
        )
    )

    print("=" * 80)
    print(
        "TEXT + INTEGRATION GROUNDING JUDGE"
    )
    print("=" * 80)

    print(
        "Split:",
        split,
    )

    print(
        "Input:",
        input_path,
    )

    print(
        "Output dir:",
        outdir,
    )

    rows = read_jsonl(
        input_path
    )

    if args.limit is not None:

        rows = rows[
            :args.limit
        ]

    print(
        "Rows:",
        len(rows),
    )

    text_template = load_prompt(
        TEXT_PROMPT_FILE
    )

    integration_template = (
        load_prompt(
            INTEGRATION_PROMPT_FILE
        )
    )

    print()
    print(
        "Loading model:",
        MODEL_NAME,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME,
        )
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    # Avoid sampling-related warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    run_component(
        component="text",
        rows=rows,
        output_path=(
            outdir
            / "text_labels.jsonl"
        ),
        prompt_builder=(
            build_text_prompt
        ),
        template=text_template,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    run_component(
        component="integration",
        rows=rows,
        output_path=(
            outdir
            / "integration_labels.jsonl"
        ),
        prompt_builder=(
            build_integration_prompt
        ),
        template=integration_template,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
