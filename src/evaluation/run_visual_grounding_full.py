#!/usr/bin/env python3

import argparse
import ast
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

PROMPT_FILE = Path(
    "src/prompts/grounding_judge/"
    "visual_grounding_binary_v4_final.txt"
)

FPS = 4


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
                x = json.loads(line)

                if (
                    x.get("judge_id") is not None
                    and x.get("parse_ok", False)
                ):
                    completed.add(
                        str(x["judge_id"])
                    )

            except Exception:
                continue

    return completed


def parse_judge_output(text):

    raw = text.strip()

    candidates = [raw]

    if "```" in raw:

        candidates.append(
            raw
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

    # Handle escaped JSON:
    # {\"verdict\": \"SUPPORTED\", ...}
    if '\\"' in raw:

        candidates.append(
            raw.replace(
                '\\"',
                '"',
            )
        )

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

    # Robust fallback.
    verdict_match = re.search(
        r"[\"']?verdict[\"']?\s*:\s*[\\\"']*"
        r"(SUPPORTED|UNSUPPORTED)"
        r"[\\\"']*",
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


def build_prompt(
    template,
    row,
):

    return template.replace(
        "{{VISUAL_EVIDENCE}}",
        row["sections"][
            "visual_evidence"
        ],
    )


@torch.inference_mode()
def judge_one(
    *,
    model,
    processor,
    row,
    template,
    max_new_tokens,
):

    video_path = row["video"]

    prompt = build_prompt(
        template,
        row,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": FPS,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = (
        processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    # Some processor versions return this,
    # but Qwen3-VL generate does not need it.
    inputs.pop(
        "token_type_ids",
        None,
    )

    inputs = inputs.to(
        model.device
    )

    input_length = (
        inputs["input_ids"]
        .shape[1]
    )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    generated = generated_ids[
        :,
        input_length:
    ]

    output = (
        processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    return output


def append_failure(
    path,
    row,
    reason,
    raw_output=None,
):

    item = {
        "judge_id":
            str(row["judge_id"]),

        "source_id":
            str(row["source_id"]),

        "sample_idx":
            row.get("sample_idx"),

        "split":
            row.get("split"),

        "reason":
            reason,

        "raw_output":
            raw_output,
    }

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
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

    output_path = (
        outdir
        / "visual_labels.jsonl"
    )

    failure_path = (
        outdir
        / "visual_failures.jsonl"
    )

    print("=" * 80)
    print(
        "VISUAL GROUNDING JUDGE"
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
        "Output:",
        output_path,
    )

    print(
        "FPS:",
        FPS,
    )

    rows = read_jsonl(
        input_path
    )

    if args.limit is not None:
        rows = rows[
            :args.limit
        ]

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

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = PROMPT_FILE.read_text(
        encoding="utf-8"
    )

    print()
    print(
        "Loading processor:",
        MODEL_NAME,
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_NAME,
        )
    )

    print(
        "Loading model:",
        MODEL_NAME,
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    parsed_count = 0
    supported_count = 0
    unsupported_count = 0
    failure_count = 0

    with open(
        output_path,
        "a",
        encoding="utf-8",
    ) as fout:

        for i, row in enumerate(
            pending,
            start=1,
        ):

            try:

                raw_output = judge_one(
                    model=model,
                    processor=processor,
                    row=row,
                    template=template,
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )

            except Exception as exc:

                failure_count += 1

                append_failure(
                    failure_path,
                    row,
                    reason=(
                        "generation_error: "
                        + repr(exc)
                    ),
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                continue

            parsed = parse_judge_output(
                raw_output
            )

            if not parsed[
                "parse_ok"
            ]:

                failure_count += 1

                append_failure(
                    failure_path,
                    row,
                    reason="parse_error",
                    raw_output=raw_output,
                )

                continue

            result = {
                "judge_id":
                    str(
                        row["judge_id"]
                    ),

                "source_id":
                    str(
                        row["source_id"]
                    ),

                "sample_idx":
                    row.get(
                        "sample_idx"
                    ),

                "split":
                    split,

                "component":
                    "visual",

                "grounded":
                    parsed[
                        "grounded"
                    ],

                "verdict":
                    parsed[
                        "verdict"
                    ],

                "parse_ok":
                    True,

                "evidence":
                    parsed[
                        "evidence"
                    ],

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

            parsed_count += 1

            if (
                parsed["grounded"]
                == 1
            ):
                supported_count += 1
            else:
                unsupported_count += 1

            if (
                i % 25 == 0
                or i == len(pending)
            ):

                print(
                    f"[visual] "
                    f"{i}/{len(pending)} "
                    f"| parsed={parsed_count} "
                    f"| supported={supported_count} "
                    f"| unsupported={unsupported_count} "
                    f"| failures={failure_count}"
                )

    print()
    print("=" * 80)
    print(
        "VISUAL FINISHED"
    )
    print("=" * 80)

    print(
        "New parsed:",
        parsed_count,
    )

    print(
        "SUPPORTED:",
        supported_count,
    )

    print(
        "UNSUPPORTED:",
        unsupported_count,
    )

    print(
        "Failures:",
        failure_count,
    )

    print(
        "Output:",
        output_path,
    )


if __name__ == "__main__":
    main()
