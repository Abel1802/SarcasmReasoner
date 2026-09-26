#!/usr/bin/env python3

import argparse
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


MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

PROMPT_FILE = Path(
    "src/prompts/grounding_judge/"
    "audio_grounding_binary_v3_final.txt"
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

                if (
                    row.get("judge_id") is not None
                    and row.get("parse_ok", False)
                ):
                    completed.add(
                        str(row["judge_id"])
                    )

            except Exception:
                continue

    return completed


def parse_judge_output(text):

    raw = text.strip()

    candidates = [raw]

    if "```" in raw:

        cleaned = (
            raw
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

        candidates.append(cleaned)

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
            obj = json.loads(candidate)

        except Exception:

            try:
                value = ast.literal_eval(
                    candidate
                )

                if isinstance(value, dict):
                    obj = value

            except Exception:
                pass

        if isinstance(obj, dict):

            verdict = str(
                obj.get("verdict", "")
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

    # Robust fallback for malformed JSON.
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


def build_prompt(
    template,
    row,
):

    return template.replace(
        "{{AUDIO_EVIDENCE}}",
        row["sections"][
            "audio_evidence"
        ],
    )


def load_waveform(
    path,
    sampling_rate,
):

    waveform, _ = librosa.load(
        path,
        sr=sampling_rate,
        mono=True,
    )

    return waveform


def make_chat_text(
    processor,
    prompt,
    audio_path,
):

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": audio_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    return processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_batch(
    *,
    model,
    processor,
    rows,
    template,
    sampling_rate,
    max_new_tokens,
):

    # Reuse decoded waveform inside this batch.
    waveform_by_path = {}

    chat_texts = []
    waveforms = []

    for row in rows:

        audio_path = row["audio"]

        if audio_path not in waveform_by_path:

            waveform_by_path[
                audio_path
            ] = load_waveform(
                audio_path,
                sampling_rate,
            )

        waveform = waveform_by_path[
            audio_path
        ]

        prompt = build_prompt(
            template,
            row,
        )

        chat_text = make_chat_text(
            processor,
            prompt,
            audio_path,
        )

        chat_texts.append(
            chat_text
        )

        waveforms.append(
            waveform
        )

    inputs = processor(
        text=chat_texts,
        audios=waveforms,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: (
            value.to(model.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    input_width = (
        inputs["input_ids"].shape[1]
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    generated_ids = output_ids[
        :,
        input_width:
    ]

    outputs = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return outputs


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


def write_success(
    fout,
    row,
    parsed,
    raw_output,
    split,
):

    result = {
        "judge_id":
            str(row["judge_id"]),

        "source_id":
            str(row["source_id"]),

        "sample_idx":
            row.get("sample_idx"),

        "split":
            split,

        "component":
            "audio",

        "grounded":
            parsed["grounded"],

        "verdict":
            parsed["verdict"],

        "parse_ok":
            True,

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
        / "audio_labels.jsonl"
    )

    failure_path = (
        outdir
        / "audio_failures.jsonl"
    )

    print("=" * 80)
    print("AUDIO GROUNDING JUDGE")
    print("=" * 80)

    print("Split:", split)
    print("Input:", input_path)
    print("Output:", output_path)
    print("Batch size:", args.batch_size)

    rows = read_jsonl(
        input_path
    )

    if args.limit is not None:
        rows = rows[:args.limit]

    completed = load_completed(
        output_path
    )

    pending = [
        row
        for row in rows
        if str(row["judge_id"])
        not in completed
    ]

    print("Total rows:", len(rows))
    print(
        "Already completed:",
        len(completed),
    )
    print("Pending:", len(pending))

    if not pending:
        print("Nothing to do.")
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

    # Important for batched decoder-only generation.
    if hasattr(
        processor,
        "tokenizer",
    ):
        processor.tokenizer.padding_side = (
            "left"
        )

    print(
        "Loading model:",
        MODEL_NAME,
    )

    model = (
        Qwen2AudioForConditionalGeneration
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

    sampling_rate = (
        processor
        .feature_extractor
        .sampling_rate
    )

    print(
        "Sampling rate:",
        sampling_rate,
    )

    parsed_count = 0
    supported_count = 0
    unsupported_count = 0
    failure_count = 0

    with open(
        output_path,
        "a",
        encoding="utf-8",
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):

            batch = pending[
                start:
                start + args.batch_size
            ]

            # --------------------------------
            # First try true batch inference.
            # --------------------------------

            try:

                outputs = generate_batch(
                    model=model,
                    processor=processor,
                    rows=batch,
                    template=template,
                    sampling_rate=sampling_rate,
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )

                batch_results = list(
                    zip(
                        batch,
                        outputs,
                    )
                )

            except Exception as exc:

                print()
                print(
                    "Batch failed; "
                    "falling back to single-item inference."
                )
                print(
                    "Error:",
                    repr(exc)[:500],
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_results = []

                for row in batch:

                    try:

                        outputs = generate_batch(
                            model=model,
                            processor=processor,
                            rows=[row],
                            template=template,
                            sampling_rate=sampling_rate,
                            max_new_tokens=(
                                args.max_new_tokens
                            ),
                        )

                        batch_results.append(
                            (
                                row,
                                outputs[0],
                            )
                        )

                    except Exception as one_exc:

                        failure_count += 1

                        append_failure(
                            failure_path,
                            row,
                            reason=(
                                "generation_error: "
                                + repr(one_exc)
                            ),
                        )

            # --------------------------------
            # Parse / save outputs.
            # --------------------------------

            for row, raw_output in batch_results:

                parsed = (
                    parse_judge_output(
                        raw_output
                    )
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

                write_success(
                    fout=fout,
                    row=row,
                    parsed=parsed,
                    raw_output=raw_output,
                    split=split,
                )

                parsed_count += 1

                if (
                    parsed["grounded"]
                    == 1
                ):
                    supported_count += 1
                else:
                    unsupported_count += 1

            processed = min(
                start
                + len(batch),
                len(pending),
            )

            if (
                processed % 50
                < args.batch_size
                or processed
                == len(pending)
            ):

                print(
                    f"[audio] "
                    f"{processed}/{len(pending)} "
                    f"| parsed={parsed_count} "
                    f"| supported={supported_count} "
                    f"| unsupported={unsupported_count} "
                    f"| failures={failure_count}"
                )

    print()
    print("=" * 80)
    print("AUDIO FINISHED")
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
