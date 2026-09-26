#!/usr/bin/env python3

import ast
import json
import re
from pathlib import Path

import torch
from transformers import AutoModel
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


MODEL = "/scratch-shared/yzhang1/models/MiniCPM-o-4_5"

AUDIO_INPUT = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/audio_input.jsonl"
)

VISUAL_INPUT = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/visual_input.jsonl"
)

GOLD = Path(
    "data/mustard/processed/genrm/"
    "judge_calibration10_binary/human_gold.jsonl"
)

OUTDIR = Path(
    "results/mustard/genrm/"
    "judge_calibration10_binary"
)

AUDIO_OUTPUT = OUTDIR / "audio_minicpmo_4_5_pred.jsonl"
VISUAL_OUTPUT = OUTDIR / "visual_minicpmo_4_5_pred.jsonl"


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


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_output(text):
    raw = str(text).strip()

    candidates = [raw]

    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.DOTALL,
    )

    if match:
        candidates.append(match.group(0))

    if '\\"' in raw:
        candidates.append(
            raw.replace('\\"', '"')
        )

    obj = None

    for candidate in candidates:
        try:
            x = json.loads(candidate)
            if isinstance(x, dict):
                obj = x
                break
        except Exception:
            pass

        try:
            x = ast.literal_eval(candidate)
            if isinstance(x, dict):
                obj = x
                break
        except Exception:
            pass

    if obj is None:
        return {
            "parse_ok": False,
            "grounded": None,
            "issue": None,
            "issue_valid": False,
            "rationale": None,
        }

    try:
        grounded = int(obj.get("grounded"))
    except Exception:
        grounded = None

    if grounded not in {0, 1}:
        grounded = None

    issue = obj.get("issue")

    if issue is None:
        issue = "NONE"
    else:
        issue = str(issue).strip().upper()

    aliases = {
        "OVERINTERPRETED": "OVERINTERPRETATION",
        "HALLUCINATION": "HALLUCINATED_CUE",
        "MISATTRIBUTION": "ENTITY_BINDING_ERROR",
    }

    issue = aliases.get(issue, issue)

    return {
        "parse_ok": grounded is not None,
        "grounded": grounded,
        "issue": issue,
        "issue_valid": issue in VALID_ISSUES,
        "rationale": obj.get("rationale", ""),
    }


def print_metrics(name, results):
    valid = [
        x for x in results
        if x["pred_grounded"] is not None
    ]

    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    for x in results:
        print(
            f"id={x['judge_id']:>4}  "
            f"human={x['human_grounded']}  "
            f"pred={x['pred_grounded']}  "
            f"issue={x['pred_issue']}  "
            f"match="
            f"{x['human_grounded'] == x['pred_grounded']}"
        )

    if not valid:
        print("No valid predictions.")
        return

    y_true = [
        x["human_grounded"]
        for x in valid
    ]

    y_pred = [
        x["pred_grounded"]
        for x in valid
    ]

    print()
    print(
        f"Parsed:     {len(valid)}/{len(results)}"
    )

    print(
        f"Agreement:  "
        f"{accuracy_score(y_true, y_pred):.3f}"
    )

    print(
        f"Macro-F1:   "
        f"{f1_score(y_true, y_pred, average='macro'):.3f}"
    )

    print(
        f"Cohen kappa:"
        f" {cohen_kappa_score(y_true, y_pred):.3f}"
    )

    print(
        "Confusion matrix "
        "(rows=human, cols=pred):"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
    )


def main():
    audio_rows = read_jsonl(AUDIO_INPUT)
    visual_rows = read_jsonl(VISUAL_INPUT)
    gold_rows = read_jsonl(GOLD)

    gold_by_id = {
        str(x["judge_id"]): x
        for x in gold_rows
    }

    if len(audio_rows) != 10:
        raise RuntimeError(
            f"Expected 10 audio rows, got {len(audio_rows)}"
        )

    if len(visual_rows) != 10:
        raise RuntimeError(
            f"Expected 10 visual rows, got {len(visual_rows)}"
        )

    print("=" * 90)
    print("Loading MiniCPM-o 4.5")
    print("=" * 90)

    model = AutoModel.from_pretrained(
        MODEL,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )

    model.eval().cuda()

    print("Model loaded.")
    print(
        "GPU allocated:",
        round(
            torch.cuda.memory_allocated()
            / 1024**3,
            2,
        ),
        "GB",
    )

    # ======================================================
    # AUDIO
    # ======================================================

    audio_results = []

    print()
    print("=" * 90)
    print("AUDIO BINARY JUDGE")
    print("=" * 90)

    for idx, row in enumerate(
        audio_rows,
        start=1,
    ):
        judge_id = str(row["id"])

        human = int(
            gold_by_id[judge_id][
                "human_audio_grounded"
            ]
        )

        audio_path = str(
            Path(row["audios"][0]).resolve()
        )

        prompt = row["messages"][0]["content"]

        # Remove textual placeholder because the real audio
        # is supplied separately.
        prompt = prompt.replace(
            "<audio>",
            "",
        ).strip()

        print()
        print(
            f"[Audio {idx}/10] "
            f"id={judge_id} human={human}"
        )

        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": audio_path
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        raw = model.chat(
            msgs=msgs,
            max_new_tokens=128,
            do_sample=False,
            use_tts_template=False,
            enable_thinking=False,
        )

        parsed = parse_output(raw)

        result = {
            "judge_id": int(judge_id),
            "audio": audio_path,
            "human_grounded": human,
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
            "raw_output": str(raw),
            "judge_model":
                "OpenBMB/MiniCPM-o-4_5",
        }

        audio_results.append(result)

        print("raw:")
        print(raw)

        print(
            "pred:",
            parsed["grounded"],
            "match:",
            parsed["grounded"] == human,
        )

        write_jsonl(
            AUDIO_OUTPUT,
            audio_results,
        )

    print_metrics(
        "MINICPM-O 4.5 AUDIO",
        audio_results,
    )

    # ======================================================
    # VISUAL
    # ======================================================

    visual_results = []

    print()
    print("=" * 90)
    print("VISUAL BINARY JUDGE")
    print("=" * 90)

    for idx, row in enumerate(
        visual_rows,
        start=1,
    ):
        judge_id = str(row["id"])

        human = int(
            gold_by_id[judge_id][
                "human_visual_grounded"
            ]
        )

        video_path = str(
            Path(row["videos"][0]).resolve()
        )

        prompt = row["messages"][0]["content"]

        prompt = prompt.replace(
            "<video>",
            "",
        ).strip()

        print()
        print(
            f"[Visual {idx}/10] "
            f"id={judge_id} human={human}"
        )

        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_path,

                            # CRITICAL:
                            # visual judge must not see audio.
                            "use_audio": False,

                            "stack_frames": 1,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        raw = model.chat(
            msgs=msgs,
            max_new_tokens=128,
            do_sample=False,
            use_tts_template=False,
            enable_thinking=False,
            max_slice_nums=1,
        )

        parsed = parse_output(raw)

        result = {
            "judge_id": int(judge_id),
            "video": video_path,
            "human_grounded": human,
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
            "raw_output": str(raw),
            "judge_model":
                "OpenBMB/MiniCPM-o-4_5",
            "video_use_audio": False,
        }

        visual_results.append(result)

        print("raw:")
        print(raw)

        print(
            "pred:",
            parsed["grounded"],
            "match:",
            parsed["grounded"] == human,
        )

        write_jsonl(
            VISUAL_OUTPUT,
            visual_results,
        )

    print_metrics(
        "MINICPM-O 4.5 VISUAL",
        visual_results,
    )

    print()
    print("=" * 90)
    print("OUTPUTS")
    print("=" * 90)
    print(AUDIO_OUTPUT)
    print(VISUAL_OUTPUT)


if __name__ == "__main__":
    main()
