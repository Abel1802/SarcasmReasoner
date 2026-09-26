#!/usr/bin/env python3

import os

# ============================================================
# Runtime environment
# ============================================================

os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
os.environ["USE_AUDIO_IN_VIDEO"] = "False"
os.environ["FPS_MAX_FRAMES"] = "12"
os.environ["VIDEO_MAX_PIXELS"] = "50176"
os.environ["MAX_PIXELS"] = "1003520"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import importlib.util
import json
from pathlib import Path

from swift.infer_engine import RequestConfig, TransformersEngine


# ============================================================
# Helpers
# ============================================================

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_source_id(row):
    for key in ("source_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    videos = row.get("videos")
    if isinstance(videos, list) and videos:
        return Path(str(videos[0])).stem

    audios = row.get("audios")
    if isinstance(audios, list) and audios:
        return Path(str(audios[0])).stem

    return None


def align_predictions(gold_rows, pred_rows):
    gold_ids = [get_source_id(x) for x in gold_rows]
    pred_ids = [get_source_id(x) for x in pred_rows]

    can_align = (
        all(x is not None for x in gold_ids)
        and all(x is not None for x in pred_ids)
        and len(set(gold_ids)) == len(gold_ids)
        and len(set(pred_ids)) == len(pred_ids)
        and set(gold_ids) == set(pred_ids)
    )

    if can_align:
        pred_map = {
            get_source_id(x): x
            for x in pred_rows
        }
        return [
            pred_map[get_source_id(g)]
            for g in gold_rows
        ]

    if len(gold_rows) != len(pred_rows):
        raise RuntimeError(
            f"Cannot align predictions: "
            f"{len(gold_rows)} gold vs {len(pred_rows)} pred"
        )

    print("WARNING: falling back to file-order alignment.")
    return pred_rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return 100 * x if x is not None else None


# ============================================================
# Summary
# ============================================================

def summarize(rows, subset_name, selector):
    selected = [x for x in rows if selector(x)]

    valid = [
        x for x in selected
        if x["ordinary"]["judgment"] is not None
        and x["genrm"]["judgment"] is not None
    ]

    def model_stats(model_name):
        js = [
            x[model_name]["judgment"]
            for x in valid
        ]

        text_rate = mean([j["text"] for j in js])
        audio_rate = mean([j["audio"] for j in js])
        visual_rate = mean([j["visual"] for j in js])
        integration_rate = mean([j["integration"] for j in js])

        tav = [
            (
                j["text"]
                + j["audio"]
                + j["visual"]
            ) / 3.0
            for j in js
        ]

        all_three = mean([
            int(
                j["text"] == 1
                and j["audio"] == 1
                and j["visual"] == 1
            )
            for j in js
        ])

        return {
            "n": len(valid),
            "text_support": text_rate,
            "audio_support": audio_rate,
            "visual_support": visual_rate,
            "integration_support": integration_rate,
            "tav_grounding": mean(tav),
            "all_three_supported": all_three,
        }

    ordinary = model_stats("ordinary")
    genrm = model_stats("genrm")

    ordinary_tav = []
    genrm_tav = []

    for x in valid:
        jo = x["ordinary"]["judgment"]
        jg = x["genrm"]["judgment"]

        ordinary_tav.append(
            (jo["text"] + jo["audio"] + jo["visual"]) / 3
        )
        genrm_tav.append(
            (jg["text"] + jg["audio"] + jg["visual"]) / 3
        )

    wins = sum(
        g > o
        for o, g in zip(ordinary_tav, genrm_tav)
    )

    ties = sum(
        g == o
        for o, g in zip(ordinary_tav, genrm_tav)
    )

    losses = sum(
        g < o
        for o, g in zip(ordinary_tav, genrm_tav)
    )

    return {
        "subset": subset_name,
        "selected": len(selected),
        "valid_pairs": len(valid),

        "ordinary": ordinary,
        "genrm": genrm,

        "delta_genrm_minus_ordinary": {
            key: (
                genrm[key] - ordinary[key]
                if isinstance(genrm[key], float)
                else None
            )
            for key in [
                "text_support",
                "audio_support",
                "visual_support",
                "integration_support",
                "tav_grounding",
                "all_three_supported",
            ]
        },

        "pairwise_tav": {
            "genrm_better": wins,
            "tie": ties,
            "ordinary_better": losses,
        },
    }


def print_summary(summary):
    print()
    print("=" * 72)
    print(summary["subset"])
    print("=" * 72)

    print(
        f"Selected:    {summary['selected']}"
    )
    print(
        f"Valid pairs: {summary['valid_pairs']}"
    )
    print()

    print(
        f"{'Metric':<24}"
        f"{'Ordinary':>12}"
        f"{'+GenRM':>12}"
        f"{'Delta':>12}"
    )
    print("-" * 60)

    labels = [
        ("text_support", "Text support"),
        ("audio_support", "Audio support"),
        ("visual_support", "Visual support"),
        ("tav_grounding", "T/A/V mean"),
        ("all_three_supported", "All T/A/V supported"),
        ("integration_support", "Integration support"),
    ]

    for key, label in labels:
        o = summary["ordinary"][key]
        g = summary["genrm"][key]
        d = summary["delta_genrm_minus_ordinary"][key]

        print(
            f"{label:<24}"
            f"{pct(o):>11.2f}%"
            f"{pct(g):>11.2f}%"
            f"{pct(d):>+11.2f}"
        )

    p = summary["pairwise_tav"]

    print()
    print(
        "Pairwise T/A/V score: "
        f"+GenRM better={p['genrm_better']}, "
        f"tie={p['tie']}, "
        f"Ordinary better={p['ordinary_better']}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold", required=True)
    parser.add_argument("--ordinary", required=True)
    parser.add_argument("--genrm", required=True)

    parser.add_argument(
        "--genrm-model",
        default="Qwen/Qwen2.5-Omni-3B",
    )

    parser.add_argument(
        "--genrm-adapter",
        default=(
            "results/mustard/genrm/qwen25_omni_3b/"
            "v1-20260926-001310/checkpoint-1000"
        ),
    )

    parser.add_argument(
        "--plugin",
        default="src/plugins/sarcasm_grpo_reward.py",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    # ========================================================
    # Load exact reward-plugin helpers
    # ========================================================

    plugin_path = Path(args.plugin).resolve()

    spec = importlib.util.spec_from_file_location(
        "sarcasm_grpo_reward",
        plugin_path,
    )

    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    # ========================================================
    # Data
    # ========================================================

    gold_rows = read_jsonl(args.gold)
    ordinary_rows = align_predictions(
        gold_rows,
        read_jsonl(args.ordinary),
    )
    genrm_rows = align_predictions(
        gold_rows,
        read_jsonl(args.genrm),
    )

    if len(gold_rows) != len(ordinary_rows):
        raise RuntimeError("Ordinary count mismatch.")

    if len(gold_rows) != len(genrm_rows):
        raise RuntimeError("GenRM count mismatch.")

    print("Examples:", len(gold_rows))

    # ========================================================
    # Load GenRM
    # ========================================================

    print()
    print("Loading GenRM:")
    print("  model:  ", args.genrm_model)
    print("  adapter:", args.genrm_adapter)

    engine = TransformersEngine(
        args.genrm_model,
        adapters=[args.genrm_adapter],
        max_batch_size=args.batch_size,
    )

    request_config = RequestConfig(
        max_tokens=64,
        temperature=0,
    )

    # ========================================================
    # Build paired requests
    # ========================================================

    rows = []

    for i, (gold, ordinary, genrm) in enumerate(
        zip(gold_rows, ordinary_rows, genrm_rows)
    ):
        source_id = get_source_id(gold) or str(i)

        transcript = gold.get("transcript")
        audios = gold.get("audios")
        videos = gold.get("videos")

        if not transcript:
            raise RuntimeError(
                f"{source_id}: missing transcript"
            )

        if not audios:
            raise RuntimeError(
                f"{source_id}: missing audios"
            )

        if not videos:
            raise RuntimeError(
                f"{source_id}: missing videos"
            )

        gold_label = int(gold["label"])

        row = {
            "source_id": source_id,
            "gold": gold_label,
        }

        for name, pred_row in [
            ("ordinary", ordinary),
            ("genrm", genrm),
        ]:
            response = pred_row.get("response", "")

            prediction = plugin.extract_answer(response)

            reasoning = (
                plugin.extract_reasoning_for_genrm(
                    response
                )
            )

            row[name] = {
                "prediction": prediction,
                "correct": prediction == gold_label,
                "response": response,
                "reasoning": reasoning,
                "judgment": None,
                "raw_genrm_output": None,
            }

        rows.append(row)

    # ========================================================
    # Run GenRM WITHOUT correctness gating
    # ========================================================

    jobs = []

    for row_idx, row in enumerate(rows):
        gold = gold_rows[row_idx]

        for model_name in ("ordinary", "genrm"):
            reasoning = row[model_name]["reasoning"]

            if reasoning is None:
                continue

            infer_request = {
                "audios": gold["audios"],
                "videos": gold["videos"],
            }

            request = (
                plugin.SarcasmGroundingRMPlugin
                .build_genrm_request(
                    infer_request,
                    gold["transcript"].strip(),
                    reasoning,
                )
            )

            jobs.append(
                (
                    row_idx,
                    model_name,
                    request,
                )
            )

    print()
    print("GenRM judgments:", len(jobs))

    batch_size = args.batch_size

    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]

        requests = [
            x[2]
            for x in batch
        ]

        results = engine.infer(
            requests,
            request_config,
            use_tqdm=False,
        )

        if len(results) != len(batch):
            raise RuntimeError(
                "Unexpected GenRM batch size."
            )

        for (
            row_idx,
            model_name,
            _
        ), result in zip(batch, results):

            raw = (
                result
                .choices[0]
                .message
                .content
            )

            judgment = plugin.parse_genrm_output(
                raw
            )

            rows[row_idx][model_name][
                "raw_genrm_output"
            ] = raw

            rows[row_idx][model_name][
                "judgment"
            ] = judgment

        done = min(
            start + batch_size,
            len(jobs),
        )

        print(
            f"\rJudged {done}/{len(jobs)}",
            end="",
            flush=True,
        )

    print()

    # ========================================================
    # Save per-example judgments
    # ========================================================

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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

    # ========================================================
    # Summaries
    # ========================================================

    summaries = []

    summaries.append(
        summarize(
            rows,
            "ALL TEST ITEMS",
            lambda x: True,
        )
    )

    summaries.append(
        summarize(
            rows,
            "BOTH CLASSIFICATIONS CORRECT",
            lambda x: (
                x["ordinary"]["correct"]
                and x["genrm"]["correct"]
            ),
        )
    )

    summaries.append(
        summarize(
            rows,
            "ORDINARY WRONG -> GENRM CORRECT",
            lambda x: (
                not x["ordinary"]["correct"]
                and x["genrm"]["correct"]
            ),
        )
    )

    summaries.append(
        summarize(
            rows,
            "ORDINARY CORRECT -> GENRM WRONG",
            lambda x: (
                x["ordinary"]["correct"]
                and not x["genrm"]["correct"]
            ),
        )
    )

    summaries.append(
        summarize(
            rows,
            "BOTH CLASSIFICATIONS WRONG",
            lambda x: (
                not x["ordinary"]["correct"]
                and not x["genrm"]["correct"]
            ),
        )
    )

    for summary in summaries:
        print_summary(summary)

    summary_path = output_path.with_suffix(
        ".summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summaries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Saved:")
    print("  judgments:", output_path)
    print("  summary:  ", summary_path)


if __name__ == "__main__":
    main()
