#!/usr/bin/env python3
"""One multimodal call per trajectory; no label/prediction is sent to the judge.

MiniCPM-o-4.5 and Qwen3-Omni use separate Transformers environments. Heavy
dependencies are imported only for inference; --dry-run uses the standard library.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave

VERSION = "omni_quality_v1_1"
MODALITIES = ("text", "audio", "visual")
SECTION_KEYS = tuple(f"{m}_evidence" for m in MODALITIES)
DEFAULT_MODELS = {
    "minicpm": "openbmb/MiniCPM-o-4_5",
    "qwen": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def append_json(f, value):
    f.write(json.dumps(value, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def load_pool(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                for key in ("judge_id", "source_id", "text", "audio", "video"):
                    if key not in row or row[key] is None or str(row[key]).strip() == "":
                        raise ValueError(f"missing/empty {key}")
                if not isinstance(row["text"], str):
                    raise ValueError("text must be a transcript string")
                jid = str(row["judge_id"])
                if jid in seen:
                    raise ValueError(f"duplicate judge_id: {jid}")
                sections = row.get("sections")
                if not isinstance(sections, dict):
                    raise ValueError("sections must be an object")
                for key in SECTION_KEYS:
                    if not isinstance(sections.get(key), str):
                        raise ValueError(f"sections.{key} must be a string")
                # Empty evidence is allowed: its quality is judged, not a parse failure.
                row["judge_id"] = jid
                row["source_id"] = str(row["source_id"])
                seen.add(jid)
                rows.append(row)
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    if not rows:
        raise ValueError("input pool is empty")
    return rows


def select_rows(rows, sources, per_source, seed):
    groups = collections.defaultdict(list)
    for row in rows:
        # The split prevents collisions if a pooled input contains several splits.
        groups[(str(row.get("split", "")), row["source_id"])].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    if sources:
        keys = sorted(rng.sample(keys, min(sources, len(keys))))
    selected_ids = set()
    for key in keys:
        candidates = groups[key]
        chosen = rng.sample(candidates, min(per_source, len(candidates))) if per_source else candidates
        selected_ids.update(row["judge_id"] for row in chosen)
    return [row for row in rows if row["judge_id"] in selected_ids]


def payload_text(row):
    # JSON escaping preserves boundaries. Gold, prediction and integration are omitted.
    data = {"TRANSCRIPT": row["text"], "CANDIDATE_SECTIONS": {
        m: row["sections"][m + "_evidence"] for m in MODALITIES}}
    return ("Evaluate this supplied sample. Return one JSON object with exactly "
            "the keys text, audio, visual; each value is an integer 1, 2, or 3. "
            "Score sections independently. No explanations or extra fields.\n"
            + json.dumps(data, ensure_ascii=False, indent=2))


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate output key: {key}")
        result[key] = value
    return result


def parse_scores(raw):
    if not isinstance(raw, str):
        raise ValueError("model response is not a string")
    text = raw.strip()
    mode = "json"
    # Accept a completed thinking prefix, but never extract scores from inside it.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
        mode = "after_think"
    if "<think>" in text:
        raise ValueError("unfinished thinking output")
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
        mode += "+fence"
    value = json.loads(text, object_pairs_hook=no_duplicate_keys)
    # QUALITY_SCHEMA_FIX_V1_1: normalize exact aliases; preserve score values.
    if isinstance(value, dict) and set(value) == set(SECTION_KEYS):
        value = {m: value[m + "_evidence"] for m in MODALITIES}
        mode += "+evidence_keys"
    if not isinstance(value, dict) or set(value) != set(MODALITIES):
        raise ValueError("expected exactly text/audio/visual or their _evidence aliases")
    for key in MODALITIES:
        if type(value[key]) is not int or value[key] not in (1, 2, 3):
            raise ValueError(f"{key}: score must be integer 1, 2 or 3")
    return value, mode


def read_journal(path):
    """Strict journals: reject a partial final line instead of silently discarding it."""
    values = []
    if not Path(path).exists():
        return values
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"journal {path}:{lineno} is damaged; repair before resume") from e
    return values


def summaries(records, failures, selected, args):
    valid = {r["judge_id"]: r for r in records}
    latest_failures = {r["judge_id"]: r for r in failures if r["judge_id"] not in valid}
    result = {"judge_version": VERSION, "dataset": args.dataset,
              "backend": args.backend, "model": args.model,
              "total": len(selected), "parsed": len(valid),
              "failures": len(latest_failures),
              "pending": len(selected) - len(valid) - len(latest_failures),
              "parse_rate": len(valid) / len(selected),
              "retry_recovered": sum(r["attempt_count"] > 1 for r in valid.values()),
              "parse_modes": dict(collections.Counter(r["parse_mode"] for r in valid.values())),
              "modality_stats": {}, "diagnostics": {}}
    for m in MODALITIES:
        scores = [r["quality_scores"][m] for r in valid.values()]
        rewards = [(s - 1) / 2 for s in scores]
        result["modality_stats"][m] = {
            "n": len(scores), "score_distribution": {str(s): scores.count(s) for s in (1, 2, 3)},
            "mean_score": statistics.mean(scores) if scores else None,
            "mean_reward": statistics.mean(rewards) if rewards else None,
            "reward_std": statistics.pstdev(rewards) if rewards else None}
    # These fields are used only after inference, never in model prompts.
    by_id = {r["judge_id"]: r for r in selected}
    for field in ("label", "prediction", "prediction_correct"):
        groups = collections.defaultdict(list)
        for jid, output in valid.items():
            if field in by_id[jid]:
                groups[str(by_id[jid][field])].append(output)
        result["diagnostics"][field] = {key: {
            "n": len(items), "mean_scores": {
                m: statistics.mean(r["quality_scores"][m] for r in items) for m in MODALITIES}}
            for key, items in groups.items()}
    return result


def run_command(cmd, timeout=180):
    result = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(cmd[0]).name} failed: " + result.stderr.decode(errors="replace")[-1500:])
    return result.stdout


class MediaCache:
    """Reuse decoded media across trajectories; never silently crop a clip."""
    def __init__(self, args):
        self.args = args
        self.key = None
        self.media = None
        self.temp = None

    def close(self):
        if self.temp:
            self.temp.cleanup()
        self.temp = self.media = self.key = None

    def get(self, row):
        import numpy as np
        from PIL import Image
        root = Path(self.args.media_root).resolve()
        audio = (root / row["audio"]).resolve()
        video = (root / row["video"]).resolve()
        for path in (audio, video):
            if not path.is_file():
                raise FileNotFoundError(f"media missing: {path}")
        key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (audio, video))
        if key == self.key:
            return self.media
        self.close()
        self.temp = tempfile.TemporaryDirectory(prefix="omni_quality_")
        temp = Path(self.temp.name)
        info = json.loads(run_command([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=duration,width,height:format=duration", "-of", "json", str(video)]))
        streams = info.get("streams", [])
        if not streams:
            raise ValueError("video has no decodable video stream")
        stream = streams[0]
        try:
            duration = float(stream.get("duration", "nan"))
        except (ValueError, TypeError):
            duration = float("nan")
        if not math.isfinite(duration):
            duration = float(info.get("format", {}).get("duration", "nan"))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid video duration")
        if math.ceil(duration) > self.args.max_frames:
            raise ValueError(f"clip needs about {math.ceil(duration)} frames at 1 FPS; "
                             f"exceeds --max-frames {self.args.max_frames}; not truncated")
        wav = temp / "audio.wav"
        run_command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio),
                     "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
        with wave.open(str(wav), "rb") as f:
            if (f.getnchannels(), f.getframerate(), f.getsampwidth()) != (1, 16000, 2):
                raise ValueError("unexpected canonical audio format")
            waveform = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
        audio_duration = len(waveform) / 16000
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError("empty/nonfinite decoded audio")
        if abs(audio_duration - duration) > self.args.duration_tolerance:
            raise ValueError(f"audio/video duration mismatch: {audio_duration:.3f}s vs {duration:.3f}s; "
                             "check clip alignment or explicitly change --duration-tolerance")
        width, height = int(stream["width"]), int(stream["height"])
        ratio = min(1.0, self.args.max_side / max(width, height))
        w, h = max(2, round(width * ratio)), max(2, round(height * ratio))
        run_command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                     "-map", "0:v:0", "-an", "-vf", f"fps=fps=1:start_time=0:round=up,scale={w}:{h}",
                     "-q:v", "2", str(temp / "frame_%06d.jpg")])
        paths = sorted(temp.glob("frame_*.jpg"))
        if not paths or len(paths) > self.args.max_frames:
            raise ValueError(f"invalid/excessive decoded frame count: {len(paths)}")
        frames = []
        for p in paths:
            with Image.open(p) as im:
                frames.append(im.convert("RGB").copy())
        self.media = {"frames": frames, "frame_paths": paths, "waveform": waveform,
                      "wav_path": wav, "metadata": {
                          "audio_path": str(audio), "video_path": str(video),
                          "audio_sha256": digest_file(audio), "video_sha256": digest_file(video),
                          "audio_duration": audio_duration, "video_duration": duration,
                          "video_fps": 1, "num_frames": len(paths), "frame_size": [w, h],
                          "audio_sample_rate": 16000,
                          "audio_source": "row.audio (video audio track is not used)"}}
        self.key = key
        return self.media


class GuardedProcessor:
    """Disable MiniCPM's silent max_length slicing, then check the actual MM length."""
    def __init__(self, processor, limit, reserve):
        self.inner, self.limit, self.reserve = processor, limit, reserve
        self.last_length = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        kwargs["max_length"] = None
        inputs = self.inner(*args, **kwargs)
        self.last_length = int(inputs["input_ids"].shape[-1])
        if self.last_length + self.reserve > self.limit:
            raise ValueError(f"input {self.last_length} + output reserve {self.reserve} "
                             f"exceeds context limit {self.limit}; not truncated")
        return inputs


def context_limit(config, requested):
    limits = [requested]
    for obj in (config, getattr(config, "thinker_config", None)):
        if obj is not None:
            text_config = getattr(obj, "text_config", obj)
            value = getattr(text_config, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                limits.append(value)
    return min(limits)


class MiniCPMBackend:
    def __init__(self, args, revision):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.args, self.torch = args, torch
        self.model = AutoModel.from_pretrained(
            args.model, revision=revision, code_revision=revision,
            trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            init_vision=True, init_audio=True, init_tts=False).eval().cuda()
        processor = AutoProcessor.from_pretrained(
            args.model, revision=revision, code_revision=revision, trust_remote_code=True)
        self.guard = GuardedProcessor(processor, context_limit(self.model.config, args.max_context_tokens),
                                      args.max_new_tokens)

    def judge(self, media, prompt, row, retry):
        contents = []
        waveform = media["waveform"]
        for i, frame in enumerate(media["frames"]):
            contents.append(frame)
            part = waveform[i * 16000:(i + 1) * 16000]
            if len(part):
                contents.append(part)
        tail = waveform[len(media["frames"]) * 16000:]
        if len(tail):
            contents.append(tail)
        contents.append(payload_text(row) + retry)
        with self.torch.inference_mode():
            raw = self.model.chat(
                msgs=[{"role": "system", "content": [prompt]},
                      {"role": "user", "content": contents}],
                processor=self.guard, omni_mode=True,
                enable_thinking=False, generate_audio=False,
                use_tts_template=True, merge_audio_from_same_content=True,
                max_slice_nums=self.args.max_slice_nums,
                max_inp_length=None, max_new_tokens=self.args.max_new_tokens,
                do_sample=False, num_beams=1, repetition_penalty=1.0,
                temperature=None, top_p=None, top_k=None)
        return raw, self.guard.last_length


class QwenBackend:
    def __init__(self, args, revision):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        self.args, self.torch = args, torch
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model, revision=revision, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation=args.attn_implementation).eval()
        self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, revision=revision)
        self.limit = context_limit(self.model.config, args.max_context_tokens)

    def judge(self, media, prompt, row, retry):
        # Same canonical frames and external audio as MiniCPM, not a second decode.
        from qwen_omni_utils import process_mm_info
        messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]},
                    {"role": "user", "content": [
                        {"type": "video", "video": [str(p) for p in media["frame_paths"]],
                         "fps": 1.0, "max_pixels": self.args.qwen_max_pixels},
                        {"type": "audio", "audio": str(media["wav_path"])},
                        {"type": "text", "text": payload_text(row) + retry}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=False, fps=1.0)
        n = int(inputs["input_ids"].shape[-1])
        if n + self.args.max_new_tokens > self.limit:
            raise ValueError(f"input {n} + output reserve exceeds {self.limit}; not truncated")
        inputs = inputs.to(self.model.device)
        # Cast only floating tensors, preserving integer token IDs and metadata.
        for key, value in list(inputs.items()):
            if self.torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, return_audio=False, use_audio_in_video=False,
                thinker_max_new_tokens=self.args.max_new_tokens,
                thinker_do_sample=False, thinker_num_beams=1,
                thinker_repetition_penalty=1.0,
                thinker_temperature=None, thinker_top_p=None, thinker_top_k=None)
        if isinstance(generated, tuple):
            generated = generated[0]
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        raw = self.processor.batch_decode(generated[:, n:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        return raw, n


def environment_versions():
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "accelerate", "huggingface-hub",
                 "numpy", "Pillow", "minicpmo-utils", "qwen-omni-utils"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def resolve_revision(model, revision):
    if Path(model).is_dir():
        raise ValueError("v1 uses a Hugging Face model ID with a pinned commit; local model paths are not supported")
    from huggingface_hub import model_info
    return model_info(model, revision=revision).sha


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="successful labels JSONL; sidecars hold metadata/stats/failures")
    p.add_argument("--dataset", choices=("mcsd", "mustard"), required=True)
    p.add_argument("--prompt", default="src/genrm_judge/prompts/quality.txt")
    p.add_argument("--backend", choices=tuple(DEFAULT_MODELS), default="minicpm")
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default="main")
    p.add_argument("--media-root", default=".", help="project root for relative audio/video paths")
    p.add_argument("--sources", type=int, default=0, help="0=all sources; pilot example: 20")
    p.add_argument("--per-source", type=int, default=0, help="0=all trajectories; pilot example: 2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-context-tokens", type=int, default=32768)
    p.add_argument("--max-frames", type=int, default=120, help="1 FPS; fail rather than truncate longer clips")
    p.add_argument("--max-side", type=int, default=448)
    p.add_argument("--max-slice-nums", type=int, default=1, help="MiniCPM internal image slices")
    p.add_argument("--qwen-max-pixels", type=int, default=200704)
    p.add_argument("--duration-tolerance", type=float, default=0.75)
    p.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    p.add_argument("--max-retries", type=int, default=1, help="extra calls for invalid JSON only")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failures", action="store_true", help="with --resume, revisit terminal failures")
    p.add_argument("--dry-run", action="store_true", help="check pool/prompt/selection without model or media")
    p.add_argument("--check-media", action="store_true", help="decode selected clips without loading a model")
    a = p.parse_args(argv)
    a.model = a.model or DEFAULT_MODELS[a.backend]
    for name in ("sources", "per_source", "max_retries"):
        if getattr(a, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >=0")
    for name in ("max_new_tokens", "max_context_tokens", "max_frames", "max_side", "max_slice_nums",
                 "qwen_max_pixels", "log_every"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be >0")
    if a.duration_tolerance < 0:
        p.error("--duration-tolerance must be >=0")
    if a.retry_failures and not a.resume:
        p.error("--retry-failures requires --resume")
    return a


def run(args):
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("prompt is empty")
    rows = select_rows(load_pool(args.input), args.sources, args.per_source, args.seed)
    config = {"version": VERSION, "dataset": args.dataset, "backend": args.backend, "model": args.model,
              "input_sha256": digest_file(args.input),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "selected_judge_ids": [r["judge_id"] for r in rows],
              "settings": {key: getattr(args, key) for key in (
                  "seed", "sources", "per_source", "max_new_tokens", "max_context_tokens", "max_frames",
                  "max_side", "max_slice_nums", "qwen_max_pixels", "duration_tolerance",
                  "attn_implementation", "max_retries")},
              "media_root": str(Path(args.media_root).resolve()),
              "video_fps": 1, "enable_thinking": False, "do_sample": False}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "selected": len(rows),
                          "sources": len({(r.get('split'), r['source_id']) for r in rows}),
                          "config": config}, ensure_ascii=False, indent=2))
        return 0
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} is not installed")
    media_cache = MediaCache(args)
    if args.check_media:
        checked, failed = set(), 0
        try:
            for row in rows:
                key = (row["audio"], row["video"])
                if key in checked:
                    continue
                checked.add(key)
                try:
                    media = media_cache.get(row)
                    print(json.dumps({"source_id": row["source_id"], "ok": True,
                                      **media["metadata"]}, ensure_ascii=False))
                except Exception as e:
                    failed += 1
                    print(json.dumps({"source_id": row["source_id"], "ok": False,
                                      "error": str(e)}, ensure_ascii=False))
        finally:
            media_cache.close()
        return int(failed > 0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_suffix(".meta.json")
    stats_path = output.with_suffix(".stats.json")
    failure_path = output.with_suffix(".failures.jsonl")
    # Never overwrite an experiment; resume requires identical pool, prompt and settings.
    if args.resume:
        if not meta_path.is_file():
            raise ValueError("--resume requires the existing .meta.json")
        meta = json.loads(meta_path.read_text())
        if meta["config"] != config or meta["requested_revision"] != args.revision:
            raise ValueError("resume settings/input/prompt changed; use a new output path")
        revision = meta["model_commit"]
    else:
        if any(p.exists() for p in (output, meta_path, stats_path, failure_path)):
            raise FileExistsError("output or sidecars exist; use --resume or a new output path")
        revision = resolve_revision(args.model, args.revision)
        meta = {"config": config, "requested_revision": args.revision, "model_commit": revision,
                "prompt": prompt, "arguments": vars(args), "environment": environment_versions(),
                "script_sha256": digest_file(__file__), "created_at_unix": time.time(),
                "reward_mapping": {"1": 0.0, "2": 0.5, "3": 1.0}}
        atomic_json(meta_path, meta)
    if meta["script_sha256"] != digest_file(__file__):
        raise ValueError("script changed since run creation; use a new output path")
    records, failures = read_journal(output), read_journal(failure_path)
    allowed = {r["judge_id"] for r in rows}
    if any(r.get("judge_id") not in allowed for r in records + failures):
        raise ValueError("journals contain IDs outside the selected input")
    completed = {r["judge_id"] for r in records}
    if len(completed) != len(records):
        raise ValueError("duplicate successful IDs in output")
    for record in records:
        parse_scores(json.dumps(record["quality_scores"]))
    if args.resume:
        checked_media = set()
        for record in records:
            for modality in ("audio", "video"):
                path = record["media"][f"{modality}_path"]
                expected = record["media"][f"{modality}_sha256"]
                if (path, expected) not in checked_media:
                    if not Path(path).is_file() or digest_file(path) != expected:
                        raise ValueError(f"media used by completed records changed: {path}")
                    checked_media.add((path, expected))
    if not args.retry_failures:
        completed.update(r["judge_id"] for r in failures)
    pending = [r for r in rows if r["judge_id"] not in completed]
    if not pending:
        stats = summaries(records, failures, rows, args)
        atomic_json(stats_path, stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return int(stats["failures"] > 0)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for inference")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[{VERSION}] backend={args.backend} model={args.model} commit={revision} "
          f"selected={len(rows)} pending={len(pending)}", flush=True)
    backend_cls = MiniCPMBackend if args.backend == "minicpm" else QwenBackend
    backend = backend_cls(args, revision)
    with output.open("a", encoding="utf-8") as out, failure_path.open("a", encoding="utf-8") as fail:
        try:
            for index, row in enumerate(pending, 1):
                start = time.monotonic()
                attempts = []
                media_meta = None
                try:
                    media = media_cache.get(row)
                    media_meta = media["metadata"]
                    for attempt in range(args.max_retries + 1):
                        retry = "" if attempt == 0 else (
                            '\nReturn only the required JSON with integer text/audio/visual scores; no other text.')
                        raw, n_tokens = backend.judge(media, prompt, row, retry)
                        attempts.append({"raw_output": raw, "input_tokens": n_tokens})
                        try:
                            scores, mode = parse_scores(raw)
                            break
                        except ValueError as e:
                            attempts[-1]["parse_error"] = str(e)
                            if attempt == args.max_retries:
                                raise
                    result = {"judge_id": row["judge_id"], "source_id": row["source_id"],
                              "sample_idx": row.get("sample_idx"), "split": row.get("split"),
                              "judge_version": VERSION, "judge_model": args.model, "model_commit": revision,
                              "quality_scores": scores,
                              "quality_rewards": {m: (scores[m] - 1) / 2 for m in MODALITIES},
                              "quality_reward_mean": sum((scores[m] - 1) / 2 for m in MODALITIES) / 3,
                              "parse_mode": mode, "attempt_count": len(attempts), "attempts": attempts,
                              "media": media_meta, "elapsed_seconds": time.monotonic() - start}
                    append_json(out, result)
                    records.append(result)
                except Exception as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    result = {"judge_id": row["judge_id"], "source_id": row["source_id"],
                              "split": row.get("split"), "error_type": type(e).__name__,
                              "error": str(e), "attempts": attempts, "media": media_meta,
                              "elapsed_seconds": time.monotonic() - start}
                    append_json(fail, result)
                    failures.append(result)
                    print(f"[{VERSION}] FAILED {row['judge_id']}: {type(e).__name__}: {e}", flush=True)
                if index % args.log_every == 0 or index == len(pending):
                    stats = summaries(records, failures, rows, args)
                    atomic_json(stats_path, stats)
                    print(f"[{VERSION}] parsed={stats['parsed']}/{stats['total']} "
                          f"failures={stats['failures']} pending={stats['pending']} "
                          f"recovered={stats['retry_recovered']}", flush=True)
        finally:
            media_cache.close()
    stats = summaries(records, failures, rows, args)
    atomic_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return int(stats["failures"] > 0)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run or args.check_media:
        return run(args)
    # Acquire ownership before reading journals, creating metadata or loading models.
    import fcntl
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("another process is writing this experiment") from e
        return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
