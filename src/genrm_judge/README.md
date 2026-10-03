# Omni quality judge v1

One joint audio/video/text call per candidate trajectory. The model returns
`{"text": 3, "audio": 2, "visual": 3}`. Each score covers observation reliability
AND evidence-to-interpretation reasoning in that section. It does not score
integration or independently predict sarcasm. Both datasets use the same prompt.

## Files and installation

Copy `src/` into the SarcasmReasoner project root. This package does not change
policy training or existing evaluation scripts.

MiniCPM-o-4.5's official model card specifies Transformers 4.51.0 and Torch
2.3–2.8. Your existing ms2 environment has a different Transformers/Torch setup:
create a separate environment rather than downgrading ms2. Python 3.10 is the
model card's tested version. FFmpeg and ffprobe must be installed on PATH.

```bash
python3.10 -m venv .venv-quality-minicpm
source .venv-quality-minicpm/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r src/genrm_judge/requirements-minicpm.txt
```

For Qwen comparison, use a second environment and `requirements-qwen.txt`.
FlashAttention is optional; the default is SDPA. v1 uses **Transformers**, not
vLLM. It uses one sample per generation and caches decoded media for consecutive
trajectories from the same source. MiniCPM's implicit beam search is explicitly
disabled (`num_beams=1`); both models use greedy generation without thinking.

## Input contract

The existing grounding judge pool JSONL is supported directly:

```json
{"judge_id":"valid:832:0","source_id":"832","sample_idx":0,"split":"valid","text":"original transcript","audio":"data/mcsd/raw/audios/832.wav","video":"data/mcsd/raw/videos/832.mp4","sections":{"text_evidence":"candidate text section","audio_evidence":"candidate audio section","visual_evidence":"candidate visual section","integration":"not sent"}}
```

`label`, `prediction`, and `prediction_correct` may exist, but are used only in
post-inference diagnostics. They and integration are never sent to the model.
Relative media paths are resolved against `--media-root` (default current dir).
Rows must have unique judge IDs. Empty evidence strings are scored; missing
evidence fields are input errors. Train/valid source IDs are grouped separately.

## MCSD pilot: 20 sources, 2 trajectories per source

Run from the SarcasmReasoner project root:

```bash
CUDA_VISIBLE_DEVICES=0 python src/genrm_judge/run_omni_quality_judge.py \
  --dataset mcsd \
  --input data/mcsd/processed/genrm/grounding_judge_pool_valid.jsonl \
  --output results/mcsd/genrm/quality_judging_v1/minicpm_o45/valid/pilot_labels.jsonl \
  --prompt src/genrm_judge/prompts/quality.txt \
  --backend minicpm \
  --sources 20 --per-source 2 --seed 42
```

Add `--dry-run` to check fields, prompt and deterministic selection without
importing model dependencies or requiring media. Add `--check-media` to decode
selected sources without loading model weights. Both check modes write no labels.

The selected IDs are recorded in metadata. Equal seed, input and sampling flags
select the same trajectories for either backend.

## MCSD full valid / train

```bash
CUDA_VISIBLE_DEVICES=0 python src/genrm_judge/run_omni_quality_judge.py \
  --dataset mcsd \
  --input data/mcsd/processed/genrm/grounding_judge_pool_valid.jsonl \
  --output results/mcsd/genrm/quality_judging_v1/minicpm_o45/valid/quality_labels.jsonl \
  --backend minicpm --sources 0 --per-source 0
```

For train change both `valid` occurrences in the paths to `train`. Full mode
judges every trajectory, including correct and incorrect predictions; no
correctness gate is applied to offline supervision.

## MUStARD++ pilot

```bash
CUDA_VISIBLE_DEVICES=0 python src/genrm_judge/run_omni_quality_judge.py \
  --dataset mustard \
  --input data/mustard/processed/genrm/grounding_judge_pool_valid.jsonl \
  --output results/mustard/genrm/quality_judging_v1/minicpm_o45/valid/pilot_labels.jsonl \
  --backend minicpm --sources 20 --per-source 2 --seed 42
```

## Qwen3-Omni-Instruct comparison

Activate the separate Qwen environment, then run the same input/prompt/selection:

```bash
CUDA_VISIBLE_DEVICES=0 python src/genrm_judge/run_omni_quality_judge.py \
  --dataset mcsd \
  --input data/mcsd/processed/genrm/grounding_judge_pool_valid.jsonl \
  --output results/mcsd/genrm/quality_judging_v1/qwen3_omni_instruct/valid/pilot_labels.jsonl \
  --backend qwen --sources 20 --per-source 2 --seed 42
```

Model defaults:
- MiniCPM: `openbmb/MiniCPM-o-4_5`.
- Qwen: `Qwen/Qwen3-Omni-30B-A3B-Instruct`.

Both load an exact Hugging Face commit resolved from `--revision` (default main).
The resolved commit is saved and reused on resume. No talker/TTS is generated.
The same canonical video frames and explicit `row.audio` recording are used for
both models. Their native vision/audio encoders and chat templates still differ.

## Media coverage and context limits

- Both backends receive video sampled at **1 FPS**, resized to a maximum side
  of 448 pixels by default, and the complete external audio resampled to 16 kHz.
- MiniCPM receives interleaved image/audio segments in one user message.
  Qwen receives a video frame list and a separate audio part in one user message.
- Sampling can miss short gestures. An unobserved event cannot automatically
  be classified as absent. This v1 is a pilot, not a high-FPS gesture verifier.
- Duration mismatch greater than 0.75 seconds is a recorded media failure.
  Check synchronization before deliberately increasing `--duration-tolerance`.
- Clips exceeding `--max-frames` fail; they are not silently cropped or
  downsampled. Increase this limit if the model context and GPU memory allow it.
- MiniCPM's processor silently slices tokens by default. A wrapper disables that
  slicing and checks actual multimodal tokens + output reservation before
  generation. Qwen also checks processed token length. Both default to 32768.
- Increasing sampling resolution, frame count or MiniCPM slices changes the
  experiment configuration; use a fresh output path.

## Output, retries and resume

For `pilot_labels.jsonl`, sidecars are:
- `pilot_labels.meta.json`: original arguments, prompt, hashes, selection,
  exact model commit, environment and script checksum.
- `pilot_labels.stats.json`: per-modality score distributions/rewards and
  outcome-based diagnostics, calculated outside model inference.
- `pilot_labels.failures.jsonl`: media, context, inference or JSON failures.
- `pilot_labels.lock`: an advisory lock file; it contains no model output.

Each successful row includes `quality_scores`, per-modality `quality_rewards`,
their mean, all raw responses, parse mode, attempt count, media hashes and timing.
Rewards map 1→0, 2→0.5, 3→1. These fields are deliberately named quality fields;
they do not silently replace old binary grounding labels. A later data builder
must explicitly read `quality_scores` to construct upgraded GenRM SFT targets.

An invalid JSON response gets at most one extra call with a format reminder.
Completed `<think>...</think>` prefixes and a single JSON code fence can be
parsed, but extra keys, duplicate keys, floats, booleans and arbitrary surrounding
prose are rejected. Media/context/inference failures are not assigned score 1.

Repeat the exact original command with `--resume` to skip completed successes
and terminal failures. Add `--retry-failures` to revisit failures after fixing
the infrastructure. Recovered successes are excluded from failure counts.
Input, prompt, sampling, script and model revision cannot change within a run.
Media previously used by successful records is re-hashed before resume.
No concurrent writers are allowed. Exit 0 means no terminal failures; exit 1
means some trajectories failed; exit 2 means a configuration/startup error.

## Validation status

Standard-library tests cover schema, leakage prevention, sampling, parsing,
context guards and failure recovery counts. A real FFmpeg fixture checks media
decoding and cache reuse. `--dry-run` was also checked on the supplied 3192-row
MCSD valid pool. GPU inference cannot be tested in the artifact workspace, so
start with the 40-trajectory pilot on your GPU node before launching full runs.

Run helper tests:

```bash
python -m unittest discover -s src/genrm_judge/tests -v
```

Official references:
- https://huggingface.co/openbmb/MiniCPM-o-4_5
- https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
