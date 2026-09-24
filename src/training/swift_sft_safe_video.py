#!/usr/bin/env python

"""
ms-swift SFT launcher with safe single-threaded decord decoding.

Some MCSD videos trigger FFmpeg/decord threaded-decoder errors.
This launcher forces decord.VideoReader(..., num_threads=1)
while leaving all training settings unchanged.
"""

import decord


# ============================================================
# Patch decord BEFORE ms-swift / qwen video utilities are loaded
# ============================================================

_original_video_reader = decord.VideoReader


def _safe_video_reader(*args, **kwargs):
    kwargs.setdefault("num_threads", 1)
    return _original_video_reader(*args, **kwargs)


decord.VideoReader = _safe_video_reader

print(
    "[SarcasmReasoner] "
    "Patched decord.VideoReader: default num_threads=1",
    flush=True,
)


# ============================================================
# Match `swift sft` initialization
# ============================================================

from swift.cli.utils import try_use_single_device_mode
from swift.cli.sft import try_init_unsloth
from swift.ray import try_init_ray

try_use_single_device_mode()
try_init_unsloth()
try_init_ray()


# ============================================================
# Launch SFT
# ============================================================

from swift.pipelines import sft_main

sft_main()
