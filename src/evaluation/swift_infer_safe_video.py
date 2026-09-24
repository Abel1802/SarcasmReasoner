#!/usr/bin/env python3

import os

# Force Qwen2.5-Omni to use decord rather than torchvision.
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

# MCSD contains videos that can fail with Decord's default
# multi-threaded reader. Force single-threaded decoding.
import decord

_original_video_reader = decord.VideoReader


def _safe_video_reader(*args, **kwargs):
    kwargs.setdefault("num_threads", 1)
    return _original_video_reader(*args, **kwargs)


decord.VideoReader = _safe_video_reader


# Import Swift only AFTER patching Decord.
from swift.pipelines import infer_main


if __name__ == "__main__":
    infer_main()