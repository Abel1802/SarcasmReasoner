#!/usr/bin/env python3

"""Launch ms-swift RLHF with single-threaded Decord video decoding."""

import os

os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

import decord

_original_video_reader = decord.VideoReader


def _safe_video_reader(*args, **kwargs):
    kwargs.setdefault("num_threads", 1)
    return _original_video_reader(*args, **kwargs)


decord.VideoReader = _safe_video_reader

print("[SarcasmReasoner] Decord video decoding: num_threads=1", flush=True)

from swift.cli.utils import try_use_single_device_mode

try_use_single_device_mode()

from swift.pipelines import rlhf_main


if __name__ == "__main__":
    rlhf_main()
