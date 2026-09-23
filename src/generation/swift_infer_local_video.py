"""
Wrapper around ms-swift inference.

This wrapper applies two compatibility fixes:

1. For local video files read with decord, keep the original file path
   instead of converting the video to BytesIO.

2. Add a compatibility property required by vLLM 0.13.0 for Qwen2
   tokenizers when using recent Transformers versions.
"""

# ---------------------------------------------------------------------
# Fix 1: Qwen2Tokenizer compatibility with vLLM tokenizer caching
# ---------------------------------------------------------------------

from transformers import PreTrainedTokenizerBase


if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):

    @property
    def all_special_tokens_extended(self):
        # vLLM only needs a cached collection of special tokens here.
        # Fall back to the standard special-token list.
        return self.all_special_tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = (
        all_special_tokens_extended
    )


# ---------------------------------------------------------------------
# Fix 2: Keep local video paths for decord
# ---------------------------------------------------------------------

from swift.model.models import qwen as qwen_module


_original_get_new_read_video_func = (
    qwen_module._get_new_read_video_func
)


def _get_new_read_video_func(read_video_func, read_backend):
    if read_backend != "decord":
        return _original_get_new_read_video_func(
            read_video_func,
            read_backend,
        )

    original_reader = _original_get_new_read_video_func(
        read_video_func,
        read_backend,
    )

    def _reader(ele):
        video = ele.get("video")

        if (
            isinstance(video, str)
            and not video.startswith(
                ("http://", "https://", "data:")
            )
        ):
            return read_video_func(ele)

        return original_reader(ele)

    return _reader


qwen_module._get_new_read_video_func = (
    _get_new_read_video_func
)


# ---------------------------------------------------------------------
# Run ms-swift inference
# ---------------------------------------------------------------------

from swift.cli.infer import infer_main


if __name__ == "__main__":
    infer_main()
