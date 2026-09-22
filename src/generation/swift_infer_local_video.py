"""
Wrapper around ms-swift inference.

For local video files read with decord, keep the original file path
instead of converting the video to BytesIO. This avoids decoding
failures observed for some MP4 files with decord 0.6.0.
"""

from swift.model.models import qwen as qwen_module


_original_get_new_read_video_func = qwen_module._get_new_read_video_func


def _get_new_read_video_func(read_video_func, read_backend):
    if read_backend != "decord":
        return _original_get_new_read_video_func(
            read_video_func,
            read_backend,
        )

    def _reader(ele):
        video = ele.get("video")

        # Our datasets use local file paths.
        # Keep local paths intact for decord.
        if isinstance(video, str) and not video.startswith(
            ("http://", "https://", "data:")
        ):
            return read_video_func(ele)

        # Preserve the original ms-swift behavior for remote/base64 inputs.
        return _original_get_new_read_video_func(
            read_video_func,
            read_backend,
        )(ele)

    return _reader


qwen_module._get_new_read_video_func = _get_new_read_video_func


from swift.cli.infer import infer_main


if __name__ == "__main__":
    infer_main()
