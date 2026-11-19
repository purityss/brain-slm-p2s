"""Audio-only compatibility shim for Baichuan's unused video dependency."""


def cpu(index=0):
    return ("cpu", index)


class VideoReader:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "VideoReader is unavailable in the audio-only activation pipeline"
        )
