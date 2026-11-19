from contextlib import contextmanager

MODEL_KEY = "qwen_audio_chat"
DECODER_POSITION_POLICY = "exact Step01 audio-pad positions from native tokenizer audio_info"

def module_layout(adapter):
    model = adapter.model
    encoder = model.transformer.audio
    return ([(encoder.conv1, 1, True), (encoder.conv2, 1, True)]
            + [(layer, -1, True) for layer in encoder.blocks]
            + [(encoder.proj, -1, True)]
            + [(layer, -1, True) for layer in model.transformer.h])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    query = adapter.tokenizer.from_list_format(
        [{"audio": str(audio_path)}, {"text": adapter.prompt}]
    )
    audio_info = adapter.tokenizer.process_audio(query)
    expected = int(audio_info["input_audio_lengths"][0, 1].item())
    captured = {"mask": False}

    def capture_prefill_mask(_module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] <= 1:
            return
        mask = input_ids[0].eq(int(adapter.tokenizer.audio_pad_id))
        count = int(mask.sum())
        if count != expected:
            raise RuntimeError(
                f"Qwen-Audio-Chat runtime audio-pad count {count} "
                f"!= projected length {expected}"
            )
        controller.set_audio_mask(mask)
        captured["mask"] = True

    handle = adapter.model.transformer.register_forward_pre_hook(
        capture_prefill_mask, with_kwargs=True
    )
    try:
        yield
        if not captured["mask"]:
            raise RuntimeError("Qwen-Audio-Chat decoder prefill mask was not captured")
    finally:
        handle.remove()
