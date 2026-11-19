from contextlib import contextmanager

MODEL_KEY = "qwen_audio"
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
    query = f"<audio>{audio_path}</audio>{adapter.control_prompt}"
    audio_info = adapter.tokenizer.process_audio(query)
    inputs = adapter.tokenizer(query, return_tensors="pt", audio_info=audio_info)
    mask = inputs["input_ids"][0].eq(int(adapter.tokenizer.audio_pad_id))
    expected = int(audio_info["input_audio_lengths"][0, 1].item())
    if int(mask.sum()) != expected:
        raise RuntimeError(f"Qwen-Audio audio-pad count {int(mask.sum())} != projected length {expected}")
    controller.set_audio_mask(mask)
    yield
