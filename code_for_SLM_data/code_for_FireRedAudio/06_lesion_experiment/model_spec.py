from contextlib import contextmanager
import numpy as np

MODEL_KEY = "firered_audio"
DECODER_POSITION_POLICY = "exact Step01 audio-special-token placeholder positions"

def module_layout(adapter):
    model = adapter.engine.model
    encoder = model.audio_encoder
    return ([(encoder.conv1, 1, True), (encoder.conv2, 1, True)]
            + [(layer, -1, True) for layer in encoder.layers]
            + [(encoder.adapter, -1, True)]
            + [(layer, -1, True) for layer in model.backbone_llm.model.layers])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    import soundfile as sf
    from scipy.signal import resample_poly
    model = adapter.engine.model
    chatml = adapter.build_understand_prompt(
        adapter.prompt, 1, model.config.audio_special_token, enable_thinking=False
    )
    waveform, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if rate != adapter.UNDERSTAND_SAMPLE_RATE:
        waveform = resample_poly(
            waveform, adapter.UNDERSTAND_SAMPLE_RATE, rate
        ).astype(np.float32)
    batch = adapter.engine.encoder.encode(chatml, [{
        "feat_type": adapter.FEAT_TYPE_UNDERSTAND,
        "audio_understand": waveform, "audio_generation": None, "role": "user",
    }])
    mask = batch["input_ids"][0].eq(int(model.config.audio_special_token_id))
    if not bool(mask.any()):
        raise RuntimeError("FireRedAudio input contains no audio placeholder positions")
    controller.set_audio_mask(mask)
    yield
