from contextlib import contextmanager
import json

MODEL_KEY = "baichuan_audio_instruct"
DECODER_POSITION_POLICY = "exact Step01 audio-pad positions; prompt and generated tokens excluded"

def module_layout(adapter):
    model = adapter.model
    encoder = model.audio_tokenizer.audio_model
    bridge = model.audio_tokenizer.audio_bridge_model
    return ([(encoder.conv2, 1, True)]
            + [(layer, -1, True) for layer in encoder.layers]
            + [(bridge.layer_norm, -1, True)]
            + [(layer, -1, True) for layer in model.model.layers])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    prompt = adapter.prompt + adapter.audio_start + json.dumps(
        {"path": str(audio_path)}, ensure_ascii=False
    ) + adapter.audio_end
    inputs = adapter.model.processor([prompt])
    mask = inputs.input_ids[0].eq(
        int(adapter.model.config.audio_config.audio_pad_token_id)
    )
    # Decoder audio positions are the actual audio-pad token locations consumed
    # by get_multimodal_embed.  For one valid long FLEURS utterance the
    # processor's bridge_length metadata is shorter than this mask, while the
    # model itself still consumes every audio-pad position.
    controller.set_audio_mask(mask)
    yield
