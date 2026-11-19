from contextlib import contextmanager
from pipeline_code.step06_lesion.spec_utils import qwen25_audio_mask

MODEL_KEY = "qwen2_5_omni"
DECODER_POSITION_POLICY = "exact Step01 Thinker audio-token positions; Talker/prompt/generated tokens excluded"

def module_layout(adapter):
    model = adapter.model
    encoder = model.thinker.audio_tower
    return ([(encoder.conv1, 1, True), (encoder.conv2, 1, True)]
            + [(layer, -1, True) for layer in encoder.layers]
            + [(encoder.proj, -1, True)]
            + [(layer, -1, True) for layer in model.thinker.model.layers])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    controller.set_audio_mask(qwen25_audio_mask(adapter, audio_path))
    yield
