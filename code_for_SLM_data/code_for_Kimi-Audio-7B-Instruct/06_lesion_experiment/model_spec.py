from contextlib import contextmanager
from pipeline_code.step06_lesion.spec_utils import kimi_audio_mask

MODEL_KEY = "kimi_audio_7b_instruct"
DECODER_POSITION_POLICY = "exact Step01 official is_continuous_mask source-audio positions"

def module_layout(adapter):
    encoder = adapter.kimi.prompt_manager.whisper_model.speech_encoder
    model = adapter.kimi.alm.model
    return ([(encoder.conv1, 1, True), (encoder.conv2, 1, True)]
            + [(layer, -1, True) for layer in encoder.layers]
            + [(model.vq_adaptor, -1, True)]
            + [(layer, -1, True) for layer in model.layers])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    controller.set_audio_mask(kimi_audio_mask(adapter, audio_path))
    yield
