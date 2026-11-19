from contextlib import contextmanager
from pipeline_code.step06_lesion.spec_utils import qwen2_audio_mask

MODEL_KEY = "qwen2_audio_instruct"
DECODER_POSITION_POLICY = "exact Step01 audio-token positions; prompt and generated tokens excluded"

def module_layout(adapter):
    model = adapter.model
    return ([(model.audio_tower.conv1, 1, True), (model.audio_tower.conv2, 1, True)]
            + [(layer, -1, True) for layer in model.audio_tower.layers]
            + [(model.multi_modal_projector, -1, True)]
            + [(layer, -1, True) for layer in model.language_model.model.layers])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    controller.set_audio_mask(qwen2_audio_mask(adapter, audio_path))
    yield
