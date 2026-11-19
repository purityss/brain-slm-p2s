from contextlib import contextmanager
from pipeline_code.step06_lesion.spec_utils import minicpm45_audio_mask

MODEL_KEY = "minicpm_o_4_5"
DECODER_POSITION_POLICY = "exact Step01 processor audio_bounds; prompt and generated tokens excluded"

def module_layout(adapter):
    model = adapter.model
    return ([(model.apm.conv1, 1, True), (model.apm.conv2, 1, True)]
            + [(layer, -1, True) for layer in model.apm.layers]
            + [(model.audio_avg_pooler, 1, True)]
            + [(layer, -1, True) for layer in model.llm.model.layers])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    controller.set_audio_mask(minicpm45_audio_mask(adapter, audio_path))
    yield
