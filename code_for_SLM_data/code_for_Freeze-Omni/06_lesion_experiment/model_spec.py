from contextlib import contextmanager
import math

MODEL_KEY = "freeze_omni"
DECODER_POSITION_POLICY = "exact Step01 valid source-audio tail per streaming chunk; cached prefix/generated tokens excluded"

def module_layout(adapter):
    model = adapter.pipeline.model
    conv1 = model.encoder.enc[0].core.conv[0]
    conv2 = model.encoder.enc[0].core.conv[2]
    audio_layers = list(model.encoder.enc[1].encoders)
    projector = model.adpter.project
    decoder_layers = list(model.llm_decoder.model.layers)
    return ([(conv1, 1, True), (conv2, 1, True)]
            + [(layer, -1, True, "infer") for layer in audio_layers]
            + [(projector, -1, True)]
            + [(layer, -1, True) for layer in decoder_layers])

@contextmanager
def audio_position_context(adapter, audio_path, controller):
    import torch
    runtime = adapter.runtime
    padded, original_samples = runtime.load_waveform(
        audio_path, adapter.samples_per_chunk
    )
    valid_fbank = [
        int(valid) for _index, _samples, valid in runtime.chunk_plan(
            padded, original_samples, adapter.samples_per_chunk
        )
    ]
    projector = adapter.pipeline.model.adpter.project
    state = {"index": 0}

    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(value) or value.ndim != 3:
            return
        index = state["index"]
        if index >= len(valid_fbank):
            return
        actual = int(value.shape[1])
        valid = min(max(1, math.ceil(valid_fbank[index] / 8)), actual)
        controller.set_audio_tail(actual, valid)
        state["index"] += 1

    handle = projector.register_forward_hook(hook)
    original_speech_dialogue = adapter.pipeline.speech_dialogue

    def speech_dialogue(audio, *args, **kwargs):
        if audio is None:
            # Initial system prompt, chat suffix, and generation are not source
            # audio positions. Clear the final chunk tail before these calls.
            controller.clear_audio_positions()
        return original_speech_dialogue(audio, *args, **kwargs)

    adapter.pipeline.speech_dialogue = speech_dialogue
    try:
        yield
    finally:
        adapter.pipeline.speech_dialogue = original_speech_dialogue
        handle.remove()
    if state["index"] != len(valid_fbank):
        raise RuntimeError(
            f"Freeze-Omni observed {state['index']} audio chunks; expected {len(valid_fbank)}"
        )
