from contextlib import contextmanager

MODEL_KEY = "covo_audio_chat"
DECODER_POSITION_POLICY = "exact Step01 cAUDIO placeholder positions, refreshed for every native audio chunk"

def module_layout(adapter):
    model = adapter.model
    return ([(model.encoder.conv1, 1, True), (model.encoder.conv2, 1, True)]
            + [(layer, -1, True) for layer in model.encoder.layers]
            + [(model.audio_adapter, -1, True)]
            + [(layer, -1, True) for layer in model.llm.model.layers])

@contextmanager
def audio_position_context(adapter, _audio_path, controller):
    # Covo's custom generate path bypasses the model-level forward pre-hook.
    # Capture the actual ChatML input IDs handed to native generation instead.
    original_generate = adapter.model.generate

    def generate_with_audio_mask(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or input_ids.ndim != 2:
            raise RuntimeError("Covo native generation did not receive input_ids")
        mask = input_ids[0].eq(int(adapter.c_audio_id)).detach().cpu()
        if not bool(mask.any()):
            raise RuntimeError("Covo native generation has no cAUDIO positions")
        controller.set_audio_mask(mask)
        try:
            return original_generate(*args, **kwargs)
        finally:
            controller.clear_audio_positions()

    adapter.model.generate = generate_with_audio_mask
    try:
        yield
    finally:
        adapter.model.generate = original_generate
