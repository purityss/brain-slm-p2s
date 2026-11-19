from contextlib import contextmanager

MODEL_KEY = "glm_4_voice"
DECODER_POSITION_POLICY = "exact Step01 positions strictly between begin/end audio markers"

def module_layout(adapter):
    return ([(adapter.speech_model.conv1, 1, True), (adapter.speech_model.conv2, 1, True)]
            + [(layer, -1, True) for layer in adapter.speech_model.layers]
            # Step01's returned continuous post-VQ state is not upstream of the
            # discrete token IDs consumed by the decoder, hence non-causal.
            + [(None, -1, False)]
            + [(layer, -1, True) for layer in adapter.model.transformer.encoder.layers])

@contextmanager
def audio_position_context(adapter, _audio_path, controller):
    begin_id = int(adapter.tokenizer.convert_tokens_to_ids("<|begin_of_audio|>"))
    end_id = int(adapter.tokenizer.convert_tokens_to_ids("<|end_of_audio|>"))

    def pre_hook(_module, args, kwargs):
        import torch
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args and getattr(args[0], "ndim", 0) == 2:
            input_ids = args[0]
        if input_ids is None or input_ids.shape[1] <= 1:
            return
        begin = torch.where(input_ids[0].eq(begin_id))[0]
        end = torch.where(input_ids[0].eq(end_id))[0]
        if begin.numel() == 1 and end.numel() == 1:
            mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
            mask[int(begin.item()) + 1:int(end.item())] = True
            controller.set_audio_mask(mask)

    handle = adapter.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
