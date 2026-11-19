from pipeline_code.step06_lesion.spec_utils import capture_input_id_mask

MODEL_KEY = "minicpm_o_2_6"
DECODER_POSITION_POLICY = "exact Step01 UNK audio-placeholder positions captured from decoder embedding input"

def module_layout(adapter):
    model = adapter.model
    return ([(model.apm.conv1, 1, True), (model.apm.conv2, 1, True)]
            + [(layer, -1, True) for layer in model.apm.layers]
            + [(model.audio_avg_pooler, 1, True)]
            + [(layer, -1, True) for layer in model.llm.model.layers])

def audio_position_context(adapter, _audio_path, controller):
    return capture_input_id_mask(
        adapter.model.llm.model.embed_tokens,
        int(adapter.tokenizer.unk_token_id), controller,
    )
