"""Canonical human-readable order for every cross-model command and table."""

CANONICAL_MODEL_KEYS = (
    "baichuan_audio_base",
    "baichuan_audio_instruct",
    "covo_audio_chat",
    "firered_audio",
    "freeze_omni",
    "glm_4_voice",
    "kimi_audio_7b_instruct",
    "minicpm_o_2_6",
    "minicpm_o_4_5",
    "qwen_audio",
    "qwen_audio_chat",
    "qwen2_audio",
    "qwen2_audio_instruct",
    "qwen2_5_omni",
)

MODEL_ORDER_INDEX = {
    model_key: index for index, model_key in enumerate(CANONICAL_MODEL_KEYS)
}


def validate_model_keys(model_keys) -> None:
    """Reject missing, extra, duplicated, or differently ordered model keys."""
    keys = tuple(model_keys)
    if keys != CANONICAL_MODEL_KEYS:
        raise RuntimeError(
            "Model order differs from pipeline_code.model_order.CANONICAL_MODEL_KEYS: "
            f"{keys}"
        )
