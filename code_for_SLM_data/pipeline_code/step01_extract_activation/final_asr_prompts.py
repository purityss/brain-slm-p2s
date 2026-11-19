"""Final ASR prompts used by the active activation-extraction scripts."""

from __future__ import annotations

import os


FINAL_USER_PROMPTS = {
    "qwen2_audio": "Transcribe the speech in the audio:",
    "qwen2_audio_instruct": "What does the person say?",
    "qwen_audio": (
        "<|startoftranscript|><|zh|><|transcribe|><|zh|><|notimestamps|><|wo_itn|>"
    ),
    "qwen_audio_chat": "Recognize the speech",
    "baichuan_audio_base": "将语音转录为文本:",
    "baichuan_audio_instruct": (
        "请逐字转写以下中文语音，只输出听到的词，"
        "不要解释、扩展或补充任何内容："
    ),
    "firered_audio": "Transcribe speech to text.",
    "covo_audio_chat": "Transcribe the speech in the audio:",
    "glm_4_voice": (
        "你是一个语音识别模型。严格转写用户提供的中文语音，"
        "不允许解释、总结、发挥、重复或添加内容。将以下音频转写为文本："
    ),
    "freeze_omni": (
        "Transcribe the speech. Output only the transcription "
        "and nothing else."
    ),
    "kimi_audio_7b_instruct": "请将音频内容转换为文字。",
    "minicpm_o_2_6": "请仔细听这段音频片段，并将其内容逐字记录。",
    "minicpm_o_4_5": "请仔细听这段音频片段，并将其内容逐字记录。",
    "qwen2_5_omni": "Transcribe the speech in the audio:",
}

ENGLISH_USER_PROMPTS = {
    "qwen2_audio": "Transcribe the English speech exactly. Output only the word you hear:",
    "qwen2_audio_instruct": "Transcribe the English speech exactly. Output only the word you hear.",
    "qwen_audio": (
        "<|startoftranscript|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
    ),
    "qwen_audio_chat": "Transcribe the English speech exactly. Output only the word you hear.",
    "baichuan_audio_base": "Transcribe the English speech exactly. Output only the word you hear:",
    "baichuan_audio_instruct": (
        "Transcribe the following English speech verbatim. Output only the word "
        "you hear, with no explanation or additional text:"
    ),
    "firered_audio": "Transcribe the English speech exactly. Output only the word you hear.",
    "covo_audio_chat": "Transcribe the English speech exactly. Output only the word you hear:",
    "glm_4_voice": (
        "You are an English speech recognition model. Transcribe the supplied audio "
        "verbatim and output only the word you hear, without explanation."
    ),
    "freeze_omni": "Transcribe the English speech. Output only the word you hear.",
    "kimi_audio_7b_instruct": "Transcribe the English audio. Output only the word you hear.",
    "minicpm_o_2_6": "Listen carefully and transcribe the English speech verbatim. Output only the word.",
    "minicpm_o_4_5": "Listen carefully and transcribe the English speech verbatim. Output only the word.",
    "qwen2_5_omni": "Transcribe the English speech exactly. Output only the word you hear:",
}

INPUT_LANGUAGE = os.environ.get("SLM_INPUT_LANGUAGE", "zh").strip().lower()
ACTIVE_USER_PROMPTS = ENGLISH_USER_PROMPTS if INPUT_LANGUAGE.startswith("en") else FINAL_USER_PROMPTS

QWEN2_AUDIO_WRAPPED_PROMPT = (
    "<|audio_bos|><|AUDIO|><|audio_eos|>"
    + ACTIVE_USER_PROMPTS["qwen2_audio"]
)
QWEN_AUDIO_CONTROL_PROMPT = ACTIVE_USER_PROMPTS["qwen_audio"]
QWEN2_AUDIO_INSTRUCT_SYSTEM_PROMPT = "You are a speech recognition model."
QWEN25_OMNI_SYSTEM_PROMPT = (
    "You are an English speech recognition model."
    if INPUT_LANGUAGE.startswith("en")
    else "You are a Chinese speech recognition model."
)


def final_prompt(model_key: str) -> str:
    try:
        return ACTIVE_USER_PROMPTS[model_key]
    except KeyError as error:
        raise KeyError(f"No final ASR prompt registered for {model_key!r}") from error
