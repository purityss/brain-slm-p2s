"""Utilities shared by model-local Step06 lesion specifications."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np


@contextmanager
def static_audio_mask(controller, mask):
    controller.set_audio_mask(mask)
    yield


@contextmanager
def capture_input_id_mask(module, token_id, controller, validate=None):
    """Capture the exact decoder prefill IDs used by the native generation path."""
    def pre_hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            candidate = args[0]
            if getattr(candidate, "ndim", 0) == 2:
                input_ids = candidate
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] <= 1:
            return
        mask = input_ids[0].eq(int(token_id)).detach().cpu()
        if bool(mask.any()):
            if validate is not None:
                validate(mask)
            controller.set_audio_mask(mask)
    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def qwen2_audio_mask(adapter, audio_path: Path):
    import librosa
    sampling_rate = int(adapter.processor.feature_extractor.sampling_rate)
    audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
    if adapter.instruct:
        from pipeline_code.step01_extract_activation.final_asr_prompts import (
            QWEN2_AUDIO_INSTRUCT_SYSTEM_PROMPT,
        )
        conversation = [
            {"role": "system", "content": QWEN2_AUDIO_INSTRUCT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": adapter.prompt},
            ]},
        ]
        text = adapter.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio_arg = [audio]
    else:
        text = "<|audio_bos|><|AUDIO|><|audio_eos|>" + adapter.prompt
        audio_arg = audio
    inputs = adapter.processor(
        text=text, audio=audio_arg, sampling_rate=sampling_rate,
        return_tensors="pt", padding=True, return_attention_mask=True,
    )
    mask = inputs["input_ids"][0].eq(int(adapter.model.config.audio_token_index))
    if not bool(mask.any()):
        raise RuntimeError("Qwen2-Audio processor produced no audio-token positions")
    return mask


def minicpm45_audio_mask(adapter, audio_path: Path):
    import librosa
    import torch
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    messages = [{"role": "user", "content": [adapter.prompt, audio]}]
    model, processor, tokenizer = adapter.model, adapter.processor, adapter.tokenizer
    model.prepare_processor(processor=processor, tokenizer=tokenizer)
    copied = deepcopy(messages)
    images, audios, audio_parts = [], [], []
    for message_index, message in enumerate(copied):
        parts = message["content"] if isinstance(message["content"], list) else [message["content"]]
        rendered = []
        for part in parts:
            if isinstance(part, np.ndarray):
                audios.append(part)
                audio_parts.append(message_index)
                rendered.append("<audio>./</audio>")
            else:
                rendered.append(str(part))
        message["content"] = "\n".join(rendered)
    prompt = model.processor.tokenizer.apply_chat_template(
        copied, tokenize=False, add_generation_prompt=True,
        use_tts_template=True, enable_thinking=False,
    )
    data = model.processor(
        [prompt], [images], [audios], [audio_parts], max_slice_nums=None,
        use_image_id=None, stream_input=False, return_tensors="pt", max_length=8192,
    )
    mask = torch.zeros(data["input_ids"].shape[1], dtype=torch.bool)
    bounds = data.get("audio_bounds")
    if bounds is not None:
        sample_bounds = bounds[0] if isinstance(bounds, (list, tuple)) else bounds[0]
        for bound in sample_bounds:
            mask[int(bound[0]):int(bound[1])] = True
    if not bool(mask.any()):
        raise RuntimeError("MiniCPM-o 4.5 processor produced no audio-bound positions")
    return mask


def kimi_audio_mask(adapter, audio_path: Path):
    messages = [
        {"role": "user", "message_type": "text", "content": adapter.prompt},
        {"role": "user", "message_type": "audio", "content": str(audio_path)},
    ]
    history = adapter.kimi.prompt_manager.get_prompt(messages, output_type="text")
    _audio_ids, _text_ids, is_continuous_mask, _a, _b = history.to_tensor()
    mask = is_continuous_mask[0].detach().cpu().bool()
    if not bool(mask.any()):
        raise RuntimeError("Kimi official is_continuous_mask contains no audio positions")
    return mask


def qwen25_audio_mask(adapter, audio_path: Path):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": adapter.system_prompt}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": str(audio_path)},
            {"type": "text", "text": adapter.prompt},
        ]},
    ]
    text = adapter.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = adapter.process_mm_info(messages, use_audio_in_video=True)
    inputs = adapter.processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    mask = inputs["input_ids"][0].eq(int(adapter.model.thinker.config.audio_token_index))
    if not bool(mask.any()):
        raise RuntimeError("Qwen2.5-Omni processor produced no audio-token positions")
    return mask
