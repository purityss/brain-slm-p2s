#!/usr/bin/env python3
"""Shared extractor implementation for Qwen-Audio base and Chat checkpoints."""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

# Remote-code audio loading launches ffmpeg by name. Directly invoking a conda
# environment's Python does not always prepend that environment's bin folder.
os.environ["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + os.environ.get("PATH", "")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .activation_extraction_common import (
    LayerSpec, save_final_outputs, selected_stimuli,
    validate_activation_set, validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import QWEN_AUDIO_CONTROL_PROMPT, final_prompt


ASR_CONTROL_PROMPT = QWEN_AUDIO_CONTROL_PROMPT
CHAT_TASK_PROMPT = final_prompt("qwen_audio_chat")


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def clean_chat_transcription(text: str) -> str:
    """Remove Qwen-Audio-Chat's habitual English acknowledgement wrapper."""
    text = str(text).strip()
    chinese_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text)
    # A long transcript commonly contains several Chinese runs separated by
    # punctuation. Keeping only the final run silently discarded almost all
    # of meeting-style utterances; concatenate every run while still removing
    # the model's habitual English acknowledgement wrapper.
    return "".join(chinese_runs) if chinese_runs else text


def run_qwen_audio_extraction(
    *, model_name: str, file_prefix: str, model_path: str, audio_dir: str,
    output_dir: str, chat_mode: bool, max_stimuli: int | None, seed: int,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_path_obj, audio_dir_obj, output_dir_obj = (
        Path(model_path).resolve(), Path(audio_dir).resolve(), Path(output_dir).resolve()
    )
    stimuli = selected_stimuli(max_stimuli)
    validate_audio_files(audio_dir_obj, stimuli)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_obj, trust_remote_code=True, local_files_only=True
    )

    # Qwen-Audio's remote tokenizer shells out to an ffmpeg executable merely
    # to decode the source waveform. The study stimuli are already 16-kHz mono
    # WAV files, so use the pinned librosa decoder and leave all subsequent
    # official Qwen feature extraction untouched. This removes an unnecessary
    # system-binary dependency from the reproducible Python environments.
    def load_stimulus_wav(file_path, sr=16000):
        waveform, _ = librosa.load(file_path, sr=sr, mono=True)
        return np.asarray(waveform, dtype=np.float32)

    process_audio_function = getattr(tokenizer.process_audio, "__func__", tokenizer.process_audio)
    process_audio_globals = getattr(process_audio_function, "__globals__", {})
    if "load_audio" not in process_audio_globals:
        raise RuntimeError("Could not patch Qwen-Audio's remote WAV loader")
    process_audio_globals["load_audio"] = load_stimulus_wav

    model = AutoModelForCausalLM.from_pretrained(
        model_path_obj, trust_remote_code=True, local_files_only=True,
        device_map="cuda:0", bf16=True, fp16=False, fp32=False,
    ).eval()
    audio_encoder = model.transformer.audio
    audio_layers = list(audio_encoder.blocks)
    decoder_layers = list(model.transformer.h)
    audio_dim = int(model.config.audio["n_state"])
    text_dim = int(model.config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native", int(audio_encoder.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native", int(audio_encoder.conv2.out_channels)),
    ]
    specs.extend(LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(audio_layers)))
    specs.append(LayerSpec("multimodal_projector", "audio_projected", text_dim))
    specs.extend(LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers)))
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _ground_truth) in enumerate(stimuli, start=1):
        audio_path = audio_dir_obj / audio_file
        if chat_mode:
            query = tokenizer.from_list_format([
                {"audio": str(audio_path)}, {"text": CHAT_TASK_PROMPT},
            ])
        else:
            query = f"<audio>{audio_path}</audio>{ASR_CONTROL_PROMPT}"
        audio_info = tokenizer.process_audio(query)
        raw_inputs = tokenizer(query, return_tensors="pt", audio_info=audio_info)
        conv2_length = int(audio_info["input_audio_lengths"][0, 0].item())
        projector_length = int(audio_info["input_audio_lengths"][0, 1].item())
        conv1_length = conv2_length * 2
        inputs = raw_inputs.to(model.device)

        # Pass 1: preserve each checkpoint's native behavioral interface, with
        # no activation hooks installed.
        with torch.inference_mode():
            if chat_mode:
                raw_transcription, _history = model.chat(
                    tokenizer, query=query, history=None, do_sample=False,
                    num_beams=1, max_new_tokens=32, repetition_penalty=1.0,
                    use_cache=True,
                )
                transcription = clean_chat_transcription(raw_transcription)
            else:
                generated = model.generate(
                    **inputs, audio_info=audio_info, do_sample=False, num_beams=1,
                    max_new_tokens=32, repetition_penalty=1.0, use_cache=True,
                )
                new_ids = generated[:, inputs["input_ids"].shape[1]:]
                transcription = tokenizer.decode(
                    new_ids[0].cpu(), skip_special_tokens=True, audio_info=audio_info
                ).strip()

        captured, handles = {}, []
        decoder_mask_holder = {"mask": None}

        def transformer_pre_hook(_module, args, kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is not None and input_ids.shape[1] > 1 and decoder_mask_holder["mask"] is None:
                mask = input_ids[0].eq(int(tokenizer.audio_pad_id)).detach().cpu()
                if int(mask.sum()) != projector_length:
                    raise RuntimeError(
                        f"decoder audio-pad count {int(mask.sum())} != projected length {projector_length}"
                    )
                decoder_mask_holder["mask"] = mask

        handles.append(model.transformer.register_forward_pre_hook(transformer_pre_hook, with_kwargs=True))

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        handles.append(audio_encoder.conv1.register_forward_hook(save_once(
            "audio_conv1", lambda x: F.gelu(x[0, :, :conv1_length]).transpose(0, 1)
        )))
        handles.append(audio_encoder.conv2.register_forward_hook(save_once(
            "audio_conv2", lambda x: F.gelu(x[0, :, :conv2_length]).transpose(0, 1)
        )))
        for index, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(save_once(
                f"audio_block_{index:02d}", lambda x, n=conv2_length: x[0, :n]
            )))
        handles.append(audio_encoder.proj.register_forward_hook(save_once(
            "multimodal_projector", lambda x: x[0, :projector_length]
        )))
        for index, layer in enumerate(decoder_layers):
            def select_decoder_audio(x):
                mask = decoder_mask_holder["mask"]
                if mask is None or x.shape[1] != mask.numel():
                    raise RuntimeError("Decoder block was called before a valid audio prefill mask")
                return x[0, mask.to(x.device)]
            handles.append(layer.register_forward_hook(save_once(
                f"decoder_block_{index:02d}", select_decoder_audio
            )))
        try:
            with torch.inference_mode():
                # Pass 2: one non-autoregressive multimodal prefill forward.
                model(
                    **inputs, audio_info=audio_info, use_cache=False,
                    output_hidden_states=False, return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        validate_activation_set(captured, specs)
        all_activations.append(captured); transcriptions.append(str(transcription))
        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; "
            f"shapes={captured['audio_conv1'].shape}/{captured['audio_block_31'].shape}/"
            f"{captured['multimodal_projector'].shape}/{captured['decoder_block_31'].shape}",
            flush=True,
        )

    result = save_final_outputs(
        model_name=model_name, file_prefix=file_prefix, output_dir=output_dir_obj,
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(model).__name__), ("model_checkpoint", str(model_path_obj)),
            ("parameter_count", str(sum(p.numel() for p in model.parameters()))),
            ("architecture_family", "Qwen-Audio legacy audio encoder + causal decoder"),
            ("prompt", CHAT_TASK_PROMPT if chat_mode else ASR_CONTROL_PROMPT),
            ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            (
                "generation_mode",
                "behavior: deterministic native generation; activation: separate "
                "non-generate multimodal prefill forward, use_cache=False",
            ),
            ("decoder_sequence_definition", "prefill audio-pad positions only; audio BOS/EOS excluded"),
            ("transcription_postprocessing", "Chat only: remove acknowledgement wrapper and concatenate all Chinese spans" if chat_mode else "none"),
            ("padding_policy", "all audio layers cropped using tokenizer input_audio_lengths"),
        ],
    )
    print(result, flush=True)
