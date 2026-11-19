#!/usr/bin/env python3
"""Extract MiniCPM-o 4.5 source-audio states with separate behavior/prefill passes."""

from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec,
    save_final_outputs,
    selected_stimuli,
    validate_activation_set,
    validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt  # noqa: E402

MODEL_NAME = "MiniCPM-o 4.5"
FILE_PREFIX = "MiniCPM_o_4_5"
MODEL_PATH = "/data/zss/SLMs/All_Models/MiniCPM-o-4_5"
AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
OUTPUT_DIR = str(PROJECT_ROOT / "code_for_MiniCPM-o 4.5/01_extract_activation/MiniCPM_o_4_5_hidden_dimension_activations")
TASK_PROMPT = final_prompt("minicpm_o_4_5")


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def build_prefill_inputs(model, processor, tokenizer, msgs):
    model.prepare_processor(processor=processor, tokenizer=tokenizer)
    copied = deepcopy(msgs)
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
        copied,
        tokenize=False,
        add_generation_prompt=True,
        use_tts_template=True,
        enable_thinking=False,
    )
    data = model.processor(
        [prompt], [images], [audios], [audio_parts],
        max_slice_nums=None, use_image_id=None, stream_input=False,
        return_tensors="pt", max_length=8192,
    ).to(model.device)
    data.pop("image_sizes", None)
    if "position_ids" not in data:
        position_ids = data["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(data["attention_mask"].eq(0), 0)
        data["position_ids"] = position_ids
    return data


def audio_position_mask(data) -> torch.Tensor:
    mask = torch.zeros(data["input_ids"].shape[1], dtype=torch.bool)
    bounds = data.get("audio_bounds")
    if bounds is not None:
        sample_bounds = bounds[0] if isinstance(bounds, (list, tuple)) else bounds[0]
        for bound in sample_bounds:
            start, stop = int(bound[0]), int(bound[1])
            mask[start:stop] = True
    if int(mask.sum()) == 0:
        raise RuntimeError("MiniCPM-o 4.5 processor did not provide valid audio bounds")
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path, audio_dir, output_dir = Path(args.model_path), Path(args.audio_dir), Path(args.output_dir)
    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(audio_dir, stimuli)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    # The downloaded checkpoint predates three sampling-only fields now read by
    # its remote-code TTS constructor. They do not affect text understanding or
    # the Thinker prefill, but must exist for the top-level model to initialize.
    for field, value in (("top_p", 0.8), ("top_k", 20), ("repetition_penalty", 1.05)):
        if not hasattr(config.tts_config, field):
            setattr(config.tts_config, field, value)
    # Transformers 5 finalizes checkpoint loading through an attribute added
    # after the released MiniCPM-o 4.5 remote class was authored. Resolve the
    # checkpoint class first so the compatibility field is present on that
    # exact class (patching only the generic AutoModel base is insufficient).
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModel"], model_path, local_files_only=True
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    model = model_class.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, config=config,
    ).eval().to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    audio_layers = list(model.apm.layers)
    decoder_layers = list(model.llm.model.layers)
    audio_dim = int(model.apm.config.d_model)
    text_dim = int(model.llm.config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native", int(model.apm.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native", int(model.apm.conv2.out_channels)),
    ]
    specs.extend(LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(audio_layers)))
    specs.append(LayerSpec("multimodal_projector", "audio_projected", text_dim))
    specs.extend(LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers)))
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _truth) in enumerate(stimuli, 1):
        audio, _ = librosa.load(audio_dir / audio_file, sr=16000, mono=True)
        # Follow the official MiniCPM-o ASR example exactly: the task text
        # precedes the audio object in the multimodal user content list.
        msgs = [{"role": "user", "content": [TASK_PROMPT, audio]}]

        # Pass 1: model-native behavioral generation without hooks.
        with torch.inference_mode():
            response = model.chat(
                msgs=msgs, tokenizer=tokenizer, processor=processor,
                do_sample=False, max_new_tokens=32, enable_thinking=False,
                use_tts_template=True, generate_audio=False, stream=False,
            )
        transcription = str(response).strip()

        # Pass 2: rebuild the same multimodal prefill, then call forward once.
        data = build_prefill_inputs(model, processor, tokenizer, msgs)
        audio_mask = audio_position_mask(data)
        captured, handles = {}, []

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        handles.append(model.apm.conv1.register_forward_hook(save_once(
            "audio_conv1", lambda x: F.gelu(x[0]).transpose(0, 1))))
        handles.append(model.apm.conv2.register_forward_hook(save_once(
            "audio_conv2", lambda x: F.gelu(x[0]).transpose(0, 1))))
        for i, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(save_once(f"audio_block_{i:02d}", lambda x: x[0])))
        handles.append(model.audio_avg_pooler.register_forward_hook(save_once(
            "multimodal_projector", lambda x: x[0].transpose(0, 1))))
        for i, layer in enumerate(decoder_layers):
            def select_audio(x, mask=audio_mask):
                if x.shape[1] != mask.numel():
                    raise RuntimeError(f"MiniCPM-o 4.5 decoder length {x.shape[1]} != input length {mask.numel()}")
                return x[0, mask.to(x.device)]
            handles.append(layer.register_forward_hook(save_once(f"decoder_block_{i:02d}", select_audio)))
        try:
            with torch.inference_mode():
                model(
                    data=data, attention_mask=data["attention_mask"],
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        validate_activation_set(captured, specs)
        all_activations.append(captured); transcriptions.append(transcription)
        last_audio = f"audio_block_{len(audio_layers) - 1:02d}"
        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; "
            f"layers={len(captured)}, shapes={captured['audio_conv1'].shape}/"
            f"{captured[last_audio].shape}/{captured['multimodal_projector'].shape}/"
            f"{captured['decoder_block_00'].shape}", flush=True,
        )

    result = save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=output_dir,
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(model).__name__), ("model_checkpoint", str(model_path)),
            ("parameter_count", str(sum(p.numel() for p in model.parameters()))),
            ("architecture_family", "MiniCPM-o 4.5 Whisper-medium audio encoder + pooling/projector + Qwen3 decoder"),
            ("prompt", TASK_PROMPT),
            ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", "behavior: deterministic native chat using the official Chinese-ASR message layout; activation: separate non-generate prefill forward, use_cache=False"),
            ("decoder_sequence_definition", "prefill processor audio-bound positions only"),
            ("padding_policy", "single-stimulus offline processor; only exact processor audio bounds retained in decoder"),
        ],
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
