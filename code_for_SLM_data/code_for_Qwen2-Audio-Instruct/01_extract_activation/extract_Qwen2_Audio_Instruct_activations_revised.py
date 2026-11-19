#!/usr/bin/env python3
"""Extract Qwen2-Audio-Instruct encoder and decoder audio-position states."""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec, save_final_outputs, selected_stimuli,
    validate_activation_set, validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import (  # noqa: E402
    QWEN2_AUDIO_INSTRUCT_SYSTEM_PROMPT,
    final_prompt,
)

MODEL_NAME = "Qwen2-Audio-Instruct"
FILE_PREFIX = "Qwen2_Audio_Instruct"
DEFAULT_MODEL_PATH = "/data/zss/SLMs/All_Models/Qwen2-Audio-7B-Instruct"
DEFAULT_AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
DEFAULT_OUTPUT_DIR = str(
    PROJECT_ROOT
    / "code_for_Qwen2-Audio-Instruct/01_extract_activation/Qwen2_Audio_Instruct_hidden_dimension_activations"
)
TASK_PROMPT = final_prompt("qwen2_audio_instruct")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def extract_transcription_text(decoded: str) -> str:
    """Keep only the quoted transcript emitted by the official example prompt."""
    text = str(decoded).strip()
    quoted = re.findall(r'["“\'‘]([^"”\'’]+)["”\'’]', text)
    return quoted[-1].strip() if quoted else text


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path, audio_dir, output_dir = (
        Path(args.model_path).resolve(), Path(args.audio_dir).resolve(), Path(args.output_dir).resolve()
    )
    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(audio_dir, stimuli)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, device_map="auto", torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).eval()
    audio_layers = list(model.audio_tower.layers)
    decoder_layers = list(model.language_model.model.layers)
    audio_dim = int(model.config.audio_config.d_model)
    text_dim = int(model.config.text_config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native", int(model.audio_tower.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native", int(model.audio_tower.conv2.out_channels)),
    ]
    specs.extend(LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(audio_layers)))
    specs.append(LayerSpec("multimodal_projector", "audio_projected", text_dim))
    specs.extend(LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers)))
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _ground_truth) in enumerate(stimuli, start=1):
        audio_path = audio_dir / audio_file
        conversation = [
            {"role": "system", "content": QWEN2_AUDIO_INSTRUCT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": TASK_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
        inputs = processor(
            text=text, audio=[audio], sampling_rate=sampling_rate,
            return_tensors="pt", padding=True, return_attention_mask=True,
        )
        feature_length = int(inputs["feature_attention_mask"].sum().item())
        conv2_len_t, projector_len_t = model.audio_tower._get_feat_extract_output_lengths(
            torch.tensor([feature_length], dtype=torch.long)
        )
        conv1_length = feature_length
        conv2_length = int(conv2_len_t.item())
        projector_length = int(projector_len_t.item())
        audio_token_id = int(model.config.audio_token_index)
        audio_mask = inputs["input_ids"][0].eq(audio_token_id).cpu()
        if int(audio_mask.sum()) != projector_length:
            raise RuntimeError("Audio-token count does not match projector length")
        captured, handles = {}, []

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        handles.append(model.audio_tower.conv1.register_forward_hook(save_once(
            "audio_conv1", lambda x: F.gelu(x[0, :, :conv1_length]).transpose(0, 1)
        )))
        handles.append(model.audio_tower.conv2.register_forward_hook(save_once(
            "audio_conv2", lambda x: F.gelu(x[0, :, :conv2_length]).transpose(0, 1)
        )))
        for index, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(save_once(
                f"audio_block_{index:02d}", lambda x, n=conv2_length: x[0, :n]
            )))
        handles.append(model.multi_modal_projector.register_forward_hook(save_once(
            "multimodal_projector", lambda x: x[0, :projector_length]
        )))
        for index, layer in enumerate(decoder_layers):
            def select_audio(x, mask=audio_mask):
                if x.shape[1] != mask.numel():
                    raise RuntimeError(f"First decoder call is not prefill: {x.shape[1]} vs {mask.numel()}")
                return x[0, mask.to(x.device)]
            handles.append(layer.register_forward_hook(save_once(f"decoder_block_{index:02d}", select_audio)))
        model_inputs = inputs.to(next(model.parameters()).device)
        # Pass 1: behavioral transcription only; activation hooks are not yet installed.
        for handle in handles:
            handle.remove()
        handles = []
        captured = {}
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs, max_new_tokens=32, do_sample=False, num_beams=5,
                early_stopping=True, repetition_penalty=1.0, use_cache=True,
            )

        # Re-register hooks only for the independent, non-autoregressive prefill.
        handles.append(model.audio_tower.conv1.register_forward_hook(save_once(
            "audio_conv1", lambda x: F.gelu(x[0, :, :conv1_length]).transpose(0, 1)
        )))
        handles.append(model.audio_tower.conv2.register_forward_hook(save_once(
            "audio_conv2", lambda x: F.gelu(x[0, :, :conv2_length]).transpose(0, 1)
        )))
        for index, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(save_once(
                f"audio_block_{index:02d}", lambda x, n=conv2_length: x[0, :n]
            )))
        handles.append(model.multi_modal_projector.register_forward_hook(save_once(
            "multimodal_projector", lambda x: x[0, :projector_length]
        )))
        for index, layer in enumerate(decoder_layers):
            def select_audio(x, mask=audio_mask):
                if x.shape[1] != mask.numel():
                    raise RuntimeError(f"Decoder prefill length mismatch: {x.shape[1]} vs {mask.numel()}")
                return x[0, mask.to(x.device)]
            handles.append(layer.register_forward_hook(save_once(f"decoder_block_{index:02d}", select_audio)))
        try:
            with torch.inference_mode():
                model(**model_inputs, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        new_ids = generated[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        transcription = extract_transcription_text(decoded)
        validate_activation_set(captured, specs)
        all_activations.append(captured); transcriptions.append(transcription)
        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; "
            f"shapes={captured['audio_conv1'].shape}/{captured['audio_block_31'].shape}/"
            f"{captured['multimodal_projector'].shape}/{captured['decoder_block_31'].shape}",
            flush=True,
        )

    result = save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=output_dir,
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(model).__name__), ("model_checkpoint", str(model_path)),
            ("parameter_count", str(sum(p.numel() for p in model.parameters()))),
            ("architecture_family", "Qwen2-Audio encoder-decoder instruct"),
            ("sampling_rate_hz", str(sampling_rate)),
            ("prompt", TASK_PROMPT),
            ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", "behavior: deterministic 5-beam generate; activation: separate non-generate prefill forward, use_cache=False"),
            ("transcription_postprocessing", "extract quoted transcript for the official prompt; otherwise retain stripped decoded text"),
            ("decoder_sequence_definition", "prefill audio-token positions only"),
            ("padding_policy", "all encoder/projector positions cropped to per-stimulus valid length"),
        ],
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
