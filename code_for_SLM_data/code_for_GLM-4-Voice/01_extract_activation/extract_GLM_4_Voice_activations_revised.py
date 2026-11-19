#!/usr/bin/env python3
"""Extract GLM-4-Voice VQ speech-tokenizer and decoder audio-token states."""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import types
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec, save_final_outputs, selected_stimuli,
    validate_activation_set, validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt  # noqa: E402

MODEL_NAME = "GLM-4-Voice"
FILE_PREFIX = "GLM_4_Voice"
REFERENCE_ROOT = str(PROJECT_ROOT / "code_for_GLM-4-Voice/model_runtime/GLM-4-Voice-main")
MODEL_ROOT = "/data/zss/SLMs/All_Models/GLM-4-Voice"
AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
OUTPUT_DIR = str(PROJECT_ROOT / "code_for_GLM-4-Voice/01_extract_activation/GLM_4_Voice_hidden_dimension_activations")
TASK_PROMPT = final_prompt("glm_4_voice")


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def install_chatglm_generation_compat(model):
    """Restore the legacy cache extractor expected by GLM-4-Voice remote code.

    GLM-4-Voice's pinned ``modeling_chatglm.py`` overrides
    ``_update_model_kwargs_for_generation`` and calls a GenerationMixin helper
    that Transformers 4.51 removed.  Bind the former helper to this model
    instance only, without modifying Transformers or the downloaded checkpoint.
    """
    if hasattr(model, "_extract_past_from_model_output"):
        return

    def _extract_past_from_model_output(
        _model, outputs, standardize_cache_format=False,
    ):
        del standardize_cache_format
        for cache_name in ("past_key_values", "mems", "past_buckets_states"):
            cache = getattr(outputs, cache_name, None)
            if cache is not None:
                return cache_name, cache
        return "past_key_values", None

    model._extract_past_from_model_output = types.MethodType(  # type: ignore[attr-defined]
        _extract_past_from_model_output, model,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default=REFERENCE_ROOT)
    parser.add_argument("--model-root", default=MODEL_ROOT)
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    reference_root = Path(args.reference_root).resolve()
    model_root = Path(args.model_root).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(audio_dir, stimuli)
    sys.path.insert(0, str(reference_root))
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder  # type: ignore

    tokenizer_path = model_root / "glm-4-voice-tokenizer"
    language_path = model_root / "GLM-4-Voice-9B"
    tokenizer = AutoTokenizer.from_pretrained(language_path, trust_remote_code=True, local_files_only=True)
    language_model = AutoModelForCausalLM.from_pretrained(
        language_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval().to(device)
    install_chatglm_generation_compat(language_model)
    speech_model = WhisperVQEncoder.from_pretrained(tokenizer_path, local_files_only=True).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path, local_files_only=True)
    audio_layers = list(speech_model.layers)
    decoder_layers = list(language_model.transformer.encoder.layers)
    audio_dim = int(speech_model.config.d_model)
    text_dim = int(language_model.config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native", int(speech_model.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native", int(speech_model.conv2.out_channels)),
    ]
    specs.extend(LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(audio_layers)))
    specs.append(LayerSpec("speech_vq_projection", "audio_discrete_projected", audio_dim))
    specs.extend(LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers)))
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _ground_truth) in enumerate(stimuli, 1):
        audio, _ = librosa.load(audio_dir / audio_file, sr=16000, mono=True)
        pooling = int(speech_model.config.pooling_kernel_size or 1)
        stride_samples = int(speech_model.conv1.stride[0] * speech_model.conv2.stride[0] * pooling * feature_extractor.hop_length)
        features = feature_extractor(
            [audio], sampling_rate=16000, return_attention_mask=True,
            return_tensors="pt", padding="longest", pad_to_multiple_of=stride_samples,
        ).to(device)
        feature_length = int(features.attention_mask[0].sum().item())
        conv1_length = feature_length
        conv2_length = (feature_length + 1) // 2
        pooled_mask = features.attention_mask[:, ::2][:, ::pooling].bool()

        # Pass 1: derive the discrete speech tokens and obtain the behavioral
        # transcription without installing any activation hooks.
        with torch.inference_mode():
            behavior_speech_output = speech_model(**features)
        speech_tokens = behavior_speech_output.quantized_token_ids[0, pooled_mask[0]].tolist()
        if not speech_tokens:
            raise RuntimeError("No valid GLM speech tokens were extracted")
        audio_token_text = "<|begin_of_audio|>" + "".join(
            f"<|audio_{token_id}|>" for token_id in speech_tokens
        ) + "<|end_of_audio|>"
        assistant_prefix = "Transcription:" if os.environ.get("SLM_INPUT_LANGUAGE", "zh").lower().startswith("en") else "请转写："
        prompt = f"<|system|>\n{TASK_PROMPT}\n<|user|>\n{audio_token_text}<|assistant|>\n{assistant_prefix}"
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        begin_id = int(tokenizer.convert_tokens_to_ids("<|begin_of_audio|>"))
        end_id = int(tokenizer.convert_tokens_to_ids("<|end_of_audio|>"))
        begin_pos = torch.where(input_ids[0].eq(begin_id))[0]
        end_pos = torch.where(input_ids[0].eq(end_id))[0]
        if begin_pos.numel() != 1 or end_pos.numel() != 1:
            raise RuntimeError("Could not locate GLM begin/end audio markers")
        audio_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        audio_mask[int(begin_pos.item()) + 1:int(end_pos.item())] = True
        if int(audio_mask.sum()) != len(speech_tokens):
            raise RuntimeError("GLM tokenizer audio span does not match extracted VQ token count")
        with torch.inference_mode():
            output_ids = language_model.generate(
                input_ids=input_ids, max_new_tokens=32, do_sample=False,
                num_beams=1, use_cache=True, attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
            )

        captured, handles = {}, []

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        handles.append(speech_model.conv1.register_forward_hook(
            save_once("audio_conv1", lambda x: F.gelu(x[0, :, :conv1_length]).transpose(0, 1))))
        handles.append(speech_model.conv2.register_forward_hook(
            save_once("audio_conv2", lambda x: F.gelu(x[0, :, :conv2_length]).transpose(0, 1))))
        for i, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(
                save_once(f"audio_block_{i:02d}", lambda x: x[0, :conv2_length])))
        try:
            with torch.inference_mode():
                speech_output = speech_model(**features)
            captured["speech_vq_projection"] = speech_output.last_hidden_state[
                0, pooled_mask[0]
            ].detach().float().cpu().contiguous().numpy()

            def select_decoder_audio(x):
                if x.ndim != 3:
                    raise RuntimeError(f"Unexpected GLM layer shape {tuple(x.shape)}")
                if x.shape[0] == audio_mask.numel():  # sequence, batch, hidden
                    return x[:, 0][audio_mask.to(x.device)]
                if x.shape[1] == audio_mask.numel():  # batch, sequence, hidden
                    return x[0][audio_mask.to(x.device)]
                raise RuntimeError("GLM decoder prefill did not align with prompt positions")

            for i, layer in enumerate(decoder_layers):
                handles.append(layer.register_forward_hook(
                    save_once(f"decoder_block_{i:02d}", select_decoder_audio)))
            with torch.inference_mode():
                # Pass 2 decoder stage: one prefill forward, no autoregression.
                language_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
        finally:
            for handle in handles: handle.remove()
        transcription = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        transcription = re.sub(r"<\|audio_\d+\|>", "", transcription)
        transcription = re.sub(r"\s+", " ", transcription).strip()
        validate_activation_set(captured, specs)
        all_activations.append(captured); transcriptions.append(transcription)
        last_audio = f"audio_block_{len(audio_layers) - 1:02d}"
        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; "
            f"layers={len(captured)}, shapes={captured['audio_conv1'].shape}/"
            f"{captured[last_audio].shape}/{captured['speech_vq_projection'].shape}/"
            f"{captured['decoder_block_00'].shape}", flush=True,
        )

    result = save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=output_dir,
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(language_model).__name__), ("model_checkpoint", str(model_root)),
            ("parameter_count", str(sum(p.numel() for p in speech_model.parameters()) + sum(p.numel() for p in language_model.parameters()))),
            ("architecture_family", "Whisper VQ speech tokenizer + discrete audio tokens + ChatGLM decoder"),
            ("prompt", TASK_PROMPT),
            ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", "behavior: deterministic greedy generate; activation: separate speech-tokenizer and decoder prefill forwards, use_cache=False"),
            ("audio_encoder_sequence_definition", "valid unpadded acoustic frames; first 16 VQ encoder blocks"),
            ("vq_sequence_definition", "valid pooled quantized speech tokens"),
            ("decoder_sequence_definition", "prefill discrete audio-token positions only"),
            ("padding_policy", "feature-extractor attention mask applied before pooling; padded VQ positions excluded"),
        ],
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
