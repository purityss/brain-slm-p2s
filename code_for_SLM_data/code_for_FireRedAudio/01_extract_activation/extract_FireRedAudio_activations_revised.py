#!/usr/bin/env python3
"""Extract FireRedAudio encoder/projector and prefill-audio LLM hidden dimensions."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec,
    decoder_audio_positions,
    first_tensor,
    require_complete_sharded_checkpoint,
    save_final_outputs,
    selected_stimuli,
    validate_activation_set,
    validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt  # noqa: E402

MODEL_NAME = "FireRedAudio"
FILE_PREFIX = "FireRedAudio"
MODEL_PATH = "/data/zss/SLMs/All_Models/FireRedAudio/FireRedAudio"
SOURCE_ROOT = str(PROJECT_ROOT / "code_for_FireRedAudio/model_runtime/FireRedAudio-main")
AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
OUTPUT_DIR = str(PROJECT_ROOT / "code_for_FireRedAudio/01_extract_activation/FireRedAudio_hidden_dimension_activations")
TASK_PROMPT = final_prompt("firered_audio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path, audio_dir = Path(args.model_path), Path(args.audio_dir)
    require_complete_sharded_checkpoint(model_path)
    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(audio_dir, stimuli)
    sys.path.insert(0, str(Path(args.source_root)))
    from inference import (  # type: ignore
        DEFAULT_ASR_PROMPT, FEAT_TYPE_UNDERSTAND, FireRedAudioInference,
        UNDERSTAND_SAMPLE_RATE, build_understand_prompt, split_thinking,
    )

    engine = FireRedAudioInference(str(model_path), device="cuda:0")
    model = engine.model
    encoder = model.audio_encoder
    encoder_layers = list(encoder.layers)
    decoder_layers = list(model.backbone_llm.model.layers)
    audio_dim = int(encoder.config.d_model)
    text_dim = int(model.config.backbone_config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native", int(encoder.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native", int(encoder.conv2.out_channels)),
    ]
    specs += [LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(encoder_layers))]
    specs += [LayerSpec("multimodal_projector", "audio_projected", text_dim)]
    specs += [LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers))]
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _truth) in enumerate(stimuli, 1):
        audio_path = str(audio_dir / audio_file)
        chatml = build_understand_prompt(
            TASK_PROMPT, 1, model.config.audio_special_token, enable_thinking=False,
        )
        waveform, source_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if source_rate != UNDERSTAND_SAMPLE_RATE:
            waveform = resample_poly(waveform, UNDERSTAND_SAMPLE_RATE, source_rate).astype(np.float32)
        input_audios = [{
            "feat_type": FEAT_TYPE_UNDERSTAND,
            "audio_understand": waveform,
            "audio_generation": None, "role": "user",
        }]
        batch = engine.encoder.encode(chatml, input_audios)
        audio_mask = batch["input_ids"][0].eq(int(model.config.audio_special_token_id)).cpu()
        feature_len = int(batch["audio_feature_attention_mask"].sum().item())
        aftercnn_len, projected_len = encoder._get_feat_extract_output_lengths(
            torch.tensor([feature_len], dtype=torch.long)
        )
        aftercnn_len, projected_len = int(aftercnn_len.item()), int(projected_len.item())
        if int(audio_mask.sum()) != projected_len:
            raise RuntimeError(f"audio mask {int(audio_mask.sum())} != projected {projected_len}")
        chunk_frames = int(encoder.n_window * 2)
        chunk_lengths = []
        remaining = feature_len
        while remaining:
            chunk_lengths.append(min(chunk_frames, remaining)); remaining -= chunk_lengths[-1]
        conv2_lengths = [(value - 1) // 2 + 1 for value in chunk_lengths]
        # Pass 1: behavioral transcription only, without activation hooks.
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=batch["input_ids"].to("cuda:0"),
                attention_mask=batch["attention_mask"].to("cuda:0"),
                audio_features=batch["audio_features"].to("cuda:0"),
                audio_feature_attention_mask=batch["audio_feature_attention_mask"].to("cuda:0"),
                generation_config=engine._gen_config("asr", max_new_tokens=32),
            )
        raw = engine.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        _reasoning, transcription = split_thinking(raw)
        transcription = (transcription or raw).strip()

        captured, handles = {}, []

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        def unpack_conv(x, lengths):
            return torch.cat([x[i, :, :length].transpose(0, 1) for i, length in enumerate(lengths)], dim=0)

        handles.append(encoder.conv1.register_forward_hook(save_once("audio_conv1", lambda x: unpack_conv(x, chunk_lengths))))
        handles.append(encoder.conv2.register_forward_hook(save_once("audio_conv2", lambda x: unpack_conv(x, conv2_lengths))))
        for i, layer in enumerate(encoder_layers):
            handles.append(layer.register_forward_hook(save_once(f"audio_block_{i:02d}", lambda x: x[:aftercnn_len])))
        handles.append(encoder.adapter.register_forward_hook(save_once("multimodal_projector", lambda x: x[:projected_len])))
        for i, layer in enumerate(decoder_layers):
            handles.append(layer.register_forward_hook(save_once(
                f"decoder_block_{i:02d}", lambda x, mask=audio_mask: decoder_audio_positions(x, mask)
            )))
        try:
            with torch.inference_mode():
                # Pass 2: reproduce FireRedAudio's understanding prefill without
                # invoking generate, then run the Qwen backbone once with cache off.
                input_ids = batch["input_ids"].to("cuda:0")
                attention_mask = batch["attention_mask"].to("cuda:0")
                input_embeds = model.backbone_llm.model.get_input_embeddings()(input_ids)
                audio_output = model.get_audio_features(
                    input_features=batch["audio_features"].to("cuda:0"),
                    feature_attention_mask=batch["audio_feature_attention_mask"].to("cuda:0"),
                )
                placeholder = input_ids.eq(int(model.config.audio_special_token_id)).unsqueeze(-1).expand_as(input_embeds)
                input_embeds = input_embeds.masked_scatter(placeholder, audio_output)
                model.backbone_llm(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles: handle.remove()
        validate_activation_set(captured, specs)
        all_activations.append(captured); transcriptions.append(transcription)
        print(f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; encoder={aftercnn_len}; projected={projected_len}", flush=True)

    print(save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=Path(args.output_dir),
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions, summary_rows=[
            ("model_class", type(model).__name__), ("model_checkpoint", str(model_path)),
            ("parameter_count", str(sum(p.numel() for p in model.parameters()))),
            ("architecture_family", "FireRedAudio continuous audio encoder + adapter + Qwen3.5 backbone"),
            ("prompt", TASK_PROMPT), ("official_asr_prompt", DEFAULT_ASR_PROMPT),
            ("audio_encoder_layer_count", str(len(encoder_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", "behavior: official deterministic 4-beam ASR; activation: separate non-generate prefill forward, use_cache=False"),
            ("attention_runtime", "official SDPA/torch fallback; flash-attn and liger-kernel not installed"),
            ("decoder_sequence_definition", "prefill audio-placeholder positions only; prompt/generated positions excluded"),
            ("padding_policy", "feature mask propagated through conv2 and adapter; only valid positions saved"),
        ],
    ), flush=True)


if __name__ == "__main__": main()
