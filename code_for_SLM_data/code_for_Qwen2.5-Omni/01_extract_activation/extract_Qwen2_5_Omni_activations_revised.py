#!/usr/bin/env python3
"""Extract Qwen2.5-Omni thinker audio and decoder audio-position states."""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
warnings.filterwarnings("ignore")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec, save_final_outputs, selected_stimuli,
    validate_activation_set, validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import (  # noqa: E402
    QWEN25_OMNI_SYSTEM_PROMPT,
    final_prompt,
)

MODEL_NAME = "Qwen2.5-Omni"
FILE_PREFIX = "Qwen2_5_Omni"
REFERENCE_ROOT = str(PROJECT_ROOT / "code_for_Qwen2.5-Omni/model_runtime/Qwen2.5-Omni-main")
MODEL_PATH = "/data/zss/SLMs/All_Models/Qwen2.5-Omni-7B"
AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
OUTPUT_DIR = str(PROJECT_ROOT / "code_for_Qwen2.5-Omni/01_extract_activation/Qwen2_5_Omni_hidden_dimension_activations")
SYSTEM_PROMPT = QWEN25_OMNI_SYSTEM_PROMPT
TASK_PROMPT = final_prompt("qwen2_5_omni")


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default=REFERENCE_ROOT)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    reference_root = Path(args.reference_root).resolve()
    model_path = Path(args.model_path).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(audio_dir, stimuli)
    sys.path.insert(0, str(reference_root / "qwen-omni-utils/src"))
    sys.path.insert(0, str(reference_root / "transformers/src"))
    from qwen_omni_utils import process_mm_info  # type: ignore
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto",
    ).eval()
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
    audio_tower = model.thinker.audio_tower
    audio_layers = list(audio_tower.layers)
    decoder_layers = list(model.thinker.model.layers)
    audio_dim = int(audio_tower.config.d_model)
    text_dim = int(model.thinker.config.text_config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native", int(audio_tower.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native", int(audio_tower.conv2.out_channels)),
    ]
    specs.extend(LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(audio_layers)))
    specs.append(LayerSpec("multimodal_projector", "audio_projected", int(audio_tower.proj.out_features)))
    specs.extend(LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers)))
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _ground_truth) in enumerate(stimuli, 1):
        audio_path = str(audio_dir / audio_file)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": TASK_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        input_ids_cpu = inputs["input_ids"]
        audio_mask = input_ids_cpu[0].eq(int(model.thinker.config.audio_token_index)).cpu()
        if int(audio_mask.sum()) == 0: raise RuntimeError("No Qwen2.5-Omni audio placeholder tokens")
        feature_length = int(inputs["feature_attention_mask"].sum().item())
        aftercnn_length = int((feature_length - 1) // 2 + 1)
        projected_length = int((aftercnn_length - 2) // 2 + 1)
        if int(audio_mask.sum()) != projected_length:
            raise RuntimeError(f"Audio placeholder count {int(audio_mask.sum())} != projected length {projected_length}")
        device = next(model.thinker.parameters()).device
        model_inputs = inputs.to(device)
        model_inputs = model_inputs.to(model.thinker.dtype)

        # Pass 1: behavioral Thinker generation only, without hooks.
        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs, use_audio_in_video=True, return_audio=False,
                thinker_max_new_tokens=32, thinker_do_sample=False,
                thinker_num_beams=1, thinker_use_cache=True,
            )

        captured, handles = {}, []

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        handles.append(audio_tower.conv1.register_forward_hook(
            save_once("audio_conv1", lambda x: x[0, :, :feature_length].transpose(0, 1))))
        handles.append(audio_tower.conv2.register_forward_hook(
            save_once("audio_conv2", lambda x: x[0, :, :aftercnn_length].transpose(0, 1))))
        for i, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(
                save_once(f"audio_block_{i:02d}", lambda x: x[:aftercnn_length])))
        handles.append(audio_tower.proj.register_forward_hook(
            save_once("multimodal_projector", lambda x: x[:projected_length])))
        for i, layer in enumerate(decoder_layers):
            def select_audio(x):
                if x.shape[1] != audio_mask.numel():
                    raise RuntimeError("Qwen2.5-Omni decoder prefill did not align with input positions")
                return x[0, audio_mask.to(x.device)]
            handles.append(layer.register_forward_hook(save_once(f"decoder_block_{i:02d}", select_audio)))
        try:
            with torch.inference_mode():
                # Pass 2: Thinker-only multimodal prefill, Talker remains disabled.
                model.thinker(
                    **model_inputs, use_audio_in_video=True, use_cache=False,
                    output_hidden_states=False, return_dict=True,
                )
        finally:
            for handle in handles: handle.remove()
        generated_ids = output_ids[:, input_ids_cpu.shape[1]:] if output_ids.shape[1] > input_ids_cpu.shape[1] else output_ids
        transcription = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        validate_activation_set(captured, specs)
        all_activations.append(captured); transcriptions.append(transcription)
        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; "
            f"layers={len(captured)}, shapes={captured['audio_conv1'].shape}/"
            f"{captured[f'audio_block_{len(audio_layers)-1:02d}'].shape}/"
            f"{captured['multimodal_projector'].shape}/{captured['decoder_block_00'].shape}", flush=True,
        )

    result = save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=output_dir,
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(model).__name__), ("model_checkpoint", str(model_path)),
            ("parameter_count_thinker", str(sum(p.numel() for p in model.thinker.parameters()))),
            ("architecture_family", "Qwen2.5-Omni thinker audio encoder + projection + text decoder; talker disabled"),
            ("prompt", f"system={SYSTEM_PROMPT}; user={TASK_PROMPT}"),
            ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", "behavior: deterministic Thinker generate; activation: separate Thinker-only non-generate prefill forward, use_cache=False"),
            ("audio_encoder_sequence_definition", "valid unpadded audio frames"),
            ("projector_sequence_definition", "valid pooled projected audio tokens"),
            ("decoder_sequence_definition", "prefill audio-placeholder positions only"),
            ("padding_policy", "feature_attention_mask lengths propagated through both convolution/pooling stages"),
        ],
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
