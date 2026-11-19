#!/usr/bin/env python3
"""Extract Baichuan-Audio-Base encoder, pre-VQ bridge, and audio-position LLM states."""

from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "baichuan_runtime_compat"))
from pipeline_code.step01_extract_activation.activation_extraction_common import (
    LayerSpec,
    decoder_audio_positions,
    first_tensor,
    require_complete_sharded_checkpoint,
    save_final_outputs,
    selected_stimuli,
    validate_activation_set,
    validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt

MODEL_NAME = "Baichuan-Audio-Base"
FILE_PREFIX = "Baichuan_Audio_Base"
MODEL_PATH = "/data/zss/SLMs/All_Models/Baichuan-Audio-Base"
AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
OUTPUT_DIR = str(PROJECT_ROOT / "code_for_Baichuan-Audio-Base/01_extract_activation/Baichuan_Audio_Base_hidden_dimension_activations")
TASK_PROMPT = final_prompt("baichuan_audio_base")
CHECKPOINT_VARIANT = "Base"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=MODEL_PATH); p.add_argument("--audio-dir", default=AUDIO_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR); p.add_argument("--max-stimuli", type=int)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model_path, audio_dir = Path(args.model_path), Path(args.audio_dir)
    require_complete_sharded_checkpoint(model_path)
    stimuli = selected_stimuli(args.max_stimuli); validate_audio_files(audio_dir, stimuli)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True, model_max_length=128000)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda",
    ).eval()
    model.config.use_cache = True
    model.bind_processor(tokenizer, training=False, relative_path="/")
    encoder = model.audio_tokenizer.audio_model
    bridge = model.audio_tokenizer.audio_bridge_model
    encoder_layers, decoder_layers = list(encoder.layers), list(model.model.layers)
    audio_dim = int(model.config.audio_config.d_model); text_dim = int(model.config.hidden_size)
    bridge_dim = int(bridge.intermediate_dim)
    specs = [LayerSpec("audio_conv2", "audio_native", int(encoder.conv2.out_channels))]
    specs += [LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(encoder_layers))]
    specs += [LayerSpec("audio_bridge_pre_vq", "audio_projected_pre_vq", bridge_dim)]
    specs += [LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers))]
    audio_start = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
    audio_end = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _truth) in enumerate(stimuli, 1):
        prompt = TASK_PROMPT + audio_start + json.dumps({"path": str(audio_dir / audio_file)}, ensure_ascii=False) + audio_end
        ret = model.processor([prompt])
        audio_mask = ret.input_ids[0].eq(int(model.config.audio_config.audio_pad_token_id)).cpu()
        encoder_len = int(ret.encoder_length[0].item()); bridge_len = int(ret.bridge_length[0].item())
        if int(audio_mask.sum()) != bridge_len:
            raise RuntimeError(f"audio mask {int(audio_mask.sum())} != bridge length {bridge_len}")
        # Pass 1: behavioral transcription only, without activation hooks.
        with torch.inference_mode():
            predicted_ids = model.generate(
                input_ids=ret.input_ids.cuda(), attention_mask=ret.attention_mask.cuda(),
                audios=ret.audios.cuda(), encoder_length=ret.encoder_length.cuda(),
                bridge_length=ret.bridge_length.cuda(), max_new_tokens=32,
                num_beams=5, do_sample=False, repetition_penalty=1.1, use_cache=True,
            )
        transcription = tokenizer.batch_decode(
            predicted_ids[:, ret.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        captured, handles = {}, []

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output)); captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        handles.append(encoder.conv2.register_forward_hook(save_once("audio_conv2", lambda x: x[0, :, :encoder_len].transpose(0, 1))))
        for i, layer in enumerate(encoder_layers):
            handles.append(layer.register_forward_hook(save_once(f"audio_block_{i:02d}", lambda x: x[:encoder_len])))
        handles.append(bridge.layer_norm.register_forward_hook(save_once("audio_bridge_pre_vq", lambda x: x[0, :bridge_len])))
        for i, layer in enumerate(decoder_layers):
            handles.append(layer.register_forward_hook(save_once(
                f"decoder_block_{i:02d}", lambda x, mask=audio_mask: decoder_audio_positions(x, mask)
            )))
        try:
            with torch.inference_mode():
                # Pass 2: a single multimodal prefill forward; no generation.
                model(
                    input_ids=ret.input_ids.cuda(), attention_mask=ret.attention_mask.cuda(),
                    audios=ret.audios.cuda(), encoder_length=ret.encoder_length.cuda(),
                    bridge_length=ret.bridge_length.cuda(), use_cache=False,
                    output_hidden_states=False, return_dict=True,
                )
        finally:
            for handle in handles: handle.remove()
        validate_activation_set(captured, specs); all_activations.append(captured); transcriptions.append(transcription)
        print(f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; encoder={encoder_len}; bridge={bridge_len}", flush=True)

    print(save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=Path(args.output_dir), stimuli=stimuli,
        specs=specs, all_activations=all_activations, transcriptions=transcriptions, summary_rows=[
            ("model_class", type(model).__name__), ("model_checkpoint", str(model_path)),
            ("checkpoint_variant", CHECKPOINT_VARIANT),
            ("parameter_count", str(sum(p.numel() for p in model.parameters()))),
            ("architecture_family", "Whisper-style encoder + pooled RVQ audio tokenizer + decoder-only Audio LLM"),
            ("prompt", TASK_PROMPT + "<audio>"), ("audio_encoder_layer_count", str(len(encoder_layers))),
            ("decoder_layer_count", str(len(decoder_layers))), ("generation_mode", "behavior: deterministic 5-beam generate; activation: separate non-generate prefill forward, use_cache=False"),
            ("audio_encoder_sequence_definition", "valid post-conv frames; padded tail excluded"),
            ("bridge_sequence_definition", "continuous layer-normalized 12.5-Hz states immediately before RVQ"),
            ("decoder_sequence_definition", "prefill audio-pad positions only; prompt/generated positions excluded"),
        ],
    ), flush=True)


if __name__ == "__main__": main()
