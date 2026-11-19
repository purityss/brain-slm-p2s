#!/usr/bin/env python3
"""Extract MiniCPM-o 2.6 audio encoder and decoder audio-placeholder states."""

from __future__ import annotations

import argparse, os, random, sys
from copy import deepcopy
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
import transformers

# MiniCPM-o 2.6 imports this former Transformers base class only for type
# annotations. Transformers 4.51 no longer exports it, while its concrete
# TopK/TopP warpers remain available. Alias the equivalent processor base
# before trust_remote_code imports the checkpoint module.
if not hasattr(transformers, "LogitsWarper"):
    transformers.LogitsWarper = transformers.LogitsProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec, save_final_outputs, selected_stimuli,
    validate_activation_set, validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt  # noqa: E402

MODEL_NAME = "MiniCPM-o 2.6"
FILE_PREFIX = "MiniCPM_o_2_6"
MODEL_PATH = "/data/zss/SLMs/All_Models/MiniCPM-o-2_6"
AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
OUTPUT_DIR = str(PROJECT_ROOT / "code_for_MiniCPM-o 2.6/01_extract_activation/MiniCPM_o_2_6_hidden_dimension_activations")
TASK_PROMPT = final_prompt("minicpm_o_2_6")

def first_tensor(output): return output[0] if isinstance(output, (tuple, list)) else output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH); parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR); parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260922); args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model_path, audio_dir, output_dir = Path(args.model_path).resolve(), Path(args.audio_dir).resolve(), Path(args.output_dir).resolve()
    stimuli = selected_stimuli(args.max_stimuli); validate_audio_files(audio_dir, stimuli)
    # This study uses text-only ASR (``generate_audio=False``), so do not
    # initialize the checkpoint's optional speech-synthesis branch. The stock
    # remote code otherwise asserts that Vocos/FSQ TTS packages are installed
    # even though neither is used for transcription or activation extraction.
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    config.init_tts = False
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        config=config,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    ).eval().to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    audio_layers, decoder_layers = list(model.apm.layers), list(model.llm.model.layers)
    audio_dim, text_dim = int(model.apm.config.d_model), int(model.llm.config.hidden_size)
    specs = [LayerSpec("audio_conv1", "audio_native", int(model.apm.conv1.out_channels)), LayerSpec("audio_conv2", "audio_native", int(model.apm.conv2.out_channels))]
    specs.extend(LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim) for i in range(len(audio_layers)))
    specs.append(LayerSpec("multimodal_projector", "audio_projected", text_dim))
    specs.extend(LayerSpec(f"decoder_block_{i:02d}", "decoder_prefill_audio", text_dim) for i in range(len(decoder_layers)))
    all_activations, transcriptions = [], []

    for stimulus_index, (audio_file, _gt) in enumerate(stimuli, 1):
        audio, _ = librosa.load(audio_dir / audio_file, sr=16000, mono=True)
        # Follow the official MiniCPM-o ASR example exactly: the task text
        # precedes the audio object in the multimodal user content list.
        msgs = [{"role": "user", "content": [TASK_PROMPT, audio]}]

        # Pass 1: native behavioral chat, without activation hooks.
        with torch.inference_mode():
            response = model.chat(
                msgs=msgs, tokenizer=tokenizer, processor=processor,
                sampling=False, max_new_tokens=32, use_tts_template=True,
                generate_audio=False, stream=False,
            )

        # Reproduce MiniCPM-o's processor path, but stop before its internal
        # generate call. This data object feeds the ordinary model.forward below.
        prefill_msgs = deepcopy(msgs)
        prompt_parts, audios, audio_parts = [], [], []
        for message_index, message in enumerate(prefill_msgs):
            parts = message["content"] if isinstance(message["content"], list) else [message["content"]]
            rendered = []
            for part in parts:
                if isinstance(part, np.ndarray):
                    audios.append(part); audio_parts.append(message_index)
                    rendered.append("(<audio>./</audio>)")
                else:
                    rendered.append(str(part))
            message["content"] = "\n".join(rendered)
        prompt = processor.tokenizer.apply_chat_template(
            prefill_msgs, tokenize=False, add_generation_prompt=True,
            chat_template=model.default_tts_chat_template,
        )
        prefill_inputs = processor(
            [prompt], [[]], [audios], [audio_parts], max_slice_nums=None,
            use_image_id=None, chunk_input=True, return_tensors="pt",
            max_length=32768,
        ).to(model.device)
        prefill_inputs.pop("image_sizes", None)
        if "position_ids" not in prefill_inputs:
            position_ids = prefill_inputs["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(prefill_inputs["attention_mask"].eq(0), 0)
            prefill_inputs["position_ids"] = position_ids

        captured, handles, mask_holder = {}, [], {"mask": None}
        def save_once(name, transform):
            def hook(_m, _i, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook
        handles.append(model.apm.conv1.register_forward_hook(save_once("audio_conv1", lambda x: F.gelu(x[0]).transpose(0, 1))))
        handles.append(model.apm.conv2.register_forward_hook(save_once("audio_conv2", lambda x: F.gelu(x[0]).transpose(0, 1))))
        for i, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(save_once(f"audio_block_{i:02d}", lambda x: x[0])))
        # AvgPool output is [batch, hidden, projected_time].
        handles.append(model.audio_avg_pooler.register_forward_hook(save_once("multimodal_projector", lambda x: x[0].transpose(0, 1))))
        def embedding_hook(_m, inputs, _output):
            ids = inputs[0]
            if ids.shape[1] > 1 and mask_holder["mask"] is None:
                mask_holder["mask"] = ids[0].eq(int(tokenizer.unk_token_id)).detach().cpu()
        handles.append(model.llm.model.embed_tokens.register_forward_hook(embedding_hook))
        for i, layer in enumerate(decoder_layers):
            def select_audio(x):
                mask = mask_holder["mask"]
                if mask is None or x.shape[1] != mask.numel() or int(mask.sum()) == 0:
                    raise RuntimeError("Could not align MiniCPM decoder prefill with audio placeholders")
                return x[0, mask.to(x.device)]
            handles.append(layer.register_forward_hook(save_once(f"decoder_block_{i:02d}", select_audio)))
        try:
            with torch.inference_mode():
                # Pass 2: one non-autoregressive multimodal prefill forward.
                model(
                    data=prefill_inputs,
                    attention_mask=prefill_inputs["attention_mask"],
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
        finally:
            for handle in handles: handle.remove()
        audio_mask = mask_holder["mask"]
        if audio_mask is None:
            raise RuntimeError("MiniCPM-o 2.6 did not expose decoder prefill token IDs")
        audio_position_count = int(audio_mask.sum().item())
        projector_length = int(captured["multimodal_projector"].shape[0])
        if audio_position_count != projector_length:
            raise RuntimeError(
                "MiniCPM-o 2.6 audio-placeholder count does not match the "
                f"projector length: placeholders={audio_position_count}, "
                f"projector={projector_length}"
            )
        for i in range(len(decoder_layers)):
            decoder_length = int(captured[f"decoder_block_{i:02d}"].shape[0])
            if decoder_length != projector_length:
                raise RuntimeError(
                    f"MiniCPM-o 2.6 decoder block {i} retained {decoder_length} "
                    f"positions; expected {projector_length} audio positions"
                )
        validate_activation_set(captured, specs)
        transcription = str(response)
        all_activations.append(captured); transcriptions.append(transcription)
        last_audio_name = f"audio_block_{len(audio_layers) - 1:02d}"
        print(f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; layers={len(captured)}, shapes={captured['audio_conv1'].shape}/{captured[last_audio_name].shape}/{captured['multimodal_projector'].shape}/{captured['decoder_block_00'].shape}", flush=True)

    result = save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=output_dir, stimuli=stimuli,
        specs=specs, all_activations=all_activations, transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(model).__name__), ("model_checkpoint", str(model_path)),
            ("parameter_count", str(sum(p.numel() for p in model.parameters()))),
            ("architecture_family", "MiniCPM-o Whisper-like audio encoder + Qwen2 decoder"),
            ("prompt", TASK_PROMPT), ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", "behavior: deterministic native chat using the official Chinese-ASR message layout; activation: separate non-generate prefill forward, use_cache=False"),
            ("decoder_sequence_definition", "prefill <unk> audio-placeholder positions only"),
            ("padding_policy", "single-stimulus processor lengths; no batch padding retained"),
        ],
    ); print(result, flush=True)

if __name__ == "__main__": main()
