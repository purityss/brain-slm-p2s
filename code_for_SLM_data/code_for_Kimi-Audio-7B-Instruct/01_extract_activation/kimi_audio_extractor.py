#!/usr/bin/env python3
"""Extract Kimi-Audio hybrid-Whisper and audio-position decoder states.

Pass 1 uses the official autoregressive text-generation loop for behavioral ASR.
Pass 2 rebuilds the same prompt and makes exactly one non-generate prefill forward
with ``use_cache=False``. Only source-audio positions are retained. The GLM
semantic tokenizer is treated as discrete input preprocessing; its hidden states
and Kimi's six audio-output MIMO layers are deliberately not saved.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import torch

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

MODEL_NAME = "Kimi-Audio-7B"
FILE_PREFIX = "Kimi_Audio_7B"
MODEL_PATH = "/data/zss/SLMs/All_Models/Kimi-Audio-7B"
GLM_TOKENIZER_PATH = "/data/zss/SLMs/All_Models/GLM-4-Voice/glm-4-voice-tokenizer"
SOURCE_ROOT = str(PROJECT_ROOT / "code_for_Kimi-Audio-7B-Instruct/model_runtime")
AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
OUTPUT_DIR = str(PROJECT_ROOT / "code_for_Kimi-Audio-7B-Instruct/01_extract_activation/Kimi_Audio_7B_Instruct_hidden_dimension_activations")
PROMPT = final_prompt("kimi_audio_7b_instruct")
CHECKPOINT_VARIANT = "Base (not instruction-tuned)"
COMPARABILITY_NOTE = (
    "Base checkpoint is not instruction-tuned; behavioral ASR accuracy is not "
    "directly homogeneous with instruct/chat checkpoints"
)


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--glm-tokenizer-path", default=GLM_TOKENIZER_PATH)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    compatibility_root = Path(__file__).resolve().parent / "kimi_runtime_compat"
    sys.path.insert(0, str(compatibility_root))

    # The released checkpoint hard-requires FlashAttention during module
    # import. This project supplies the same inference API with Torch SDPA so
    # it remains compatible with the pinned Torch/CUDA build.
    import transformers
    transformers.utils.is_flash_attn_2_available = lambda: True

    # Kimi's released attention code selects ``cos[position_ids]`` and
    # ``sin[position_ids]`` before calling Qwen2's helper. In Transformers
    # 4.44 the helper indexes them a second time, corrupting the sequence axis.
    # Restore the legacy helper contract expected by the checkpoint before its
    # trust_remote_code module imports the function.
    import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

    def apply_rotary_pos_emb_preindexed(
        query, key, cosine, sine, position_ids=None, unsqueeze_dim=1,
    ):
        del position_ids
        cosine = cosine.unsqueeze(unsqueeze_dim)
        sine = sine.unsqueeze(unsqueeze_dim)
        query_embed = (query * cosine) + (qwen2_modeling.rotate_half(query) * sine)
        key_embed = (key * cosine) + (qwen2_modeling.rotate_half(key) * sine)
        return query_embed, key_embed

    qwen2_modeling.apply_rotary_pos_emb = apply_rotary_pos_emb_preindexed

    # The bundled Whisper checkpoint is a valid safetensors archive but was
    # published without the optional metadata header. Transformers 4.44.1
    # assumes that metadata is always a dictionary and crashes before loading
    # the tensors. Fall back to safetensors' native loader only for archives
    # whose metadata is absent; retain the normal Transformers validation for
    # every other checkpoint.
    import transformers.modeling_utils as modeling_utils
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file

    class SafeOpenWithDefaultMetadata:
        def __init__(self, *args, **kwargs):
            self._context = safe_open(*args, **kwargs)
            self._archive = None

        def __enter__(self):
            self._archive = self._context.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._context.__exit__(exc_type, exc_value, traceback)

        def metadata(self):
            metadata = self._archive.metadata()
            return {"format": "pt"} if metadata is None else metadata

        def __getattr__(self, name):
            return getattr(self._archive, name)

    modeling_utils.safe_open = SafeOpenWithDefaultMetadata
    original_load_state_dict = modeling_utils.load_state_dict

    def load_state_dict_allow_missing_safetensors_metadata(
        checkpoint_file, is_quantized=False,
    ):
        checkpoint_path = str(checkpoint_file)
        if checkpoint_path.endswith(".safetensors"):
            with safe_open(checkpoint_path, framework="pt") as archive:
                if archive.metadata() is None:
                    return safe_load_file(checkpoint_path)
        return original_load_state_dict(checkpoint_file, is_quantized=is_quantized)

    modeling_utils.load_state_dict = load_state_dict_allow_missing_safetensors_metadata

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path = Path(args.model_path).resolve()
    glm_tokenizer_path = Path(args.glm_tokenizer_path).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(audio_dir, stimuli)
    for required in ("model.safetensors.index.json", "modeling_moonshot_kimia.py", "tokenizer_config.json"):
        if not (model_path / required).is_file():
            raise FileNotFoundError(f"Incomplete Kimi checkpoint: missing {model_path / required}")

    import kimia_infer.api.prompt_manager as prompt_manager_module
    from kimia_infer.api.kimia import KimiAudio

    original_glm_tokenizer = prompt_manager_module.Glm4Tokenizer
    prompt_manager_module.Glm4Tokenizer = lambda _hub_id: original_glm_tokenizer(str(glm_tokenizer_path))
    try:
        kimi = KimiAudio(model_path=str(model_path), load_detokenizer=False)
    finally:
        prompt_manager_module.Glm4Tokenizer = original_glm_tokenizer

    kimi.alm.eval()
    whisper_encoder = kimi.prompt_manager.whisper_model.speech_encoder
    audio_layers = list(whisper_encoder.layers)
    decoder_layers = list(kimi.alm.model.layers)
    audio_dim = int(whisper_encoder.config.d_model)
    text_dim = int(kimi.alm.config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native_continuous", int(whisper_encoder.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native_continuous", int(whisper_encoder.conv2.out_channels)),
    ]
    specs.extend(
        LayerSpec(f"audio_block_{index:02d}", "audio_native_continuous", audio_dim)
        for index in range(len(audio_layers))
    )
    specs.append(LayerSpec("multimodal_projector", "audio_projected_hybrid", text_dim))
    specs.extend(
        LayerSpec(f"decoder_block_{index:02d}", "decoder_prefill_audio", text_dim)
        for index in range(len(decoder_layers))
    )

    all_activations: list[dict[str, np.ndarray]] = []
    transcriptions: list[str] = []
    for stimulus_index, (audio_file, _ground_truth) in enumerate(stimuli, 1):
        audio_path = str(audio_dir / audio_file)
        messages = [
            {"role": "user", "message_type": "text", "content": PROMPT},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]

        # Pass 1: official behavior generation, with no activation hooks.
        with torch.inference_mode():
            _wav, transcription = kimi.generate(
                messages, output_type="text", text_temperature=0.0,
                text_top_k=5, audio_temperature=0.8, audio_top_k=10,
                max_new_tokens=64,
            )

        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        token_length = (len(waveform) - 1) // (160 * 8) + 1
        conv2_valid_length = token_length * 4
        conv1_valid_length = conv2_valid_length * 2
        captured: dict[str, np.ndarray] = {}
        handles = []

        def save_once(name, transform):
            def hook(_module, _inputs, output):
                if name not in captured:
                    value = transform(first_tensor(output))
                    captured[name] = value.detach().float().cpu().contiguous().numpy()
            return hook

        # Pass 2a: rebuilding the prompt runs Whisper once. Crop its fixed
        # 30-second padded tensors back to the exact valid source-audio length.
        handles.append(whisper_encoder.conv1.register_forward_hook(
            save_once("audio_conv1", lambda x: x[0, :, :conv1_valid_length].transpose(0, 1))))
        handles.append(whisper_encoder.conv2.register_forward_hook(
            save_once("audio_conv2", lambda x: x[0, :, :conv2_valid_length].transpose(0, 1))))
        for index, layer in enumerate(audio_layers):
            handles.append(layer.register_forward_hook(
                save_once(f"audio_block_{index:02d}", lambda x: x[0, :conv2_valid_length, :])))
        try:
            history = kimi.prompt_manager.get_prompt(messages, output_type="text")
        finally:
            for handle in handles:
                handle.remove()
        handles.clear()

        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature
        audio_mask_cpu = is_continuous_mask[0].bool().cpu()
        projected_length = int(audio_mask_cpu.sum().item())
        if projected_length != token_length or projected_length != int(audio_features[0].shape[1]):
            raise RuntimeError(
                f"Kimi audio alignment mismatch: mask={projected_length}, "
                f"expected={token_length}, feature={audio_features[0].shape[1]}"
            )

        def select_projected_audio(x):
            if x.shape[0] != audio_mask_cpu.numel():
                raise RuntimeError(f"Unexpected Kimi projector sequence: {tuple(x.shape)}")
            return x[:, 0, :][audio_mask_cpu.to(x.device)]

        def select_decoder_audio(x):
            if x.shape[1] != audio_mask_cpu.numel():
                raise RuntimeError(f"Unexpected Kimi decoder sequence: {tuple(x.shape)}")
            return x[0, audio_mask_cpu.to(x.device), :]

        handles.append(kimi.alm.model.vq_adaptor.register_forward_hook(
            save_once("multimodal_projector", select_projected_audio)))
        for index, layer in enumerate(decoder_layers):
            handles.append(layer.register_forward_hook(
                save_once(f"decoder_block_{index:02d}", select_decoder_audio)))
        device = next(kimi.alm.parameters()).device
        try:
            with torch.inference_mode():
                kimi.alm(
                    input_ids=audio_input_ids.to(device),
                    text_input_ids=text_input_ids.to(device),
                    whisper_input_feature=[feature.to(device) for feature in audio_features],
                    is_continuous_mask=is_continuous_mask.to(device),
                    use_cache=False, output_hidden_states=False, return_dict=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        validate_activation_set(captured, specs)
        all_activations.append(captured)
        transcriptions.append(str(transcription).strip())
        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: {transcription!r}; "
            f"layers={len(captured)}, shapes={captured['audio_conv1'].shape}/"
            f"{captured[f'audio_block_{len(audio_layers)-1:02d}'].shape}/"
            f"{captured['multimodal_projector'].shape}/{captured['decoder_block_00'].shape}",
            flush=True,
        )

    parameter_count = (
        sum(parameter.numel() for parameter in kimi.alm.parameters())
        + sum(parameter.numel() for parameter in kimi.prompt_manager.whisper_model.parameters())
        + sum(parameter.numel() for parameter in kimi.prompt_manager.audio_tokenizer.parameters())
    )
    result = save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=output_dir,
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(kimi.alm).__name__),
            ("model_checkpoint", str(model_path)),
            ("checkpoint_variant", CHECKPOINT_VARIANT),
            ("parameter_count_loaded_components", str(parameter_count)),
            ("architecture_family", "hybrid GLM semantic tokens + Whisper continuous encoder + VQ adaptor + 28-layer decoder"),
            ("prompt", PROMPT),
            ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", "behavior: official deterministic text generation; activation: separate non-generate prefill forward, use_cache=False"),
            ("audio_encoder_sequence_definition", "valid unpadded continuous Whisper positions; fixed 30-second padding excluded"),
            ("projector_sequence_definition", "source-audio positions selected by official is_continuous_mask"),
            ("decoder_sequence_definition", "prefill source-audio positions selected by official is_continuous_mask"),
            ("semantic_tokenizer_policy", "GLM tokenizer supplies discrete input IDs; tokenizer hidden states are not saved as Kimi hidden units"),
            ("mimo_policy", "six audio-output MIMO layers excluded; main 28-layer decoder retained"),
            ("padding_policy", "valid waveform length propagated through Whisper and official continuous-position mask"),
            ("comparability_note", COMPARABILITY_NOTE),
        ],
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
