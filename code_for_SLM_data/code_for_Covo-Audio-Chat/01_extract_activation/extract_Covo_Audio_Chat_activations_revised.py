#!/usr/bin/env python3
"""Extract sequence-resolved hidden-dimension activations from Covo-Audio-Chat.

Pass 1 uses native generation only for transcription. Pass 2 is a separate
non-generative prefill forward for activations. Decoder arrays contain only
the acoustic placeholder positions.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from transformers import AutoConfig, AutoModel, AutoTokenizer, LogitsProcessorList


MODEL_NAME = "Covo-Audio-Chat"
FILE_PREFIX = "Covo_Audio_Chat"
MODEL_PATH = Path("/data/zss/SLMs/All_Models/Covo-Audio-Chat")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "code_for_Covo-Audio-Chat/01_extract_activation"
    / f"{FILE_PREFIX}_hidden_dimension_activations"
)
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 32
SAVE_INDIVIDUAL_STIMULI = False

sys.path.insert(0, str(LOCAL_PROJECT_ROOT))
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt  # noqa: E402
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec,
    save_final_outputs,
    selected_stimuli,
    validate_activation_set,
    validate_audio_files,
)

COVO_TASK_PROMPT = final_prompt("covo_audio_chat")


def calc_projected_length(frame_count_10ms: int) -> int:
    """Match Covo's four stride-2 length reductions."""
    length = int(frame_count_10ms)
    for _ in range(4):
        length = (length + 1) // 2
    return length


def patch_dynamic_module_assets(model: Any) -> None:
    """Point cached Covo remote code at its non-Python mel-filter asset."""
    asset_path = MODEL_PATH / "assets/mel_filters.npz"
    if not asset_path.is_file():
        raise FileNotFoundError(f"Missing Covo mel-filter asset: {asset_path}")
    modeling_module = sys.modules.get(model.__class__.__module__)
    if modeling_module is None:
        raise RuntimeError("Cannot locate the loaded Covo modeling module")

    @lru_cache(maxsize=None)
    def local_mel_filters(device: torch.device, n_mels: int) -> torch.Tensor:
        if n_mels not in (80, 128):
            raise ValueError(f"Unsupported mel count: {n_mels}")
        with np.load(asset_path, allow_pickle=False) as archive:
            return torch.from_numpy(archive[f"mel_{n_mels}"]).to(device)

    modeling_module.mel_filters = local_mel_filters


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Cannot locate tensor in hook output type {type(output)!r}")


def load_audio_24k(audio_path: Path) -> torch.Tensor:
    # torchaudio 2.10 routes file decoding through TorchCodec, which requires
    # system FFmpeg shared libraries. These stimuli are ordinary PCM WAV files,
    # so SciPy is both sufficient and independent of TorchCodec/FFmpeg.
    try:
        sample_rate, samples = wavfile.read(str(audio_path))
    except ValueError:
        # English validation stimuli are MP3 files.  The pinned Covo
        # environment can decode them through torchaudio's soundfile backend.
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(dtype=torch.float32)
    else:
        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            scale = float(max(abs(info.min), info.max))
            samples = samples.astype(np.float32) / scale
        else:
            samples = samples.astype(np.float32, copy=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    if sample_rate != 24000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)
    waveform = waveform.squeeze(0)

    # Official Covo preprocessing uses a 10-ms hop and pads to 480 samples.
    hop_size = 240
    waveform = waveform[: (waveform.shape[-1] // hop_size) * hop_size]
    if waveform.shape[-1] == 0:
        raise ValueError(f"Audio shorter than one 10-ms frame: {audio_path}")
    pad_samples = (-waveform.shape[-1]) % 480
    if pad_samples:
        waveform = torch.nn.functional.pad(waveform, (0, pad_samples))
    return waveform


def build_native_prompt(
    audio_path: Path,
    tokenizer: Any,
    c_audio_id: int,
) -> dict[str, Any]:
    waveform = load_audio_24k(audio_path)
    frame_count = waveform.shape[-1] * 100 // 24000
    conv1_length = int(frame_count)
    conv2_length = (conv1_length + 1) // 2
    projected_length = calc_projected_length(conv1_length)

    audio_placeholders = "<|cAUDIO|>" * projected_length
    prompt = (
        f"<|im_start|>system\n{COVO_TASK_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|begofcAUDIO|>{audio_placeholders}<|endofcAUDIO|>"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(DEVICE)
    attention_mask = encoded.attention_mask.to(DEVICE)
    audio_mask = input_ids.eq(c_audio_id)
    actual_audio_tokens = int(audio_mask.sum().item())
    if actual_audio_tokens != projected_length:
        raise RuntimeError(
            f"Expected {projected_length} audio tokens, got {actual_audio_tokens}"
        )
    return {
        "waveform": waveform.to(DEVICE, dtype=torch.float32),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_mask": audio_mask,
        "conv1_length": conv1_length,
        "conv2_length": conv2_length,
        "projected_length": projected_length,
        "prompt": prompt,
    }


def inject_audio_embeddings(model: Any, prepared: dict[str, Any]) -> torch.Tensor:
    """Reproduce Covo audio injection without invoking generation."""
    inputs_embeds = model.llm.get_input_embeddings()(prepared["input_ids"])
    audio_features = model.audio_encoder([prepared["waveform"]], DEVICE)
    projected_length = prepared["projected_length"]
    if audio_features.ndim != 3 or audio_features.shape[0] != 1:
        raise RuntimeError(f"Unexpected audio feature shape {audio_features.shape}")
    if audio_features.shape[1] < projected_length:
        raise RuntimeError(
            f"Projector T={audio_features.shape[1]} < required {projected_length}"
        )
    audio_features = audio_features[:, :projected_length, :]
    flat_features = audio_features.reshape(-1, audio_features.shape[-1])
    audio_mask = prepared["audio_mask"]
    if flat_features.shape[0] != int(audio_mask.sum().item()):
        raise RuntimeError("Audio features and placeholder counts do not match")
    return inputs_embeds.masked_scatter(
        audio_mask.unsqueeze(-1), flat_features.to(inputs_embeds.dtype)
    )


def make_layer_specs(model: Any) -> list[LayerSpec]:
    specs = [
        LayerSpec("audio_conv1", "audio_native_continuous", 1280),
        LayerSpec("audio_conv2", "audio_native_continuous", 1280),
    ]
    specs.extend(
        LayerSpec(f"audio_block_{index:02d}", "audio_native_continuous", 1280)
        for index in range(len(model.encoder.layers))
    )
    specs.append(LayerSpec("multimodal_projector", "audio_projected", 3584))
    specs.extend(
        LayerSpec(f"decoder_block_{index:02d}", "decoder_prefill_audio", 3584)
        for index in range(len(model.llm.model.layers))
    )
    return specs


def register_activation_hooks(
    model: Any,
    prepared: dict[str, Any],
    layer_specs: list[LayerSpec],
    store: list[np.ndarray | None],
) -> list[Any]:
    handles: list[Any] = []
    conv1_length = prepared["conv1_length"]
    conv2_length = prepared["conv2_length"]
    projected_length = prepared["projected_length"]
    audio_mask_1d = prepared["audio_mask"][0]

    def add_hook(
        module: torch.nn.Module,
        layer_index: int,
        transform: Callable[[Any], torch.Tensor],
    ) -> None:
        def hook_fn(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = transform(output).detach().float().cpu().contiguous()
            array = tensor.numpy()
            expected_hidden = layer_specs[layer_index].hidden_dim
            if array.ndim != 2 or array.shape[1] != expected_hidden:
                raise RuntimeError(
                    f"{layer_specs[layer_index].name}: expected [T,"
                    f"{expected_hidden}], got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise RuntimeError(
                    f"{layer_specs[layer_index].name}: non-finite activation"
                )
            store[layer_index] = array

        handles.append(module.register_forward_hook(hook_fn))

    add_hook(
        model.encoder.conv1,
        0,
        lambda out: first_tensor(out)[0, :, :conv1_length].transpose(0, 1),
    )
    add_hook(
        model.encoder.conv2,
        1,
        lambda out: first_tensor(out)[0, :, :conv2_length].transpose(0, 1),
    )
    for block_index, block in enumerate(model.encoder.layers):
        add_hook(
            block,
            2 + block_index,
            lambda out, n=conv2_length: first_tensor(out)[0, :n, :],
        )

    projector_index = 2 + len(model.encoder.layers)
    add_hook(
        model.audio_adapter,
        projector_index,
        lambda out: first_tensor(out)[0, :projected_length, :],
    )
    decoder_start = projector_index + 1
    for block_index, block in enumerate(model.llm.model.layers):
        add_hook(
            block,
            decoder_start + block_index,
            lambda out, mask=audio_mask_1d: first_tensor(out)[0, mask, :],
        )
    return handles


@torch.inference_mode()
def generate_transcription(
    model: Any,
    tokenizer: Any,
    prepared: dict[str, Any],
) -> str:
    eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    modeling_module = sys.modules.get(model.__class__.__module__)
    penalty_class = getattr(
        modeling_module, "WindowedRepetitionPenaltyLogitsProcessor", None
    )
    if penalty_class is None:
        raise RuntimeError(
            "The released Covo modeling module does not expose its official "
            "WindowedRepetitionPenaltyLogitsProcessor"
        )
    logits_processor = LogitsProcessorList(
        [penalty_class(penalty=1.05, window_size=8)]
    )
    result = model.generate(
        input_ids=prepared["input_ids"],
        attention_mask=prepared["attention_mask"],
        wavs=[prepared["waveform"]],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=logits_processor,
        repetition_penalty=1.05,
        return_dict_in_generate=True,
    )
    sequences = result.sequences if hasattr(result, "sequences") else result
    generated_ids = sequences[0, prepared["input_ids"].shape[1] :]
    generated_ids = generated_ids[generated_ids < len(tokenizer)]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def extract_prefill_activations(
    model: Any,
    prepared: dict[str, Any],
    layer_specs: list[LayerSpec],
) -> list[np.ndarray]:
    store: list[np.ndarray | None] = [None] * len(layer_specs)
    handles = register_activation_hooks(model, prepared, layer_specs, store)
    try:
        inputs_embeds = inject_audio_embeddings(model, prepared)
        model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared["attention_mask"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = [
        layer_specs[index].name
        for index, activation in enumerate(store)
        if activation is None
    ]
    if missing:
        raise RuntimeError(f"Hooks did not fire for {', '.join(missing)}")
    return [activation for activation in store if activation is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-stimuli", type=int)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    c_audio_id = tokenizer.convert_tokens_to_ids("<|cAUDIO|>")
    # The released config contains a stale audio_token_index. The released
    # tokenizer and Covo model implementation both use ID 151666.
    if c_audio_id != 151666:
        raise RuntimeError(
            f"Unexpected cAUDIO ID {c_audio_id}; expected released ID 151666"
        )

    # CovoAudioConfig.from_dict does not implement Transformers'
    # return_unused_kwargs contract. Consequently, letting AutoModel load the
    # config internally raises "too many values to unpack". Load the custom
    # config explicitly first and pass the object into AutoModel to bypass that
    # incompatible code path while retaining sharded low-memory loading.
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    patch_dynamic_module_assets(model)

    layer_specs = make_layer_specs(model)
    if len(layer_specs) != 63:
        raise RuntimeError(
            f"Expected 63 layers: 34 encoder, 1 projector, 28 decoder; "
            f"found {len(layer_specs)}"
        )

    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(AUDIO_DIR, stimuli)
    all_activations: list[dict[str, np.ndarray]] = []
    transcriptions: list[str] = []

    print(f"Model: {MODEL_NAME}")
    print(f"Stimuli: {len(stimuli)} | Layers: {len(layer_specs)}")
    print(f"Output: {output_dir}")

    for stimulus_index, (audio_file, ground_truth) in enumerate(stimuli, start=1):
        audio_path = AUDIO_DIR / audio_file
        prepared = build_native_prompt(audio_path, tokenizer, c_audio_id)

        # Pass 1: behavioral output, with no activation hooks.
        transcription = generate_transcription(model, tokenizer, prepared)

        # Pass 2: independent non-generative full-prefill activation forward.
        activation_list = extract_prefill_activations(
            model, prepared, layer_specs
        )
        activations = {
            spec.name: value
            for spec, value in zip(layer_specs, activation_list)
        }
        validate_activation_set(activations, layer_specs)
        all_activations.append(activations)
        transcriptions.append(transcription)

        # Retained but intentionally disabled. The analysis only needs the
        # final aggregate MAT and NPY files.
        if SAVE_INDIVIDUAL_STIMULI:
            individual_dir = output_dir / "individual_stimuli"
            individual_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                individual_dir / f"{Path(audio_file).stem}_activations.npy",
                activations,
                allow_pickle=True,
            )

        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file} | "
            f"GT={ground_truth} | transcription={transcription!r} | "
            f"T={prepared['conv1_length']}->{prepared['conv2_length']}->"
            f"{prepared['projected_length']}"
        )
        del prepared, activations
        gc.collect()
        torch.cuda.empty_cache()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = save_final_outputs(
        model_name=MODEL_NAME,
        file_prefix=FILE_PREFIX,
        output_dir=output_dir,
        stimuli=stimuli,
        specs=layer_specs,
        all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(model).__name__),
            ("model_checkpoint", str(MODEL_PATH)),
            ("parameter_count", str(parameter_count)),
            ("parameter_count_billions", f"{parameter_count / 1e9:.4f}"),
            (
                "architecture_family",
                "Whisper large-v3 encoder + stride-8 audio adapter + 28-layer Qwen decoder",
            ),
            ("prompt", COVO_TASK_PROMPT),
            ("system_prompt", COVO_TASK_PROMPT),
            (
                "prompt_wrapper",
                "Covo native ChatML and native audio placeholders",
            ),
            ("audio_encoder_layer_count", str(len(model.encoder.layers))),
            ("decoder_layer_count", str(len(model.llm.model.layers))),
            (
                "generation_mode",
                "behavior: official first-round a2t generation (endoftext EOS and official windowed repetition penalty); activation: separate non-generate prefill, use_cache=False",
            ),
            (
                "decoder_positions_saved",
                "only cAUDIO acoustic positions; text prompt and generated tokens excluded",
            ),
            ("audio_sample_rate_hz", "24000"),
            ("tokenizer_size", str(len(tokenizer))),
            ("cAUDIO_token_id", str(c_audio_id)),
            (
                "padding_policy",
                "30-second Whisper padding cropped at every encoder layer",
            ),
        ],
    )
    print(
        f"Completed successfully. Accuracy={result['accuracy']:.2%}; "
        f"MAT={result['matlab_path']}; NPY={result['numpy_path']}"
    )


if __name__ == "__main__":
    main()
