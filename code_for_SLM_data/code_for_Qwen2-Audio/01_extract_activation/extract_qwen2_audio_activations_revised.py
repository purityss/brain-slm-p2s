#!/usr/bin/env python3
"""Extract sequence-resolved Qwen2-Audio hidden-dimension activations.

The script performs one cached ``generate`` call per stimulus and keeps only
activations needed by the revised primary analysis:

* activated outputs of audio conv1 and conv2;
* outputs of every audio-encoder transformer block;
* multimodal-projector output;
* audio-token positions from the prefill output of every decoder block.

Prompt positions, padding positions, repeated decoding histories, generated-
token activations, final norms, and logits are not saved.  Per-stimulus/per-layer
checkpoint code is retained as a commented optional block, but the default run
saves only the final aggregate files and two compact reports:

* a MATLAB file whose ``data_cell`` is stimuli x layers; and
* a NumPy ``.npy`` bundle containing the same object array and its metadata;
* a tab-separated transcription report; and
* a CSV model/extraction summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import unicodedata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Fix the physical GPU before importing torch. Inside this process it is cuda:0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import io
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.final_asr_prompts import QWEN2_AUDIO_WRAPPED_PROMPT  # noqa: E402
AUDIO_STIMULI = [
    ("caomei.wav", "草莓"), ("daiyu.wav", "带鱼"), ("dangao.wav", "蛋糕"),
    ("doufu.wav", "豆腐"), ("haidai.wav", "海带"), ("hongshu.wav", "红薯"),
    ("jidan.wav", "鸡蛋"), ("jianbing.wav", "煎饼"), ("lizhi.wav", "荔枝"),
    ("longxia.wav", "龙虾"), ("luobo.wav", "萝卜"), ("lvdou.wav", "绿豆"),
    ("mangguo.wav", "芒果"), ("miji.wav", "蜜桔"), ("mianbao.wav", "面包"),
    ("mogu.wav", "蘑菇"), ("niurou.wav", "牛肉"), ("pingguo.wav", "苹果"),
    ("ruantang.wav", "软糖"), ("shutiao.wav", "薯条"), ("xigua.wav", "西瓜"),
    ("xiangchang.wav", "香肠"), ("yangcong.wav", "洋葱"),
    ("yingtao.wav", "樱桃"), ("yumi.wav", "玉米"), ("zhuti.wav", "猪蹄"),
    ("caoping.wav", "草坪"), ("daibiao.wav", "代表"), ("dangong.wav", "弹弓"),
    ("douhao.wav", "逗号"), ("haitan.wav", "海滩"), ("hongdeng.wav", "红灯"),
    ("jihui.wav", "机会"), ("jianbang.wav", "肩膀"), ("liliang.wav", "力量"),
    ("longzi.wav", "笼子"), ("luoxuan.wav", "螺旋"), ("lvshi.wav", "律师"),
    ("mangren.wav", "盲人"), ("mima.wav", "密码"), ("mianju.wav", "面具"),
    ("mote.wav", "模特"), ("niuzai.wav", "牛仔"), ("pingwei.wav", "评委"),
    ("ruanjian.wav", "软件"), ("shujia.wav", "暑假"), ("xigai.wav", "膝盖"),
    ("xiangcun.wav", "乡村"), ("yangguang.wav", "阳光"),
    ("yingxiong.wav", "英雄"), ("yushi.wav", "浴室"), ("zhubao.wav", "珠宝"),
]
AUDIO_FILES = [audio_file for audio_file, _ground_truth in AUDIO_STIMULI]

DEFAULT_MODEL_PATH = "/data/zss/SLMs/All_Models/Qwen2-Audio-7B"
DEFAULT_AUDIO_DIR = str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)")
DEFAULT_OUTPUT_DIR = str(
    PROJECT_ROOT
    / "code_for_Qwen2-Audio/01_extract_activation/Qwen2_Audio_hidden_dimension_activations"
)
DEFAULT_PROMPT = QWEN2_AUDIO_WRAPPED_PROMPT


@dataclass(frozen=True)
class LayerSpec:
    name: str
    sequence_type: str
    hidden_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def first_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Hook output does not begin with a tensor: {type(output)!r}")


def to_float32_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def build_layer_specs(model: Qwen2AudioForConditionalGeneration) -> list[LayerSpec]:
    audio_dim = int(model.config.audio_config.d_model)
    text_dim = int(model.config.text_config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native", int(model.audio_tower.conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native", int(model.audio_tower.conv2.out_channels)),
    ]
    specs.extend(
        LayerSpec(f"audio_block_{index:02d}", "audio_native", audio_dim)
        for index in range(len(model.audio_tower.layers))
    )
    specs.append(LayerSpec("multimodal_projector", "audio_projected", text_dim))
    specs.extend(
        LayerSpec(f"decoder_block_{index:02d}", "decoder_prefill_audio", text_dim)
        for index in range(len(model.language_model.model.layers))
    )
    return specs


def register_hooks(
    model: Qwen2AudioForConditionalGeneration,
    captured: dict[str, np.ndarray],
    valid_lengths: dict[str, int],
    decoder_audio_mask: torch.Tensor,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def save_once(name: str, transform: Callable[[torch.Tensor], torch.Tensor]):
        def hook(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
            if name in captured:
                return
            tensor = transform(first_tensor(output))
            if tensor.ndim != 2 or tensor.shape[0] == 0:
                raise ValueError(f"Invalid activation shape for {name}: {tuple(tensor.shape)}")
            captured[name] = to_float32_numpy(tensor)

        return hook

    conv1_length = valid_lengths["conv1"]
    conv2_length = valid_lengths["conv2"]
    projector_length = valid_lengths["projector"]

    handles.append(
        model.audio_tower.conv1.register_forward_hook(
            save_once(
                "audio_conv1",
                lambda x: F.gelu(x[0, :, :conv1_length]).transpose(0, 1),
            )
        )
    )
    handles.append(
        model.audio_tower.conv2.register_forward_hook(
            save_once(
                "audio_conv2",
                lambda x: F.gelu(x[0, :, :conv2_length]).transpose(0, 1),
            )
        )
    )

    for index, layer in enumerate(model.audio_tower.layers):
        name = f"audio_block_{index:02d}"
        handles.append(
            layer.register_forward_hook(
                save_once(name, lambda x, length=conv2_length: x[0, :length, :])
            )
        )

    handles.append(
        model.multi_modal_projector.register_forward_hook(
            save_once(
                "multimodal_projector",
                lambda x: x[0, :projector_length, :],
            )
        )
    )

    for index, layer in enumerate(model.language_model.model.layers):
        name = f"decoder_block_{index:02d}"

        def select_decoder_audio(x: torch.Tensor, mask=decoder_audio_mask) -> torch.Tensor:
            if x.shape[1] != mask.numel():
                raise ValueError(
                    "The first decoder hook call is not the expected prefill: "
                    f"sequence={x.shape[1]}, mask={mask.numel()}"
                )
            device_mask = mask.to(device=x.device)
            return x[0, device_mask, :]

        handles.append(layer.register_forward_hook(save_once(name, select_decoder_audio)))

    return handles


def extract_one_stimulus(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    audio_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> tuple[dict[str, np.ndarray], str]:
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)

    inputs = processor(
        text=prompt,
        audio=audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    required = {"input_ids", "input_features", "feature_attention_mask"}
    missing = required.difference(inputs.keys())
    if missing:
        raise KeyError(f"Processor output is missing required fields: {sorted(missing)}")

    feature_length = int(inputs["feature_attention_mask"].sum().item())
    conv2_length_tensor, projector_length_tensor = (
        model.audio_tower._get_feat_extract_output_lengths(
            torch.tensor([feature_length], dtype=torch.long)
        )
    )
    conv2_length = int(conv2_length_tensor.item())
    projector_length = int(projector_length_tensor.item())
    valid_lengths = {
        "conv1": feature_length,
        "conv2": conv2_length,
        "projector": projector_length,
    }

    audio_token_id = int(model.config.audio_token_index)
    decoder_audio_mask = inputs["input_ids"][0].eq(audio_token_id).cpu()
    audio_token_count = int(decoder_audio_mask.sum().item())
    if audio_token_count != projector_length:
        raise ValueError(
            "Audio-token count does not match valid projected-audio length: "
            f"tokens={audio_token_count}, projected={projector_length}, "
            f"stimulus={audio_path.name}"
        )

    model_device = next(model.parameters()).device
    model_inputs = inputs.to(model_device)

    # Pass 1: behavioral transcription only. No activation hooks are installed.
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.0,
            use_cache=True,
        )

    prompt_length = int(inputs["input_ids"].shape[1])
    new_token_ids = generated_ids[:, prompt_length:]
    transcription = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Pass 2: one non-autoregressive multimodal prefill forward. Only this pass
    # is hooked, so generated text can never enter the activation bundle.
    captured: dict[str, np.ndarray] = {}
    handles = register_hooks(model, captured, valid_lengths, decoder_audio_mask)
    try:
        with torch.inference_mode():
            model(**model_inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    print(f"    transcription: {transcription}")
    return captured, transcription


def validate_stimulus_activations(
    captured: dict[str, np.ndarray], specs: list[LayerSpec]
) -> None:
    expected_names = [spec.name for spec in specs]
    missing = sorted(set(expected_names).difference(captured))
    unexpected = sorted(set(captured).difference(expected_names))
    if missing or unexpected:
        raise RuntimeError(
            f"Activation set mismatch; missing={missing}, unexpected={unexpected}"
        )

    for spec in specs:
        activation = captured[spec.name]
        if activation.ndim != 2:
            raise ValueError(f"{spec.name} is not 2-D: {activation.shape}")
        if activation.shape[1] != spec.hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch for {spec.name}: "
                f"expected={spec.hidden_dim}, found={activation.shape[1]}"
            )
def aggregate_outputs(
    all_activations: list[dict[str, np.ndarray]],
    output_dir: Path,
    specs: list[LayerSpec],
) -> tuple[Path, Path, np.ndarray]:
    n_stimuli = len(AUDIO_FILES)
    n_layers = len(specs)
    if len(all_activations) != n_stimuli:
        raise ValueError(
            f"Expected {n_stimuli} stimulus activation sets, "
            f"found {len(all_activations)}"
        )
    data_cell = np.empty((n_stimuli, n_layers), dtype=object)
    valid_lengths = np.zeros((n_stimuli, n_layers), dtype=np.int32)

    for stimulus_offset, captured in enumerate(all_activations):
        for layer_index, spec in enumerate(specs):
            activation = captured[spec.name]
            if activation.ndim != 2 or activation.shape[1] != spec.hidden_dim:
                raise ValueError(
                    f"Invalid activation for stimulus={stimulus_offset + 1}, "
                    f"layer={spec.name}: {activation.shape}"
                )
            data_cell[stimulus_offset, layer_index] = activation
            valid_lengths[stimulus_offset, layer_index] = activation.shape[0]

    layer_names = np.asarray([spec.name for spec in specs], dtype=object)
    sequence_types = np.asarray([spec.sequence_type for spec in specs], dtype=object)
    hidden_dims = np.asarray([spec.hidden_dim for spec in specs], dtype=np.int32)
    stimulus_files = np.asarray(AUDIO_FILES, dtype=object)

    numpy_bundle = {
        "data_cell": data_cell,
        "stimulus_files": stimulus_files,
        "layer_names": layer_names,
        "sequence_types": sequence_types,
        "hidden_dims": hidden_dims,
        "valid_lengths": valid_lengths,
        "activation_layout": "sequence_by_hidden_dimension",
    }
    numpy_path = output_dir / "Qwen2_Audio_hidden_dimension_activations.npy"
    np.save(numpy_path, numpy_bundle, allow_pickle=True)

    matlab_path = output_dir / "Qwen2_Audio_hidden_dimension_activations.mat"
    io.savemat(
        matlab_path,
        {
            "data_cell": data_cell,
            "stimulus_files": stimulus_files.reshape(-1, 1),
            "layer_names": layer_names.reshape(1, -1),
            "sequence_types": sequence_types.reshape(1, -1),
            "hidden_dims": hidden_dims.reshape(1, -1),
            "valid_lengths": valid_lengths,
            "activation_layout": "sequence_by_hidden_dimension",
        },
        do_compression=False,
        long_field_names=True,
    )
    return matlab_path, numpy_path, valid_lengths


def normalize_transcription(text: str) -> str:
    """Apply the project-wide exact-match ASR normalization."""
    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S", "C", "Z"))
    )


def write_transcription_report(
    output_dir: Path, transcriptions: list[str]
) -> tuple[Path, int, int, float]:
    if len(transcriptions) != len(AUDIO_STIMULI):
        raise ValueError(
            f"Expected {len(AUDIO_STIMULI)} transcriptions, found {len(transcriptions)}"
        )

    rows = []
    correct_count = 0
    for (audio_file, ground_truth), prediction in zip(AUDIO_STIMULI, transcriptions):
        sanitized_prediction = " ".join(prediction.replace("\t", " ").splitlines()).strip()
        is_correct = normalize_transcription(sanitized_prediction) == normalize_transcription(
            ground_truth
        )
        correct_count += int(is_correct)
        rows.append((audio_file, ground_truth, sanitized_prediction))

    total_count = len(rows)
    error_count = total_count - correct_count
    accuracy = correct_count / total_count if total_count else float("nan")
    path = output_dir / "Qwen2_Audio_transcriptions.txt"
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            f"正确: {correct_count}\t错误: {error_count}\t总数: {total_count}"
            f"\t转录正确率: {accuracy:.2%}\n"
        )
        stream.write("音频文件名\tGroundTruth\t模型转录\n")
        for audio_file, ground_truth, prediction in rows:
            stream.write(f"{audio_file}\t{ground_truth}\t{prediction}\n")
    return path, correct_count, error_count, accuracy


def sequence_range_text(lengths: np.ndarray) -> str:
    return (
        f"min={int(np.min(lengths))}; "
        f"median={float(np.median(lengths)):.1f}; "
        f"max={int(np.max(lengths))}"
    )


def write_model_summary(
    output_dir: Path,
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    model_path: Path,
    prompt: str,
    specs: list[LayerSpec],
    valid_lengths: np.ndarray,
    correct_count: int,
    error_count: int,
    accuracy: float,
) -> Path:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    audio_layers = len(model.audio_tower.layers)
    decoder_layers = len(model.language_model.model.layers)
    audio_dim = int(model.config.audio_config.d_model)
    text_dim = int(model.config.text_config.hidden_size)

    rows = [
        ("model_name", "Qwen2_Audio"),
        ("model_class", model.__class__.__name__),
        ("model_checkpoint", str(model_path)),
        ("parameter_count", str(total_parameters)),
        ("parameter_count_billions", f"{total_parameters / 1e9:.3f}"),
        ("prompt", prompt),
        ("stimulus_count", str(len(AUDIO_STIMULI))),
        ("sampling_rate_hz", str(int(processor.feature_extractor.sampling_rate))),
        ("unit_definition", "channel for CNN; block-output hidden dimension otherwise"),
        ("activation_layout", "sequence x hidden_dimension"),
        ("saved_dtype", "float32"),
        ("extracted_layer_count", str(len(specs))),
        ("audio_conv_layer_count", "2"),
        (
            "audio_conv_units_per_layer",
            f"conv1={model.audio_tower.conv1.out_channels}; "
            f"conv2={model.audio_tower.conv2.out_channels}",
        ),
        ("audio_encoder_transformer_layer_count", str(audio_layers)),
        ("audio_encoder_units_per_layer", str(audio_dim)),
        ("multimodal_projector_layer_count", "1"),
        ("multimodal_projector_units", str(text_dim)),
        ("decoder_transformer_layer_count", str(decoder_layers)),
        ("decoder_units_per_layer", str(text_dim)),
        ("conv1_valid_sequence_lengths", sequence_range_text(valid_lengths[:, 0])),
        ("conv2_valid_sequence_lengths", sequence_range_text(valid_lengths[:, 1])),
        (
            "audio_encoder_valid_sequence_lengths",
            sequence_range_text(valid_lengths[:, 2]),
        ),
        (
            "projector_valid_sequence_lengths",
            sequence_range_text(valid_lengths[:, 34]),
        ),
        (
            "decoder_valid_sequence_lengths",
            sequence_range_text(valid_lengths[:, 35]),
        ),
        (
            "encoder_sequence_definition",
            "native valid acoustic positions; padding excluded independently per stimulus",
        ),
        (
            "decoder_sequence_definition",
            "prefill audio-token positions only; prompt, special-token, padding, and "
            "generated-token positions excluded",
        ),
        (
            "generation_mode",
            "one deterministic generate call; do_sample=False; num_beams=5; "
            "behavior: deterministic 5-beam generate; activation: separate non-generate prefill forward, use_cache=False",
        ),
        ("transcription_correct", str(correct_count)),
        ("transcription_errors", str(error_count)),
        ("transcription_accuracy", f"{accuracy:.6f}"),
    ]

    path = output_dir / "Qwen2_Audio_model_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(rows)
    return path


def main() -> None:
    global AUDIO_STIMULI, AUDIO_FILES
    args = parse_args()
    set_reproducible_seed(args.seed)

    if args.max_stimuli is not None:
        if args.max_stimuli <= 0:
            raise ValueError("--max-stimuli must be positive")
        AUDIO_STIMULI = AUDIO_STIMULI[:args.max_stimuli]
        AUDIO_FILES = [audio_file for audio_file, _ground_truth in AUDIO_STIMULI]

    model_path = Path(args.model_path).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    missing_audio = [name for name in AUDIO_FILES if not (audio_dir / name).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"Missing stimulus files: {missing_audio}")

    print(f"Loading processor from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    print(f"Loading model from {model_path}")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    specs = build_layer_specs(model)
    if len(specs) != 67:
        raise RuntimeError(f"Expected 67 extraction layers, found {len(specs)}")

    print(
        json.dumps(
            {
                "stimuli": len(AUDIO_FILES),
                "layers": len(specs),
                "model": str(model_path),
                "audio_dir": str(audio_dir),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )

    all_activations: list[dict[str, np.ndarray]] = []
    transcriptions: list[str] = []
    for stimulus_index, (audio_file, _ground_truth) in enumerate(
        AUDIO_STIMULI, start=1
    ):
        print(f"[{stimulus_index:02d}/{len(AUDIO_STIMULI)}] {audio_file}: extracting")
        captured, transcription = extract_one_stimulus(
            model=model,
            processor=processor,
            audio_path=audio_dir / audio_file,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        validate_stimulus_activations(captured, specs)
        all_activations.append(captured)
        transcriptions.append(transcription)

        # Optional per-stimulus/per-layer checkpoint saving (disabled by design).
        # Uncomment this block only if checkpoint files are explicitly needed:
        # per_stimulus_dir = (
        #     output_dir
        #     / "Qwen2_Audio_hidden_dimension_activations_per_stimulus"
        #     / f"stimulus_{stimulus_index:03d}_{Path(audio_file).stem}"
        # )
        # per_stimulus_dir.mkdir(parents=True, exist_ok=True)
        # for layer_index, spec in enumerate(specs):
        #     np.save(
        #         per_stimulus_dir / f"layer_{layer_index:03d}_{spec.name}.npy",
        #         captured[spec.name],
        #         allow_pickle=False,
        #     )

        shape_summary = {
            "conv1": captured["audio_conv1"].shape,
            "audio_block_31": captured["audio_block_31"].shape,
            "projector": captured["multimodal_projector"].shape,
            "decoder_block_31": captured["decoder_block_31"].shape,
        }
        print(
            f"    captured {len(captured)} layers; "
            f"representative shapes={shape_summary}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matlab_path, numpy_path, valid_lengths = aggregate_outputs(
        all_activations=all_activations,
        output_dir=output_dir,
        specs=specs,
    )
    transcription_path, correct_count, error_count, accuracy = (
        write_transcription_report(output_dir, transcriptions)
    )
    summary_path = write_model_summary(
        output_dir=output_dir,
        model=model,
        processor=processor,
        model_path=model_path,
        prompt=args.prompt,
        specs=specs,
        valid_lengths=valid_lengths,
        correct_count=correct_count,
        error_count=error_count,
        accuracy=accuracy,
    )
    print(f"Complete MATLAB file: {matlab_path}")
    print(f"Complete NumPy file:  {numpy_path}")
    print(f"Transcription report: {transcription_path}")
    print(f"Model summary:        {summary_path}")


if __name__ == "__main__":
    main()
