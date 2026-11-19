#!/usr/bin/env python3
# Pipeline step 01 shared implementation.
"""Shared output contract for the 14-model activation extraction project."""

from __future__ import annotations

import csv
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import io


_CHINESE_AUDIO_STIMULI = [
    ("caomei.wav", "草莓"), ("daiyu.wav", "带鱼"), ("dangao.wav", "蛋糕"),
    ("doufu.wav", "豆腐"), ("haidai.wav", "海带"), ("hongshu.wav", "红薯"),
    ("jidan.wav", "鸡蛋"), ("jianbing.wav", "煎饼"), ("lizhi.wav", "荔枝"),
    ("longxia.wav", "龙虾"), ("luobo.wav", "萝卜"), ("lvdou.wav", "绿豆"),
    ("mangguo.wav", "芒果"), ("miji.wav", "蜜桔"), ("mianbao.wav", "面包"),
    ("mogu.wav", "蘑菇"), ("niurou.wav", "牛肉"), ("pingguo.wav", "苹果"),
    ("ruantang.wav", "软糖"), ("shutiao.wav", "薯条"), ("xigua.wav", "西瓜"),
    ("xiangchang.wav", "香肠"), ("yangcong.wav", "洋葱"), ("yingtao.wav", "樱桃"),
    ("yumi.wav", "玉米"), ("zhuti.wav", "猪蹄"), ("caoping.wav", "草坪"),
    ("daibiao.wav", "代表"), ("dangong.wav", "弹弓"), ("douhao.wav", "逗号"),
    ("haitan.wav", "海滩"), ("hongdeng.wav", "红灯"), ("jihui.wav", "机会"),
    ("jianbang.wav", "肩膀"), ("liliang.wav", "力量"), ("longzi.wav", "笼子"),
    ("luoxuan.wav", "螺旋"), ("lvshi.wav", "律师"), ("mangren.wav", "盲人"),
    ("mima.wav", "密码"), ("mianju.wav", "面具"), ("mote.wav", "模特"),
    ("niuzai.wav", "牛仔"), ("pingwei.wav", "评委"), ("ruanjian.wav", "软件"),
    ("shujia.wav", "暑假"), ("xigai.wav", "膝盖"), ("xiangcun.wav", "乡村"),
    ("yangguang.wav", "阳光"), ("yingxiong.wav", "英雄"), ("yushi.wav", "浴室"),
    ("zhubao.wav", "珠宝"),
]


def _configured_audio_stimuli() -> list[tuple[str, str]]:
    """Load an optional language-specific stimulus manifest.

    The Chinese pipeline remains the default.  English launchers set
    ``SLM_STIMULUS_MANIFEST`` to a CSV containing ``file`` and ``word``
    columns, which lets all model extractors share the same audited 52-word
    order without duplicating stimulus constants in fourteen scripts.
    """
    manifest = os.environ.get("SLM_STIMULUS_MANIFEST", "").strip()
    if not manifest:
        return list(_CHINESE_AUDIO_STIMULI)
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SLM_STIMULUS_MANIFEST not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 52:
        raise ValueError(f"Expected 52 English stimuli in {path}, found {len(rows)}")
    stimuli = [(str(row["file"]).strip(), str(row["word"]).strip()) for row in rows]
    if any(not filename or not word for filename, word in stimuli):
        raise ValueError(f"Blank file/word value in {path}")
    if len({filename for filename, _ in stimuli}) != 52:
        raise ValueError(f"Stimulus filenames are not unique in {path}")
    return stimuli


AUDIO_STIMULI = _configured_audio_stimuli()


@dataclass(frozen=True)
class LayerSpec:
    name: str
    sequence_type: str
    hidden_dim: int


def require_complete_sharded_checkpoint(model_path: Path) -> None:
    """Fail before model loading when a sharded checkpoint is incomplete."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in expected if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete checkpoint at {model_path}: missing {len(missing)}/{len(expected)} "
            f"weight shards: {missing}"
        )


def first_tensor(output):
    """Return the tensor payload used by hooks that may receive tuple outputs."""
    return output[0] if isinstance(output, (tuple, list)) else output


def decoder_audio_positions(hidden_states, audio_mask):
    """Select only audio-placeholder positions from a decoder prefill output."""
    if hidden_states.ndim != 3:
        raise RuntimeError(f"Expected [batch, sequence, hidden], got {hidden_states.shape}")
    if hidden_states.shape[1] != audio_mask.numel():
        raise RuntimeError(
            f"Decoder call is not the prefill pass: sequence={hidden_states.shape[1]}, "
            f"prompt={audio_mask.numel()}"
        )
    return hidden_states[0, audio_mask.to(hidden_states.device)]


def selected_stimuli(max_stimuli: int | None) -> list[tuple[str, str]]:
    if max_stimuli is None:
        return list(AUDIO_STIMULI)
    if max_stimuli <= 0 or max_stimuli > len(AUDIO_STIMULI):
        raise ValueError(f"--max-stimuli must be in [1, {len(AUDIO_STIMULI)}]")
    return list(AUDIO_STIMULI[:max_stimuli])


def validate_audio_files(audio_dir: Path, stimuli: Iterable[tuple[str, str]]) -> None:
    audio_dir = Path(audio_dir)
    missing = [name for name, _ in stimuli if not (audio_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing stimulus files in {audio_dir}: {missing}")


def to_float32_matrix(value, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu", dtype=getattr(__import__("torch"), "float32"))
        value = value.contiguous().numpy()
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty 2-D matrix, found {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def validate_activation_set(activations: dict[str, np.ndarray], specs: list[LayerSpec]) -> None:
    expected = [spec.name for spec in specs]
    missing = sorted(set(expected).difference(activations))
    unexpected = sorted(set(activations).difference(expected))
    if missing or unexpected:
        raise RuntimeError(f"Activation set mismatch; missing={missing}, unexpected={unexpected}")
    for spec in specs:
        value = np.asarray(activations[spec.name])
        if value.ndim != 2 or value.shape[1] != spec.hidden_dim:
            raise ValueError(
                f"Invalid {spec.name}: expected sequence x {spec.hidden_dim}, found {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"NaN/Inf in {spec.name}")


def normalize_transcription(text: str) -> str:
    """Canonical text used for exact-match ASR scoring across all models."""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return "".join(
        char for char in text
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "S", "C", "Z"))
    )


def sequence_range_text(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.int64)
    return f"min={values.min()}; median={np.median(values):.1f}; max={values.max()}"


def save_final_outputs(
    *,
    model_name: str,
    file_prefix: str,
    output_dir: Path,
    stimuli: list[tuple[str, str]],
    specs: list[LayerSpec],
    all_activations: list[dict[str, np.ndarray]],
    transcriptions: list[str],
    summary_rows: list[tuple[str, str]],
) -> dict[str, object]:
    if len(stimuli) != len(all_activations) or len(stimuli) != len(transcriptions):
        raise ValueError("Stimulus, activation, and transcription counts differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_stimuli, n_layers = len(stimuli), len(specs)
    data_cell = np.empty((n_stimuli, n_layers), dtype=object)
    valid_lengths = np.zeros((n_stimuli, n_layers), dtype=np.int32)
    for stimulus_index, activations in enumerate(all_activations):
        validate_activation_set(activations, specs)
        for layer_index, spec in enumerate(specs):
            value = np.asarray(activations[spec.name], dtype=np.float32)
            data_cell[stimulus_index, layer_index] = value
            valid_lengths[stimulus_index, layer_index] = value.shape[0]

    stimulus_files = np.asarray([name for name, _ in stimuli], dtype=object)
    layer_names = np.asarray([spec.name for spec in specs], dtype=object)
    sequence_types = np.asarray([spec.sequence_type for spec in specs], dtype=object)
    hidden_dims = np.asarray([spec.hidden_dim for spec in specs], dtype=np.int32)
    bundle = {
        "data_cell": data_cell,
        "stimulus_files": stimulus_files,
        "layer_names": layer_names,
        "sequence_types": sequence_types,
        "hidden_dims": hidden_dims,
        "valid_lengths": valid_lengths,
        "activation_layout": "sequence_by_hidden_dimension",
        "model_name": model_name,
    }
    numpy_path = output_dir / f"{file_prefix}_hidden_dimension_activations.npy"
    np.save(numpy_path, bundle, allow_pickle=True)
    matlab_path = output_dir / f"{file_prefix}_hidden_dimension_activations.mat"
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
            "model_name": model_name,
        },
        do_compression=False,
        long_field_names=True,
    )

    correct = 0
    clean_predictions = []
    for (_, ground_truth), prediction in zip(stimuli, transcriptions):
        clean = " ".join(str(prediction).replace("\t", " ").splitlines()).strip()
        clean_predictions.append(clean)
        correct += int(normalize_transcription(clean) == normalize_transcription(ground_truth))
    errors = n_stimuli - correct
    accuracy = correct / n_stimuli
    transcription_path = output_dir / f"{file_prefix}_transcriptions.txt"
    with transcription_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            f"正确: {correct}\t错误: {errors}\t总数: {n_stimuli}"
            f"\t转录正确率: {accuracy:.2%}\n"
        )
        stream.write("音频文件名\tGroundTruth\t模型转录\n")
        for (audio_file, ground_truth), prediction in zip(stimuli, clean_predictions):
            stream.write(f"{audio_file}\t{ground_truth}\t{prediction}\n")

    rows = [
        ("model_name", model_name),
        ("stimulus_count", str(n_stimuli)),
        ("unit_definition", "channel for convolution; block-output hidden dimension otherwise"),
        ("activation_layout", "sequence x hidden_dimension"),
        ("saved_dtype", "float32"),
        ("extracted_layer_count", str(n_layers)),
        ("layer_names", "; ".join(layer_names.tolist())),
        ("sequence_types", "; ".join(sequence_types.tolist())),
        ("hidden_dims", "; ".join(map(str, hidden_dims.tolist()))),
    ]
    for layer_index, spec in enumerate(specs):
        rows.append((f"valid_length_{layer_index:03d}_{spec.name}", sequence_range_text(valid_lengths[:, layer_index])))
    rows.extend(summary_rows)
    rows.extend([
        ("transcription_correct", str(correct)),
        ("transcription_errors", str(errors)),
        ("transcription_accuracy", f"{accuracy:.6f}"),
    ])
    summary_path = output_dir / f"{file_prefix}_model_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("field", "value"))
        writer.writerows(rows)

    return {
        "matlab_path": matlab_path,
        "numpy_path": numpy_path,
        "transcription_path": transcription_path,
        "summary_path": summary_path,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
        "valid_lengths": valid_lengths,
    }
