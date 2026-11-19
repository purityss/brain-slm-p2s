#!/usr/bin/env python3
# Pipeline step 02 shared implementation.
"""Shared GPU engine for unit-wise temporal RDM construction.

Model-specific entry points provide paths and layer-selection rules.  The
primary analysis uses native, non-overlapping windows.  The supplementary
overlap analysis linearly maps each eligible native sequence to 101 relative
time points and applies 16 windows of width 0.25 and step 0.05.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .validate_activation_inputs import validate_activation_input_bundle


N_STIMULI = 52
GRID_POINTS = 101
OVERLAP_WINDOWS = 16
OVERLAP_WIDTH_POINTS = 26
OVERLAP_STEP_POINTS = 5
OVERLAP_START_INDICES = np.arange(OVERLAP_WINDOWS, dtype=np.int32) * OVERLAP_STEP_POINTS
OVERLAP_END_INDICES = OVERLAP_START_INDICES + OVERLAP_WIDTH_POINTS - 1
OVERLAP_STARTS = OVERLAP_START_INDICES.astype(np.float32) / (GRID_POINTS - 1)
OVERLAP_ENDS = OVERLAP_END_INDICES.astype(np.float32) / (GRID_POINTS - 1)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    default_input: str
    default_output: str


@dataclass(frozen=True)
class SelectionRule:
    minimum_length: int | None = None
    exact_length: int | None = None
    require_constant_length: bool = True

    def describe(self) -> str:
        parts = []
        if self.exact_length is not None:
            parts.append(f"native length == {self.exact_length}")
        if self.minimum_length is not None:
            parts.append(f"native length >= {self.minimum_length}")
        if self.require_constant_length:
            parts.append("equal length across all stimuli")
        return "; ".join(parts) or "all layers"


def build_parser(description: str, config: ModelConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", default=config.default_input)
    parser.add_argument("--output", default=config.default_output)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--unit-chunk", type=int, default=128)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def as_strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).reshape(-1)]


def load_activation_bundle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Activation bundle not found: {path}")
    bundle = np.load(path, allow_pickle=True).item()
    required = {
        "data_cell",
        "stimulus_files",
        "layer_names",
        "sequence_types",
        "hidden_dims",
        "valid_lengths",
    }
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"Activation bundle is missing fields: {sorted(missing)}")
    data_cell = np.asarray(bundle["data_cell"], dtype=object)
    layer_names = np.asarray(bundle["layer_names"], dtype=object).reshape(-1)
    hidden_dims = np.asarray(bundle["hidden_dims"], dtype=np.int64).reshape(-1)
    valid_lengths = np.asarray(bundle["valid_lengths"], dtype=np.int64)
    if data_cell.shape != (N_STIMULI, len(layer_names)):
        raise ValueError(f"Expected data_cell shape (52, layers), found {data_cell.shape}")
    if valid_lengths.shape != data_cell.shape:
        raise ValueError("valid_lengths shape does not match data_cell")
    if len(hidden_dims) != len(layer_names):
        raise ValueError("hidden_dims length does not match layer_names")
    return bundle


def select_bundle(bundle: dict, rule: SelectionRule) -> tuple[dict, list[dict]]:
    lengths = np.asarray(bundle["valid_lengths"], dtype=np.int64)
    layer_names = as_strings(bundle["layer_names"])
    sequence_types = as_strings(bundle["sequence_types"])
    keep = np.ones(lengths.shape[1], dtype=bool)
    if rule.require_constant_length:
        keep &= np.ptp(lengths, axis=0) == 0
    if rule.minimum_length is not None:
        keep &= lengths.min(axis=0) >= rule.minimum_length
    if rule.exact_length is not None:
        keep &= (lengths.min(axis=0) == rule.exact_length) & (
            lengths.max(axis=0) == rule.exact_length
        )
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError(f"No layers satisfy selection rule: {rule.describe()}")

    excluded = []
    for index in np.flatnonzero(~keep):
        excluded.append(
            {
                "source_layer_index_zero_based": int(index),
                "layer_name": layer_names[index],
                "sequence_type": sequence_types[index],
                "minimum_length": int(lengths[:, index].min()),
                "maximum_length": int(lengths[:, index].max()),
            }
        )

    selected = dict(bundle)
    selected["data_cell"] = np.asarray(bundle["data_cell"], dtype=object)[:, indices]
    selected["layer_names"] = np.asarray(bundle["layer_names"], dtype=object).reshape(-1)[indices]
    selected["sequence_types"] = np.asarray(bundle["sequence_types"], dtype=object).reshape(-1)[indices]
    selected["hidden_dims"] = np.asarray(bundle["hidden_dims"], dtype=np.int64).reshape(-1)[indices]
    selected["valid_lengths"] = lengths[:, indices]
    selected["source_layer_indices"] = indices.astype(np.int32)
    selected["original_layer_count"] = int(lengths.shape[1])
    return selected, excluded


def validate_layer_activations(
    data_cell: np.ndarray, layer_index: int, hidden_dim: int
) -> list[np.ndarray]:
    activations = []
    for stimulus_index in range(N_STIMULI):
        activation = np.asarray(data_cell[stimulus_index, layer_index])
        if activation.ndim != 2:
            raise ValueError(
                f"Stimulus {stimulus_index + 1}, layer {layer_index}: "
                f"expected 2-D activation, found {activation.shape}"
            )
        if activation.shape[0] < 1:
            raise ValueError("Empty activation sequence")
        if activation.shape[1] != hidden_dim:
            raise ValueError(
                f"Stimulus {stimulus_index + 1}, layer {layer_index}: "
                f"expected {hidden_dim} units, found {activation.shape[1]}"
            )
        if not np.isfinite(activation).all():
            raise ValueError(
                f"NaN/Inf in stimulus {stimulus_index + 1}, layer {layer_index}"
            )
        activations.append(activation.astype(np.float32, copy=False))
    return activations


def cosine_distance_chunk(
    window_data: np.ndarray, device: torch.device, eps: float = 1e-12
) -> np.ndarray:
    """Compute unit-wise RDMs from [stimulus, sequence, unit] input."""
    tensor = torch.from_numpy(
        np.ascontiguousarray(window_data.transpose(2, 0, 1))
    ).to(device=device, dtype=torch.float32)
    norms = torch.linalg.vector_norm(tensor, ord=2, dim=-1)
    invalid = norms <= eps
    normalized = tensor / norms.clamp_min(eps).unsqueeze(-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2)).clamp(-1.0, 1.0)
    distance = (1.0 - similarity).clamp(0.0, 2.0)
    distance = 0.5 * (distance + distance.transpose(1, 2))
    distance.masked_fill_(invalid.unsqueeze(2) | invalid.unsqueeze(1), float("nan"))
    diagonal = torch.arange(N_STIMULI, device=device)
    distance[:, diagonal, diagonal] = torch.where(
        invalid,
        torch.full_like(norms, float("nan")),
        torch.zeros_like(norms),
    )
    return distance.cpu().numpy().astype(np.float32, copy=False)


def resample_layer(
    activations: list[np.ndarray], hidden_dim: int, device: torch.device
) -> np.ndarray:
    resampled = np.empty((N_STIMULI, GRID_POINTS, hidden_dim), dtype=np.float32)
    with torch.inference_mode():
        for stimulus_index, activation in enumerate(activations):
            if activation.shape[0] < 2:
                raise ValueError("Overlap analysis requires native sequence length >= 2")
            tensor = torch.from_numpy(activation.T).unsqueeze(0).to(device)
            interpolated = F.interpolate(
                tensor, size=GRID_POINTS, mode="linear", align_corners=True
            )
            resampled[stimulus_index] = interpolated.squeeze(0).transpose(0, 1).cpu().numpy()
    return resampled


def resample_native_layer_to_length(
    activations: list[np.ndarray], target_length: int, device: torch.device
) -> list[np.ndarray]:
    """Align variable-duration English stimuli to a common relative-time grid."""
    aligned: list[np.ndarray] = []
    with torch.inference_mode():
        for activation in activations:
            if activation.shape[0] == target_length:
                aligned.append(activation)
                continue
            tensor = torch.from_numpy(activation.T).unsqueeze(0).to(device)
            interpolated = F.interpolate(
                tensor, size=target_length, mode="linear", align_corners=True
            )
            aligned.append(
                interpolated.squeeze(0).transpose(0, 1).cpu().numpy().astype(
                    np.float32, copy=False
                )
            )
    return aligned


def matlab_class(value: str) -> np.bytes_:
    return np.bytes_(value)


def hdf5_reference_dtype():
    return h5py.ref_dtype if hasattr(h5py, "ref_dtype") else h5py.special_dtype(ref=h5py.Reference)


def write_matlab_header(path: Path) -> None:
    description = (
        "MATLAB 7.3 MAT-file, Platform: GLNXA64, "
        f"Created on: {time.ctime()}, HDF5 schema 1.00 ."
    ).encode("ascii", errors="replace")
    header = description[:116].ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM"
    with path.open("r+b") as stream:
        stream.seek(0)
        stream.write(header)


def compression_kwargs(name: str) -> dict:
    if name == "none":
        return {}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def metadata_payload(
    *,
    config: ModelConfig,
    bundle: dict,
    input_path: Path,
    analysis_id: str,
    window_count: int,
    window_definition: str,
    temporal_interpolation: bool,
    selection_rule: SelectionRule,
    excluded_layers: list[dict],
) -> dict:
    hidden_dims = np.asarray(bundle["hidden_dims"], dtype=np.int64).reshape(-1)
    source_indices = np.asarray(bundle["source_layer_indices"], dtype=np.int64).reshape(-1)
    return {
        "analysis": analysis_id,
        "model": config.model_name,
        "source_activation_file": str(input_path),
        "rdm_metric": "cosine_distance",
        "rdm_shape_per_layer_matlab": f"52 x 52 x {window_count} x units",
        "activation_layout": "sequence x hidden_dimension",
        "window_count": window_count,
        "window_definition": window_definition,
        "temporal_interpolation": temporal_interpolation,
        "selection_rule": selection_rule.describe(),
        "stimulus_count": N_STIMULI,
        "original_layer_count": int(bundle["original_layer_count"]),
        "included_layer_count": int(len(hidden_dims)),
        "excluded_layer_count": int(len(excluded_layers)),
        "source_layer_indices_zero_based": source_indices.tolist(),
        "excluded_layers": excluded_layers,
        "total_unit_count": int(hidden_dims.sum()),
        "uncompressed_rdm_bytes": int(hidden_dims.sum()) * window_count * N_STIMULI**2 * 4,
        "unit_order": "hidden dimension index within each layer, zero-based in Python",
        "zero_norm_policy": "RDM row/column set to NaN for zero-norm window vector",
        "stimulus_files": as_strings(bundle["stimulus_files"]),
        "layer_names": as_strings(bundle["layer_names"]),
        "sequence_types": as_strings(bundle["sequence_types"]),
        "hidden_dims": hidden_dims.tolist(),
        "native_valid_lengths": np.asarray(bundle["valid_lengths"], dtype=np.int64).tolist(),
    }


def create_or_open_output(
    *,
    output_path: Path,
    config: ModelConfig,
    bundle: dict,
    input_path: Path,
    analysis_id: str,
    window_count: int,
    window_definition: str,
    temporal_interpolation: bool,
    selection_rule: SelectionRule,
    excluded_layers: list[dict],
    overwrite: bool,
) -> h5py.File:
    if output_path.exists() and overwrite:
        output_path.unlink()
    if output_path.exists():
        h5 = h5py.File(output_path, "r+")
        if (
            str(h5.attrs.get("analysis", "")) != analysis_id
            or str(h5.attrs.get("model", "")) != config.model_name
            or int(h5.attrs.get("window_count", -1)) != window_count
        ):
            h5.close()
            raise ValueError("Existing output uses a different model or analysis definition")
        return h5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(output_path, "w", userblock_size=512, libver="latest")
    h5.attrs["analysis"] = analysis_id
    h5.attrs["model"] = config.model_name
    h5.attrs["window_count"] = window_count
    h5.attrs["temporal_interpolation"] = int(temporal_interpolation)
    h5.attrs["python_storage_order"] = "units, windows, stimuli, stimuli"
    h5.attrs["matlab_visible_order"] = "stimuli, stimuli, windows, units"
    h5.create_group("#refs#")
    rdm_cell = h5.create_dataset(
        "rdm_cell",
        shape=(len(bundle["layer_names"]), 1),
        dtype=hdf5_reference_dtype(),
    )
    rdm_cell.attrs["MATLAB_class"] = matlab_class("cell")
    metadata = metadata_payload(
        config=config,
        bundle=bundle,
        input_path=input_path,
        analysis_id=analysis_id,
        window_count=window_count,
        window_definition=window_definition,
        temporal_interpolation=temporal_interpolation,
        selection_rule=selection_rule,
        excluded_layers=excluded_layers,
    )
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    metadata_codes = np.asarray([ord(char) for char in metadata_json], dtype=np.uint16)
    metadata_dataset = h5.create_dataset(
        "metadata_json", data=metadata_codes.reshape(-1, 1)
    )
    metadata_dataset.attrs["MATLAB_class"] = matlab_class("char")
    metadata_dataset.attrs["MATLAB_int_decode"] = np.int64(2)
    h5.flush()
    return h5


def prepare_analysis(
    args: argparse.Namespace,
    config: ModelConfig,
    rule: SelectionRule,
) -> tuple[Path, Path, dict, list[dict], torch.device]:
    if args.unit_chunk <= 0:
        raise ValueError("--unit-chunk must be positive")
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    full_bundle = load_activation_bundle(input_path)
    bundle, excluded = select_bundle(full_bundle, rule)
    device = choose_device(args.device)
    return input_path, output_path, bundle, excluded, device


def run_nonoverlap(
    *,
    config: ModelConfig,
    window_count: int,
    rule: SelectionRule,
    analysis_id: str,
    minimum_points_per_window: int,
) -> None:
    description = (
        f"Construct {window_count} native non-overlapping unit-wise RDM windows "
        f"for {config.model_name}. Included layers: {rule.describe()}."
    )
    args = build_parser(description, config).parse_args()
    input_path, output_path, bundle, excluded, device = prepare_analysis(args, config, rule)
    lengths = np.asarray(bundle["valid_lengths"], dtype=np.int64)
    if np.any(np.ptp(lengths, axis=0) != 0):
        raise ValueError("Native non-overlap analysis requires equal lengths across stimuli")
    layer_names = as_strings(bundle["layer_names"])
    sequence_types = as_strings(bundle["sequence_types"])
    hidden_dims = np.asarray(bundle["hidden_dims"], dtype=np.int64).reshape(-1)
    source_indices = np.asarray(bundle["source_layer_indices"], dtype=np.int64).reshape(-1)
    total_units = int(hidden_dims.sum())
    window_definition = f"native boundaries floor(k*T/{window_count}), k=0..{window_count}"
    print(json.dumps({
        "analysis": analysis_id,
        "model": config.model_name,
        "input": str(input_path),
        "output": str(output_path),
        "device": str(device),
        "included_layers": len(layer_names),
        "excluded_layers": len(excluded),
        "total_units": total_units,
        "windows": window_count,
        "temporal_interpolation": False,
        "uncompressed_output_GiB": round(total_units * window_count * N_STIMULI**2 * 4 / 1024**3, 2),
        "compression": args.compression,
        "unit_chunk": args.unit_chunk,
    }, indent=2))
    is_new = not output_path.exists() or args.overwrite
    h5 = create_or_open_output(
        output_path=output_path,
        config=config,
        bundle=bundle,
        input_path=input_path,
        analysis_id=analysis_id,
        window_count=window_count,
        window_definition=window_definition,
        temporal_interpolation=False,
        selection_rule=rule,
        excluded_layers=excluded,
        overwrite=args.overwrite,
    )
    if is_new:
        h5.close()
        write_matlab_header(output_path)
        h5 = h5py.File(output_path, "r+")
    try:
        refs_group = h5["#refs#"]
        rdm_cell = h5["rdm_cell"]
        compression = compression_kwargs(args.compression)
        for layer_index, (layer_name, sequence_type, hidden_dim, source_index) in enumerate(
            zip(layer_names, sequence_types, hidden_dims, source_indices)
        ):
            dataset_name = f"layer_{layer_index:03d}"
            if dataset_name in refs_group:
                existing = refs_group[dataset_name]
                if int(existing.attrs.get("complete", 0)) == 1:
                    rdm_cell[layer_index, 0] = existing.ref
                    print(f"[{layer_index + 1:02d}/{len(layer_names)}] {layer_name}: complete; skip")
                    continue
                del refs_group[dataset_name]
            activations = validate_layer_activations(bundle["data_cell"], layer_index, int(hidden_dim))
            sequence_length = int(lengths[0, layer_index])
            edges = np.floor(
                np.arange(window_count + 1, dtype=np.float64) * sequence_length / window_count
            ).astype(np.int32)
            window_lengths = np.diff(edges)
            if np.any(window_lengths < minimum_points_per_window):
                raise ValueError(
                    f"Layer {layer_name} has window lengths {window_lengths.tolist()}, "
                    f"below required minimum {minimum_points_per_window}"
                )
            print(
                f"[{layer_index + 1:02d}/{len(layer_names)}] {layer_name}: "
                f"native T={sequence_length}, windows={window_lengths.tolist()}, units={hidden_dim}"
            )
            dataset = refs_group.create_dataset(
                dataset_name,
                shape=(int(hidden_dim), window_count, N_STIMULI, N_STIMULI),
                dtype=np.float32,
                chunks=(min(args.unit_chunk, int(hidden_dim)), 1, N_STIMULI, N_STIMULI),
                fillvalue=np.nan,
                **compression,
            )
            dataset.attrs["MATLAB_class"] = matlab_class("single")
            dataset.attrs["source_layer_index_zero_based"] = int(source_index)
            dataset.attrs["layer_name"] = layer_name
            dataset.attrs["sequence_type"] = sequence_type
            dataset.attrs["unit_count"] = int(hidden_dim)
            dataset.attrs["native_sequence_length"] = sequence_length
            dataset.attrs["window_edges_zero_based_stop_exclusive"] = edges
            dataset.attrs["window_lengths"] = window_lengths
            dataset.attrs["complete"] = 0
            rdm_cell[layer_index, 0] = dataset.ref
            h5.flush()
            for window_index in range(window_count):
                start, stop = int(edges[window_index]), int(edges[window_index + 1])
                for unit_start in range(0, int(hidden_dim), args.unit_chunk):
                    unit_stop = min(unit_start + args.unit_chunk, int(hidden_dim))
                    window_data = np.stack(
                        [activation[start:stop, unit_start:unit_stop] for activation in activations],
                        axis=0,
                    )
                    dataset[unit_start:unit_stop, window_index, :, :] = cosine_distance_chunk(
                        window_data, device
                    )
                print(f"    window {window_index + 1}/{window_count}: [{start}:{stop})")
                h5.flush()
            dataset.attrs.modify("complete", 1)
            h5.flush()
            del activations
            if device.type == "cuda":
                torch.cuda.empty_cache()
        h5.attrs["complete"] = 1
        h5.flush()
    finally:
        h5.close()
    print(f"Complete RDM MAT file: {output_path}")


def run_overlap16(
    *,
    config: ModelConfig,
    rule: SelectionRule,
    analysis_id: str = "supp_overlap16",
) -> None:
    description = (
        f"Construct 16 overlapping normalized-time unit-wise RDM windows for "
        f"{config.model_name}. Included layers: {rule.describe()}."
    )
    args = build_parser(description, config).parse_args()
    input_path, output_path, bundle, excluded, device = prepare_analysis(args, config, rule)
    layer_names = as_strings(bundle["layer_names"])
    sequence_types = as_strings(bundle["sequence_types"])
    hidden_dims = np.asarray(bundle["hidden_dims"], dtype=np.int64).reshape(-1)
    source_indices = np.asarray(bundle["source_layer_indices"], dtype=np.int64).reshape(-1)
    total_units = int(hidden_dims.sum())
    window_definition = "101-point normalized grid; width=0.25; step=0.05; starts=0.00..0.75"
    print(json.dumps({
        "analysis": analysis_id,
        "model": config.model_name,
        "input": str(input_path),
        "output": str(output_path),
        "device": str(device),
        "included_layers": len(layer_names),
        "excluded_layers": len(excluded),
        "total_units": total_units,
        "windows": OVERLAP_WINDOWS,
        "temporal_interpolation": True,
        "uncompressed_output_GiB": round(total_units * OVERLAP_WINDOWS * N_STIMULI**2 * 4 / 1024**3, 2),
        "compression": args.compression,
        "unit_chunk": args.unit_chunk,
    }, indent=2))
    is_new = not output_path.exists() or args.overwrite
    h5 = create_or_open_output(
        output_path=output_path,
        config=config,
        bundle=bundle,
        input_path=input_path,
        analysis_id=analysis_id,
        window_count=OVERLAP_WINDOWS,
        window_definition=window_definition,
        temporal_interpolation=True,
        selection_rule=rule,
        excluded_layers=excluded,
        overwrite=args.overwrite,
    )
    if is_new:
        h5.close()
        write_matlab_header(output_path)
        h5 = h5py.File(output_path, "r+")
    try:
        refs_group = h5["#refs#"]
        rdm_cell = h5["rdm_cell"]
        compression = compression_kwargs(args.compression)
        for layer_index, (layer_name, sequence_type, hidden_dim, source_index) in enumerate(
            zip(layer_names, sequence_types, hidden_dims, source_indices)
        ):
            dataset_name = f"layer_{layer_index:03d}"
            if dataset_name in refs_group:
                existing = refs_group[dataset_name]
                if int(existing.attrs.get("complete", 0)) == 1:
                    rdm_cell[layer_index, 0] = existing.ref
                    print(f"[{layer_index + 1:02d}/{len(layer_names)}] {layer_name}: complete; skip")
                    continue
                del refs_group[dataset_name]
            activations = validate_layer_activations(bundle["data_cell"], layer_index, int(hidden_dim))
            print(
                f"[{layer_index + 1:02d}/{len(layer_names)}] {layer_name}: "
                f"resampling to {GRID_POINTS}; units={hidden_dim}"
            )
            resampled = resample_layer(activations, int(hidden_dim), device)
            dataset = refs_group.create_dataset(
                dataset_name,
                shape=(int(hidden_dim), OVERLAP_WINDOWS, N_STIMULI, N_STIMULI),
                dtype=np.float32,
                chunks=(min(args.unit_chunk, int(hidden_dim)), 1, N_STIMULI, N_STIMULI),
                fillvalue=np.nan,
                **compression,
            )
            dataset.attrs["MATLAB_class"] = matlab_class("single")
            dataset.attrs["source_layer_index_zero_based"] = int(source_index)
            dataset.attrs["layer_name"] = layer_name
            dataset.attrs["sequence_type"] = sequence_type
            dataset.attrs["unit_count"] = int(hidden_dim)
            dataset.attrs["native_sequence_lengths"] = np.asarray(
                bundle["valid_lengths"][:, layer_index], dtype=np.int32
            )
            dataset.attrs["complete"] = 0
            rdm_cell[layer_index, 0] = dataset.ref
            h5.flush()
            for window_index, start in enumerate(OVERLAP_START_INDICES):
                stop = int(start + OVERLAP_WIDTH_POINTS)
                for unit_start in range(0, int(hidden_dim), args.unit_chunk):
                    unit_stop = min(unit_start + args.unit_chunk, int(hidden_dim))
                    window_data = resampled[:, int(start):stop, unit_start:unit_stop]
                    dataset[unit_start:unit_stop, window_index, :, :] = cosine_distance_chunk(
                        window_data, device
                    )
                print(
                    f"    window {window_index + 1:02d}/{OVERLAP_WINDOWS}: "
                    f"relative {OVERLAP_STARTS[window_index]:.2f}-{OVERLAP_ENDS[window_index]:.2f}"
                )
                h5.flush()
            dataset.attrs.modify("complete", 1)
            h5.flush()
            del resampled, activations
            if device.type == "cuda":
                torch.cuda.empty_cache()
        h5.attrs["complete"] = 1
        h5.flush()
    finally:
        h5.close()
    print(f"Complete RDM MAT file: {output_path}")


# ---------------------------------------------------------------------------
# Adaptive native non-overlap analysis (2, 3, or 4 windows by native T).
# ---------------------------------------------------------------------------

ADAPTIVE_ANALYSIS_ID = "adaptive_nonoverlap_unit_rdm_v1"


def adaptive_window_count(model_name: str, layer_name: str, sequence_length: int) -> int:
    if sequence_length > 7:
        return 4
    if sequence_length in (6, 7):
        return 3
    if sequence_length in (4, 5):
        return 2
    raise ValueError(
        f"Unsupported native sequence length: model={model_name}, "
        f"layer={layer_name}, T={sequence_length}. Expected T>7, T=6/7, or T=4/5."
    )


def adaptive_edges(
    model_name: str, layer_name: str, sequence_length: int, window_count: int
) -> np.ndarray:
    edges = np.round(
        np.linspace(0, sequence_length, window_count + 1)
    ).astype(np.int64)
    lengths = np.diff(edges)
    if edges[0] != 0 or edges[-1] != sequence_length:
        raise AssertionError(
            f"Window coverage error: model={model_name}, layer={layer_name}, "
            f"T={sequence_length}, edges={edges.tolist()}"
        )
    if not np.all(lengths > 0):
        raise AssertionError(
            f"Window edges are not strictly increasing: model={model_name}, "
            f"layer={layer_name}, edges={edges.tolist()}"
        )
    if not np.all(lengths >= 2):
        raise AssertionError(
            f"A window has fewer than two sequence points: model={model_name}, "
            f"layer={layer_name}, lengths={lengths.tolist()}"
        )
    if int(lengths.sum()) != sequence_length:
        raise AssertionError("Adaptive windows do not cover the complete sequence")
    return edges


def validate_formal_activation_path(path: Path) -> None:
    expected_file = path.parent.name + ".npy"
    if path.name != expected_file:
        raise ValueError(
            f"Activation filename/directory mismatch: expected {expected_file}, found {path.name}"
        )
    if not path.parent.name.endswith("_hidden_dimension_activations"):
        raise ValueError(f"Not a formally named activation directory: {path.parent}")
    forbidden = (
        "past",
        "test",
        "singlepass",
        "standardized_prompt",
        "pre_202",
    )
    lowered_parts = [part.lower() for part in path.parts]
    hits = [token for token in forbidden if any(token in part for part in lowered_parts)]
    if hits:
        raise ValueError(
            f"Refusing non-final activation path {path}; forbidden markers={hits}"
        )
    activation_root = path.parent.parent
    exact_matches = [
        candidate
        for candidate in activation_root.iterdir()
        if candidate.is_dir() and candidate.name == path.parent.name
    ]
    if len(exact_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one formal activation directory named {path.parent.name} "
            f"under {activation_root}, found {len(exact_matches)}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Formal activation NPY not found: {path}")


def adaptive_layer_plan(bundle: dict, config: ModelConfig) -> list[dict]:
    lengths = np.asarray(bundle["valid_lengths"], dtype=np.int64)
    layer_names = as_strings(bundle["layer_names"])
    sequence_types = as_strings(bundle["sequence_types"])
    hidden_dims = np.asarray(bundle["hidden_dims"], dtype=np.int64).reshape(-1)
    if lengths.shape != (N_STIMULI, len(layer_names)):
        raise ValueError(
            f"valid_lengths has shape {lengths.shape}; expected {(N_STIMULI, len(layer_names))}"
        )
    plan = []
    for layer_index, (layer_name, sequence_type, hidden_dim) in enumerate(
        zip(layer_names, sequence_types, hidden_dims)
    ):
        unique_lengths = np.unique(lengths[:, layer_index])
        variable_allowed = os.environ.get(
            "SLM_ALLOW_VARIABLE_SEQUENCE_LENGTHS", ""
        ).lower() in {"1", "true", "yes"}
        if unique_lengths.size != 1 and not variable_allowed:
            raise ValueError(
                f"Unequal native lengths across 52 stimuli: model={config.model_name}, "
                f"layer={layer_name}, lengths={unique_lengths.tolist()}"
            )
        # English TTS durations vary.  Use the shortest native sequence to
        # choose the shared 2/3/4-segment design, then align every stimulus to
        # that relative-time grid during construction.  The formal design is
        # only defined from T=4 onward, so exceptionally short projected
        # sequences (for example T=3 in MiniCPM-o 4.5) are linearly upsampled
        # to T=4 before the same published rule is applied.  Equal-length
        # Chinese inputs with T>=4 retain the original no-interpolation path.
        sequence_length = max(4, int(unique_lengths.min()))
        window_count = adaptive_window_count(
            config.model_name, layer_name, sequence_length
        )
        edges = adaptive_edges(
            config.model_name, layer_name, sequence_length, window_count
        )
        window_lengths = np.diff(edges)
        plan.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "sequence_type": sequence_type,
                "hidden_units": int(hidden_dim),
                "native_T": sequence_length,
                "native_T_min": int(unique_lengths.min()),
                "native_T_max": int(unique_lengths.max()),
                "window_count": window_count,
                "window_edges": edges.tolist(),
                "window_lengths": window_lengths.tolist(),
                "dataset_name": f"layer_{layer_index:03d}",
                "dataset_shape": (
                    int(hidden_dim), window_count, N_STIMULI, N_STIMULI
                ),
            }
        )
    return plan


def adaptive_summary_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_summary.csv")


def adaptive_estimates(plan: list[dict]) -> dict:
    rdm_values = sum(
        layer["hidden_units"] * layer["window_count"] * N_STIMULI**2
        for layer in plan
    )
    cosine_pair_products = sum(
        layer["hidden_units"] * layer["native_T"] * N_STIMULI**2
        for layer in plan
    )
    counts = {
        str(window_count): sum(
            layer["window_count"] == window_count for layer in plan
        )
        for window_count in (2, 3, 4)
    }
    return {
        "layer_count": len(plan),
        "window_count_layers": counts,
        "rdm_float32_values": int(rdm_values),
        "uncompressed_output_bytes": int(rdm_values * 4),
        "uncompressed_output_GiB": round(rdm_values * 4 / 1024**3, 3),
        "approximate_cosine_pair_products": int(cosine_pair_products),
        "approximate_cosine_pair_products_G": round(cosine_pair_products / 1e9, 3),
    }


def adaptive_metadata(
    *,
    config: ModelConfig,
    bundle: dict,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    plan: list[dict],
    complete: int,
) -> dict:
    input_stat = input_path.stat()
    return {
        "analysis": ADAPTIVE_ANALYSIS_ID,
        "model_name": config.model_name,
        "stimulus_files": as_strings(bundle["stimulus_files"]),
        "layer_names": [layer["layer_name"] for layer in plan],
        "sequence_types": [layer["sequence_type"] for layer in plan],
        "hidden_dims": [layer["hidden_units"] for layer in plan],
        "native_sequence_lengths": [layer["native_T"] for layer in plan],
        "window_counts": [layer["window_count"] for layer in plan],
        "window_edges": [layer["window_edges"] for layer in plan],
        "window_lengths": [layer["window_lengths"] for layer in plan],
        "rdm_metric": "cosine_distance",
        "rdm_storage_order": "units x windows x 52 stimuli x 52 stimuli",
        "matlab_cell_note": "rdm_cell contains one numeric array per layer; window dimension varies by layer",
        "window_rule": (
            "native T>7: 4 windows; T=6/7: 3 windows; T=4/5: 2 windows; "
            "edges=round(linspace(0,T,n_windows+1)); variable-duration English "
            "layers are linearly aligned to max(4, minimum native T) before segmentation"
        ),
        "activation_source_absolute_path": str(input_path),
        "activation_source_size_bytes": int(input_stat.st_size),
        "activation_source_mtime_ns": int(input_stat.st_mtime_ns),
        "output_absolute_path": str(output_path),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_inside_process": str(device),
        "complete": int(complete),
        "estimates": adaptive_estimates(plan),
    }


def write_adaptive_metadata(h5: h5py.File, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    codes = np.asarray([ord(char) for char in text], dtype=np.uint16).reshape(-1, 1)
    if "metadata_json" in h5:
        del h5["metadata_json"]
    dataset = h5.create_dataset("metadata_json", data=codes)
    dataset.attrs["MATLAB_class"] = matlab_class("char")
    dataset.attrs["MATLAB_int_decode"] = np.int64(2)


def write_adaptive_summary(
    summary_path: Path,
    config: ModelConfig,
    plan: list[dict],
    completion: list[int],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    fieldnames = [
        "model",
        "layer_index",
        "layer_name",
        "sequence_type",
        "hidden_units",
        "native_T",
        "window_count",
        "window_edges",
        "window_lengths",
        "output_dataset_shape",
        "complete",
    ]
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for layer, is_complete in zip(plan, completion):
            writer.writerow(
                {
                    "model": config.model_name,
                    "layer_index": layer["layer_index"],
                    "layer_name": layer["layer_name"],
                    "sequence_type": layer["sequence_type"],
                    "hidden_units": layer["hidden_units"],
                    "native_T": layer["native_T"],
                    "window_count": layer["window_count"],
                    "window_edges": json.dumps(layer["window_edges"]),
                    "window_lengths": json.dumps(layer["window_lengths"]),
                    "output_dataset_shape": " x ".join(
                        str(value) for value in layer["dataset_shape"]
                    ),
                    "complete": int(is_complete),
                }
            )
    temporary_path.replace(summary_path)


def adaptive_completion(h5: h5py.File, plan: list[dict]) -> list[int]:
    refs_group = h5["#refs#"]
    return [
        int(
            layer["dataset_name"] in refs_group
            and int(refs_group[layer["dataset_name"]].attrs.get("complete", 0)) == 1
        )
        for layer in plan
    ]


def validate_adaptive_structure(
    h5: h5py.File, plan: list[dict], *, sample_values: bool
) -> None:
    if h5["rdm_cell"].shape != (len(plan), 1):
        raise ValueError(f"Bad rdm_cell shape: {h5['rdm_cell'].shape}")
    refs_group = h5["#refs#"]
    for layer_index, layer in enumerate(plan):
        name = layer["dataset_name"]
        if name not in refs_group:
            raise ValueError(f"Missing output dataset for layer {layer_index}: {name}")
        dataset = refs_group[name]
        if dataset.shape != layer["dataset_shape"]:
            raise ValueError(
                f"Bad dataset shape for {name}: {dataset.shape}, expected {layer['dataset_shape']}"
            )
        if dataset.dtype != np.dtype(np.float32):
            raise ValueError(f"Bad dtype for {name}: {dataset.dtype}")
        if int(dataset.attrs.get("complete", 0)) != 1:
            raise ValueError(f"Layer dataset is incomplete: {name}")
        if int(dataset.attrs.get("native_sequence_length", -1)) != layer["native_T"]:
            raise ValueError(f"Native T metadata mismatch for {name}")
        if not np.array_equal(
            np.asarray(dataset.attrs["window_edges"], dtype=np.int64),
            np.asarray(layer["window_edges"], dtype=np.int64),
        ):
            raise ValueError(f"Window edge metadata mismatch for {name}")
        if sample_values:
            unit_indices = sorted({0, layer["hidden_units"] - 1})
            for unit_index in unit_indices:
                for window_index in range(layer["window_count"]):
                    matrix = np.asarray(
                        dataset[unit_index, window_index, :, :], dtype=np.float32
                    )
                    if not np.allclose(matrix, matrix.T, atol=1e-5, equal_nan=True):
                        raise ValueError(
                            f"RDM symmetry check failed: {name}, unit={unit_index}, "
                            f"window={window_index}"
                        )
                    diagonal = np.diag(matrix)
                    valid_diagonal = diagonal[np.isfinite(diagonal)]
                    if valid_diagonal.size and not np.allclose(
                        valid_diagonal, 0.0, atol=1e-6
                    ):
                        raise ValueError(
                            f"RDM diagonal check failed: {name}, unit={unit_index}, "
                            f"window={window_index}"
                        )


def build_adaptive_parser(config: ModelConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Construct adaptive native non-overlapping unit-wise cosine RDMs "
            f"for {config.model_name}."
        )
    )
    parser.add_argument("--input", default=config.default_input)
    parser.add_argument("--output", default=config.default_output)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--unit-chunk", type=int, default=128)
    parser.add_argument(
        "--compression", choices=("lzf", "gzip", "none"), default="lzf"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete matching output; a complete output is validated and kept.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a matching output and rebuild it from the first layer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input metadata and print the layer plan without writing output.",
    )
    return parser


def run_adaptive_nonoverlap(config: ModelConfig) -> None:
    args = build_adaptive_parser(config).parse_args()
    if args.unit_chunk <= 0:
        raise ValueError("--unit-chunk must be positive")
    configured_input = Path(args.input).expanduser()
    validate_formal_activation_path(configured_input)
    input_path = configured_input.resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = adaptive_summary_path(output_path)
    bundle = load_activation_bundle(input_path)
    activation_layout = str(bundle.get("activation_layout", ""))
    if activation_layout != "sequence_by_hidden_dimension":
        raise ValueError(
            f"Unsupported activation_layout={activation_layout!r}; expected "
            "'sequence_by_hidden_dimension'"
        )
    input_audit = validate_activation_input_bundle(bundle, input_path)
    print(
        "INPUT VALIDATION OK: "
        f"{config.model_name}; stimuli={input_audit['stimulus_count']}; "
        f"layers={input_audit['layer_count']}; cells={input_audit['cell_count']}; "
        f"dtype={input_audit['dtype']}"
    )
    plan = adaptive_layer_plan(bundle, config)
    device = choose_device(args.device)
    estimates = adaptive_estimates(plan)
    report = {
        "analysis": ADAPTIVE_ANALYSIS_ID,
        "model": config.model_name,
        "input": str(input_path),
        "output": str(output_path),
        "summary_csv": str(summary_path),
        "device": str(device),
        "unit_chunk": args.unit_chunk,
        "compression": args.compression,
        **estimates,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for layer in plan:
        print(
            f"[{layer['layer_index'] + 1:03d}/{len(plan):03d}] "
            f"{layer['layer_name']}: T={layer['native_T']}, "
            f"windows={layer['window_count']}, edges={layer['window_edges']}, "
            f"lengths={layer['window_lengths']}, units={layer['hidden_units']}"
        )
    if args.dry_run:
        print(f"DRY RUN OK: {config.model_name}; no output was written")
        return

    if output_path.exists() and args.overwrite:
        output_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    if output_path.exists():
        h5 = h5py.File(output_path, "r+")
        try:
            if str(h5.attrs.get("analysis", "")) != ADAPTIVE_ANALYSIS_ID:
                raise ValueError("Existing output belongs to a different analysis")
            if str(h5.attrs.get("model_name", "")) != config.model_name:
                raise ValueError("Existing output belongs to a different model")
            if int(h5.attrs.get("layer_count", -1)) != len(plan):
                raise ValueError("Existing output has a different layer count")
            input_stat = input_path.stat()
            stored_size = int(h5.attrs.get("activation_source_size_bytes", -1))
            stored_mtime_ns = int(h5.attrs.get("activation_source_mtime_ns", -1))
            if (
                stored_size != int(input_stat.st_size)
                or stored_mtime_ns != int(input_stat.st_mtime_ns)
            ):
                raise RuntimeError(
                    "Existing output was built from a different or older activation "
                    f"file: {output_path}. Use --overwrite once to rebuild it from "
                    "the current formal activation NPY."
                )
            if int(h5.attrs.get("complete", 0)) == 1:
                validate_adaptive_structure(h5, plan, sample_values=True)
                if not summary_path.is_file():
                    raise FileNotFoundError(
                        f"Complete MAT exists but summary CSV is missing: {summary_path}"
                    )
                print(f"Complete output already exists and passed validation: {output_path}")
                return
            if not args.resume:
                raise RuntimeError(
                    f"Incomplete output exists: {output_path}. Use --resume to continue "
                    "or --overwrite to rebuild."
                )
        except Exception:
            h5.close()
            raise
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        h5 = h5py.File(output_path, "w", userblock_size=512, libver="latest")
        h5.attrs["analysis"] = ADAPTIVE_ANALYSIS_ID
        h5.attrs["model_name"] = config.model_name
        h5.attrs["layer_count"] = len(plan)
        h5.attrs["rdm_metric"] = "cosine_distance"
        h5.attrs["complete"] = 0
        h5.attrs["cuda_visible_devices"] = "0"
        h5.attrs["device_inside_process"] = str(device)
        input_stat = input_path.stat()
        h5.attrs["activation_source_size_bytes"] = int(input_stat.st_size)
        h5.attrs["activation_source_mtime_ns"] = int(input_stat.st_mtime_ns)
        h5.attrs["python_storage_order"] = "units, windows, stimuli, stimuli"
        h5.attrs["matlab_visible_order"] = "stimuli, stimuli, windows, units"
        h5.create_group("#refs#")
        rdm_cell = h5.create_dataset(
            "rdm_cell", shape=(len(plan), 1), dtype=hdf5_reference_dtype()
        )
        rdm_cell.attrs["MATLAB_class"] = matlab_class("cell")
        write_adaptive_metadata(
            h5,
            adaptive_metadata(
                config=config,
                bundle=bundle,
                input_path=input_path,
                output_path=output_path,
                device=device,
                plan=plan,
                complete=0,
            ),
        )
        h5.flush()
        h5.close()
        write_matlab_header(output_path)
        h5 = h5py.File(output_path, "r+")

    try:
        refs_group = h5["#refs#"]
        rdm_cell = h5["rdm_cell"]
        compression = compression_kwargs(args.compression)
        write_adaptive_summary(
            summary_path, config, plan, adaptive_completion(h5, plan)
        )
        for layer in plan:
            layer_index = layer["layer_index"]
            dataset_name = layer["dataset_name"]
            if dataset_name in refs_group:
                existing = refs_group[dataset_name]
                if (
                    int(existing.attrs.get("complete", 0)) == 1
                    and existing.shape == layer["dataset_shape"]
                    and existing.dtype == np.dtype(np.float32)
                ):
                    rdm_cell[layer_index, 0] = existing.ref
                    print(
                        f"[{layer_index + 1:03d}/{len(plan):03d}] "
                        f"{layer['layer_name']}: complete; skip"
                    )
                    continue
                del refs_group[dataset_name]

            activations = validate_layer_activations(
                bundle["data_cell"], layer_index, layer["hidden_units"]
            )
            observed_lengths = np.asarray(
                [activation.shape[0] for activation in activations], dtype=np.int64
            )
            variable_allowed = os.environ.get(
                "SLM_ALLOW_VARIABLE_SEQUENCE_LENGTHS", ""
            ).lower() in {"1", "true", "yes"}
            if variable_allowed and not np.all(observed_lengths == layer["native_T"]):
                activations = resample_native_layer_to_length(
                    activations, layer["native_T"], device
                )
                observed_lengths = np.asarray(
                    [activation.shape[0] for activation in activations], dtype=np.int64
                )
            if not np.all(observed_lengths == layer["native_T"]):
                raise ValueError(
                    f"Activation cell lengths disagree with valid_lengths: "
                    f"model={config.model_name}, layer={layer['layer_name']}, "
                    f"observed={np.unique(observed_lengths).tolist()}, "
                    f"metadata_T={layer['native_T']}"
                )
            print(
                f"[{layer_index + 1:03d}/{len(plan):03d}] {layer['layer_name']}: "
                f"T={layer['native_T']}, windows={layer['window_lengths']}, "
                f"units={layer['hidden_units']}"
            )
            dataset = refs_group.create_dataset(
                dataset_name,
                shape=layer["dataset_shape"],
                dtype=np.float32,
                chunks=(
                    min(args.unit_chunk, layer["hidden_units"]),
                    1,
                    N_STIMULI,
                    N_STIMULI,
                ),
                fillvalue=np.nan,
                **compression,
            )
            dataset.attrs["MATLAB_class"] = matlab_class("single")
            dataset.attrs["layer_index_zero_based"] = layer_index
            dataset.attrs["layer_name"] = layer["layer_name"]
            dataset.attrs["sequence_type"] = layer["sequence_type"]
            dataset.attrs["hidden_units"] = layer["hidden_units"]
            dataset.attrs["native_sequence_length"] = layer["native_T"]
            dataset.attrs["native_sequence_length_min"] = layer["native_T_min"]
            dataset.attrs["native_sequence_length_max"] = layer["native_T_max"]
            dataset.attrs["window_count"] = layer["window_count"]
            dataset.attrs["window_edges"] = np.asarray(
                layer["window_edges"], dtype=np.int64
            )
            dataset.attrs["window_lengths"] = np.asarray(
                layer["window_lengths"], dtype=np.int64
            )
            dataset.attrs["complete"] = 0
            rdm_cell[layer_index, 0] = dataset.ref
            h5.flush()

            edges = layer["window_edges"]
            for window_index in range(layer["window_count"]):
                start, stop = edges[window_index], edges[window_index + 1]
                for unit_start in range(0, layer["hidden_units"], args.unit_chunk):
                    unit_stop = min(
                        unit_start + args.unit_chunk, layer["hidden_units"]
                    )
                    window_data = np.stack(
                        [
                            activation[start:stop, unit_start:unit_stop]
                            for activation in activations
                        ],
                        axis=0,
                    )
                    dataset[unit_start:unit_stop, window_index, :, :] = (
                        cosine_distance_chunk(window_data, device)
                    )
                print(
                    f"    window {window_index + 1}/{layer['window_count']}: "
                    f"native [{start}:{stop}), length={stop - start}"
                )
                h5.flush()

            dataset.attrs.modify("complete", 1)
            h5.flush()
            write_adaptive_summary(
                summary_path, config, plan, adaptive_completion(h5, plan)
            )
            del activations
            if device.type == "cuda":
                torch.cuda.empty_cache()

        validate_adaptive_structure(h5, plan, sample_values=True)
        write_adaptive_metadata(
            h5,
            adaptive_metadata(
                config=config,
                bundle=bundle,
                input_path=input_path,
                output_path=output_path,
                device=device,
                plan=plan,
                complete=1,
            ),
        )
        h5.flush()
        write_adaptive_summary(summary_path, config, plan, [1] * len(plan))
        # The file-level flag is deliberately the final mutation: if metadata
        # or CSV finalization fails, the MAT remains resumable (complete=0).
        h5.attrs.modify("complete", 1)
        h5.flush()
    finally:
        h5.close()
    print(f"Complete and validated MAT file: {output_path}")
    print(f"Layer summary CSV: {summary_path}")
