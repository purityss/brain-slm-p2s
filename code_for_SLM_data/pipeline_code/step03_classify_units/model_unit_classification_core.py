#!/usr/bin/env python3
"""Shared implementation for the step03 adaptive-window unit classifiers.

The public entry points are the three small ``step03_*``/``supp_step03_*``
scripts next to this file.  Keeping all statistics here guarantees that the
primary and sensitivity analyses differ only in their final decision rule.

The P and S contrasts, restricted permutations, one-sided window p-values,
and positive-effect rule reproduce the operational definitions in the human
contact script ``main_classify_factorControlled_with_downstream_exports.m``.
No temporal interpolation is performed during classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat


N_STIMULI = 52
N_PAIRS = 1326
MAX_WINDOWS = 4
HUMAN_SEED = 20260918
CLASS_NAMES = {-1: "Invalid", 0: "None", 1: "P-only", 2: "S-only", 3: "Shared"}
METHODS = (
    "human_nominal_consecutive2",
    "persistence_fdr",
    "anywindow_max_fdr",
    "acoustic_residual_human_nominal_consecutive2",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    directory: str
    prefix: str


MODELS = (
    ModelSpec("Baichuan-Audio-Base", "baichuan_audio_base", "code_for_Baichuan-Audio-Base", "Baichuan_Audio_Base"),
    ModelSpec("Baichuan-Audio-Instruct", "baichuan_audio_instruct", "code_for_Baichuan-Audio-Instruct", "Baichuan_Audio_Instruct"),
    ModelSpec("Covo-Audio-Chat", "covo_audio_chat", "code_for_Covo-Audio-Chat", "Covo_Audio_Chat"),
    ModelSpec("FireRedAudio", "firered_audio", "code_for_FireRedAudio", "FireRedAudio"),
    ModelSpec("Freeze-Omni", "freeze_omni", "code_for_Freeze-Omni", "Freeze_Omni"),
    ModelSpec("GLM-4-Voice", "glm_4_voice", "code_for_GLM-4-Voice", "GLM_4_Voice"),
    ModelSpec("Kimi-Audio-7B-Instruct", "kimi_audio_7b_instruct", "code_for_Kimi-Audio-7B-Instruct", "Kimi_Audio_7B_Instruct"),
    ModelSpec("MiniCPM-o 2.6", "minicpm_o_2_6", "code_for_MiniCPM-o 2.6", "MiniCPM_o_2_6"),
    ModelSpec("MiniCPM-o 4.5", "minicpm_o_4_5", "code_for_MiniCPM-o 4.5", "MiniCPM_o_4_5"),
    ModelSpec("Qwen-Audio", "qwen_audio", "code_for_Qwen-Audio", "Qwen_Audio"),
    ModelSpec("Qwen-Audio-Chat", "qwen_audio_chat", "code_for_Qwen-Audio-Chat", "Qwen_Audio_Chat"),
    ModelSpec("Qwen2-Audio", "qwen2_audio", "code_for_Qwen2-Audio", "Qwen2_Audio"),
    ModelSpec("Qwen2-Audio-Instruct", "qwen2_audio_instruct", "code_for_Qwen2-Audio-Instruct", "Qwen2_Audio_Instruct"),
    ModelSpec("Qwen2.5-Omni", "qwen2_5_omni", "code_for_Qwen2.5-Omni", "Qwen2_5_Omni"),
)


def parse_args(method: str, default_output_name: str, description: str):
    data_root = Path(__file__).resolve().parents[2]
    group_dir = data_root / "code_for_all_models_group"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--output-root", type=Path, default=group_dir / default_output_name)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=HUMAN_SEED)
    parser.add_argument("--unit-chunk", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument("--models", nargs="*", choices=[m.slug for m in MODELS])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-statistics-root", type=Path,
        help="Reuse per-window statistics from a completed step03 output root. "
             "Recommended for the two supplementary methods.",
    )
    parser.add_argument(
        "--materialize-selected-rdms", action="store_true",
        help="Also copy selected P-only/S-only/Shared RDMs into selected_unit_rdms.mat. "
             "This can require tens of GB; exact source references are always saved.",
    )
    parser.add_argument("--rdm-copy-chunk", type=int, default=8)
    parser.add_argument(
        "--acoustic-rdm", type=Path,
        default=Path(__file__).resolve().parent / "reference_data" / "acoustic_RDM_logMel_first250ms.mat",
        help="52x52 first-250-ms log-Mel acoustic RDM MAT file (acoustic-residual method only).",
    )
    parser.add_argument(
        "--acoustic-metric", choices=("correlation", "cosine", "euclideanRMS"),
        default="correlation", help="Acoustic MAT variable to regress out.",
    )
    args = parser.parse_args()
    args.method = method
    return args


def decode(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def load_acoustic_matrix(path, variable):
    """Load a 52x52 nuisance RDM from classic MAT or MATLAB-v7.3 HDF5."""
    try:
        content = loadmat(path)
        if variable not in content:
            raise KeyError(f"{path} lacks {variable}")
        return np.asarray(content[variable], dtype=np.float64)
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if variable not in handle:
                raise KeyError(f"{path} lacks {variable}")
            # MATLAB v7.3 reverses array dimensions. The RDM is symmetric,
            # but transpose explicitly to preserve general MATLAB ordering.
            return np.asarray(handle[variable], dtype=np.float64).T


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def rdm_path(root: Path, spec: ModelSpec):
    return root / spec.directory / "02_construct_RDMs" / "adaptive_nonoverlap" / f"{spec.prefix}_adaptive_nonoverlap_unit_RDMs.mat"


def layer_datasets(handle):
    datasets = [value for key, value in handle["#refs#"].items() if key.startswith("layer_")]
    datasets.sort(key=lambda ds: int(ds.attrs["layer_index_zero_based"]))
    return datasets


def pair_layout():
    rows, cols = np.tril_indices(N_STIMULI, k=-1)
    labels = np.r_[np.ones(26, dtype=np.int8), np.full(26, 2, dtype=np.int8)]
    cross = labels[rows] != labels[cols]
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    true_pair = np.zeros(N_PAIRS, dtype=bool)
    for i in range(26):
        true_pair[lookup[(i + 26, i)]] = True
    return rows, cols, cross, true_pair, lookup


def build_weights(permutations, seed):
    """Build the same restricted P/S null definitions used by the human code."""
    rows, cols, cross, true_pair, lookup = pair_layout()
    rng = np.random.default_rng(seed)
    count = permutations + 1

    # P = mean(P-different/S-different) - mean(P-same/S-different).
    p_weights = np.zeros((N_PAIRS, count), dtype=np.float32)
    p_weights[cross] = np.float32(1.0 / 650.0)
    matching = np.empty((count, 26), dtype=np.int16)
    matching[0] = np.arange(26)
    for b in range(1, count):
        matching[b] = rng.permutation(26)
    for b in range(count):
        indices = [lookup[(26 + int(matching[b, i]), i)] for i in range(26)]
        p_weights[indices, b] = np.float32(-1.0 / 26.0)

    # S = mean(P-different/S-different) - mean(P-different/S-same).
    flips = np.zeros((count, 26), dtype=bool)
    flips[1:] = rng.random((permutations, 26)) < 0.5
    perm_labels = np.empty((count, N_STIMULI), dtype=np.int8)
    perm_labels[:, :26] = 1 + flips
    perm_labels[:, 26:] = 2 - flips
    within = perm_labels[:, rows] == perm_labels[:, cols]
    if np.any(within[:, true_pair]):
        raise AssertionError("true phonological pairs must remain cross-semantic")
    s_weights = np.where(within, -1.0 / 650.0, 1.0 / 650.0).astype(np.float32)
    s_weights[:, true_pair] = 0.0
    return rows, cols, p_weights, np.ascontiguousarray(s_weights.T)


def bh(values):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if not valid.size:
        return out.astype(np.float32)
    order = valid[np.argsort(values[valid], kind="mergesort")]
    adjusted = values[order] * valid.size / np.arange(1, valid.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.clip(adjusted, 0, 1)
    return out.astype(np.float32)


def dimension_statistics(flat, weights, n_selected, n_windows, permutations):
    pool = (flat @ weights).reshape(n_selected, n_windows, permutations + 1)
    observed = pool[:, :, 0]
    null = pool[:, :, 1:]
    null_mean = torch.mean(null, dim=2)
    null_sd = torch.std(null, dim=2, unbiased=True)
    estimable = torch.isfinite(null_sd) & (null_sd > torch.finfo(null_sd.dtype).eps)
    z_all = (pool - null_mean[:, :, None]) / null_sd[:, :, None]
    z_all[~estimable] = float("nan")
    p_window = (1.0 + torch.sum(null >= observed[:, :, None], dim=2, dtype=torch.int64)) / float(permutations + 1)
    p_window[~estimable] = float("nan")

    observed_z = z_all[:, :, 0]
    positive = observed > 0
    candidate = torch.where(positive, observed_z, torch.full_like(observed_z, -torch.inf))
    any_observed, any_index = torch.max(candidate, dim=1)
    any_null = torch.nan_to_num(z_all[:, :, 1:], nan=-torch.inf).amax(dim=1)
    p_any = (1.0 + torch.sum(any_null >= any_observed[:, None], dim=1, dtype=torch.int64)) / float(permutations + 1)
    p_any[~torch.isfinite(any_observed)] = 1.0

    pair_z = torch.minimum(observed_z[:, :-1], observed_z[:, 1:])
    pair_positive = positive[:, :-1] & positive[:, 1:]
    pair_candidate = torch.where(pair_positive, pair_z, torch.full_like(pair_z, -torch.inf))
    persist_observed, persist_index = torch.max(pair_candidate, dim=1)
    null_z = torch.nan_to_num(z_all[:, :, 1:], nan=-torch.inf)
    persist_null = torch.minimum(null_z[:, :-1], null_z[:, 1:]).amax(dim=1)
    p_persist = (1.0 + torch.sum(persist_null >= persist_observed[:, None], dim=1, dtype=torch.int64)) / float(permutations + 1)
    p_persist[~torch.isfinite(persist_observed)] = 1.0

    def cpu(value, dtype=np.float32):
        return value.detach().cpu().numpy().astype(dtype, copy=False)

    return {
        "effect": cpu(observed), "z": cpu(observed_z), "p_window": cpu(p_window),
        "null_mean": cpu(null_mean), "null_sd": cpu(null_sd),
        "p_any": cpu(p_any), "p_persist": cpu(p_persist),
        "any_index": cpu(any_index, np.int8), "persist_index": cpu(persist_index, np.int8),
        "estimable": cpu(estimable, bool),
    }


def evaluate_layer(dataset, rows, cols, p_weights, s_weights, args, device, acoustic_z=None):
    if dataset.ndim != 4 or tuple(dataset.shape[2:]) != (52, 52):
        raise ValueError(f"unexpected RDM shape {dataset.shape} at {dataset.name}")
    n_units, n_windows = int(dataset.shape[0]), int(dataset.shape[1])
    if n_windows not in (2, 3, 4):
        raise ValueError(f"adaptive analysis requires 2/3/4 windows, got {n_windows}")

    two_d_metrics = ("effect", "z", "p_window", "null_mean", "null_sd")
    one_d_metrics = ("p_any", "p_persist", "any_index", "persist_index")
    out = {
        f"{dim}_{metric}": np.full((n_units, n_windows), np.nan, np.float32)
        for dim in ("P", "S") for metric in two_d_metrics
    }
    for dim in ("P", "S"):
        for metric in one_d_metrics:
            dtype = np.int8 if metric.endswith("index") else np.float32
            fill = -1 if metric.endswith("index") else np.nan
            out[f"{dim}_{metric}"] = np.full(n_units, fill, dtype=dtype)
    out["eligible"] = np.zeros(n_units, dtype=bool)

    for start in range(0, n_units, args.unit_chunk):
        stop = min(n_units, start + args.unit_chunk)
        rdm = np.asarray(dataset[start:stop], dtype=np.float32)
        lower = rdm[:, :, rows, cols]
        complete = np.isfinite(lower).all(axis=(1, 2))
        if not np.any(complete):
            continue
        selected = lower[complete]
        flat = torch.from_numpy(np.ascontiguousarray(selected.reshape(-1, N_PAIRS))).to(device)
        if acoustic_z is not None:
            # Human acoustic-control definition: y = b0 + bA*A + residual.
            # acoustic_z is centered, so intercept removal and acoustic
            # projection can be applied row-wise without forming a large
            # projection matrix. P and S are not partialled from one another.
            centered = flat - torch.mean(flat, dim=1, keepdim=True)
            denominator = torch.dot(acoustic_z, acoustic_z)
            beta = (centered @ acoustic_z) / denominator
            flat = centered - beta[:, None] * acoustic_z[None, :]
        results = {
            "P": dimension_statistics(flat, p_weights, selected.shape[0], n_windows, args.permutations),
            "S": dimension_statistics(flat, s_weights, selected.shape[0], n_windows, args.permutations),
        }
        complete_indices = np.flatnonzero(complete) + start
        # The consecutive-2 rule only needs one adjacent estimable pair per
        # dimension.  Requiring every native window to be estimable wrongly
        # invalidates a whole unit when (for example) a shared pre-audio
        # window has an exactly degenerate semantic permutation null.
        p_estimable = results["P"]["estimable"]
        s_estimable = results["S"]["estimable"]
        p_has_adjacent_pair = np.any(p_estimable[:, :-1] & p_estimable[:, 1:], axis=1)
        s_has_adjacent_pair = np.any(s_estimable[:, :-1] & s_estimable[:, 1:], axis=1)
        estimable = p_has_adjacent_pair & s_has_adjacent_pair
        out["eligible"][complete_indices[estimable]] = True
        for dim in ("P", "S"):
            for metric in two_d_metrics + one_d_metrics:
                out[f"{dim}_{metric}"][complete_indices] = results[dim][metric]
        del flat, rdm, lower, selected, results

    excluded = ~out["eligible"]
    for key, value in out.items():
        if key == "eligible":
            continue
        value[excluded] = -1 if key.endswith("index") else np.nan
    return out


def pad(array, width=MAX_WINDOWS, fill=np.nan):
    result = np.full((array.shape[0], width), fill, dtype=array.dtype)
    result[:, :array.shape[1]] = array
    return result


def run_information(hit):
    """Return persistent flag, longest run, and first qualifying run bounds."""
    n_units, n_windows = hit.shape
    persistent = np.zeros(n_units, dtype=bool)
    max_run = np.zeros(n_units, dtype=np.int8)
    first_start = np.full(n_units, -1, dtype=np.int8)
    first_end = np.full(n_units, -1, dtype=np.int8)
    for w in range(n_windows):
        for i in np.flatnonzero(hit[:, w]):
            run = 1
            while w - run >= 0 and hit[i, w - run]:
                run += 1
            if run > max_run[i]:
                max_run[i] = run
            if run >= 2 and first_start[i] < 0:
                persistent[i] = True
                first_start[i] = w - run + 1
                first_end[i] = w
    return persistent, max_run, first_start, first_end


def assign_classes(p_present, s_present, eligible):
    result = p_present.astype(np.int8) + 2 * s_present.astype(np.int8)
    result[~eligible] = -1
    return result


def add_all_classifications(data, alpha):
    eligible = data["eligible"].astype(bool)
    valid = np.arange(MAX_WINDOWS)[None, :] < data["window_count"][:, None]

    p_hit = eligible[:, None] & valid & (data["P_effect"] > 0) & (data["P_p_window"] <= alpha)
    s_hit = eligible[:, None] & valid & (data["S_effect"] > 0) & (data["S_p_window"] <= alpha)
    data["P_nominal_significant_window"] = p_hit
    data["S_nominal_significant_window"] = s_hit
    for dim, hit in (("P", p_hit), ("S", s_hit)):
        persistent, max_run, first_start, first_end = run_information(hit)
        data[f"{dim}_nominal_consecutive2"] = persistent
        data[f"{dim}_nominal_max_run"] = max_run
        data[f"{dim}_nominal_first_run_start"] = first_start
        data[f"{dim}_nominal_first_run_end"] = first_end
    data["class_human_nominal_consecutive2"] = assign_classes(
        data["P_nominal_consecutive2"], data["S_nominal_consecutive2"], eligible)
    data["class_acoustic_residual_human_nominal_consecutive2"] = data[
        "class_human_nominal_consecutive2"].copy()

    split = len(eligible)
    pooled = np.r_[data["P_p_persist"], data["S_p_persist"]]
    q = bh(pooled)
    data["P_q_persist"] = q[:split]
    data["S_q_persist"] = q[split:]
    data["class_persistence_fdr"] = assign_classes(q[:split] <= alpha, q[split:] <= alpha, eligible)

    pooled = np.r_[data["P_p_any"], data["S_p_any"]]
    q = bh(pooled)
    data["P_q_any"] = q[:split]
    data["S_q_any"] = q[split:]
    data["class_anywindow_max_fdr"] = assign_classes(q[:split] <= alpha, q[split:] <= alpha, eligible)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(spec, data, method):
    classes = data["class_code"]
    eligible = data["eligible"].astype(bool)
    eligible_units = int(np.sum(eligible))
    total_units = int(len(eligible))
    ineligible_units = total_units - eligible_units
    rows = []
    for code in range(4):
        count = int(np.sum(classes == code))
        rows.append({
            "Model": spec.name, "ModelSlug": spec.slug, "Method": method,
            "ClassCode": code, "Class": CLASS_NAMES[code], "Count": count,
            "EligibleUnits": eligible_units, "TotalUnits": total_units,
            "IneligibleUnits": ineligible_units,
            "PercentTotal": 100.0 * count / total_units if total_units else math.nan,
            "PercentEligible": 100.0 * count / eligible_units if eligible_units else math.nan,
        })
    return rows


def save_npz(path, data):
    with path.open("wb") as stream:
        np.savez_compressed(stream, **data)


def h5_write_dataset(group, name, value):
    value = np.asarray(value)
    kwargs = {}
    if value.ndim and value.size > 32:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    group.create_dataset(name, data=value, **kwargs)


def save_mat_hdf5(path, data, layer_rows, metadata):
    """Write HDF5-backed .mat content, readable in MATLAB with h5read."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temp, "w") as handle:
        handle.attrs["format"] = "HDF5 MATLAB exchange file (use h5read/h5info)"
        handle.attrs["complete"] = np.int8(0)
        for key, value in metadata.items():
            if value is None:
                continue
            handle.attrs[key] = str(value)
        stats = handle.create_group("unit_statistics")
        for key, value in data.items():
            h5_write_dataset(stats, key, value)

        layer_group = handle.create_group("layer_metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        fields = list(layer_rows[0])
        for field in fields:
            values = [row[field] for row in layer_rows]
            if field in {"LayerOrderZeroBased", "SourceLayerIndexZeroBased", "Units", "WindowCount", "NativeSequenceLength"}:
                layer_group.create_dataset(field, data=np.asarray(values, dtype=np.int64))
            else:
                layer_group.create_dataset(field, data=np.asarray(values, dtype=string_dtype), dtype=string_dtype)

        selected = handle.create_group("selected_units")
        classes = data["class_code"]
        for code, label in ((1, "P_only"), (2, "S_only"), (3, "Shared")):
            subgroup = selected.create_group(label)
            idx = np.flatnonzero(classes == code)
            h5_write_dataset(subgroup, "global_unit_index_zero_based", idx.astype(np.int64))
            h5_write_dataset(subgroup, "global_unit_index_one_based", (idx + 1).astype(np.int64))
            h5_write_dataset(subgroup, "layer_order_zero_based", data["layer_order_zero_based"][idx])
            h5_write_dataset(subgroup, "layer_order_one_based", data["layer_order_one_based"][idx])
            h5_write_dataset(subgroup, "source_layer_index_zero_based", data["source_layer_index_zero_based"][idx])
            h5_write_dataset(subgroup, "unit_index_within_layer_zero_based", data["unit_index_within_layer_zero_based"][idx])
            h5_write_dataset(subgroup, "unit_index_within_layer_one_based", data["unit_index_within_layer_one_based"][idx])
        handle.attrs["complete"] = np.int8(1)
    os.replace(temp, path)


def selected_manifest_rows(spec, data, layer_rows, source_rdm):
    layers = {int(row["LayerOrderZeroBased"]): row for row in layer_rows}
    classes = data["class_code"]
    relative_rdm = source_rdm_relative_path(spec)
    for global_index in np.flatnonzero(classes > 0):
        order = int(data["layer_order_zero_based"][global_index])
        layer = layers[order]
        n_windows = int(data["window_count"][global_index])
        yield {
            "Model": spec.name, "ModelSlug": spec.slug,
            "ClassCode": int(classes[global_index]), "Class": CLASS_NAMES[int(classes[global_index])],
            "GlobalUnitIndexZeroBased": int(global_index), "GlobalUnitIndexOneBased": int(global_index + 1),
            "LayerOrderZeroBased": order, "LayerOrderOneBased": order + 1,
            "SourceLayerIndexZeroBased": int(data["source_layer_index_zero_based"][global_index]),
            "LayerName": layer["LayerName"], "SequenceType": layer["SequenceType"],
            "UnitIndexWithinLayerZeroBased": int(data["unit_index_within_layer_zero_based"][global_index]),
            "UnitIndexWithinLayerOneBased": int(data["unit_index_within_layer_one_based"][global_index]),
            "WindowCount": n_windows, "NativeSequenceLength": layer["NativeSequenceLength"],
            "WindowEdges": layer["WindowEdges"], "SourceDatasetPath": layer["SourceDatasetPath"],
            "SourceRDMFile": str(source_rdm), "SourceRDMRelativePath": str(relative_rdm),
            "PStrengthZ": json.dumps(data["P_z"][global_index, :n_windows].astype(float).tolist()),
            "SStrengthZ": json.dumps(data["S_z"][global_index, :n_windows].astype(float).tolist()),
            "PEffect": json.dumps(data["P_effect"][global_index, :n_windows].astype(float).tolist()),
            "SEffect": json.dumps(data["S_effect"][global_index, :n_windows].astype(float).tolist()),
        }


def source_rdm_relative_path(spec):
    return Path(spec.directory) / "02_construct_RDMs" / "adaptive_nonoverlap" / f"{spec.prefix}_adaptive_nonoverlap_unit_RDMs.mat"


def load_referenced_unit_rdm(source_rdm_file, source_dataset_path, unit_index_within_layer_zero_based):
    """Load one unit's native-window RDMs from a saved Step 03 reference.

    ``source_rdm_file`` and ``source_dataset_path`` come from
    ``analysis_metadata.json``/``selected_units_manifest.csv`` and
    ``layer_metadata.csv``.  The returned shape is ``windows x 52 x 52``.
    """
    with h5py.File(Path(source_rdm_file), "r") as handle:
        return np.asarray(
            handle[source_dataset_path][int(unit_index_within_layer_zero_based)],
            dtype=np.float32,
        )


def save_source_reference(model_dir, source_rdm, spec):
    """Remove obsolete duplicate source-reference files.

    The project-relative RDM path is stored in ``analysis_metadata.json``.
    The former symlink and two one-line text files duplicated that reference
    and were not consumed by downstream analysis.
    """
    del source_rdm, spec
    for name in (
        "source_adaptive_unit_rdms.mat",
        "source_adaptive_unit_rdms.path.txt",
        "source_adaptive_unit_rdms.relative_path.txt",
    ):
        legacy = model_dir / name
        if legacy.is_symlink() or legacy.is_file():
            legacy.unlink()


def materialize_selected_rdms(path, source_rdm, data, layer_rows, chunk_units):
    temp = path.with_suffix(path.suffix + ".tmp")
    classes = data["class_code"]
    with h5py.File(source_rdm, "r") as source, h5py.File(temp, "w") as target:
        target.attrs["complete"] = np.int8(0)
        target.attrs["source_rdm_file"] = source_rdm.name
        source_layers = layer_datasets(source)
        root = target.create_group("layers")
        for row in layer_rows:
            order = int(row["LayerOrderZeroBased"])
            indices = np.flatnonzero((data["layer_order_zero_based"] == order) & (classes > 0))
            if not indices.size:
                continue
            source_unit_indices = data["unit_index_within_layer_zero_based"][indices].astype(np.int64)
            source_ds = source_layers[order]
            n_windows = int(row["WindowCount"])
            group = root.create_group(f"layer_{order:04d}")
            group.attrs["layer_name"] = str(row["LayerName"])
            group.attrs["source_dataset_path"] = str(row["SourceDatasetPath"])
            shape = (len(indices), n_windows, N_STIMULI, N_STIMULI)
            output = group.create_dataset(
                "rdm", shape=shape, dtype=np.float32,
                chunks=(min(chunk_units, len(indices)), 1, N_STIMULI, N_STIMULI),
                compression="gzip", compression_opts=4, shuffle=True,
            )
            for start in range(0, len(indices), chunk_units):
                stop = min(len(indices), start + chunk_units)
                # HDF5 fancy indices must be increasing; model unit indices are.
                output[start:stop] = np.asarray(source_ds[source_unit_indices[start:stop]], dtype=np.float32)
            h5_write_dataset(group, "global_unit_index_zero_based", indices.astype(np.int64))
            h5_write_dataset(group, "unit_index_within_layer_zero_based", source_unit_indices)
            h5_write_dataset(group, "class_code", classes[indices])
            h5_write_dataset(group, "P_z", data["P_z"][indices, :n_windows])
            h5_write_dataset(group, "S_z", data["S_z"][indices, :n_windows])
        target.attrs["complete"] = np.int8(1)
    os.replace(temp, path)


def collect_statistics(spec, source_rdm, rows, cols, p_weights, s_weights, args, device, acoustic_z=None):
    arrays = {key: [] for key in (
        "P_effect", "P_z", "P_p_window", "P_null_mean", "P_null_sd",
        "S_effect", "S_z", "S_p_window", "S_null_mean", "S_null_sd",
        "P_p_any", "P_p_persist", "S_p_any", "S_p_persist",
        "P_any_index", "P_persist_index", "S_any_index", "S_persist_index", "eligible",
    )}
    unit_layer, source_layer, unit_index, window_count = [], [], [], []
    layer_rows = []
    with h5py.File(source_rdm, "r") as handle:
        if int(handle.attrs.get("complete", 0)) != 1:
            raise RuntimeError(f"incomplete HDF5: {source_rdm}")
        datasets = layer_datasets(handle)
        for order, dataset in enumerate(datasets):
            layer_index = int(dataset.attrs["layer_index_zero_based"])
            name = decode(dataset.attrs.get("layer_name", dataset.name))
            sequence_type = decode(dataset.attrs.get("sequence_type", "unknown"))
            native_t = int(dataset.attrs["native_sequence_length"])
            edges = np.asarray(dataset.attrs["window_edges"], dtype=int).ravel()
            n_units, n_windows = int(dataset.shape[0]), int(dataset.shape[1])
            print(f"[{spec.name}] layer {order + 1}/{len(datasets)} source={layer_index} units={n_units} windows={n_windows} T={native_t}", flush=True)
            result = evaluate_layer(dataset, rows, cols, p_weights, s_weights, args, device, acoustic_z)
            for key in arrays:
                value = result[key]
                arrays[key].append(pad(value) if value.ndim == 2 else value)
            unit_layer.append(np.full(n_units, order, np.int32))
            source_layer.append(np.full(n_units, layer_index, np.int32))
            unit_index.append(np.arange(n_units, dtype=np.int32))
            window_count.append(np.full(n_units, n_windows, np.int8))
            layer_rows.append({
                "Model": spec.name, "ModelSlug": spec.slug, "LayerOrderZeroBased": order,
                "SourceLayerIndexZeroBased": layer_index, "LayerName": name,
                "SequenceType": sequence_type, "Units": n_units, "WindowCount": n_windows,
                "NativeSequenceLength": native_t, "WindowEdges": json.dumps(edges.tolist()),
                "SourceDatasetPath": dataset.name,
            })
    data = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    data["global_unit_index_zero_based"] = np.arange(len(data["eligible"]), dtype=np.int64)
    data["global_unit_index_one_based"] = data["global_unit_index_zero_based"] + 1
    data["layer_order_zero_based"] = np.concatenate(unit_layer)
    data["layer_order_one_based"] = data["layer_order_zero_based"] + 1
    data["source_layer_index_zero_based"] = np.concatenate(source_layer)
    data["unit_index_within_layer_zero_based"] = np.concatenate(unit_index)
    data["unit_index_within_layer_one_based"] = data["unit_index_within_layer_zero_based"] + 1
    data["window_count"] = np.concatenate(window_count)
    add_all_classifications(data, args.alpha)
    return data, layer_rows


def load_reused_statistics(spec, args):
    if args.reuse_statistics_root is None:
        return None
    reuse_dir = args.reuse_statistics_root.expanduser().resolve() / spec.slug
    npz_candidates = sorted(reuse_dir.glob("model_unit_classification_*.npz"))
    metadata_path = reuse_dir / "analysis_metadata.json"
    layer_path = reuse_dir / "layer_metadata.csv"
    if len(npz_candidates) != 1 or not metadata_path.is_file() or not layer_path.is_file():
        raise FileNotFoundError(f"incomplete reusable statistics for {spec.name}: {reuse_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, expected in (("permutations", args.permutations), ("seed", args.seed), ("alpha", args.alpha)):
        if float(metadata[key]) != float(expected):
            raise ValueError(f"reuse mismatch for {spec.name}: {key}={metadata[key]} (requested {expected})")
    with np.load(npz_candidates[0]) as saved:
        data = {key: saved[key] for key in saved.files}
    required = {
        "class_human_nominal_consecutive2",
        "class_persistence_fdr",
        "class_anywindow_max_fdr",
    }
    if not required.issubset(data):
        raise ValueError(f"reusable NPZ lacks method arrays: {required - set(data)}")
    print(f"[{spec.name}] reusing statistics from {npz_candidates[0]}", flush=True)
    return data, read_csv(layer_path), metadata["source_rdm_file"]


def process_model(spec, source_rdm, output_root, rows, cols, p_weights, s_weights, args, device):
    model_dir = output_root / spec.slug
    model_dir.mkdir(parents=True, exist_ok=True)
    npz_path = model_dir / f"model_unit_classification_{args.method}.npz"
    mat_path = model_dir / f"model_unit_classification_{args.method}.mat"
    layer_path = model_dir / "layer_metadata.csv"
    metadata_path = model_dir / "analysis_metadata.json"
    if all(path.exists() for path in (npz_path, mat_path, layer_path, metadata_path)) and not args.overwrite:
        print(f"[{spec.name}] complete output exists; use --overwrite to replace", flush=True)
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_source = Path(saved_metadata.get("source_rdm_file", source_rdm))
        if not saved_source.is_absolute():
            saved_source = args.data_root / saved_source
        save_source_reference(model_dir, saved_source, spec)
        with np.load(npz_path) as saved:
            data = {key: saved[key] for key in saved.files}
        return summarize(spec, data, args.method)

    reused = load_reused_statistics(spec, args)
    if reused is None:
        data, layer_rows = collect_statistics(
            spec, source_rdm, rows, cols, p_weights, s_weights, args, device,
            getattr(args, "acoustic_z", None),
        )
    else:
        data, layer_rows, reused_source = reused
        source_rdm = Path(reused_source)
        if not source_rdm.is_absolute():
            source_rdm = args.data_root / source_rdm
    data["class_code"] = data[f"class_{args.method}"].astype(np.int8, copy=True)
    data["P_selected"] = np.isin(data["class_code"], (1, 3))
    data["S_selected"] = np.isin(data["class_code"], (2, 3))
    data["Shared_selected"] = data["class_code"] == 3
    for code, prefix in ((1, "P_only"), (2, "S_only"), (3, "Shared")):
        idx = np.flatnonzero(data["class_code"] == code).astype(np.int64)
        data[f"{prefix}_global_unit_index_zero_based"] = idx
        data[f"{prefix}_global_unit_index_one_based"] = idx + 1
        data[f"{prefix}_layer_order_zero_based"] = data["layer_order_zero_based"][idx]
        data[f"{prefix}_source_layer_index_zero_based"] = data["source_layer_index_zero_based"][idx]
        data[f"{prefix}_unit_index_within_layer_zero_based"] = data["unit_index_within_layer_zero_based"][idx]
        data[f"{prefix}_unit_index_within_layer_one_based"] = data["unit_index_within_layer_one_based"][idx]

    metadata = {
        "model": spec.name, "model_slug": spec.slug, "method": args.method,
        "source_rdm_file": str(source_rdm_relative_path(spec)), "permutations": args.permutations,
        "source_rdm_relative_to_data_root": str(source_rdm_relative_path(spec)),
        "seed": args.seed, "alpha": args.alpha, "device": str(device),
        "python_version": sys.version,
        "classification_uses_native_windows_only": True,
        "interpolation_used_in_classification": False,
        "eligibility": (
            "all native RDM lower-triangle cells finite; P and S each have at least one adjacent "
            "pair of native windows with finite, nonzero restricted-null SD"
        ),
        "human_rule": "positive effect and nominal one-sided restricted-permutation p<=alpha in at least two adjacent native windows",
        "persistence_fdr_rule": "unit statistic max_j min(z_j,z_j+1); restricted-permutation p; BH over P+S eligible unit p-values within model",
        "anywindow_max_fdr_rule": "unit statistic max_window(z); restricted max-stat p; BH over P+S eligible unit p-values within model",
        "acoustic_control": (
            "per unit/window: regress 1326 lower-triangle distances on intercept + standardized acoustic RDM; "
            "apply original P/S restricted permutations to residual; P and S are not partialled from one another"
            if args.method == "acoustic_residual_human_nominal_consecutive2" else None
        ),
        "acoustic_rdm_file": args.acoustic_rdm.name if args.method == "acoustic_residual_human_nominal_consecutive2" else None,
        "acoustic_metric": args.acoustic_metric if args.method == "acoustic_residual_human_nominal_consecutive2" else None,
        "matlab_file_note": "HDF5-backed .mat exchange file; use MATLAB h5info/h5read",
        "rdm_storage": "materialized selected-unit copy" if args.materialize_selected_rdms else "lightweight source path + exact dataset/layer/unit references; no RDM file or symlink in results",
        "reused_statistics_root": args.reuse_statistics_root.name if args.reuse_statistics_root else None,
    }
    save_npz(npz_path, data)
    write_csv(layer_path, layer_rows)
    save_mat_hdf5(mat_path, data, layer_rows, metadata)
    summary = summarize(spec, data, args.method)
    write_csv(model_dir / "classification_summary.csv", summary)
    selected_rows = list(selected_manifest_rows(spec, data, layer_rows, source_rdm))
    if selected_rows:
        write_csv(model_dir / "selected_units_manifest.csv", selected_rows)
    else:
        (model_dir / "selected_units_manifest.csv").write_text(
            "Model,ModelSlug,ClassCode,Class\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    save_source_reference(model_dir, source_rdm, spec)
    if args.materialize_selected_rdms:
        materialize_selected_rdms(model_dir / "selected_unit_rdms.mat", source_rdm, data, layer_rows, args.rdm_copy_chunk)
    return summary


def run_cli(method: str, default_output_name: str, description: str):
    if method not in METHODS:
        raise ValueError(method)
    args = parse_args(method, default_output_name, description)
    if args.permutations < 999:
        raise ValueError("at least 999 permutations are required")
    if not (0 < args.alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if args.unit_chunk < 1 or args.rdm_copy_chunk < 1:
        raise ValueError("chunk sizes must be positive")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    args.acoustic_z = None
    if method == "acoustic_residual_human_nominal_consecutive2":
        if args.reuse_statistics_root is not None:
            raise ValueError("acoustic-residual statistics cannot reuse non-residualized Step 03 statistics")
        args.acoustic_rdm = args.acoustic_rdm.expanduser().resolve()
        if not args.acoustic_rdm.is_file():
            raise FileNotFoundError(args.acoustic_rdm)
        variable = {
            "correlation": "acoustic_RDM_correlation",
            "cosine": "acoustic_RDM_cosine",
            "euclideanRMS": "acoustic_RDM_euclideanRMS",
        }[args.acoustic_metric]
        matrix = load_acoustic_matrix(args.acoustic_rdm, variable)
        if matrix.shape != (N_STIMULI, N_STIMULI) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid acoustic RDM {variable}: shape={matrix.shape}")
        acoustic_vec = matrix[np.tril_indices(N_STIMULI, k=-1)]
        acoustic_z = (acoustic_vec - acoustic_vec.mean()) / acoustic_vec.std(ddof=1)
        args.acoustic_z = torch.from_numpy(acoustic_z.astype(np.float32)).to(device)
    selected = [m for m in MODELS if not args.models or m.slug in args.models]
    print(json.dumps({
        "method": method, "models": [m.name for m in selected], "device": str(device),
        "permutations": args.permutations, "alpha": args.alpha, "seed": args.seed,
        "output_root": str(args.output_root),
    }, indent=2, ensure_ascii=False), flush=True)

    # Supplementary jobs can reuse the main NPZ and skip all GPU work.
    if args.reuse_statistics_root is None:
        pair_rows, pair_cols, p_np, s_np = build_weights(args.permutations, args.seed)
        p_weights = torch.from_numpy(p_np).to(device)
        s_weights = torch.from_numpy(s_np).to(device)
    else:
        pair_rows = pair_cols = p_weights = s_weights = None

    summaries, failures = [], []
    started = time.time()
    for spec in selected:
        try:
            source = rdm_path(args.data_root, spec)
            if args.reuse_statistics_root is None and not source.is_file():
                raise FileNotFoundError(source)
            summaries.extend(process_model(
                spec, source, args.output_root, pair_rows, pair_cols,
                p_weights, s_weights, args, device,
            ))
        except Exception as exc:
            failures.append({"Model": spec.name, "ModelSlug": spec.slug, "Error": repr(exc)})
            print(f"ERROR {spec.name}: {exc!r}", file=sys.stderr, flush=True)
    # Rebuild the root summary from every completed model directory. This is
    # important for incremental additions (for example --models freeze_omni):
    # the new run must not replace the existing all-model table with one model.
    completed_summaries = []
    completed_models = []
    for spec in MODELS:
        summary_path = args.output_root / spec.slug / "classification_summary.csv"
        if summary_path.is_file():
            completed_summaries.extend(read_csv(summary_path))
            completed_models.append(spec.name)
    if completed_summaries:
        write_csv(
            args.output_root / "all_models_classification_summary.csv",
            completed_summaries,
        )
    run_metadata = {
        "method": method, "models": completed_models, "models_requested": [m.name for m in selected], "permutations": args.permutations,
        "alpha": args.alpha, "seed": args.seed, "device": str(device),
        "python_version": sys.version,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "failures": failures,
    }
    (args.output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        write_csv(args.output_root / "classification_failures.csv", failures)
        raise RuntimeError(f"{len(failures)} model(s) failed")
    print(f"completed in {run_metadata['elapsed_minutes']:.1f} minutes", flush=True)
