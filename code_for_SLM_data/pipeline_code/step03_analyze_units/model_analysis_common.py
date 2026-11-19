#!/usr/bin/env python3
"""Common I/O, styling, interpolation, and alignment helpers for Step 03."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from pipeline_code.step03_classify_units.model_unit_classification_core import MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP_ROOT = PROJECT_ROOT / "code_for_all_models_group"
DEFAULT_CLASSIFICATION_ROOT = GROUP_ROOT / "step03_classify_units_human_nominal_consecutive2"
DEFAULT_OUTPUT_ROOT = GROUP_ROOT / "step03_sequential_hierarchy_geometry"
DEFAULT_HUMAN_ROOT = GROUP_ROOT / "step00_human_reference"

CLASS_SPECS = (
    (1, "P-only", "Phonological"),
    (2, "S-only", "Semantic"),
    (3, "Shared", "Shared"),
)
CLASS_NAME = {-1: "Invalid", 0: "None", 1: "P-only", 2: "S-only", 3: "Shared"}
P_COLOR = np.asarray([186, 48, 49], dtype=float) / 256.0
S_COLOR = np.asarray([35, 100, 156], dtype=float) / 256.0
SHARED_COLOR = np.asarray([103, 38, 141], dtype=float) / 255.0
NONE_COLOR = np.asarray([185, 185, 185], dtype=float) / 255.0
CLASS_COLORS = {"P-only": P_COLOR, "S-only": S_COLOR, "Shared": SHARED_COLOR, "None": NONE_COLOR}
CANONICAL4 = np.asarray([0.125, 0.375, 0.625, 0.875], dtype=float)
HUMAN16 = (np.arange(16, dtype=float) + 0.5) / 16.0


def configure_style():
    requested_file = os.environ.get("SLM_ARIAL_FONT")
    if requested_file and Path(requested_file).is_file():
        font_manager.fontManager.addfont(requested_file)
        active_font = font_manager.FontProperties(fname=requested_file).get_name()
    else:
        try:
            font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
            active_font = "Arial"
        except ValueError:
            try:
                font_manager.findfont(font_manager.FontProperties(family="Liberation Sans"), fallback_to_default=False)
                active_font = "Liberation Sans"
            except ValueError:
                active_font = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": active_font, "font.size": 7.0,
        "axes.labelsize": 7.0, "axes.titlesize": 7.0,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.5,
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.transparent": False,
    })
    return active_font


def clean_axis(ax):
    ax.tick_params(width=0.5, colors="black", length=2.5)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.5)


def save_figure(fig, base: Path, dpi=600):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_result_paths(classification_root: Path, spec):
    model_dir = classification_root / spec.slug
    candidates = sorted(model_dir.glob("model_unit_classification_*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one classification NPZ in {model_dir}, found {len(candidates)}")
    layer_csv = model_dir / "layer_metadata.csv"
    metadata_json = model_dir / "analysis_metadata.json"
    if not layer_csv.is_file() or not metadata_json.is_file():
        raise FileNotFoundError(f"incomplete classification output: {model_dir}")
    return candidates[0], layer_csv, metadata_json


def load_model_result(classification_root: Path, spec):
    npz_path, layer_path, metadata_path = model_result_paths(classification_root, spec)
    with np.load(npz_path) as saved:
        data = {key: saved[key] for key in saved.files}
    layers = pd.read_csv(layer_path).sort_values("LayerOrderZeroBased").reset_index(drop=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"class_code", "P_z", "S_z", "eligible", "window_count",
                "layer_order_zero_based", "unit_index_within_layer_zero_based"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{npz_path} lacks required arrays: {sorted(missing)}")
    if len(data["class_code"]) != int(layers["Units"].sum()):
        raise ValueError(f"unit count disagrees with layer metadata: {spec.name}")
    return data, layers, metadata


def available_models(classification_root: Path, requested=None):
    requested = set(requested or [])
    chosen = [m for m in MODELS if not requested or m.slug in requested]
    missing = [m.name for m in chosen if not (classification_root / m.slug).is_dir()]
    if missing:
        raise FileNotFoundError("missing classification result directories: " + ", ".join(missing))
    return chosen


def layer_window_centers(layer_row):
    edges = np.asarray(json.loads(str(layer_row["WindowEdges"])), dtype=float)
    native_t = float(layer_row["NativeSequenceLength"])
    centers = (edges[:-1] + edges[1:]) / (2.0 * native_t)
    if not np.all(np.diff(centers) > 0):
        raise ValueError(f"non-increasing window centers: {centers}")
    return centers


def interpolate_rows(values, x_old, x_new):
    values = np.asarray(values, dtype=float)
    x_old = np.asarray(x_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    out = np.full((values.shape[0], len(x_new)), np.nan, dtype=np.float32)
    for i, row in enumerate(values):
        finite = np.isfinite(row) & np.isfinite(x_old)
        if finite.sum() == 1:
            out[i] = row[finite][0]
        elif finite.sum() >= 2:
            out[i] = np.interp(x_new, x_old[finite], row[finite]).astype(np.float32)
    return out


def unit_values_at_positions(data, layers, key, positions):
    values = np.asarray(data[key], dtype=float)
    out = np.full((len(values), len(positions)), np.nan, dtype=np.float32)
    layer_order = data["layer_order_zero_based"].astype(int)
    for _, layer in layers.iterrows():
        order = int(layer["LayerOrderZeroBased"])
        use = layer_order == order
        nwin = int(layer["WindowCount"])
        out[use] = interpolate_rows(values[use, :nwin], layer_window_centers(layer), positions)
    return out


def mean_sem(matrix):
    matrix = np.asarray(matrix, dtype=float)
    n = np.sum(np.isfinite(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0) if matrix.size else np.full(matrix.shape[1], np.nan)
    sd = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
    sem = np.divide(sd, np.sqrt(n), out=np.full_like(sd, np.nan), where=n > 0)
    return mean, sem, n


def safe_corr(x, y, kind="spearman"):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return math.nan
    result = spearmanr(x[valid], y[valid]) if kind == "spearman" else pearsonr(x[valid], y[valid])
    return float(result.statistic)


def load_human_sequential(human_root=DEFAULT_HUMAN_ROOT):
    table = pd.read_csv(Path(human_root) / "Sequential_participant_level_statistics.csv")
    output = {}
    for _, _, human_label in CLASS_SPECS:
        current = table[table["ContactClass"] == human_label].sort_values("WindowCenter_ms")
        if len(current) != 16:
            raise ValueError(f"human sequential reference {human_label} has {len(current)} rows, expected 16")
        p16 = current["P_mean_z"].to_numpy(float)
        s16 = current["S_mean_z"].to_numpy(float)
        output[human_label] = {
            "P16": p16, "S16": s16,
            "P4": p16.reshape(4, 4).mean(axis=1),
            "S4": s16.reshape(4, 4).mean(axis=1),
        }
    return output, table


def load_human_hierarchy(human_root=DEFAULT_HUMAN_ROOT):
    table = pd.read_csv(Path(human_root) / "Hierarchy_12ROI_distanceMedian_summary.csv")
    table = table.sort_values("HierarchyRank")
    return {
        "P-only": table["Pct_Ponly"].to_numpy(float),
        "S-only": table["Pct_Sonly"].to_numpy(float),
        "Shared": table["Pct_Shared"].to_numpy(float),
    }, table


def depth_bin(values, bins=12):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.full(bins, np.nan)
    groups = np.array_split(np.arange(len(values)), bins)
    out = np.asarray([np.nanmean(values[g]) if len(g) else np.nan for g in groups], dtype=float)
    good = np.flatnonzero(np.isfinite(out))
    if len(good) == 1:
        out[:] = out[good[0]]
    elif len(good) >= 2:
        out = np.interp(np.arange(bins), good, out[good])
    return out


def centered_moving_mean(values, width=3):
    return pd.Series(np.asarray(values, dtype=float)).rolling(width, center=True, min_periods=1).mean().to_numpy()


def module_group(value):
    text = str(value).lower()
    if any(token in text for token in ("projector", "adapter", "projection")):
        return "Projector"
    if any(token in text for token in ("decoder", "language", "llm", "text")):
        return "Language decoder"
    if any(token in text for token in ("audio", "speech", "encoder", "whisper")):
        return "Audio encoder"
    return "Other"


def bin_heatmap(matrix, max_rows=1200):
    matrix = np.asarray(matrix, dtype=float)
    if len(matrix) <= max_rows:
        return matrix
    edges = np.linspace(0, len(matrix), max_rows + 1, dtype=int)
    return np.vstack([np.nanmean(matrix[edges[i]:edges[i + 1]], axis=0) for i in range(max_rows)])


def human_diverging_colormap():
    return mpl.colors.LinearSegmentedColormap.from_list("human_PS", [S_COLOR, (1, 1, 1), P_COLOR], N=256)


def rdm_colormap():
    nodes = np.asarray([[50, 28, 90], [40, 100, 120], [45, 175, 120], [170, 215, 60], [252, 245, 65]]) / 255.0
    return mpl.colors.LinearSegmentedColormap.from_list("human_rdm", nodes, N=256)


def source_rdm_path(metadata, data_root=PROJECT_ROOT):
    absolute = Path(metadata["source_rdm_file"])
    if absolute.is_file():
        return absolute
    relative = metadata.get("source_rdm_relative_to_data_root")
    if relative and (Path(data_root) / relative).is_file():
        return Path(data_root) / relative
    raise FileNotFoundError(f"cannot resolve source RDM from metadata: {metadata}")


def factor_controlled_masks():
    rows, cols = np.tril_indices(52, k=-1)
    labels = np.r_[np.zeros(26, dtype=int), np.ones(26, dtype=int)]
    cross = labels[rows] != labels[cols]
    true_pair = np.zeros(len(rows), dtype=bool)
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    for i in range(26):
        true_pair[lookup[(i + 26, i)]] = True
    return {
        "rows": rows, "cols": cols,
        "P_same_S_diff": true_pair,
        "P_diff_S_diff": cross & ~true_pair,
        "P_diff_S_same": ~cross,
    }


def complete_rdm(rdm):
    rdm = np.asarray(rdm, dtype=float)
    if rdm.shape != (52, 52):
        raise ValueError(f"expected 52x52 RDM, got {rdm.shape}")
    out = np.nanmean(np.stack([rdm, rdm.T]), axis=0)
    np.fill_diagonal(out, 0.0)
    return out
