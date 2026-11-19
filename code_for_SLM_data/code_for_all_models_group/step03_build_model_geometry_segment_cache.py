#!/usr/bin/env python3
"""Build compact Segment 1-4 caches for the local MATLAB geometry plot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

DATA_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from pipeline_code.step03_analyze_units.model_analysis_common import (
    CLASS_SPECS,
    DEFAULT_CLASSIFICATION_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MODELS,
    available_models,
    complete_rdm,
    factor_controlled_masks,
    load_model_result,
    source_rdm_path,
)

CACHE_VERSION = 4
MAX_SEGMENTS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, default=DEFAULT_CLASSIFICATION_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "results_03_geometry" / "segment_rdms_mat",
    )
    parser.add_argument("--models", nargs="*", choices=[m.slug for m in MODELS])
    parser.add_argument("--rdm-chunk", type=int, default=32)
    return parser.parse_args()


def layer_mean(dataset, indices, n_segments, chunk):
    total = np.zeros((n_segments, 52, 52), dtype=np.float64)
    for start in range(0, len(indices), chunk):
        block = np.asarray(dataset[indices[start : start + chunk], :n_segments], dtype=np.float32)
        total += np.nansum(block, axis=0)
    result = total / len(indices)
    return np.stack([complete_rdm(r) for r in result]).astype(np.float32)


def build_model(spec, classification_root, data_root, output_root, chunk):
    data, layers, metadata = load_model_result(classification_root, spec)
    source = source_rdm_path(metadata, data_root)
    classes = data["class_code"].astype(int)
    unit_layer = data["layer_order_zero_based"].astype(int)
    unit_within = data["unit_index_within_layer_zero_based"].astype(int)

    sums = np.zeros((3, MAX_SEGMENTS, 52, 52), dtype=np.float64)
    layer_counts = np.zeros((3, MAX_SEGMENTS), dtype=np.int32)
    unit_counts = np.zeros((3, MAX_SEGMENTS), dtype=np.int64)
    layer_rdms = [[[] for _ in range(MAX_SEGMENTS)] for _ in range(3)]

    with h5py.File(source, "r") as handle:
        for _, row in layers.sort_values("LayerOrderZeroBased").iterrows():
            order = int(row.LayerOrderZeroBased)
            n_segments = int(row.WindowCount)
            if n_segments not in (2, 3, 4):
                raise ValueError(f"unsupported segment count {n_segments}")
            segment_map = {
                2: (0, 0, 1, 1),
                3: (0, 1, 2, 2),
                4: (0, 1, 2, 3),
            }[n_segments]
            dataset = handle[str(row.SourceDatasetPath)]
            for slot, (code, _, _) in enumerate(CLASS_SPECS):
                global_indices = np.flatnonzero((unit_layer == order) & (classes == code))
                if not len(global_indices):
                    continue
                selected = unit_within[global_indices]
                if not np.all(np.diff(selected) >= 0):
                    raise ValueError("unit indices must be sorted within each layer")
                means = layer_mean(dataset, selected, n_segments, chunk)
                for segment, source_segment in enumerate(segment_map):
                    rdm = complete_rdm(means[source_segment]).astype(np.float32)
                    sums[slot, segment] += rdm
                    layer_counts[slot, segment] += 1
                    unit_counts[slot, segment] += len(selected)
                    layer_rdms[slot][segment].append(rdm)

    group = np.full((3, MAX_SEGMENTS, 52, 52), np.nan, dtype=np.float32)
    max_layers = max([len(x) for row in layer_rdms for x in row] + [1])
    factors = np.full((3, max_layers, MAX_SEGMENTS, 4), np.nan, dtype=np.float32)
    masks = factor_controlled_masks()
    rr, cc = masks["rows"], masks["cols"]

    for slot in range(3):
        for segment in range(MAX_SEGMENTS):
            if not layer_counts[slot, segment]:
                continue
            group[slot, segment] = (
                sums[slot, segment] / layer_counts[slot, segment]
            ).astype(np.float32)
            off_diagonal = group[slot, segment][~np.eye(52, dtype=bool)]
            lo, hi = np.nanmin(off_diagonal), np.nanmax(off_diagonal)
            for layer_index, rdm in enumerate(layer_rdms[slot][segment]):
                normalized = (rdm - lo) / (hi - lo) if hi > lo else np.zeros((52, 52))
                values = normalized[rr, cc]
                both_different = np.nanmean(values[masks["P_diff_S_diff"]])
                factors[slot, layer_index, segment] = [
                    np.nanmean(values[masks["P_same_S_diff"]]),
                    both_different,
                    np.nanmean(values[masks["P_diff_S_same"]]),
                    both_different,
                ]

    payload = {
        "cache_version": np.asarray([[CACHE_VERSION]], dtype=np.int32),
        "class_codes": np.asarray([1, 2, 3], dtype=np.int32),
        "class_names": np.asarray(["P-only", "S-only", "Shared"], dtype=object),
        "segment_numbers": np.arange(1, MAX_SEGMENTS + 1, dtype=np.int32),
        "segment_labels": np.asarray(
            [f"Segment {i}" for i in range(1, MAX_SEGMENTS + 1)], dtype=object
        ),
        "segment_group_rdms": group,
        "segment_layer_factor_values": factors,
        "segment_layer_counts": layer_counts,
        "segment_unit_counts": unit_counts,
        "aggregation_method": (
            "units averaged within layer, then all layers averaged equally for each expanded segment"
        ),
        "segment_expansion_rule": (
            "2 -> [1 1 2 2]; 3 -> [1 2 3 3]; 4 -> [1 2 3 4]"
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{spec.slug}_class_segment_mean_rdms.mat"
    savemat(destination, payload, do_compression=True, long_field_names=True)
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "model": spec.name,
                "source_rdm": str(source),
                "segment_numbers": [1, 2, 3, 4],
                "segment_expansion_rule": payload["segment_expansion_rule"],
                "aggregation": payload["aggregation_method"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def main():
    args = parse_args()
    classification_root = args.classification_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    models = available_models(classification_root, args.models)
    for index, spec in enumerate(models, 1):
        print(f"[{index}/{len(models)}] {spec.name}", flush=True)
        path = build_model(
            spec, classification_root, data_root, output_root, args.rdm_chunk
        )
        print(f"  saved {path}", flush=True)


if __name__ == "__main__":
    main()
