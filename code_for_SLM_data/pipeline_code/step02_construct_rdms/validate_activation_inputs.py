#!/usr/bin/env python3
"""Validate every formal activation NPY configured for RDM construction."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_STIMULI = 52
REQUIRED_FIELDS = {
    "data_cell",
    "stimulus_files",
    "layer_names",
    "sequence_types",
    "hidden_dims",
    "valid_lengths",
    "activation_layout",
}


def _strings(values: object) -> list[str]:
    return [str(value) for value in np.asarray(values, dtype=object).reshape(-1)]


def validate_activation_input_bundle(
    bundle: dict, source_path: Path, *, expected_stimuli: int = EXPECTED_STIMULI
) -> dict[str, object]:
    """Fully audit one in-memory activation bundle used by Step 02.

    This validates the actual activation matrices, not only their metadata.
    It intentionally does not require the Step 01 MAT, transcript, or summary
    files because none of those files is an input to RDM construction.
    """

    missing = REQUIRED_FIELDS.difference(bundle)
    if missing:
        raise KeyError(f"{source_path}: missing fields {sorted(missing)}")

    if str(bundle["activation_layout"]) != "sequence_by_hidden_dimension":
        raise ValueError(
            f"{source_path}: unsupported activation_layout="
            f"{bundle['activation_layout']!r}"
        )

    data_cell = np.asarray(bundle["data_cell"], dtype=object)
    stimulus_files = _strings(bundle["stimulus_files"])
    layer_names = _strings(bundle["layer_names"])
    sequence_types = _strings(bundle["sequence_types"])
    hidden_dims = np.asarray(bundle["hidden_dims"], dtype=np.int64).reshape(-1)
    valid_lengths = np.asarray(bundle["valid_lengths"], dtype=np.int64)
    expected_shape = (expected_stimuli, len(layer_names))

    if data_cell.shape != expected_shape:
        raise ValueError(
            f"{source_path}: data_cell shape {data_cell.shape}; "
            f"expected {expected_shape}"
        )
    if valid_lengths.shape != expected_shape:
        raise ValueError(
            f"{source_path}: valid_lengths shape {valid_lengths.shape}; "
            f"expected {expected_shape}"
        )
    if len(stimulus_files) != expected_stimuli:
        raise ValueError(
            f"{source_path}: found {len(stimulus_files)} stimulus names; "
            f"expected {expected_stimuli}"
        )
    if len(set(stimulus_files)) != expected_stimuli or any(
        not value for value in stimulus_files
    ):
        raise ValueError(f"{source_path}: stimulus_files must be non-empty and unique")
    if not layer_names or len(set(layer_names)) != len(layer_names) or any(
        not value for value in layer_names
    ):
        raise ValueError(f"{source_path}: layer_names must be non-empty and unique")
    if len(sequence_types) != len(layer_names):
        raise ValueError(
            f"{source_path}: sequence_types has {len(sequence_types)} values; "
            f"expected {len(layer_names)}"
        )
    if len(hidden_dims) != len(layer_names) or np.any(hidden_dims <= 0):
        raise ValueError(f"{source_path}: invalid hidden_dims")
    if np.any(valid_lengths <= 0):
        raise ValueError(f"{source_path}: valid_lengths must all be positive")

    for layer_index, (layer_name, hidden_dim) in enumerate(
        zip(layer_names, hidden_dims)
    ):
        unique_lengths = np.unique(valid_lengths[:, layer_index])
        if unique_lengths.size != 1 and not os.environ.get(
            "SLM_ALLOW_VARIABLE_SEQUENCE_LENGTHS", ""
        ).lower() in {"1", "true", "yes"}:
            raise ValueError(
                f"{source_path}: unequal T across {expected_stimuli} stimuli for "
                f"layer {layer_name!r}: {unique_lengths.tolist()}"
            )
        for stimulus_index in range(expected_stimuli):
            activation = np.asarray(data_cell[stimulus_index, layer_index])
            expected_cell_shape = (
                int(valid_lengths[stimulus_index, layer_index]),
                int(hidden_dim),
            )
            if activation.ndim != 2 or activation.shape != expected_cell_shape:
                raise ValueError(
                    f"{source_path}: bad cell stimulus={stimulus_index + 1}, "
                    f"layer={layer_index + 1} ({layer_name!r}); "
                    f"shape={activation.shape}, expected={expected_cell_shape}"
                )
            if not np.issubdtype(activation.dtype, np.floating):
                raise TypeError(
                    f"{source_path}: non-floating cell stimulus={stimulus_index + 1}, "
                    f"layer={layer_index + 1}; dtype={activation.dtype}"
                )
            if activation.dtype != np.float32:
                raise TypeError(
                    f"{source_path}: cell stimulus={stimulus_index + 1}, "
                    f"layer={layer_index + 1} has dtype={activation.dtype}; "
                    "expected float32"
                )
            if not np.isfinite(activation).all():
                raise ValueError(
                    f"{source_path}: NaN/Inf in stimulus={stimulus_index + 1}, "
                    f"layer={layer_index + 1} ({layer_name!r})"
                )

    return {
        "stimulus_count": expected_stimuli,
        "layer_count": len(layer_names),
        "cell_count": int(data_cell.size),
        "native_T_min": int(valid_lengths.min()),
        "native_T_max": int(valid_lengths.max()),
        "dtype": "float32",
    }


def parse_args(model_keys: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=model_keys,
        default=model_keys,
        help="One or more model keys; default: all models in alphabetical order.",
    )
    return parser.parse_args()


def main() -> None:
    # Import lazily so rdm_construction_common can reuse the validator without
    # creating an import cycle through the all-model runner.
    from code_for_all_models_group.step02_run_construct_adaptive_nonoverlap_unit_RDMs import (
        MODEL_CONFIGS,
    )
    from pipeline_code.step02_construct_rdms.rdm_construction_common import (
        ModelConfig,
        adaptive_layer_plan,
        load_activation_bundle,
        validate_formal_activation_path,
    )

    args = parse_args(list(MODEL_CONFIGS))
    for index, model_key in enumerate(args.models, start=1):
        spec = MODEL_CONFIGS[model_key]
        input_path = Path(spec["input"])
        validate_formal_activation_path(input_path)
        resolved = input_path.resolve()
        bundle = load_activation_bundle(resolved)
        audit = validate_activation_input_bundle(bundle, resolved)
        config = ModelConfig(
            model_name=str(spec["model_name"]),
            default_input=str(resolved),
            default_output=str(spec["output"]),
        )
        plan = adaptive_layer_plan(bundle, config)
        window_counts = {
            count: sum(layer["window_count"] == count for layer in plan)
            for count in (2, 3, 4)
        }
        print(
            f"PASS {index:02d}/{len(args.models):02d}\t{spec['model_name']}\t"
            f"stimuli={audit['stimulus_count']}\tlayers={audit['layer_count']}\t"
            f"cells={audit['cell_count']}\tT={audit['native_T_min']}-"
            f"{audit['native_T_max']}\twindows="
            f"2:{window_counts[2]},3:{window_counts[3]},4:{window_counts[4]}"
        )
    print(f"VALIDATION OK: {len(args.models)}/{len(args.models)} models")


if __name__ == "__main__":
    main()
