#!/usr/bin/env python3
"""Unified Step06 activation-zero lesion runner for all 14 retained SLMs.

Model-specific layer layouts and decoder audio-position definitions live in
``code_for_<MODEL>/06_lesion_experiment/model_spec.py``.  This file is the
only Step06 executable kept at the root of ``code_for_all_models_group``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_ROOT = PROJECT_ROOT / "code_for_all_models_group"
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_CODES = {"p_only": 1, "s_only": 2, "shared": 3}
MODEL_REGISTRY = {
    "baichuan_audio_base": ("code_for_Baichuan-Audio-Base", "py310_SLM"),
    "baichuan_audio_instruct": ("code_for_Baichuan-Audio-Instruct", "py310_SLM"),
    "covo_audio_chat": ("code_for_Covo-Audio-Chat", "py311_covo"),
    "firered_audio": ("code_for_FireRedAudio", "py310_fireredaudio"),
    "freeze_omni": ("code_for_Freeze-Omni", "py310_SLM"),
    "glm_4_voice": ("code_for_GLM-4-Voice", "py310_SLM"),
    "kimi_audio_7b_instruct": ("code_for_Kimi-Audio-7B-Instruct", "py310_kimi"),
    "minicpm_o_2_6": ("code_for_MiniCPM-o 2.6", "py310_SLM"),
    "minicpm_o_4_5": ("code_for_MiniCPM-o 4.5", "py310_SLM"),
    "qwen_audio": ("code_for_Qwen-Audio", "py310_SLM"),
    "qwen_audio_chat": ("code_for_Qwen-Audio-Chat", "py310_SLM"),
    "qwen2_audio": ("code_for_Qwen2-Audio", "py310_SLM"),
    "qwen2_audio_instruct": ("code_for_Qwen2-Audio-Instruct", "py310_SLM"),
    "qwen2_5_omni": ("code_for_Qwen2.5-Omni", "py310_SLM"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_REGISTRY))
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help=argparse.SUPPRESS)
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-root", type=Path, default=Path(os.environ.get("SLM_CONDA_ROOT", Path.home() / "anaconda3" / "envs")))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dose-step", type=int, default=500)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CLASS_CODES), default=list(CLASS_CODES))
    parser.add_argument(
        "--manifest-root", type=Path,
        default=GROUP_ROOT / "step04_manifests/fleurs_zh",
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=GROUP_ROOT / "step04_model_performance/fleurs_zh",
    )
    parser.add_argument(
        "--classification-root", type=Path,
        default=GROUP_ROOT / "step03_classify_units_human_nominal_consecutive2",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=GROUP_ROOT / "step06_lesion_results",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.dose_step != 500:
        parser.error("The formal lesion schedule requires --dose-step 500")
    return args


def load_model_spec(model_key: str):
    folder, _environment = MODEL_REGISTRY[model_key]
    path = PROJECT_ROOT / folder / "06_lesion_experiment/model_spec.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model lesion specification: {path}")
    module_name = f"step06_model_spec_{model_key}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MODEL_KEY != model_key:
        raise RuntimeError(f"{path}: MODEL_KEY={module.MODEL_KEY!r}, expected {model_key!r}")
    return module


def dispatch(args):
    if args.list:
        for key, (folder, environment) in MODEL_REGISTRY.items():
            print(f"{key}\t{environment}\t{folder}")
        return 0
    models = args.models or list(MODEL_REGISTRY)
    script = Path(__file__).resolve()
    for model_key in models:
        _folder, default_environment = MODEL_REGISTRY[model_key]
        environment = os.environ.get(f"STEP06_ENV_{model_key.upper()}", default_environment)
        python = args.conda_root / environment / "bin/python"
        if not python.is_file() and model_key == "covo_audio_chat" and environment == "py311_covo":
            environment = "py310_covo"
            python = args.conda_root / environment / "bin/python"
        if not python.is_file():
            raise FileNotFoundError(f"{model_key}: missing Python environment {python}")
        command = [
            str(python), str(script), "--internal-worker", "--model", model_key,
            "--gpu", str(args.gpu), "--max-new-tokens", str(args.max_new_tokens),
            "--dose-step", str(args.dose_step),
            "--manifest-root", str(args.manifest_root),
            "--baseline-root", str(args.baseline_root),
            "--classification-root", str(args.classification_root),
            "--output-root", str(args.output_root),
            "--conditions", *args.conditions,
        ]
        if args.model_path:
            command += ["--model-path", str(args.model_path)]
        for flag in ("overwrite", "retry_failures", "fail_fast", "plan_only"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        print("RUN", subprocess.list2cmdline(command), flush=True)
        child_environment = os.environ.copy()
        if model_key == "kimi_audio_7b_instruct":
            system_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
            if Path(system_libstdcpp).is_file():
                existing_preload = child_environment.get("LD_PRELOAD", "")
                child_environment["LD_PRELOAD"] = (
                    system_libstdcpp
                    if not existing_preload
                    else system_libstdcpp + os.pathsep + existing_preload
                )
        subprocess.run(
            command, check=True, cwd=PROJECT_ROOT, env=child_environment
        )
    return 0


def first_tensor(output):
    import torch
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def repack(output, changed):
    if isinstance(output, tuple):
        return (changed,) + output[1:]
    if isinstance(output, list):
        return [changed] + output[1:]
    return changed


class ActivationZeroController:
    """Zero selected coordinates; decoder edits are restricted to audio positions."""

    def __init__(self, layout, decoder_layers):
        self.layout = layout
        self.decoder_layers = set(map(int, decoder_layers))
        self.audio_mask = None
        self.audio_tail = None
        self.active = False
        self.audit = {}
        self.handles = []
        self.restorations = []

    def begin_sample(self):
        self.audio_mask = None
        self.audio_tail = None
        self.active = True

    def set_audio_mask(self, mask):
        import torch
        mask = torch.as_tensor(mask).detach().cpu().bool()
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 1 or not bool(mask.any()):
            raise RuntimeError(f"Invalid decoder audio mask shape={tuple(mask.shape)}")
        self.audio_mask = mask
        self.audio_tail = None

    def set_audio_tail(self, actual_audio_positions, valid_audio_positions):
        actual = int(actual_audio_positions)
        valid = int(valid_audio_positions)
        if actual < 1 or valid < 1 or valid > actual:
            raise RuntimeError(f"Invalid streaming audio tail actual={actual}, valid={valid}")
        self.audio_mask = None
        self.audio_tail = (actual, valid)

    def clear_audio_positions(self):
        self.audio_mask = None
        self.audio_tail = None

    def end_sample(self):
        self.active = False
        self.audio_mask = None
        self.audio_tail = None

    def _make_hook(self, layer, units, unit_axis):
        name = f"layer_{layer:03d}"
        state = self.audit[name] = {
            "calls": 0, "applied_calls": 0, "skipped_nonprefill_calls": 0,
            "values_modified": 0, "audio_positions_modified": 0,
        }

        def hook(_module, _inputs, output):
            import torch
            if not self.active:
                return output
            state["calls"] += 1
            hidden = first_tensor(output)
            if hidden is None or hidden.ndim < 2:
                return output
            axis = unit_axis if unit_axis >= 0 else hidden.ndim + unit_axis
            if axis < 0 or axis >= hidden.ndim or max(units) >= hidden.shape[axis]:
                return output
            changed = hidden.clone()
            index = torch.as_tensor(units, dtype=torch.long, device=hidden.device)
            if layer in self.decoder_layers:
                if hidden.ndim != 3 or (self.audio_mask is None and self.audio_tail is None):
                    state["skipped_nonprefill_calls"] += 1
                    return output
                if self.audio_mask is not None:
                    mask = self.audio_mask.to(hidden.device)
                else:
                    actual, valid = self.audio_tail
                    sequence_length = int(hidden.shape[1])
                    start = sequence_length - actual
                    if start < 0:
                        state["skipped_nonprefill_calls"] += 1
                        return output
                    mask = torch.zeros(sequence_length, dtype=torch.bool, device=hidden.device)
                    mask[start:start + valid] = True
                sequence_axes = [
                    candidate for candidate in range(hidden.ndim)
                    if candidate != axis and hidden.shape[candidate] == mask.numel()
                ]
                if not sequence_axes:
                    state["skipped_nonprefill_calls"] += 1
                    return output
                sequence_axis = 1 if 1 in sequence_axes and axis != 1 else sequence_axes[0]
                moved = changed.movedim((sequence_axis, axis), (-2, -1))
                selected = moved.index_select(-1, index)
                selected[..., mask, :] = 0
                moved.index_copy_(-1, index, selected)
                changed = moved.movedim((-2, -1), (sequence_axis, axis))
                positions = int(mask.sum())
                modified = int(math.prod(hidden.shape) / hidden.shape[sequence_axis] / hidden.shape[axis]) * positions * len(units)
                state["audio_positions_modified"] += positions
            else:
                slices = [slice(None)] * changed.ndim
                slices[axis] = index
                changed[tuple(slices)] = 0
                modified = int(math.prod(hidden.shape) / hidden.shape[axis] * len(units))
            state["applied_calls"] += 1
            state["values_modified"] += modified
            return repack(output, changed)
        return hook

    @contextmanager
    def installed(self, targets):
        grouped = defaultdict(list)
        for layer, unit in targets:
            grouped[int(layer)].append(int(unit))
        try:
            for layer, units in sorted(grouped.items()):
                entry = self.layout[layer]
                module, axis, causal = entry[:3]
                hook_kind = entry[3] if len(entry) > 3 else "forward"
                if not causal or module is None:
                    raise ValueError(f"Layer {layer} is not causally hookable")
                unique = sorted(set(units))
                hook = self._make_hook(layer, unique, axis)
                if hook_kind == "forward":
                    self.handles.append(module.register_forward_hook(hook))
                elif hook_kind == "infer":
                    original = module.infer
                    def wrapped(*args, __original=original, __hook=hook, __module=module, **kwargs):
                        output = __original(*args, **kwargs)
                        return __hook(__module, args, output)
                    module.infer = wrapped
                    self.restorations.append((module, original))
                else:
                    raise ValueError(f"Unknown hook kind {hook_kind!r} for layer {layer}")
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            for module, original in self.restorations:
                module.infer = original
            self.restorations.clear()


def common_magnitude(data):
    p_z, s_z = data["P_z"], data["S_z"]
    counts = data["window_count"].astype(int)
    valid = np.arange(p_z.shape[1])[None, :] < counts[:, None]
    p = np.where(valid, np.maximum(p_z, 0.0), np.nan)
    s = np.where(valid, np.maximum(s_z, 0.0), np.nan)
    magnitude = np.sqrt(p * p + s * s)
    score = np.full(len(magnitude), -np.inf, dtype=np.float64)
    keep = np.any(np.isfinite(magnitude), axis=1)
    score[keep] = np.nanmax(magnitude[keep], axis=1)
    return score


def latest_records(path):
    records = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[str(row["key"])] = row
    return records


def mean_utterance_accuracy(rows):
    values = []
    for row in rows:
        if row.get("status") != "ok":
            values.append(0.0)
        elif "ASR_accuracy_1_minus_CER" in row:
            values.append(float(row["ASR_accuracy_1_minus_CER"]))
        else:
            values.append(max(0.0, 1.0 - float(row["CER"])))
    return float(np.mean(values)) if values else float("nan")


def dose_schedule(maximum, step):
    if maximum < 1:
        raise RuntimeError("No common causally hookable units across the three classes")
    if step != 500:
        raise ValueError("The formal lesion schedule requires --dose-step 500")
    values = list(range(step, maximum + 1, step)) if maximum >= step else []
    if not values or values[-1] != maximum:
        values.append(maximum)
    return values


def classification_total_units(summary_path: Path, model_key: str) -> int:
    """Read the model-wide unit count used by the final 1% dose cap."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing Step03 all-model classification summary: {summary_path}")
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("ModelSlug") == model_key]
    if not rows:
        raise KeyError(f"{model_key}: absent from {summary_path}")
    totals = {int(row["TotalUnits"]) for row in rows}
    if len(totals) != 1:
        raise RuntimeError(f"{model_key}: inconsistent TotalUnits in {summary_path}: {sorted(totals)}")
    total_units = totals.pop()
    if total_units < 1:
        raise RuntimeError(f"{model_key}: invalid TotalUnits={total_units} in {summary_path}")
    return total_units


def final_maximum_dose(total_units: int, class_counts: dict[str, int]) -> tuple[int, int]:
    """Return min(floor(1% of all units), P-only, S-only, Shared)."""
    missing = set(CLASS_CODES) - set(class_counts)
    if missing:
        raise RuntimeError(f"Incomplete lesion class counts; missing {sorted(missing)}")
    one_percent_units = total_units // 100
    maximum = min(one_percent_units, *(int(class_counts[name]) for name in CLASS_CODES))
    if maximum < 1:
        raise RuntimeError(
            "No common lesion dose after applying the 1% total-unit cap: "
            f"TotalUnits={total_units}, floor(1%)={one_percent_units}, classes={class_counts}"
        )
    return maximum, one_percent_units


def write_design(path, data, score, order, condition, dose):
    fields = [
        "condition", "rank", "global_unit_index_zero_based",
        "layer_order_zero_based", "source_layer_index_zero_based",
        "unit_index_within_layer_zero_based", "class_code",
        "common_magnitude_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(order[:dose], 1):
            writer.writerow({
                "condition": condition, "rank": rank,
                "global_unit_index_zero_based": int(index),
                "layer_order_zero_based": int(data["layer_order_zero_based"][index]),
                "source_layer_index_zero_based": int(data["source_layer_index_zero_based"][index]),
                "unit_index_within_layer_zero_based": int(data["unit_index_within_layer_zero_based"][index]),
                "class_code": int(data["class_code"][index]),
                "common_magnitude_score": float(score[index]),
            })


def run_condition(args, adapter, model_spec, layout, decoder_layers, data, score,
                  class_orders, samples, baseline_accuracy, condition, dose):
    from pipeline_code.step04_evaluate_performance.asr_benchmark_common import (
        edit_counts, normalize_for_cer, write_summary,
    )

    selected = class_orders[condition][:dose]
    layers = data["layer_order_zero_based"].astype(int)
    targets = [
        (int(layers[index]), int(data["unit_index_within_layer_zero_based"][index]))
        for index in selected
    ]
    run_dir = args.output_root / "model_details" / args.model / condition / f"n_{dose:05d}"
    result_path = run_dir / "result.json"
    if result_path.is_file() and not args.overwrite:
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if int(completed.get("samples", -1)) != len(samples) or int(completed.get("failures", -1)) != 0:
            if not args.retry_failures:
                raise RuntimeError(f"{result_path}: incomplete; use --retry-failures")
        else:
            print(f"SKIP completed {args.model}/{condition}/N={dose}", flush=True)
            return
    run_dir.mkdir(parents=True, exist_ok=True)
    write_design(run_dir / "lesion_design.csv", data, score, class_orders[condition], condition, dose)
    config = {
        "model_key": args.model, "condition": condition, "dose": dose,
        "sample_count": len(samples), "max_new_tokens": args.max_new_tokens,
        "intervention": "activation_zero",
        "decoder_position_policy": model_spec.DECODER_POSITION_POLICY,
        "ranking": "class-agnostic max-window sqrt(max(P_z,0)^2 + max(S_z,0)^2)",
        "generation": adapter.generation_description,
        "baseline_mean_utterance_accuracy": baseline_accuracy,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    predictions = run_dir / "predictions.jsonl"
    if args.overwrite and predictions.exists():
        predictions.unlink()
    existing = latest_records(predictions)
    pending = [
        sample for sample in samples
        if sample.key not in existing
        or (args.retry_failures and existing[sample.key].get("status") != "ok")
    ]
    audit_path = run_dir / "lesion_audit.json"
    previous_audit = {}
    if not args.overwrite and audit_path.is_file():
        previous_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    controller = ActivationZeroController(layout, decoder_layers)
    started = time.time()
    with controller.installed(targets), predictions.open("a", encoding="utf-8", buffering=1) as handle:
        for number, sample in enumerate(pending, 1):
            row = {
                "key": sample.key, **asdict(sample), "model_key": args.model,
                "condition": condition, "dose": dose,
            }
            tick = time.time()
            try:
                controller.begin_sample()
                with model_spec.audio_position_context(adapter, Path(sample.audio_path), controller):
                    hypothesis = adapter.transcribe(Path(sample.audio_path))
                scored = adapter.hypothesis_for_scoring(hypothesis)
                reference = normalize_for_cer(sample.reference)
                normalized_hypothesis = normalize_for_cer(scored)
                substitutions, deletions, insertions, count = edit_counts(reference, normalized_hypothesis)
                cer = ((substitutions + deletions + insertions) / count
                       if count else (0.0 if substitutions + deletions + insertions == 0 else 1.0))
                row.update(
                    status="ok", hypothesis=hypothesis, scoring_hypothesis=scored,
                    normalized_reference=reference, normalized_hypothesis=normalized_hypothesis,
                    S=substitutions, D=deletions, I=insertions, N=count, CER=cer,
                    ASR_accuracy_1_minus_CER=max(0.0, 1.0 - cer),
                )
            except Exception as error:
                row.update(
                    status="error", hypothesis="", S=0, D=0, I=0, N=0,
                    error_type=type(error).__name__, error=str(error),
                    traceback=traceback.format_exc(limit=10),
                )
            finally:
                controller.end_sample()
            row["elapsed_seconds"] = time.time() - tick
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if number == 1 or number % 10 == 0 or row["status"] != "ok":
                print(
                    f"[{args.model}/{condition}/N={dose}] {number}/{len(pending)} "
                    f"{row['status']} {row['elapsed_seconds']:.2f}s", flush=True,
                )
            if row["status"] == "error" and args.fail_fast:
                raise RuntimeError(row["error"])

    summary = write_summary(predictions, run_dir / "dataset_summary.csv")[0]
    final_rows = list(latest_records(predictions).values())
    lesion_accuracy = mean_utterance_accuracy(final_rows)
    failures = sum(row.get("status") != "ok" for row in final_rows)
    merged_layers = {}
    for source in (previous_audit.get("layers", {}), controller.audit):
        for name, state in source.items():
            merged = merged_layers.setdefault(name, {key: 0 for key in state})
            for key, value in state.items():
                merged[key] = int(merged.get(key, 0)) + int(value)
    audit = {
        "elapsed_seconds": (
            float(previous_audit.get("elapsed_seconds", 0.0)) + time.time() - started
        ),
        "target_count": dose,
        "layers": merged_layers,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not any(state["applied_calls"] for state in merged_layers.values()):
        raise RuntimeError("No lesion hook was applied")
    # Validate the samples processed by this invocation, not every row already
    # present in predictions.jsonl. A preempted/node-failed run can leave
    # successful prediction rows before lesion_audit.json is written; on resume
    # those rows must not make the new audit appear artificially incomplete.
    # Each resumed invocation still proves that its own pending samples received
    # the decoder audio-position lesion, while final failures==0 verifies the
    # complete merged prediction set.
    targeted_decoder = {
        name: state for name, state in controller.audit.items()
        if int(name.rsplit("_", 1)[1]) in decoder_layers
    }
    missing_decoder = {
        name: state for name, state in targeted_decoder.items()
        if state["applied_calls"] < len(pending)
    }
    if missing_decoder:
        raise RuntimeError(
            "Decoder audio-position lesion was not applied at least once per sample: "
            + json.dumps(missing_decoder, ensure_ascii=False)
        )
    result = {
        "model_key": args.model, "condition": condition, "dose": dose,
        "samples": len(final_rows), "failures": failures,
        "baseline_mean_utterance_accuracy": baseline_accuracy,
        "lesion_mean_utterance_accuracy": lesion_accuracy,
        "delta_percentage_points": 100.0 * (lesion_accuracy - baseline_accuracy),
        "corpus_CER": float(summary["CER"]),
        "elapsed_seconds": audit["elapsed_seconds"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def worker(args):
    if not args.model:
        raise ValueError("--model is required for internal worker mode")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_spec = load_model_spec(args.model)
    classification = (
        args.classification_root / args.model
        / "model_unit_classification_human_nominal_consecutive2.npz"
    )
    with np.load(classification, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata_path = classification.parent / "layer_metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))

    from pipeline_code.step04_evaluate_performance.asr_benchmark_common import load_samples
    from pipeline_code.step04_evaluate_performance.asr_model_adapters import create_adapter

    samples = load_samples(args.manifest_root.resolve(), ["fleurs_zh"], None, None)
    if len(samples) != 945:
        raise RuntimeError(f"Expected the full 945-row FLEURS-zh manifest, found {len(samples)}")
    baseline_records = latest_records(args.baseline_root / args.model / "predictions.jsonl")
    missing = [sample.key for sample in samples if sample.key not in baseline_records]
    if missing:
        raise RuntimeError(f"Step04 baseline is missing {len(missing)} selected rows")
    baseline_accuracy = mean_utterance_accuracy([baseline_records[sample.key] for sample in samples])

    adapter = create_adapter(args.model, args)
    adapter.model_key = args.model
    adapter.load()
    layout = model_spec.module_layout(adapter)
    expected_layers = int(data["layer_order_zero_based"].max()) + 1
    if len(layout) != expected_layers or len(metadata) != expected_layers:
        raise RuntimeError(
            f"Layer-order mismatch: runtime={len(layout)}, metadata={len(metadata)}, "
            f"classification={expected_layers}"
        )
    decoder_layers = {
        int(row["LayerOrderZeroBased"]) for row in metadata
        if str(row["SequenceType"]).startswith("decoder_")
    }
    if not decoder_layers:
        raise RuntimeError("No decoder layers found in Step03 metadata")
    causal = np.asarray([entry[2] for entry in layout], dtype=bool)
    layers = data["layer_order_zero_based"].astype(int)
    score = common_magnitude(data)
    class_orders = {}
    class_counts = {}
    for condition, code in CLASS_CODES.items():
        candidates = np.flatnonzero((data["class_code"] == code) & causal[layers])
        order = candidates[np.argsort(-score[candidates], kind="stable")]
        class_orders[condition] = order
        class_counts[condition] = len(order)
    classification_summary = args.classification_root / "all_models_classification_summary.csv"
    total_units = classification_total_units(classification_summary, args.model)
    maximum, one_percent_units = final_maximum_dose(total_units, class_counts)
    doses = dose_schedule(maximum, args.dose_step)
    plan = {
        "model_key": args.model, "causally_hookable_class_counts": class_counts,
        "total_model_units": total_units,
        "one_percent_total_units_floor": one_percent_units,
        "maximum_equal_N": maximum, "doses": doses,
        "conditions": args.conditions, "samples": len(samples),
        "dose_step": args.dose_step, "max_new_tokens": args.max_new_tokens,
        "maximum_rule": "min(floor(0.01 * TotalUnits), p_only, s_only, shared)",
        "classification_summary": str(classification_summary.resolve()),
        "ranking": "class-agnostic max-window sqrt(max(P_z,0)^2 + max(S_z,0)^2)",
        "intervention": "activation_zero",
        "decoder_position_policy": model_spec.DECODER_POSITION_POLICY,
    }
    model_output = args.output_root / "model_details" / args.model
    model_output.mkdir(parents=True, exist_ok=True)
    plan_path = model_output / "experiment_plan.json"
    if plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous.get("doses") != doses or int(previous.get("samples", -1)) != len(samples):
            raise RuntimeError(
                f"{plan_path}: existing schedule/sample count differs; use a new --output-root"
            )
        plan["conditions"] = [name for name in CLASS_CODES if name in set(previous["conditions"]) | set(args.conditions)]
    plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return 0
    for dose in doses:
        for condition in args.conditions:
            run_condition(
                args, adapter, model_spec, layout, decoder_layers, data, score,
                class_orders, samples, baseline_accuracy, condition, dose,
            )
    return 0


def main():
    args = parse_args()
    if args.internal_worker:
        return worker(args)
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
