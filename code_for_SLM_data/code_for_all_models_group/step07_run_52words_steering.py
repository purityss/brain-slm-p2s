#!/usr/bin/env python3
"""Run the finalized Step-07 steering endpoint on the 52-word dataset.

The Step06 lesion rankings are reused without reranking.  At every dose the
P-only, S-only, and Shared populations are size matched.  The intervention is

    h' = (1 + alpha) * h

at native acoustic positions in encoder/adapter modules and at decoder prefill
audio positions in decoder modules.  Generated-token positions are excluded.
The number of steered units is fixed to the Step-06 final maximum dose,
``min(floor(1% of all units), P-only count, S-only count, Shared count)``, and
the activation-scaling coefficient is fixed to alpha=0.05.  The 52 words used
to define the units are also used for evaluation, so these results are
exploratory rather than independent gain-of-function evidence.

Omit ``--models`` to run all registered models, or pass one or more model keys.
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
from dataclasses import dataclass
from pathlib import Path

np = None


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[1]
GROUP_ROOT = PROJECT_ROOT / "code_for_all_models_group"
sys.path.insert(0, str(PROJECT_ROOT))

CLASSES = ("p_only", "s_only", "shared")
FIXED_ALPHA = 0.05
EXPECTED_SAMPLES = 52
MODEL_REGISTRY = {
    "baichuan_audio_base": ("code_for_Baichuan-Audio-Base", "Baichuan_Audio_Base", "py310_SLM"),
    "qwen2_audio": ("code_for_Qwen2-Audio", "Qwen2_Audio", "py310_SLM"),
}

EDIBLE = [
    ("caomei", "草莓"), ("daiyu", "带鱼"), ("dangao", "蛋糕"), ("doufu", "豆腐"),
    ("haidai", "海带"), ("hongshu", "红薯"), ("jidan", "鸡蛋"), ("jianbing", "煎饼"),
    ("lizhi", "荔枝"), ("longxia", "龙虾"), ("luobo", "萝卜"), ("lvdou", "绿豆"),
    ("mangguo", "芒果"), ("miji", "蜜桔"), ("mianbao", "面包"), ("mogu", "蘑菇"),
    ("niurou", "牛肉"), ("pingguo", "苹果"), ("ruantang", "软糖"), ("shutiao", "薯条"),
    ("xigua", "西瓜"), ("xiangchang", "香肠"), ("yangcong", "洋葱"), ("yingtao", "樱桃"),
    ("yumi", "玉米"), ("zhuti", "猪蹄"),
]
INEDIBLE = [
    ("caoping", "草坪"), ("daibiao", "代表"), ("dangong", "弹弓"), ("douhao", "逗号"),
    ("haitan", "海滩"), ("hongdeng", "红灯"), ("jihui", "机会"), ("jianbang", "肩膀"),
    ("liliang", "力量"), ("longzi", "笼子"), ("luoxuan", "螺旋"), ("lvshi", "律师"),
    ("mangren", "盲人"), ("mima", "密码"), ("mianju", "面具"), ("mote", "模特"),
    ("niuzai", "牛仔"), ("pingwei", "评委"), ("ruanjian", "软件"), ("shujia", "暑假"),
    ("xigai", "膝盖"), ("xiangcun", "乡村"), ("yangguang", "阳光"), ("yingxiong", "英雄"),
    ("yushi", "浴室"), ("zhubao", "珠宝"),
]
WORDS = EDIBLE + INEDIBLE


@dataclass(frozen=True)
class Sample:
    index: int
    pair: int
    semantic: int
    stem: str
    reference: str
    audio: Path


def import_step06():
    path = GROUP_ROOT / "step06_run_lesion_experiments.py"
    spec = importlib.util.spec_from_file_location("step07_step06_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S6 = None


def require_numpy():
    global np
    if np is None:
        import numpy as numpy_module
        np = numpy_module


def require_step06():
    global S6
    if S6 is None:
        S6 = import_step06()


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["record_key"])] = row
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def four_bins(values) -> np.ndarray:
    array = np.asarray(values, np.float32)
    edges = np.rint(np.linspace(0, len(array), 5)).astype(int)
    return np.stack([
        array[left:max(left + 1, right)].mean(0)
        for left, right in zip(edges[:-1], edges[1:])
    ])


def load_all_activations(model: str, metadata: list[dict], layout):
    folder, prefix, _environment = MODEL_REGISTRY[model]
    path = (
        PROJECT_ROOT / folder / "01_extract_activation"
        / f"{prefix}_hidden_dimension_activations"
        / f"{prefix}_hidden_dimension_activations.npy"
    )
    payload = np.load(path, allow_pickle=True).item()
    lookup = {Path(str(value)).stem: index for index, value in enumerate(payload["stimulus_files"])}
    order = [lookup[stem] for stem, _reference in WORDS]
    activations = {}
    for row in metadata:
        layer = int(row["LayerOrderZeroBased"])
        if layer >= len(layout) or layout[layer][0] is None or not layout[layer][2]:
            continue
        try:
            values = [payload["data_cell"][index, layer] for index in order]
            if any(value is None for value in values):
                continue
            activations[layer] = np.stack([four_bins(value) for value in values])
        except Exception:
            continue
    if not activations:
        raise RuntimeError(f"{model}: no hookable activation layers")
    return activations, path


def disable_cache(adapter) -> list[str]:
    changed, seen = [], set()
    candidates = [adapter]
    for name in ("model", "audio_model", "language_model", "llm"):
        value = getattr(adapter, name, None)
        if value is not None:
            candidates.append(value)
    model = getattr(adapter, "model", None)
    if model is not None:
        for name in ("model", "language_model", "llm", "transformer"):
            value = getattr(model, name, None)
            if value is not None:
                candidates.append(value)
    for value in candidates:
        if id(value) in seen:
            continue
        seen.add(id(value))
        generate = getattr(value, "generate", None)
        if callable(generate):
            def force_no_cache(*args, __generate=generate, **kwargs):
                kwargs["use_cache"] = False
                for key in list(kwargs):
                    if key.endswith("_use_cache"):
                        kwargs[key] = False
                return __generate(*args, **kwargs)
            try:
                value.generate = force_no_cache
                changed.append(f"{type(value).__name__}.generate(use_cache=False)")
            except Exception:
                pass
        for name in ("config", "generation_config"):
            config = getattr(value, name, None)
            if config is not None and hasattr(config, "use_cache"):
                try:
                    config.use_cache = False
                    changed.append(f"{type(value).__name__}.{name}.use_cache")
                except Exception:
                    pass
    return changed


def score_hypothesis(adapter, sample: Sample, hypothesis: str) -> dict:
    from pipeline_code.step04_evaluate_performance.asr_benchmark_common import edit_counts, normalize_for_cer

    scoring = adapter.hypothesis_for_scoring(hypothesis)
    reference = normalize_for_cer(sample.reference)
    normalized = normalize_for_cer(scoring)
    substitutions, deletions, insertions, length = edit_counts(reference, normalized)
    cer = (substitutions + deletions + insertions) / length if length else float(bool(substitutions + deletions + insertions))
    return {
        "hypothesis": hypothesis,
        "scoring_hypothesis": scoring,
        "normalized_reference": reference,
        "normalized_hypothesis": normalized,
        "S": substitutions,
        "D": deletions,
        "I": insertions,
        "N": length,
        "CER": cer,
        "utterance_accuracy": max(0.0, 1.0 - cer),
        "exact_match": normalized == reference,
    }


def population(activations: dict[int, np.ndarray], targets: list[tuple[int, int, float]]):
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for layer, unit, score in targets:
        grouped[layer].append((unit, score))
    selected = {}
    for layer, rows in grouped.items():
        units = np.asarray([unit for unit, _score in rows], dtype=np.int64)
        selected[layer] = {
            "units": units,
            "scores": np.asarray([score for _unit, score in rows], np.float32),
        }
    return selected


class GainController:
    def __init__(self, layout, decoder_layers, selected, gain, sequence_types):
        self.layout = layout
        self.decoder_layers = set(map(int, decoder_layers))
        self.selected = selected
        self.gain = float(gain)
        self.sequence_types = sequence_types
        self.active = False
        self.audio_mask = None
        self.audio_tail = None
        self.handles = []
        self.restorations = []
        self.audit = {}

    def begin_sample(self):
        self.active = True
        self.audio_mask = self.audio_tail = None

    def end_sample(self):
        self.active = False
        self.audio_mask = self.audio_tail = None

    def clear_audio_positions(self):
        self.audio_mask = self.audio_tail = None

    def set_audio_mask(self, mask):
        import torch

        value = torch.as_tensor(mask).detach().cpu().bool()
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1 or not value.any():
            raise RuntimeError("invalid decoder audio mask")
        self.audio_mask, self.audio_tail = value, None

    def set_audio_tail(self, actual_audio_positions, valid_audio_positions):
        self.audio_mask = None
        self.audio_tail = (int(actual_audio_positions), int(valid_audio_positions))

    def decoder_mask(self, hidden, unit_axis):
        import torch

        if self.audio_mask is not None:
            mask = self.audio_mask.to(hidden.device)
            axes = [axis for axis in range(hidden.ndim) if axis != unit_axis and hidden.shape[axis] == mask.numel()]
            if not axes:
                axes = [axis for axis in range(hidden.ndim) if axis != unit_axis and hidden.shape[axis] == int(mask.sum())]
                if axes:
                    mask = torch.ones(int(mask.sum()), dtype=torch.bool, device=hidden.device)
            if not axes:
                return None, None
            return mask, 1 if 1 in axes and unit_axis != 1 else axes[0]
        if self.audio_tail is None:
            return None, None
        actual, valid = self.audio_tail
        axes = [axis for axis in range(hidden.ndim) if axis != unit_axis and hidden.shape[axis] >= actual]
        if not axes:
            return None, None
        sequence_axis = 1 if 1 in axes and unit_axis != 1 else axes[0]
        start = hidden.shape[sequence_axis] - actual
        mask = torch.zeros(hidden.shape[sequence_axis], dtype=torch.bool, device=hidden.device)
        mask[start:start + valid] = True
        return mask, sequence_axis

    def hook(self, layer, unit_axis):
        import torch

        item = self.selected[layer]
        state = self.audit[f"layer_{layer:03d}"] = {
            "sequence_type": self.sequence_types.get(layer),
            "calls": 0,
            "applied_calls": 0,
            "skipped_calls": 0,
            "selected_units": len(item["units"]),
            "audio_positions_modified": 0,
            "position_modes": {},
        }

        def apply(_module, _inputs, output):
            if not self.active or self.gain == 0:
                return output
            state["calls"] += 1
            hidden = S6.first_tensor(output)
            real_axis = unit_axis if unit_axis >= 0 else (hidden.ndim + unit_axis if hidden is not None else -1)
            if hidden is None or hidden.ndim < 2 or real_axis < 0 or real_axis >= hidden.ndim:
                state["skipped_calls"] += 1
                return output
            units = torch.as_tensor(item["units"], dtype=torch.long, device=hidden.device)
            is_decoder = layer in self.decoder_layers or str(self.sequence_types.get(layer, "")).lower().startswith("decoder")
            if not is_decoder:
                values = hidden.index_select(real_axis, units).float()
                enhanced = (1.0 + self.gain) * values
                changed = hidden.clone()
                changed.index_copy_(real_axis, units, enhanced.to(hidden.dtype))
                state["position_modes"]["native_acoustic_all"] = state["position_modes"].get("native_acoustic_all", 0) + 1
                state["applied_calls"] += 1
                state["audio_positions_modified"] += int(math.prod(hidden.shape) // hidden.shape[real_axis])
                return S6.repack(output, changed)
            if hidden.ndim != 3:
                state["skipped_calls"] += 1
                return output
            mask, sequence_axis = self.decoder_mask(hidden, real_axis)
            if mask is None or not mask.any():
                state["skipped_calls"] += 1
                state["position_modes"]["decoder_non_audio"] = state["position_modes"].get("decoder_non_audio", 0) + 1
                return output
            moved = hidden.movedim((sequence_axis, real_axis), (-2, -1))
            values = moved.index_select(-1, units).float()
            enhanced = values.clone()
            enhanced[..., mask, :] = (1.0 + self.gain) * values[..., mask, :]
            changed = moved.clone()
            changed.index_copy_(-1, units, enhanced.to(hidden.dtype))
            changed = changed.movedim((-2, -1), (sequence_axis, real_axis))
            state["position_modes"]["decoder_prefill_audio"] = state["position_modes"].get("decoder_prefill_audio", 0) + 1
            state["applied_calls"] += 1
            state["audio_positions_modified"] += int(mask.sum())
            return S6.repack(output, changed)

        return apply

    @contextmanager
    def installed(self):
        try:
            for layer in sorted(self.selected):
                module, axis, causal, *rest = self.layout[layer]
                kind = rest[0] if rest else "forward"
                if module is None or not causal:
                    raise ValueError(f"unhookable selected layer {layer}")
                hook = self.hook(layer, axis)
                if kind == "forward":
                    self.handles.append(module.register_forward_hook(hook))
                elif kind == "infer":
                    original = module.infer
                    def wrapped(*args, __original=original, __hook=hook, __module=module, **kwargs):
                        return __hook(__module, args, __original(*args, **kwargs))
                    module.infer = wrapped
                    self.restorations.append((module, original))
                else:
                    raise ValueError(f"unsupported hook kind {kind}")
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            for module, original in self.restorations:
                module.infer = original
            self.handles.clear()
            self.restorations.clear()


def run_baseline(adapter, samples, path: Path, model: str, retry_failures: bool):
    done = read_jsonl(path)
    for number, sample in enumerate(samples, 1):
        key = f"baseline:{sample.index}"
        if key in done and not (retry_failures and done[key].get("status") != "ok"):
            continue
        row = {
            "record_key": key, "model": model, "condition": "baseline",
            "word_index": sample.index, "pair_id": sample.pair,
            "semantic": sample.semantic, "stem": sample.stem,
            "reference": sample.reference, "audio_path": str(sample.audio),
        }
        started = time.time()
        try:
            row.update(status="ok", **score_hypothesis(adapter, sample, adapter.transcribe(sample.audio)))
        except Exception as error:
            row.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=12))
        row["elapsed_seconds"] = time.time() - started
        append_jsonl(path, row)
        if number == 1 or number % 10 == 0 or row["status"] != "ok":
            print(f"[{model}/baseline] {number}/{len(samples)} {row['status']}", flush=True)
        if row["status"] != "ok":
            raise RuntimeError(row["error"])
    rows = {int(row["word_index"]): row for row in read_jsonl(path).values() if row.get("status") == "ok"}
    if len(rows) != len(samples):
        raise RuntimeError(f"{model}: baseline has {len(rows)}/{len(samples)} successes")
    return rows


def run_condition(adapter, spec, layout, decoder, sequence_types, selected, gain,
                  samples, path: Path, model: str, condition: str, retry_failures: bool):
    done = read_jsonl(path)
    controller = GainController(layout, decoder, selected, gain, sequence_types)
    with controller.installed():
        for number, sample in enumerate(samples, 1):
            key = f"{condition}:{sample.index}"
            if key in done and not (retry_failures and done[key].get("status") != "ok"):
                continue
            controller.begin_sample()
            row = {
                "record_key": key, "model": model, "condition": condition,
                "gain": gain, "word_index": sample.index, "pair_id": sample.pair,
                "semantic": sample.semantic, "stem": sample.stem,
                "reference": sample.reference, "audio_path": str(sample.audio),
            }
            started = time.time()
            try:
                with spec.audio_position_context(adapter, sample.audio, controller):
                    hypothesis = adapter.transcribe(sample.audio)
                row.update(status="ok", **score_hypothesis(adapter, sample, hypothesis))
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=12))
            finally:
                controller.end_sample()
            row["elapsed_seconds"] = time.time() - started
            append_jsonl(path, row)
            if number == 1 or number % 10 == 0 or row["status"] != "ok":
                print(f"[{model}/{condition}] {number}/{len(samples)} {row['status']}", flush=True)
            if row["status"] != "ok":
                raise RuntimeError(row["error"])
    rows = {int(row["word_index"]): row for row in read_jsonl(path).values() if row.get("status") == "ok"}
    if len(rows) != len(samples):
        raise RuntimeError(f"{model}/{condition}: {len(rows)}/{len(samples)} successes")
    return rows, controller.audit


def validate_audit(audit: dict, expected_units: int, expected_layers: set[int], label: str):
    observed = {int(name.replace("layer_", "")) for name in audit}
    if observed != expected_layers:
        raise RuntimeError(f"{label}: audit layers differ; missing={sorted(expected_layers-observed)}, extra={sorted(observed-expected_layers)}")
    if sum(int(row.get("selected_units", 0)) for row in audit.values()) != expected_units:
        raise RuntimeError(f"{label}: selected-unit audit total differs from {expected_units}")
    bad = [name for name, row in audit.items() if int(row.get("applied_calls", 0)) <= 0 or int(row.get("audio_positions_modified", 0)) <= 0]
    if bad:
        raise RuntimeError(f"{label}: inactive hooks {bad}")


def classification_counts(model: str, summary_path: Path) -> tuple[int, dict[str, int]]:
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8-sig", newline="")))
    wanted = [row for row in rows if row["ModelSlug"] == model]
    if not wanted:
        raise KeyError(f"{model}: absent from {summary_path}")
    total = int(wanted[0]["TotalUnits"])
    mapping = {"P-only": "p_only", "S-only": "s_only", "Shared": "shared"}
    counts = {mapping[row["Class"]]: int(row["Count"]) for row in wanted if row["Class"] in mapping}
    if set(counts) != set(CLASSES):
        raise RuntimeError(f"{model}: incomplete class counts {counts}")
    return total, counts


def final_max_n(model: str, classification_summary: Path) -> int:
    total, counts = classification_counts(model, classification_summary)
    return min(math.floor(total * 0.01), *counts.values())


def dose_grid(model: str, classification_summary: Path, step: int) -> list[int]:
    maximum = final_max_n(model, classification_summary)
    values = list(range(step, maximum + 1, step))
    if not values or values[-1] != maximum:
        values.append(maximum)
    return sorted(set(values))


def read_design(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"{path}: non-consecutive rank column")
    return rows


def canonical_design(step06_root: Path, model: str, class_name: str):
    parent = step06_root / "model_details" / model / class_name
    found = []
    for child in parent.iterdir():
        if child.name.startswith("n_") and (child / "lesion_design.csv").is_file():
            found.append((int(child.name[2:]), child / "lesion_design.csv"))
    if not found:
        raise FileNotFoundError(f"no Step06 lesion design under {parent}")
    _largest_n, largest_path = max(found)
    canonical = read_design(largest_path)
    canonical_ids = [(int(row["layer_order_zero_based"]), int(row["unit_index_within_layer_zero_based"])) for row in canonical]
    for dose, path in sorted(found):
        rows = read_design(path)
        ids = [(int(row["layer_order_zero_based"]), int(row["unit_index_within_layer_zero_based"])) for row in rows]
        if ids != canonical_ids[:dose]:
            raise RuntimeError(f"{path}: not a prefix of {largest_path}")
    return largest_path, canonical


def targets_from_rows(rows: list[dict], dose: int):
    if len(rows) < dose:
        raise RuntimeError(f"canonical design has {len(rows)} rows, requires {dose}")
    targets = []
    for row in rows[:dose]:
        score = float(row.get("common_magnitude_score") or row.get("class_effect_score") or 1.0)
        targets.append((int(row["layer_order_zero_based"]), int(row["unit_index_within_layer_zero_based"]), score))
    if len({(layer, unit) for layer, unit, _score in targets}) != dose:
        raise RuntimeError(f"duplicate unit in first {dose} canonical rows")
    return targets


def summarize(model, dose, class_name, gain, baseline, steered, source):
    baseline_exact = np.asarray([baseline[index]["exact_match"] for index in range(52)], bool)
    steered_exact = np.asarray([steered[index]["exact_match"] for index in range(52)], bool)
    baseline_accuracy = np.asarray([baseline[index]["utterance_accuracy"] for index in range(52)], float)
    steered_accuracy = np.asarray([steered[index]["utterance_accuracy"] for index in range(52)], float)
    return {
        "model": model, "dose": dose, "class": class_name, "gain": gain,
        "baseline_exact": int(baseline_exact.sum()), "steered_exact": int(steered_exact.sum()),
        "exact_delta": int(steered_exact.sum() - baseline_exact.sum()),
        "fixed": int(((~baseline_exact) & steered_exact).sum()),
        "broken": int((baseline_exact & (~steered_exact)).sum()),
        "baseline_mean_utterance_accuracy": float(baseline_accuracy.mean()),
        "steered_mean_utterance_accuracy": float(steered_accuracy.mean()),
        "delta_mean_utterance_accuracy": float((steered_accuracy - baseline_accuracy).mean()),
        "canonical_lesion_design": str(source),
        "selection": f"first {dose} ranked rows",
    }


def transitions(model, dose, class_name, gain, baseline, steered):
    rows = []
    for index in range(52):
        if baseline[index]["normalized_hypothesis"] == steered[index]["normalized_hypothesis"]:
            continue
        before = bool(baseline[index]["exact_match"])
        after = bool(steered[index]["exact_match"])
        rows.append({
            "model": model, "dose": dose, "class": class_name, "gain": gain,
            "word_index": index, "reference": baseline[index]["reference"],
            "baseline_hypothesis": baseline[index]["normalized_hypothesis"],
            "steered_hypothesis": steered[index]["normalized_hypothesis"],
            "baseline_exact": before, "steered_exact": after,
            "transition": "fixed" if not before and after else "broken" if before and not after else "changed_still_wrong",
        })
    return rows


def runtime(model: str, args):
    require_step06()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from pipeline_code.step04_evaluate_performance.asr_model_adapters import create_adapter

    metadata_path = args.classification_root / model / "layer_metadata.csv"
    metadata = list(csv.DictReader(metadata_path.open(encoding="utf-8-sig", newline="")))
    spec = S6.load_model_spec(model)
    adapter = create_adapter(model, args)
    adapter.model_key = model
    adapter.load()
    cache_changes = disable_cache(adapter)
    layout = spec.module_layout(adapter)
    activations, activation_path = load_all_activations(model, metadata, layout)
    sequence_types = {int(row["LayerOrderZeroBased"]): row["SequenceType"] for row in metadata}
    decoder = [layer for layer in activations if str(sequence_types.get(layer, "")).lower().startswith("decoder")]
    audio_root = PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)"
    samples = [Sample(index, index % 26, int(index >= 26), stem, reference, audio_root / f"{stem}.wav") for index, (stem, reference) in enumerate(WORDS)]
    return adapter, spec, layout, activations, decoder, sequence_types, samples, activation_path, metadata_path, cache_changes


def run_grid(model: str, doses: list[int], gains: list[float], args, mode: str):
    require_numpy()
    adapter, spec, layout, activations, decoder, sequence_types, samples, activation_path, metadata_path, cache_changes = runtime(model, args)
    model_root = args.output_root / "model_details" / model
    model_root.mkdir(parents=True, exist_ok=True)
    target_sets, populations, source_paths, manifest = {}, {}, {}, []
    for class_name in args.conditions:
        source, canonical = canonical_design(args.step06_root, model, class_name)
        for dose in doses:
            targets = targets_from_rows(canonical, dose)
            missing = sorted({layer for layer, _unit, _score in targets if layer not in activations})
            bad = [(layer, unit) for layer, unit, _score in targets if layer in activations and not 0 <= unit < activations[layer].shape[-1]]
            if missing or bad:
                raise RuntimeError(f"{model}/{class_name}/n={dose}: missing layers={missing}, bad units={bad[:5]}")
            target_sets[(dose, class_name)] = targets
            populations[(dose, class_name)] = population(activations, targets)
            source_paths[(dose, class_name)] = source
            for rank, (layer, unit, score) in enumerate(targets, 1):
                manifest.append({
                    "model": model, "dose": dose, "class": class_name, "rank": rank,
                    "layer_order_zero_based": layer,
                    "unit_index_within_layer_zero_based": unit,
                    "score": score,
                    "canonical_lesion_design": str(source),
                })
    write_csv(model_root / "unit_manifest.csv", manifest)

    probe_dose = max(doses)
    union = {}
    for class_name in args.conditions:
        for layer, unit, score in target_sets[(probe_dose, class_name)]:
            union.setdefault((layer, unit), score)
    probe_population = population(activations, [(layer, unit, score) for (layer, unit), score in union.items()])
    probe_dir = model_root / "runtime_probe"
    probe_rows, probe_audit = run_condition(
        adapter, spec, layout, decoder, sequence_types, probe_population, 1e-6,
        samples[:1], probe_dir / "predictions.jsonl", model, "runtime_probe",
        args.retry_failures,
    )
    validate_audit(probe_audit, len(union), set(probe_population), f"{model}/runtime_probe")
    (probe_dir / "steering_audit.json").write_text(json.dumps(probe_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline = run_baseline(adapter, samples, model_root / "baseline_predictions.jsonl", model, args.retry_failures)
    summary_rows, transition_rows = [], []
    for dose in doses:
        for class_name in args.conditions:
            selected = populations[(dose, class_name)]
            for gain in gains:
                condition = f"{class_name}_n{dose:05d}_alpha{gain:.3f}"
                condition_dir = model_root / class_name / f"n_{dose:05d}" / f"alpha_{gain:.3f}"
                steered, audit = run_condition(
                    adapter, spec, layout, decoder, sequence_types, selected, gain,
                    samples, condition_dir / "predictions.jsonl", model, condition,
                    args.retry_failures,
                )
                validate_audit(audit, dose, set(selected), f"{model}/{condition}")
                row = summarize(model, dose, class_name, gain, baseline, steered, source_paths[(dose, class_name)])
                summary_rows.append(row)
                transition_rows.extend(transitions(model, dose, class_name, gain, baseline, steered))
                design_rows = [row for row in manifest if row["dose"] == dose and row["class"] == class_name]
                write_csv(condition_dir / "steering_design.csv", design_rows)
                (condition_dir / "steering_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
                (condition_dir / "result.json").write_text(json.dumps({
                    **row, "samples": 52, "failures": 0,
                    "formula": "h'=(1+alpha)*h for selected units; h'=h otherwise",
                    "method_reference": "Zhao et al., ACL 2026, targeted neuron activation scaling",
                    "positions": "encoder/adapter native acoustic positions; decoder prefill audio positions; generated tokens excluded",
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(model_root / f"{mode}_curve_summary.csv", summary_rows)
    write_csv(model_root / f"{mode}_curve_transitions.csv", transition_rows)
    # The finalized Step-07 endpoint has one maximum dose and one fixed alpha.
    # Keep the curve-named files for backward compatibility, and also expose
    # stable endpoint filenames shared with the FLEURS-zh result layout.
    if mode == "dose" and len(doses) == 1 and len(gains) == 1:
        write_csv(model_root / "summary.csv", summary_rows)
        write_csv(model_root / "transitions.csv", transition_rows)
    plan = {
        "status": "complete", "model_key": model, "mode": mode,
        "doses": doses, "gains": gains, "conditions": list(args.conditions),
        "samples": 52,
        "max_rule": "min(floor(1% of TotalUnits), minimum P-only/S-only/Shared count)",
        "unit_source": "exact prefixes of Step06 lesion_design.csv; no Step07 reranking",
        "formula": "h'=(1+alpha)*h for selected units; h'=h otherwise, alpha>0",
        "method_reference": "Zhao et al., ACL 2026, targeted neuron activation scaling",
        "positions": "encoder/adapter native acoustic positions; decoder prefill audio positions; generated tokens excluded",
        "activation_path": str(activation_path), "metadata_path": str(metadata_path),
        "cache_changes": cache_changes,
        "warning": "the same 52 words define units and evaluate steering; exploratory only",
    }
    (model_root / "experiment_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_root / "run_complete.json").write_text(json.dumps({"status": "complete", "summary_rows": len(summary_rows)}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "model": model, "mode": mode, "summary_rows": len(summary_rows)}, ensure_ascii=False), flush=True)


def aggregate(output_root: Path, filename: str, output_name: str | None = None) -> Path:
    rows = []
    for path in sorted((output_root / "model_details").glob(f"*/{filename}")):
        rows.extend(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if filename == "summary.csv":
        rows = [{
            "model": row["model"],
            "condition": row["class"],
            "N": int(row["dose"]),
            "alpha": float(row["gain"]),
            "samples": 52,
            "failures": 0,
            "baseline_mean_accuracy_pct": 100.0 * float(row["baseline_mean_utterance_accuracy"]),
            "steered_mean_accuracy_pct": 100.0 * float(row["steered_mean_utterance_accuracy"]),
            "delta_pp": 100.0 * float(row["delta_mean_utterance_accuracy"]),
            "baseline_exact": int(row["baseline_exact"]),
            "steered_exact": int(row["steered_exact"]),
            "exact_delta": int(row["exact_delta"]),
            "fixed": int(row["fixed"]),
            "broken": int(row["broken"]),
            "formula": "h'=(1+alpha)*h for selected units; h'=h otherwise",
            "selection": row["selection"],
        } for row in rows]
    elif filename == "transitions.csv":
        rows = [{
            "model": row["model"],
            "condition": row["class"],
            "N": int(row["dose"]),
            "alpha": float(row["gain"]),
            "word_index": int(row["word_index"]),
            "reference": row["reference"],
            "baseline_hypothesis": row["baseline_hypothesis"],
            "steered_hypothesis": row["steered_hypothesis"],
            "baseline_exact": row.get("baseline_exact", ""),
            "steered_exact": row.get("steered_exact", ""),
            "transition": row["transition"],
        } for row in rows]
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    output = summary_root / (output_name or filename.replace("curve_summary", "curve_all_models"))
    # A selected-model rerun must not erase already completed models from the
    # combined table. Replace only models that have a freshly resolved file.
    if output.is_file() and rows:
        existing = list(csv.DictReader(output.open(encoding="utf-8-sig", newline="")))
        refreshed = {str(row.get("model", row.get("model_key", ""))) for row in rows}
        rows = [
            row for row in existing
            if str(row.get("model", row.get("model_key", ""))) not in refreshed
        ] + rows
    write_csv(output, rows)
    return output


def _summary_rows(results_root: Path, model: str, mode: str) -> list[dict]:
    endpoint = results_root / "model_details" / model / "summary.csv"
    direct = endpoint if mode == "dose" and endpoint.is_file() else (
        results_root / "model_details" / model / f"{mode}_curve_summary.csv"
    )
    path = direct if direct.is_file() else (
        results_root / f"{mode}_curves" / "model_details" / model
        / f"{mode}_curve_summary.csv"
    )
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def _transition_rows(results_root: Path, model: str, mode: str) -> list[dict]:
    endpoint = results_root / "model_details" / model / "transitions.csv"
    direct = endpoint if mode == "dose" and endpoint.is_file() else (
        results_root / "model_details" / model / f"{mode}_curve_transitions.csv"
    )
    path = direct if direct.is_file() else (
        results_root / f"{mode}_curves" / "model_details" / model
        / f"{mode}_curve_transitions.csv"
    )
    if not path.is_file():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        row["mode"] = mode
    return rows


def _audit_status(results_root: Path, model: str, mode: str, row: dict) -> tuple[int, int, str]:
    direct = results_root / "model_details" / model
    model_root = direct if direct.is_dir() else (
        results_root / f"{mode}_curves" / "model_details" / model
    )
    condition_dir = (
        model_root / row["class"] / f"n_{int(row['dose']):05d}"
        / f"alpha_{float(row['gain']):.3f}"
    )
    result_path = condition_dir / "result.json"
    audit_path = condition_dir / "steering_audit.json"
    samples, failures = 0, 0
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        samples = int(result.get("samples", 0))
        failures = int(result.get("failures", 0))
    if not audit_path.is_file():
        return samples, failures, "missing"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selected = sum(int(layer.get("selected_units", 0)) for layer in audit.values())
    valid = (
        selected == int(row["dose"])
        and all(
            int(layer.get("selected_units", 0)) > 0
            and int(layer.get("applied_calls", 0)) > 0
            and int(layer.get("audio_positions_modified", 0)) > 0
            for layer in audit.values()
        )
    )
    return samples, failures, "valid" if valid else "invalid"


def write_summary_workbook(results_root: Path, model: str) -> Path | None:
    """Create the Step06-style per-model Excel summary from completed CSVs.

    This is called by the top-level dispatcher, whose standard py310_SLM
    environment contains openpyxl. Model-specific workers remain independent
    of the spreadsheet dependency.
    """
    dose_rows = _summary_rows(results_root, model, "dose")
    alpha_rows = _summary_rows(results_root, model, "alpha")
    if not dose_rows and not alpha_rows:
        return None
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.table import Table, TableStyleInfo
    except ImportError as error:
        raise RuntimeError(
            "Excel summaries require openpyxl in the top-level dispatcher environment"
        ) from error

    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    def add_table_sheet(name: str, headers: list[str], data: list[list], formats: dict[int, str] | None = None):
        sheet = workbook.create_sheet(name)
        sheet.sheet_view.showGridLines = False
        sheet.append(headers)
        for values in data:
            sheet.append(values)
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        if sheet.max_row >= 2:
            table = Table(displayName=f"T_{model}_{name}"[:250].replace(".", "_"), ref=f"A1:{sheet.cell(1, sheet.max_column).column_letter}{sheet.max_row}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
            sheet.add_table(table)
            sheet.freeze_panes = "A2"
        for column, number_format in (formats or {}).items():
            for row_index in range(2, sheet.max_row + 1):
                sheet.cell(row_index, column).number_format = number_format
        for column_cells in sheet.columns:
            width = min(60, max(11, max(len(str(cell.value or "")) for cell in column_cells) + 2))
            sheet.column_dimensions[column_cells[0].column_letter].width = width
        return sheet

    def pivot(rows: list[dict], x_key: str, value_key: str, baseline: bool = True):
        grouped: dict[float, dict[str, float]] = defaultdict(dict)
        for row in rows:
            grouped[float(row[x_key])][row["class"]] = float(row[value_key])
        values = []
        if baseline and rows:
            values.append([0, *([float(rows[0]["baseline_mean_utterance_accuracy"])] * 3)])
        for x_value in sorted(grouped):
            classes = grouped[x_value]
            values.append([x_value, classes.get("p_only"), classes.get("s_only"), classes.get("shared")])
        return values

    if dose_rows:
        add_table_sheet(
            "dose_accuracy", ["steering_count", "p_only", "s_only", "shared"],
            pivot(dose_rows, "dose", "steered_mean_utterance_accuracy"),
            {2: "0.0000", 3: "0.0000", 4: "0.0000"},
        )
        dose_delta = pivot(dose_rows, "dose", "delta_mean_utterance_accuracy", baseline=False)
        for values in dose_delta:
            values[1:] = [None if value is None else value * 100 for value in values[1:]]
        dose_delta.insert(0, [0, 0, 0, 0])
        add_table_sheet(
            "dose_delta_pp", ["steering_count", "p_only", "s_only", "shared"], dose_delta,
            {2: "+0.00;-0.00;0.00", 3: "+0.00;-0.00;0.00", 4: "+0.00;-0.00;0.00"},
        )
        dose_exact = pivot(dose_rows, "dose", "steered_exact", baseline=False)
        dose_exact.insert(0, [0, *([int(dose_rows[0]["baseline_exact"])] * 3)])
        add_table_sheet("dose_exact", ["steering_count", "p_only", "s_only", "shared"], dose_exact, {2: "0", 3: "0", 4: "0"})

    if alpha_rows:
        add_table_sheet(
            "alpha_accuracy", ["alpha", "p_only", "s_only", "shared"],
            pivot(alpha_rows, "gain", "steered_mean_utterance_accuracy"),
            {1: "0.00", 2: "0.0000", 3: "0.0000", 4: "0.0000"},
        )
        alpha_delta = pivot(alpha_rows, "gain", "delta_mean_utterance_accuracy", baseline=False)
        for values in alpha_delta:
            values[1:] = [None if value is None else value * 100 for value in values[1:]]
        alpha_delta.insert(0, [0, 0, 0, 0])
        add_table_sheet(
            "alpha_delta_pp", ["alpha", "p_only", "s_only", "shared"], alpha_delta,
            {1: "0.00", 2: "+0.00;-0.00;0.00", 3: "+0.00;-0.00;0.00", 4: "+0.00;-0.00;0.00"},
        )
        alpha_exact = pivot(alpha_rows, "gain", "steered_exact", baseline=False)
        alpha_exact.insert(0, [0, *([int(alpha_rows[0]["baseline_exact"])] * 3)])
        add_table_sheet("alpha_exact", ["alpha", "p_only", "s_only", "shared"], alpha_exact, {1: "0.00", 2: "0", 3: "0", 4: "0"})

    quality = []
    for mode, rows in (("dose", dose_rows), ("alpha", alpha_rows)):
        for row in rows:
            samples, failures, audit = _audit_status(results_root, model, mode, row)
            quality.append([mode, row["class"], int(row["dose"]), float(row["gain"]), samples, failures, audit])
    add_table_sheet("quality", ["mode", "condition", "steering_count", "alpha", "samples", "failures", "hook_audit"], quality, {4: "0.00", 5: "0", 6: "0"})

    transitions = []
    for mode in ("dose", "alpha"):
        for row in _transition_rows(results_root, model, mode):
            transitions.append([
                mode, row["class"], int(row["dose"]), float(row["gain"]), int(row["word_index"]),
                row["reference"], row["baseline_hypothesis"], row["steered_hypothesis"], row["transition"],
            ])
    if transitions:
        add_table_sheet(
            "word_transitions",
            ["mode", "condition", "steering_count", "alpha", "word_index", "reference", "baseline_hypothesis", "steered_hypothesis", "transition"],
            transitions,
            {4: "0.00", 5: "0"},
        )

    metadata = [
        ["model_key", model],
        ["method", "lesion-matched targeted neuron activation scaling"],
        ["formula", "h'=(1+alpha)*h for selected units; h'=h otherwise"],
        ["positions", "encoder/adapter native acoustic positions; decoder prefill audio positions; generated tokens excluded"],
        ["unit_source", "exact ranked prefixes of Step06 lesion_design.csv"],
        ["samples", 52],
        ["dose_rows", len(dose_rows)],
        ["alpha_rows", len(alpha_rows)],
        ["interpretation", "exploratory same-52 result; unit definition, parameter screening, and evaluation reuse the same words"],
    ]
    add_table_sheet("metadata", ["field", "value"], metadata)
    summary_dir = results_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output = summary_dir / f"{model}.xlsx"
    workbook.save(output)
    return output


def write_all_summary_workbooks(results_root: Path, models: list[str] | None = None) -> list[Path]:
    available = set()
    direct = results_root / "model_details"
    if direct.is_dir():
        available.update(path.name for path in direct.iterdir() if path.is_dir())
    for mode in ("dose", "alpha"):
        details = results_root / f"{mode}_curves" / "model_details"
        if details.is_dir():
            available.update(path.name for path in details.iterdir() if path.is_dir())
    selected = sorted(available if models is None else available.intersection(models))
    outputs = []
    for model in selected:
        output = write_summary_workbook(results_root, model)
        if output is not None:
            outputs.append(output)
    return outputs


def dispatch(args, script: Path, forwarded: list[str]):
    if args.list:
        for key, (folder, _prefix, environment) in MODEL_REGISTRY.items():
            print(f"{key}\t{environment}\t{folder}")
        return
    models = [args.model] if args.model else (args.models or list(MODEL_REGISTRY))
    for model in models:
        environment = MODEL_REGISTRY[model][2]
        python = args.conda_root / environment / "bin/python"
        command = [str(python), str(script), "--internal-worker", "--model", model, "--gpu", str(args.gpu), *forwarded]
        print("RUN", subprocess.list2cmdline(command), flush=True)
        if args.plan_only:
            continue
        if not python.is_file():
            raise FileNotFoundError(f"{model}: missing interpreter {python}")
        environment_variables = os.environ.copy()
        if model == "kimi_audio_7b_instruct" and Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6").is_file():
            environment_variables["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment_variables, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_REGISTRY))
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help=argparse.SUPPRESS)
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-root", type=Path, default=Path(os.environ.get("SLM_CONDA_ROOT", Path.home() / "anaconda3" / "envs")))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--conditions", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--classification-root", type=Path, default=GROUP_ROOT / "step03_classify_units_human_nominal_consecutive2")
    parser.add_argument("--step06-root", type=Path, default=GROUP_ROOT / "step06_lesion_results")
    parser.add_argument(
        "--output-root", type=Path,
        default=GROUP_ROOT / "step07_steering_results" / "52words",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.internal_worker:
        forwarded = [
            "--max-new-tokens", str(args.max_new_tokens), "--output-root", str(args.output_root),
            "--classification-root", str(args.classification_root), "--step06-root", str(args.step06_root),
            "--conditions", *args.conditions,
        ]
        if args.retry_failures:
            forwarded.append("--retry-failures")
        if args.model_path:
            forwarded += ["--model-path", str(args.model_path)]
        dispatch(args, SCRIPT, forwarded)
        if not args.plan_only:
            print(aggregate(
                args.output_root, "summary.csv", "all_models_summary.csv"
            ))
            print(aggregate(
                args.output_root, "transitions.csv", "all_models_transitions.csv"
            ))
            for output in write_all_summary_workbooks(
                args.output_root, [args.model] if args.model else args.models
            ):
                print(output)
        return
    classification_summary = args.classification_root / "all_models_classification_summary.csv"
    dose = final_max_n(args.model, classification_summary)
    run_grid(args.model, [dose], [FIXED_ALPHA], args, "dose")


if __name__ == "__main__":
    main()
