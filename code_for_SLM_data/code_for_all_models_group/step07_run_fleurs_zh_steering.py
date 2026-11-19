#!/usr/bin/env python3
"""Evaluate the finalized Step07 steering intervention on full FLEURS-zh.

For every requested model this script reuses the canonical Step06 lesion-unit
ranking, selects the model-specific final maximum N,

    min(floor(1% of all classified units), P-only count, S-only count,
        Shared count),

and applies targeted-neuron activation scaling at fixed alpha=0.05:

    h' = (1 + alpha) * h

P-only, S-only, and Shared conditions use the same N. Encoder/adapter native
acoustic states and decoder prefill audio positions are modified; generated
tokens are excluded. Evaluation uses the full 945-item FLEURS-zh manifest.

Without --models the dispatcher runs every registered model. Use --models for
one or several models. Each model is launched in its model-specific Python
environment and writes resumable predictions plus strict hook audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import step07_run_52words_steering as core


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[1]
GROUP_ROOT = PROJECT_ROOT / "code_for_all_models_group"
CLASSES = core.CLASSES
MODEL_REGISTRY = core.MODEL_REGISTRY
FIXED_ALPHA = 0.05
EXPECTED_SAMPLES = 945


@dataclass(frozen=True)
class FleursSample:
    index: int
    pair: int
    semantic: int
    stem: str
    reference: str
    audio: Path
    key: str
    dataset: str
    subset: str
    sample_id: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_latest_step04(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "key" not in row:
            raise ValueError(f"{path}:{line_number}: missing key")
        rows[str(row["key"])] = row
    return rows


def resolve_audio_path(record: dict) -> Path:
    original = Path(str(record["audio_path"]))
    if original.is_file():
        return original
    subset = str(record.get("subset", "test"))
    sample_id = str(record["sample_id"])
    benchmark_root = os.environ.get("SLM_BENCHMARK_ROOT")
    candidates = (
        [Path(benchmark_root) / "Fleurs-zh" / subset / f"{sample_id}.wav"]
        if benchmark_root else []
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"FLEURS audio for {subset}/{sample_id} is absent; original={original}; "
        f"candidates={candidates}"
    )


def load_fleurs_samples(manifest_root: Path, baseline_path: Path) -> tuple[list[FleursSample], Path, str]:
    from pipeline_code.step04_evaluate_performance.asr_benchmark_common import load_samples

    source = manifest_root / "fleurs_zh.jsonl"
    if source.is_file():
        raw = load_samples(manifest_root.resolve(), ["fleurs_zh"], None, None)
        records = [
            {
                "key": sample.key, "dataset": sample.dataset, "subset": sample.subset,
                "sample_id": sample.sample_id, "reference": sample.reference,
                "audio_path": sample.audio_path,
            }
            for sample in raw
        ]
        source_kind = "formal_manifest"
    else:
        records = list(read_latest_step04(baseline_path).values())
        source = baseline_path
        source_kind = "reconstructed_from_step04_baseline"
    if len(records) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLES} FLEURS-zh samples, found {len(records)} from {source}")
    samples = []
    for index, record in enumerate(records):
        dataset = str(record.get("dataset", "fleurs_zh"))
        subset = str(record.get("subset", "test"))
        sample_id = str(record["sample_id"])
        key = str(record.get("key") or f"{dataset}\t{subset}\t{sample_id}")
        samples.append(FleursSample(
            index=index, pair=-1, semantic=-1, stem=key,
            reference=str(record["reference"]), audio=resolve_audio_path(record),
            key=key, dataset=dataset, subset=subset, sample_id=sample_id,
        ))
    return samples, source, source_kind


def normalize_baseline(samples: list[FleursSample], path: Path) -> dict[int, dict]:
    from pipeline_code.step04_evaluate_performance.asr_benchmark_common import normalize_for_cer

    latest = read_latest_step04(path)
    missing = [sample.key for sample in samples if sample.key not in latest]
    if missing:
        raise RuntimeError(f"Step04 baseline {path} is missing {len(missing)} FLEURS rows")
    normalized: dict[int, dict] = {}
    for sample in samples:
        row = latest[sample.key]
        if row.get("status") != "ok":
            raise RuntimeError(f"Step04 baseline failure for {sample.key}: {row.get('error')}")
        reference = str(row.get("normalized_reference") or normalize_for_cer(sample.reference))
        hypothesis = str(row.get("normalized_hypothesis") or normalize_for_cer(row.get("scoring_hypothesis", row.get("hypothesis", ""))))
        accuracy = row.get("ASR_accuracy_1_minus_CER", row.get("utterance_accuracy"))
        if accuracy is None:
            accuracy = max(0.0, 1.0 - float(row["CER"]))
        normalized[sample.index] = {
            **row,
            "reference": sample.reference,
            "normalized_reference": reference,
            "normalized_hypothesis": hypothesis,
            "utterance_accuracy": float(accuracy),
            "exact_match": hypothesis == reference,
        }
    return normalized


def write_baseline_reference(path: Path, samples: list[FleursSample], baseline: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = baseline[sample.index]
            payload = {
                "index": sample.index,
                "key": sample.key,
                "dataset": sample.dataset,
                "subset": sample.subset,
                "sample_id": sample.sample_id,
                "audio_path": str(sample.audio),
                "reference": sample.reference,
                "normalized_reference": row["normalized_reference"],
                "hypothesis": row.get("hypothesis", ""),
                "normalized_hypothesis": row["normalized_hypothesis"],
                "CER": float(row.get("CER", 1.0 - row["utterance_accuracy"])),
                "utterance_accuracy": float(row["utterance_accuracy"]),
                "exact_match": bool(row["exact_match"]),
                "source": "Step04 FLEURS-zh baseline",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def merge_audits(previous: dict, current: dict) -> dict:
    merged: dict[str, dict] = {}
    for name in sorted(set(previous) | set(current)):
        old, new = previous.get(name, {}), current.get(name, {})
        modes: dict[str, int] = {}
        for source in (old.get("position_modes", {}), new.get("position_modes", {})):
            for key, value in source.items():
                modes[key] = modes.get(key, 0) + int(value)
        merged[name] = {
            "sequence_type": new.get("sequence_type", old.get("sequence_type")),
            "calls": int(old.get("calls", 0)) + int(new.get("calls", 0)),
            "applied_calls": int(old.get("applied_calls", 0)) + int(new.get("applied_calls", 0)),
            "skipped_calls": int(old.get("skipped_calls", 0)) + int(new.get("skipped_calls", 0)),
            "selected_units": int(new.get("selected_units", old.get("selected_units", 0))),
            "audio_positions_modified": int(old.get("audio_positions_modified", 0)) + int(new.get("audio_positions_modified", 0)),
            "position_modes": modes,
        }
    return merged


def metric_arrays(rows: dict[int, dict], count: int):
    core.require_numpy()
    accuracy = core.np.asarray([float(rows[index]["utterance_accuracy"]) for index in range(count)], dtype=float)
    exact = core.np.asarray([bool(rows[index]["exact_match"]) for index in range(count)], dtype=bool)
    s = sum(int(rows[index].get("S", 0)) for index in range(count))
    d = sum(int(rows[index].get("D", 0)) for index in range(count))
    i = sum(int(rows[index].get("I", 0)) for index in range(count))
    n = sum(int(rows[index].get("N", 0)) for index in range(count))
    corpus_cer = (s + d + i) / n if n else float("nan")
    return accuracy, exact, corpus_cer


def paired_bootstrap(delta, iterations: int, seed: int) -> dict:
    core.require_numpy()
    rng = core.np.random.default_rng(seed)
    n = len(delta)
    means = core.np.empty(iterations, dtype=float)
    batch = 500
    for start in range(0, iterations, batch):
        stop = min(start + batch, iterations)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = core.np.quantile(means, [0.025, 0.975])
    lower_tail = (core.np.count_nonzero(means <= 0) + 1) / (iterations + 1)
    upper_tail = (core.np.count_nonzero(means >= 0) + 1) / (iterations + 1)
    return {
        "iterations": iterations,
        "seed": seed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_two_sided": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "probability_delta_positive": float((core.np.count_nonzero(means > 0) + 1) / (iterations + 1)),
    }


def summarize(model: str, class_name: str, dose: int, baseline: dict[int, dict],
              steered: dict[int, dict], source: Path, bootstrap_iterations: int,
              bootstrap_seed: int) -> dict:
    count = len(baseline)
    base_acc, base_exact, base_cer = metric_arrays(baseline, count)
    steer_acc, steer_exact, steer_cer = metric_arrays(steered, count)
    delta = steer_acc - base_acc
    boot = paired_bootstrap(delta, bootstrap_iterations, bootstrap_seed)
    return {
        "model": model,
        "dataset": "fleurs_zh",
        "samples": count,
        "class": class_name,
        "dose": dose,
        "alpha": FIXED_ALPHA,
        "baseline_mean_utterance_accuracy": float(base_acc.mean()),
        "steered_mean_utterance_accuracy": float(steer_acc.mean()),
        "delta_mean_utterance_accuracy": float(delta.mean()),
        "delta_percentage_points": float(100.0 * delta.mean()),
        "bootstrap_ci95_low_percentage_points": float(100.0 * boot["ci95_low"]),
        "bootstrap_ci95_high_percentage_points": float(100.0 * boot["ci95_high"]),
        "bootstrap_p_two_sided": boot["p_two_sided"],
        "bootstrap_probability_delta_positive": boot["probability_delta_positive"],
        "baseline_exact": int(base_exact.sum()),
        "steered_exact": int(steer_exact.sum()),
        "exact_delta": int(steer_exact.sum() - base_exact.sum()),
        "fixed": int(((~base_exact) & steer_exact).sum()),
        "broken": int((base_exact & (~steer_exact)).sum()),
        "baseline_corpus_CER": float(base_cer),
        "steered_corpus_CER": float(steer_cer),
        "canonical_lesion_design": str(source),
        "selection": f"first {dose} ranked rows",
    }


def direct_contrast(model: str, left_name: str, left: dict[int, dict], right_name: str,
                    right: dict[int, dict], iterations: int, seed: int) -> dict:
    left_acc, _left_exact, _ = metric_arrays(left, len(left))
    right_acc, _right_exact, _ = metric_arrays(right, len(right))
    delta = left_acc - right_acc
    boot = paired_bootstrap(delta, iterations, seed)
    return {
        "model": model,
        "contrast": f"{left_name}_minus_{right_name}",
        "delta_mean_utterance_accuracy": float(delta.mean()),
        "delta_percentage_points": float(100.0 * delta.mean()),
        "bootstrap_ci95_low_percentage_points": float(100.0 * boot["ci95_low"]),
        "bootstrap_ci95_high_percentage_points": float(100.0 * boot["ci95_high"]),
        "bootstrap_p_two_sided": boot["p_two_sided"],
    }


def runtime(model: str, args):
    core.require_numpy()
    core.require_step06()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from pipeline_code.step04_evaluate_performance.asr_model_adapters import create_adapter

    metadata_path = args.classification_root / model / "layer_metadata.csv"
    metadata = list(csv.DictReader(metadata_path.open(encoding="utf-8-sig", newline="")))
    spec = core.S6.load_model_spec(model)
    adapter = create_adapter(model, args)
    adapter.model_key = model
    adapter.load()
    cache_changes = core.disable_cache(adapter)
    layout = spec.module_layout(adapter)
    sequence_types = {int(row["LayerOrderZeroBased"]): row["SequenceType"] for row in metadata}
    # Direct targeted-neuron scaling uses only selected unit indices; unlike
    # centered gain it does not consume Step01 activation values.  Resolve the
    # hookable layers and widths from canonical metadata plus the live layout.
    layer_dims = {}
    for row in metadata:
        layer = int(row["LayerOrderZeroBased"])
        if layer >= len(layout) or layout[layer][0] is None or not layout[layer][2]:
            continue
        layer_dims[layer] = int(row["Units"])
    if not layer_dims:
        raise RuntimeError(f"{model}: no hookable layers in live layout")
    decoder = [layer for layer in layer_dims if str(sequence_types.get(layer, "")).lower().startswith("decoder")]
    baseline_path = args.baseline_root / model / "predictions.jsonl"
    samples, manifest_path, manifest_source_kind = load_fleurs_samples(args.manifest_root, baseline_path)
    return (adapter, spec, layout, layer_dims, decoder, sequence_types, samples,
            metadata_path, manifest_path, manifest_source_kind, cache_changes)


def run_model(model: str, args) -> None:
    started = time.time()
    (adapter, spec, layout, layer_dims, decoder, sequence_types, samples,
     metadata_path, manifest_path, manifest_source_kind, cache_changes) = runtime(model, args)

    classification_summary = args.classification_root / "all_models_classification_summary.csv"
    dose = core.final_max_n(model, classification_summary)
    model_root = args.output_root / "model_details" / model
    model_root.mkdir(parents=True, exist_ok=True)

    targets, populations, sources, manifest = {}, {}, {}, []
    for class_name in args.conditions:
        source, canonical = core.canonical_design(args.step06_root, model, class_name)
        selected_targets = core.targets_from_rows(canonical, dose)
        missing = sorted({layer for layer, _unit, _score in selected_targets if layer not in layer_dims})
        bad = [(layer, unit) for layer, unit, _score in selected_targets
               if layer in layer_dims and not 0 <= unit < layer_dims[layer]]
        if missing or bad:
            raise RuntimeError(f"{model}/{class_name}: missing layers={missing}; invalid units={bad[:5]}")
        targets[class_name] = selected_targets
        populations[class_name] = core.population({}, selected_targets)
        sources[class_name] = source
        for rank, (layer, unit, score) in enumerate(selected_targets, 1):
            manifest.append({
                "model": model,
                "dataset": "fleurs_zh",
                "class": class_name,
                "dose": dose,
                "alpha": FIXED_ALPHA,
                "rank": rank,
                "layer_order_zero_based": layer,
                "unit_index_within_layer_zero_based": unit,
                "score": score,
                "canonical_lesion_design": str(source),
            })
    core.write_csv(model_root / "unit_manifest.csv", manifest)

    union: dict[tuple[int, int], float] = {}
    for class_name in args.conditions:
        for layer, unit, score in targets[class_name]:
            union.setdefault((layer, unit), score)
    probe_population = core.population(
        {},
        [(layer, unit, score) for (layer, unit), score in union.items()],
    )
    probe_dir = model_root / "runtime_probe"
    probe_predictions = probe_dir / "predictions.jsonl"
    # A runtime probe must execute on every invocation.  Reusing its previous
    # successful prediction would skip inference and produce an empty current
    # hook audit, while formal condition predictions remain resumable.
    if probe_predictions.is_file():
        probe_predictions.unlink()
    _probe_rows, probe_current = core.run_condition(
        adapter, spec, layout, decoder, sequence_types, probe_population, 1e-6,
        samples[:1], probe_predictions, model, "runtime_probe", False,
    )
    probe_path = probe_dir / "steering_audit.json"
    # The runtime probe validates the exact population selected for this
    # invocation.  Do not merge a probe left by a previous invocation with a
    # different condition set, otherwise its layers appear as false extras.
    probe_audit = probe_current
    core.validate_audit(probe_audit, len(union), set(probe_population), f"{model}/runtime_probe")
    write_json(probe_path, probe_audit)

    baseline_source = args.baseline_root / model / "predictions.jsonl"
    baseline = normalize_baseline(samples, baseline_source)
    write_baseline_reference(model_root / "baseline_reference.jsonl", samples, baseline)
    resolved_manifest = model_root / "resolved_fleurs_zh_manifest.jsonl"
    with resolved_manifest.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({
                "dataset": sample.dataset, "subset": sample.subset,
                "sample_id": sample.sample_id, "audio_path": str(sample.audio),
                "reference": sample.reference,
            }, ensure_ascii=False) + "\n")

    summaries, condition_rows = [], {}
    for class_index, class_name in enumerate(args.conditions):
        condition = f"{class_name}_n{dose:05d}_alpha{FIXED_ALPHA:.3f}_fleurs_zh"
        condition_dir = model_root / class_name / f"n_{dose:05d}" / f"alpha_{FIXED_ALPHA:.3f}"
        predictions = condition_dir / "predictions.jsonl"
        steered, current_audit = core.run_condition(
            adapter, spec, layout, decoder, sequence_types, populations[class_name], FIXED_ALPHA,
            samples, predictions, model, condition, args.retry_failures,
        )
        audit_path = condition_dir / "steering_audit.json"
        previous_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        audit = merge_audits(previous_audit, current_audit)
        core.validate_audit(audit, dose, set(populations[class_name]), f"{model}/{condition}")
        write_json(audit_path, audit)
        design_rows = [row for row in manifest if row["class"] == class_name]
        core.write_csv(condition_dir / "steering_design.csv", design_rows)

        row = summarize(
            model, class_name, dose, baseline, steered, sources[class_name],
            args.bootstrap_iterations, args.bootstrap_seed + class_index,
        )
        summaries.append(row)
        condition_rows[class_name] = steered
        write_json(condition_dir / "result.json", {
            **row,
            "status": "complete",
            "failures": 0,
            "formula": "h'=(1+alpha)*h for selected units; h'=h otherwise",
            "method_reference": "Zhao et al., ACL 2026, targeted neuron activation scaling",
            "positions": "encoder/adapter native acoustic states; decoder prefill audio positions; generated tokens excluded",
        })

    core.write_csv(model_root / "summary.csv", summaries)
    contrasts = []
    if set(args.conditions) == set(CLASSES):
        contrasts = [
            direct_contrast(model, "shared", condition_rows["shared"], "p_only", condition_rows["p_only"],
                            args.bootstrap_iterations, args.bootstrap_seed + 100),
            direct_contrast(model, "shared", condition_rows["shared"], "s_only", condition_rows["s_only"],
                            args.bootstrap_iterations, args.bootstrap_seed + 101),
        ]
        core.write_csv(model_root / "direct_contrasts.csv", contrasts)

    plan = {
        "status": "complete",
        "model": model,
        "dataset": "fleurs_zh",
        "samples": len(samples),
        "dose": dose,
        "alpha": FIXED_ALPHA,
        "conditions": list(args.conditions),
        "formula": "h'=(1+alpha)*h for selected units; h'=h otherwise",
        "method_reference": "Zhao et al., ACL 2026, targeted neuron activation scaling",
        "max_rule": "min(floor(1% of TotalUnits), minimum P-only/S-only/Shared count)",
        "unit_source": "strict prefixes of canonical Step06 lesion_design.csv; no Step07 reranking",
        "activation_layout_source": "canonical layer metadata + live model layout (Step01 activations not required)",
        "positions": "encoder/adapter native acoustic states; decoder prefill audio positions; generated tokens excluded",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_source_kind": manifest_source_kind,
        "resolved_manifest_path": str(resolved_manifest),
        "resolved_manifest_sha256": sha256(resolved_manifest),
        "baseline_path": str(baseline_source),
        "baseline_sha256": sha256(baseline_source),
        "metadata_path": str(metadata_path),
        "cache_changes": cache_changes,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "summary": summaries,
        "direct_contrasts": contrasts,
        "elapsed_seconds": time.time() - started,
    }
    write_json(model_root / "run_complete.json", plan)
    print(json.dumps({
        "status": "complete",
        "model": model,
        "dose": dose,
        "alpha": FIXED_ALPHA,
        "samples_per_condition": len(samples),
        "summary": summaries,
    }, ensure_ascii=False), flush=True)


def aggregate(output_root: Path) -> tuple[Path, Path | None]:
    summaries, contrasts = [], []
    for model_dir in sorted((output_root / "model_details").glob("*")):
        summary_path = model_dir / "summary.csv"
        if summary_path.is_file():
            summaries.extend(csv.DictReader(summary_path.open(encoding="utf-8-sig", newline="")))
        contrast_path = model_dir / "direct_contrasts.csv"
        if contrast_path.is_file():
            contrasts.extend(csv.DictReader(contrast_path.open(encoding="utf-8-sig", newline="")))
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_output = summary_root / "all_models_summary.csv"
    core.write_csv(summary_output, summaries)
    contrast_output = None
    if contrasts:
        contrast_output = summary_root / "all_models_direct_contrasts.csv"
        core.write_csv(contrast_output, contrasts)
    return summary_output, contrast_output


def write_summary_workbooks(output_root: Path, models: list[str]) -> list[Path]:
    """Write one Step-06-style Excel summary per completed model."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.table import Table, TableStyleInfo
    except ImportError as error:
        raise RuntimeError("FLEURS Excel summaries require openpyxl") from error

    outputs = []
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    for model in models:
        model_root = output_root / "model_details" / model
        summary_path = model_root / "summary.csv"
        if not summary_path.is_file():
            continue
        summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8-sig", newline="")))
        contrast_path = model_root / "direct_contrasts.csv"
        contrast_rows = (
            list(csv.DictReader(contrast_path.open(encoding="utf-8-sig", newline="")))
            if contrast_path.is_file() else []
        )
        workbook = Workbook()
        workbook.remove(workbook.active)
        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(color="FFFFFF", bold=True)

        def add_sheet(name: str, rows: list[dict]):
            sheet = workbook.create_sheet(name)
            sheet.sheet_view.showGridLines = False
            fields = list(rows[0]) if rows else ["status"]
            sheet.append(fields)
            for row in rows:
                sheet.append([row.get(field) for field in fields])
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")
            if rows:
                table = Table(
                    displayName=f"T_{model}_{name}"[:250].replace(".", "_"),
                    ref=f"A1:{sheet.cell(1, sheet.max_column).column_letter}{sheet.max_row}",
                )
                table.tableStyleInfo = TableStyleInfo(
                    name="TableStyleMedium2", showRowStripes=True,
                    showColumnStripes=False,
                )
                sheet.add_table(table)
                sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = min(60, max(11, max(len(str(cell.value or "")) for cell in column) + 2))
                sheet.column_dimensions[column[0].column_letter].width = width

        add_sheet("summary", summary_rows)
        if contrast_rows:
            add_sheet("direct_contrasts", contrast_rows)
        quality_rows = []
        for row in summary_rows:
            condition_dir = (
                model_root / row["class"] / f"n_{int(row['dose']):05d}"
                / f"alpha_{float(row['alpha']):.3f}"
            )
            result = json.loads((condition_dir / "result.json").read_text(encoding="utf-8"))
            audit = json.loads((condition_dir / "steering_audit.json").read_text(encoding="utf-8"))
            selected = sum(int(value.get("selected_units", 0)) for value in audit.values())
            valid = selected == int(row["dose"]) and all(
                int(value.get("applied_calls", 0)) > 0
                and int(value.get("audio_positions_modified", 0)) > 0
                for value in audit.values()
            )
            quality_rows.append({
                "condition": row["class"], "samples": result.get("samples"),
                "failures": result.get("failures"), "selected_units": selected,
                "hook_audit": "valid" if valid else "invalid",
            })
        add_sheet("quality", quality_rows)
        add_sheet("metadata", [{
            "model": model, "dataset": "fleurs_zh", "alpha": FIXED_ALPHA,
            "dose_rule": "min(floor(1% total units), P-only, S-only, Shared)",
            "formula": "h'=(1+alpha)h for selected units",
            "unit_source": "Step06 canonical lesion ranking prefixes",
        }])
        output = summary_root / f"{model}.xlsx"
        workbook.save(output)
        outputs.append(output)
    return outputs


def dispatch(args) -> None:
    if args.list:
        for model, (folder, _prefix, environment) in MODEL_REGISTRY.items():
            print(f"{model}\t{environment}\t{folder}")
        return
    models = args.models or list(MODEL_REGISTRY)
    for model in models:
        environment = MODEL_REGISTRY[model][2]
        python = args.conda_root / environment / "bin/python"
        if not python.is_file() and model == "covo_audio_chat":
            python = args.conda_root / "py311_covo" / "bin/python"
        command = [
            str(python), str(SCRIPT), "--internal-worker", "--model", model,
            "--gpu", str(args.gpu),
            "--max-new-tokens", str(args.max_new_tokens),
            "--manifest-root", str(args.manifest_root),
            "--baseline-root", str(args.baseline_root),
            "--classification-root", str(args.classification_root),
            "--step06-root", str(args.step06_root),
            "--output-root", str(args.output_root),
            "--bootstrap-iterations", str(args.bootstrap_iterations),
            "--bootstrap-seed", str(args.bootstrap_seed),
            "--conditions", *args.conditions,
        ]
        if args.retry_failures:
            command.append("--retry-failures")
        if args.model_path:
            if len(models) != 1:
                raise ValueError("--model-path requires exactly one selected model")
            command += ["--model-path", str(args.model_path)]
        print("RUN", subprocess.list2cmdline(command), flush=True)
        if args.plan_only:
            continue
        if not python.is_file():
            raise FileNotFoundError(f"{model}: missing interpreter {python}")
        environment_variables = os.environ.copy()
        if model == "kimi_audio_7b_instruct" and Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6").is_file():
            environment_variables["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment_variables, check=True)
    if not args.plan_only:
        summary, contrasts = aggregate(args.output_root)
        print(summary)
        if contrasts:
            print(contrasts)
        for output in write_summary_workbooks(args.output_root, models):
            print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_REGISTRY),
                        help="one or more models; omit to run all registered models")
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help=argparse.SUPPRESS)
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-root", type=Path, default=Path(os.environ.get("SLM_CONDA_ROOT", Path.home() / "anaconda3" / "envs")))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--conditions", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--manifest-root", type=Path, default=GROUP_ROOT / "step04_manifests")
    parser.add_argument("--baseline-root", type=Path, default=GROUP_ROOT / "step04_model_performance/fleurs_zh")
    parser.add_argument("--classification-root", type=Path, default=GROUP_ROOT / "step03_classify_units_human_nominal_consecutive2")
    parser.add_argument("--step06-root", type=Path, default=GROUP_ROOT / "step06_lesion_results")
    parser.add_argument(
        "--output-root", type=Path,
        default=GROUP_ROOT / "step07_steering_results/fleurs_zh",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.internal_worker:
        if not args.model:
            raise ValueError("--internal-worker requires --model")
        if not math.isclose(FIXED_ALPHA, 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("finalized FLEURS steering must use alpha=0.05")
        run_model(args.model, args)
    else:
        dispatch(args)


if __name__ == "__main__":
    main()
