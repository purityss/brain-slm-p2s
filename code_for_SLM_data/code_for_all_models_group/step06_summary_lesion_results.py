#!/usr/bin/env python3
"""Validate Step06 conditions and create one plot-ready workbook per model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


GROUP_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = GROUP_ROOT / "step06_lesion_results"
CONDITIONS = ("p_only", "s_only", "shared")
DEFAULT_SUPPLEMENT = RESULTS_ROOT / "lesion_summary" / "step06_1pct_supplement_results.json"
CLASSIFICATION_SUMMARY = GROUP_ROOT / "step03_classify_units_human_nominal_consecutive2" / "all_models_classification_summary.csv"


def model_names(details_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    return sorted(path.name for path in details_root.iterdir() if path.is_dir())


def final_doses(model: str, plan: dict) -> tuple[int, int, list[int]]:
    """Use the same 1%/three-class cap and 500-step schedule as the runner."""
    with CLASSIFICATION_SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["ModelSlug"] == model]
    totals = {int(row["TotalUnits"]) for row in rows}
    if len(totals) != 1:
        raise ValueError(f"{model}: missing or inconsistent TotalUnits")
    class_counts = plan.get("causally_hookable_class_counts")
    if class_counts:
        class_min = min(int(class_counts[name]) for name in CONDITIONS)
    else:
        class_min = int(plan["maximum_equal_N"])
    maximum = min(totals.pop() // 100, class_min)
    if maximum < 1:
        raise ValueError(f"{model}: no common lesion dose")
    doses = list(range(500, maximum + 1, 500))
    if not doses or doses[-1] != maximum:
        doses.append(maximum)
    return maximum, class_min, doses


def load_model(details_root: Path, model: str, allow_incomplete: bool = False):
    folder = details_root / model
    plan_path = folder / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("model_key") != model:
        raise ValueError(f"{plan_path}: mismatched model_key")
    _, _, allowed_doses = final_doses(model, plan)
    doses = [int(value) for value in plan["doses"] if int(value) in allowed_doses]
    conditions = [name for name in CONDITIONS if name in plan["conditions"]]
    if not doses or not conditions:
        raise ValueError(f"{plan_path}: empty dose/condition plan")
    results = {}
    baseline = None
    missing = []
    for dose in doses:
        for condition in conditions:
            path = folder / condition / f"n_{dose:05d}" / "result.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row.get("model_key"), row.get("condition"), int(row.get("dose", -1))) != (model, condition, dose):
                raise ValueError(f"{path}: identity mismatch")
            if int(row["samples"]) != int(plan["samples"]) or int(row["failures"]) != 0:
                raise ValueError(f"{path}: incomplete samples or nonzero failures")
            current_baseline = float(row["baseline_mean_utterance_accuracy"])
            if baseline is not None and abs(baseline - current_baseline) > 1e-9:
                raise ValueError(f"{path}: baseline differs between conditions")
            baseline = current_baseline
            results[(dose, condition)] = row
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"{model}: {len(missing)} planned results missing; first: {missing[0]}")
    if baseline is None:
        raise ValueError(f"{model}: no completed conditions")
    return plan, doses, conditions, baseline, results, missing


def _sheet(workbook: Workbook, name: str, headers: list[str], rows: list[list]):
    sheet = workbook.create_sheet(name)
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.row_dimensions[1].height = 25
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")
    for index, header in enumerate(headers, 1):
        sheet.column_dimensions[get_column_letter(index)].width = max(17, len(header) + 4)
    for row in sheet.iter_rows(min_row=2, min_col=2):
        for cell in row:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000"
    return sheet


def load_supplement(path: Path | None) -> dict[str, list[dict]]:
    """Load optional validated result rows that add non-grid lesion doses."""
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for row in payload.get("results", []):
        model = str(row["model_key"])
        grouped.setdefault(model, []).append(row)
    return grouped


def merge_supplement(
    model: str,
    plan: dict,
    doses: list[int],
    conditions: list[str],
    baseline: float,
    results: dict[tuple[int, str], dict],
    rows: list[dict],
) -> list[int]:
    """Validate and merge supplemental doses without changing the experiment plan."""
    if not rows:
        return doses
    by_dose: dict[int, set[str]] = {}
    for row in rows:
        condition = str(row.get("condition"))
        dose = int(row.get("dose", -1))
        if row.get("model_key") != model or condition not in conditions or dose <= 0:
            raise ValueError(f"{model}: invalid supplemental identity: {row}")
        if int(row["samples"]) != int(plan["samples"]) or int(row["failures"]) != 0:
            raise ValueError(f"{model}/{condition}/n_{dose:05d}: incomplete supplemental result")
        current_baseline = float(row["baseline_mean_utterance_accuracy"])
        if abs(baseline - current_baseline) > 1e-9:
            raise ValueError(f"{model}/{condition}/n_{dose:05d}: supplemental baseline differs")
        key = (dose, condition)
        if key in results:
            existing = float(results[key]["lesion_mean_utterance_accuracy"])
            incoming = float(row["lesion_mean_utterance_accuracy"])
            if abs(existing - incoming) > 1e-12:
                raise ValueError(f"{model}/{condition}/n_{dose:05d}: conflicting result")
        results[key] = row
        by_dose.setdefault(dose, set()).add(condition)
    expected = set(conditions)
    for dose, present in by_dose.items():
        if present != expected:
            raise ValueError(
                f"{model}/n_{dose:05d}: supplemental conditions {sorted(present)}; "
                f"expected {sorted(expected)}"
            )
    return sorted(set(doses) | set(by_dose))


def summarize_model(
    results_root: Path,
    model: str,
    allow_incomplete: bool = False,
    supplement_rows: list[dict] | None = None,
) -> Path:
    plan, doses, conditions, baseline, results, missing = load_model(
        results_root / "model_details", model, allow_incomplete
    )
    maximum, class_min, allowed_doses = final_doses(model, plan)
    supplement_rows = [row for row in (supplement_rows or []) if int(row["dose"]) in allowed_doses]
    doses = merge_supplement(
        model, plan, doses, conditions, baseline, results, supplement_rows
    )
    missing_doses = [dose for dose in allowed_doses if any((dose, condition) not in results for condition in conditions)]
    if missing_doses and not allow_incomplete:
        raise ValueError(f"{model}: missing required lesion doses: {missing_doses}")
    doses = allowed_doses
    output = results_root / "lesion_summary" / f"{model}.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    headers = ["lesion_count", *CONDITIONS]
    accuracy_rows = [[0, baseline, baseline, baseline]]
    delta_rows = [[0, 0.0, 0.0, 0.0]]
    for dose in doses:
        accuracy_rows.append([dose] + [
            float(results[(dose, condition)]["lesion_mean_utterance_accuracy"])
            if (dose, condition) in results else None for condition in CONDITIONS
        ])
        delta_rows.append([dose] + [
            float(results[(dose, condition)]["delta_percentage_points"])
            if (dose, condition) in results else None for condition in CONDITIONS
        ])
    accuracy = _sheet(workbook, "lesion_accuracy", headers, accuracy_rows)
    delta = _sheet(workbook, "delta_pp", headers, delta_rows)
    for row in accuracy.iter_rows(min_row=2, min_col=2):
        for cell in row:
            cell.number_format = "0.00%"
    for row in delta.iter_rows(min_row=2, min_col=2):
        for cell in row:
            cell.number_format = '+0.00;-0.00;0.00'
    quality = _sheet(workbook, "quality", ["condition", "lesion_count", "samples", "failures"], [
        [condition, dose, results[(dose, condition)]["samples"], results[(dose, condition)]["failures"]]
        for dose in doses for condition in conditions if (dose, condition) in results
    ])
    quality.column_dimensions["A"].width = 22
    meta = _sheet(workbook, "metadata", ["field", "value"], [
        ["model_key", model], ["planned_samples", int(plan["samples"])],
        ["maximum_equal_N", maximum],
        ["class_min_N", class_min],
        ["planned_conditions", ",".join(conditions)],
        ["planned_doses", ",".join(map(str, allowed_doses))],
        ["curve_doses", ",".join(map(str, doses))],
        ["supplemental_results", sum(int(row["dose"]) in allowed_doses for row in supplement_rows)],
        ["missing_conditions", len(missing)],
        ["accuracy_definition", "mean utterance max(0, 1-CER)"],
        ["delta_unit", "percentage points"],
    ])
    meta.column_dimensions["A"].width = 26
    meta.column_dimensions["B"].width = 52
    temporary = output.with_suffix(".tmp.xlsx")
    workbook.save(temporary)
    os.replace(temporary, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", help="One model; default: all model directories")
    selection.add_argument("--models", nargs="+", help="Selected models")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--supplement-results",
        type=Path,
        default=DEFAULT_SUPPLEMENT,
        help="Optional JSON containing validated non-grid result rows",
    )
    parser.add_argument("--allow-incomplete", action="store_true", help="Leave missing doses blank")
    args = parser.parse_args()
    supplement = load_supplement(args.supplement_results)
    for model in model_names(args.results_root / "model_details", [args.model] if args.model else args.models):
        print(summarize_model(
            args.results_root,
            model,
            args.allow_incomplete,
            supplement.get(model),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
