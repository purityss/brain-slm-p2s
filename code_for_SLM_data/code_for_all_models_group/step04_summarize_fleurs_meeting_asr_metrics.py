#!/usr/bin/env python3
"""Rebuild 14-model summaries and create the final FLEURS/meeting workbook."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.model_order import CANONICAL_MODEL_KEYS, validate_model_keys
DEFAULT_INPUT = SCRIPT_DIR / "step04_model_performance"
DEFAULT_OUTPUT = DEFAULT_INPUT / "fleurs_meeting_dataset_asr_metrics.xlsx"

DATASETS = (
    ("Fleurs-zh", "fleurs_zh", "fleurs_zh", "test", 945),
    ("WenetSpeech meeting", "wenetspeech_test_meeting", "wenetspeech", "test_meeting", 8370),
)
METRICS = (
    ("mean_utterance_ASR_accuracy", "mean_utterance_ASR_accuracy"),
    ("ASR_accuracy_1_minus_CER", "ASR_accuracy_1_minus_CER"),
)
# Canonical human-readable model-family order shared by every pipeline stage.
MODELS = (
    ("baichuan_audio_base", "Baichuan-Audio-Base"),
    ("baichuan_audio_instruct", "Baichuan-Audio-Instruct"),
    ("covo_audio_chat", "Covo-Audio-Chat"),
    ("firered_audio", "FireRedAudio"),
    ("freeze_omni", "Freeze-Omni"),
    ("glm_4_voice", "GLM-4-Voice"),
    ("kimi_audio_7b_instruct", "Kimi-Audio-7B-Instruct"),
    ("minicpm_o_2_6", "MiniCPM-o 2.6"),
    ("minicpm_o_4_5", "MiniCPM-o 4.5"),
    ("qwen_audio", "Qwen-Audio"),
    ("qwen_audio_chat", "Qwen-Audio-Chat"),
    ("qwen2_audio", "Qwen2-Audio"),
    ("qwen2_audio_instruct", "Qwen2-Audio-Instruct"),
    ("qwen2_5_omni", "Qwen2.5-Omni"),
)
validate_model_keys(key for key, _ in MODELS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-sample-count-check", action="store_true",
        help="Allow pilot/incomplete result folders; formal results must not use this.",
    )
    return parser.parse_args()


def rebuild_sources(input_root: Path, check_counts: bool):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline_code.step04_evaluate_performance.asr_benchmark_common import (
        _read_latest_predictions, write_summary,
    )

    values: dict[str, dict[str, dict[str, float]]] = {}
    for label, directory_name, dataset_name, subset_name, expected_count in DATASETS:
        dataset_root = input_root / directory_name
        combined_rows = []
        values[label] = {}
        for model_key, model_name in MODELS:
            model_root = dataset_root / model_key
            predictions = model_root / "predictions.jsonl"
            if not predictions.is_file():
                raise FileNotFoundError(f"Missing predictions: {predictions}")
            latest = _read_latest_predictions(predictions)
            if check_counts and len(latest) != expected_count:
                raise ValueError(
                    f"Incomplete {label}/{model_key}: {len(latest)}/{expected_count} unique samples"
                )
            rows = write_summary(predictions, model_root / "dataset_summary.csv")
            matching = [
                row for row in rows
                if row["dataset"] == dataset_name and row["subset"] == subset_name
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"Expected one summary row for {label}/{model_key}; found {len(matching)}"
                )
            row = matching[0]
            values[label][model_key] = {
                metric: float(row[metric]) for _, metric in METRICS
            }
            combined_rows.append({"model_key": model_key, "model": model_name, **row})

        combined_path = dataset_root / "all_models_asr_summary.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with combined_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
            writer.writeheader()
            writer.writerows(combined_rows)
    return values


def style_sheet(sheet):
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = f"A1:E{len(MODELS) + 1}"
    sheet.row_dimensions[1].height = 30
    header_fill = PatternFill("solid", fgColor="1F4E78")
    divider = Side(style="thin", color="D9E2F3")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in sheet.iter_rows(min_row=2, max_row=len(MODELS) + 1):
        for column_index, cell in enumerate(row, 1):
            cell.font = Font(name="Arial", size=10, color="000000")
            cell.border = Border(bottom=divider)
            cell.alignment = Alignment(
                horizontal="right" if column_index >= 3 else "left", vertical="center"
            )
            if column_index >= 3:
                cell.number_format = "0.00%"
    sheet.column_dimensions["A"].width = 28
    sheet.column_dimensions["B"].width = 27
    for column_index in range(3, 6):
        sheet.column_dimensions[get_column_letter(column_index)].width = 22


def write_workbook(output: Path, values):
    workbook = Workbook()
    workbook.remove(workbook.active)
    headers = ["Model", "ModelKey", "Fleurs-zh", "WenetSpeech meeting", "Mean"]
    for sheet_name, metric in METRICS:
        sheet = workbook.create_sheet(sheet_name)
        sheet.append(headers)
        for row_index, (model_key, model_name) in enumerate(MODELS, start=2):
            sheet.append([
                model_name,
                model_key,
                values["Fleurs-zh"][model_key][metric],
                values["WenetSpeech meeting"][model_key][metric],
                f"=AVERAGE(C{row_index}:D{row_index})",
            ])
        style_sheet(sheet)
    workbook.active = 0
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    workbook.calculation.calcMode = "auto"
    workbook.save(output)


def verify_workbook(output: Path):
    workbook = load_workbook(output, data_only=False, read_only=True)
    expected_sheets = [name for name, _ in METRICS]
    if workbook.sheetnames != expected_sheets:
        raise RuntimeError(f"Unexpected sheets: {workbook.sheetnames}")
    expected_headers = ["Model", "ModelKey", "Fleurs-zh", "WenetSpeech meeting", "Mean"]
    for sheet in workbook.worksheets:
        headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
        if headers != expected_headers:
            raise RuntimeError(f"Unexpected headers in {sheet.title}: {headers}")
        if sheet.max_row != 15 or sheet.max_column != 5:
            raise RuntimeError(
                f"Unexpected dimensions in {sheet.title}: {sheet.max_row}x{sheet.max_column}"
            )
        for row_index in range(2, 16):
            expected_model_key = MODELS[row_index - 2][0]
            if sheet.cell(row_index, 2).value != expected_model_key:
                raise RuntimeError(
                    f"Bad ModelKey order in {sheet.title}!B{row_index}: "
                    f"expected {expected_model_key!r}, got {sheet.cell(row_index, 2).value!r}"
                )
            if sheet.cell(row_index, 5).value != f"=AVERAGE(C{row_index}:D{row_index})":
                raise RuntimeError(f"Bad Mean formula in {sheet.title}!E{row_index}")
    workbook.close()


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".xlsx":
        raise ValueError("--output must end with .xlsx")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; rerun with --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    values = rebuild_sources(input_root, not args.skip_sample_count_check)
    write_workbook(output, values)
    verify_workbook(output)
    print(f"Saved: {output}")
    print("Models: 14; datasets: FLEURS-zh and WenetSpeech meeting")


if __name__ == "__main__":
    main()
