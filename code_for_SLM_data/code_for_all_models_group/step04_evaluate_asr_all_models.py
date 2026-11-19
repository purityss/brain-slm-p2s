#!/usr/bin/env python3
"""Step04: evaluate all 14 SLMs on one independently scheduled ASR job."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.model_order import CANONICAL_MODEL_KEYS, validate_model_keys


def conda_environment_python(environment_name: str) -> str:
    """Resolve a sibling Conda environment without hard-coding a user home."""
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
    environments_root = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    candidate = environments_root / environment_name / "bin/python"
    return str(candidate) if candidate.is_file() else sys.executable


def first_conda_environment_python(*environment_names: str) -> str:
    """Resolve the first installed environment from an ordered preference list."""
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
    environments_root = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    for environment_name in environment_names:
        candidate = environments_root / environment_name / "bin/python"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


PYTHON_SLM = conda_environment_python("py310_SLM")
PYTHON_KIMI = conda_environment_python("py310_kimi")
PYTHON_COVO = first_conda_environment_python("py311_covo", "py310_covo")
PYTHON_FIRERED = conda_environment_python("py310_fireredaudio")
PREPARE_PYTHON = os.environ.get("PYTHON_PREPARE_ASR", PYTHON_SLM)


def shell_join(command: list[str]) -> str:
    join = getattr(shlex, "join", None)
    return join(command) if join is not None else " ".join(map(shlex.quote, command))


def spec(name, directory, script, python_env, default_python=PYTHON_SLM):
    return {
        "name": name,
        "directory": directory,
        "script": script,
        "python": os.environ.get(python_env, default_python),
    }


MODELS = {
    "baichuan_audio_base": spec("Baichuan-Audio-Base", "code_for_Baichuan-Audio-Base", "evaluate_Baichuan_Audio_Base_asr.py", "PYTHON_BAICHUAN"),
    "baichuan_audio_instruct": spec("Baichuan-Audio-Instruct", "code_for_Baichuan-Audio-Instruct", "evaluate_Baichuan_Audio_Instruct_asr.py", "PYTHON_BAICHUAN"),
    "covo_audio_chat": spec("Covo-Audio-Chat", "code_for_Covo-Audio-Chat", "evaluate_Covo_Audio_Chat_asr.py", "PYTHON_COVO", PYTHON_COVO),
    "firered_audio": spec("FireRedAudio", "code_for_FireRedAudio", "evaluate_FireRedAudio_asr.py", "PYTHON_FIRERED", PYTHON_FIRERED),
    "freeze_omni": spec("Freeze-Omni", "code_for_Freeze-Omni", "evaluate_Freeze_Omni_asr.py", "PYTHON_FREEZE"),
    "glm_4_voice": spec("GLM-4-Voice", "code_for_GLM-4-Voice", "evaluate_GLM_4_Voice_asr.py", "PYTHON_GLM"),
    "kimi_audio_7b_instruct": spec("Kimi-Audio-7B-Instruct", "code_for_Kimi-Audio-7B-Instruct", "evaluate_Kimi_Audio_7B_Instruct_asr.py", "PYTHON_KIMI", PYTHON_KIMI),
    "minicpm_o_2_6": spec("MiniCPM-o 2.6", "code_for_MiniCPM-o 2.6", "evaluate_MiniCPM_o_2_6_asr.py", "PYTHON_MINICPM"),
    "minicpm_o_4_5": spec("MiniCPM-o 4.5", "code_for_MiniCPM-o 4.5", "evaluate_MiniCPM_o_4_5_asr.py", "PYTHON_MINICPM"),
    "qwen2_5_omni": spec("Qwen2.5-Omni", "code_for_Qwen2.5-Omni", "evaluate_Qwen2_5_Omni_asr.py", "PYTHON_QWEN25"),
    "qwen2_audio": spec("Qwen2-Audio", "code_for_Qwen2-Audio", "evaluate_Qwen2_Audio_asr.py", "PYTHON_QWEN"),
    "qwen2_audio_instruct": spec("Qwen2-Audio-Instruct", "code_for_Qwen2-Audio-Instruct", "evaluate_Qwen2_Audio_Instruct_asr.py", "PYTHON_QWEN"),
    "qwen_audio": spec("Qwen-Audio", "code_for_Qwen-Audio", "evaluate_Qwen_Audio_asr.py", "PYTHON_QWEN"),
    "qwen_audio_chat": spec("Qwen-Audio-Chat", "code_for_Qwen-Audio-Chat", "evaluate_Qwen_Audio_Chat_asr.py", "PYTHON_QWEN"),
}
validate_model_keys(MODELS)
MODEL_KEYS = CANONICAL_MODEL_KEYS


def entry_script(model: dict[str, str]) -> Path:
    return PROJECT_ROOT / model["directory"] / "04_evaluate_performance" / model["script"]


def output_dir(model_key: str, run_name: str) -> Path:
    return (
        PROJECT_ROOT / "code_for_all_models_group" / "step04_model_performance"
        / run_name / model_key
    )


def collect_summary(model_keys: list[str], run_name: str) -> None:
    output = (
        PROJECT_ROOT / "code_for_all_models_group" / "step04_model_performance"
        / run_name / "all_models_asr_summary.csv"
    )
    rows = []
    for key in model_keys:
        model = MODELS[key]
        path = output_dir(key, run_name) / "dataset_summary.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"model_key": key, "model": model["name"], **row})
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Combined summary: {output}")


def rebuild_summaries(model_keys: list[str], run_name: str) -> None:
    """Recalculate CSV summaries from saved per-utterance predictions only."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline_code.step04_evaluate_performance.asr_benchmark_common import (
        _read_latest_predictions,
        write_summary,
    )

    rebuilt: list[str] = []
    missing: list[str] = []
    incomplete: list[str] = []
    for key in model_keys:
        model = MODELS[key]
        directory = output_dir(key, run_name)
        predictions = directory / "predictions.jsonl"
        if not predictions.is_file():
            missing.append(key)
            print(f"MISSING predictions: {predictions}", file=sys.stderr)
            continue
        config_path = directory / "run_config.json"
        expected_samples = None
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_samples = int(config["sample_count"])
        completed_samples = len(_read_latest_predictions(predictions))
        if expected_samples is not None and completed_samples != expected_samples:
            incomplete.append(key)
            print(
                f"INCOMPLETE predictions for {model['name']}: "
                f"{completed_samples}/{expected_samples}; summary not rebuilt",
                file=sys.stderr,
            )
            continue
        rows = write_summary(predictions, directory / "dataset_summary.csv")
        rebuilt.append(key)
        for row in rows:
            print(
                f"REBUILT {model['name']} {row['dataset']}/{row['subset']}: "
                f"mean utterance accuracy={float(row['mean_utterance_ASR_accuracy']):.6f}"
            )
    collect_summary(rebuilt, run_name)
    print(f"Rebuilt summaries: {len(rebuilt)}/{len(model_keys)}")
    if missing or incomplete:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument(
        "--dataset", required=True,
        choices=("fleurs_zh", "wenetspeech"),
    )
    parser.add_argument("--subset", choices=("test_meeting",))
    parser.add_argument("--run-name", help="Output namespace; inferred from dataset/subset by default.")
    parser.add_argument(
        "--max-new-tokens", type=int,
        default=512,
        help="Output-length ceiling for every model (default: 512).",
    )
    parser.add_argument("--max-samples-per-subset", type=int)
    parser.add_argument("--gpu", default="0", help="Physical GPU id(s), default: 0")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--rebuild-summaries-only", action="store_true",
        help="Read existing predictions.jsonl files and rebuild CSV summaries without inference.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dataset == "wenetspeech" and args.subset is None:
        parser.error("--subset test_meeting is required for WenetSpeech jobs")
    if args.dataset != "wenetspeech" and args.subset is not None:
        parser.error("--subset is only valid with --dataset wenetspeech")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    run_name = args.run_name or (
        f"{args.dataset}_{args.subset}" if args.subset else args.dataset
    )
    if not all(character.isalnum() or character in "_-" for character in run_name):
        parser.error("--run-name may contain only letters, digits, '_' and '-'")

    if args.rebuild_summaries_only:
        rebuild_summaries(args.models, run_name)
        return

    manifest_root = (
        PROJECT_ROOT / "code_for_all_models_group" / "step04_manifests" / run_name
    )
    manifest_path = manifest_root / f"{args.dataset}.jsonl"
    if not args.skip_prepare:
        command = [
            str(PREPARE_PYTHON),
            str(PROJECT_ROOT / "pipeline_code/step04_evaluate_performance/prepare_asr_manifests.py"),
            "--output-root", str(manifest_root),
            "--datasets", args.dataset,
        ]
        if args.subset:
            command += ["--subsets", args.subset]
        print("PREPARE: " + shell_join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    elif not manifest_path.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Missing manifest with --skip-prepare: {manifest_path}")

    succeeded: list[str] = []
    failed: list[str] = []
    for index, key in enumerate(args.models, start=1):
        model = MODELS[key]
        command = [
            model["python"], str(entry_script(model)),
            "--manifest-root", str(manifest_root),
            "--datasets", args.dataset,
            "--output-dir", str(output_dir(key, run_name)),
            "--cuda-visible-devices", str(args.gpu),
        ]
        if args.subset:
            command += ["--subsets", args.subset]
        command += ["--max-new-tokens", str(args.max_new_tokens)]
        if args.max_samples_per_subset is not None:
            command += ["--max-samples-per-subset", str(args.max_samples_per_subset)]
        for flag in ("overwrite", "retry_failures", "fail_fast"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        print(f"\n[{index}/{len(args.models)}] {model['name']}")
        print(shell_join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            failed.append(key)
            print(f"FAILED {model['name']}: exit={result.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
        else:
            succeeded.append(key)
            # Include every existing model summary so a selected-model rerun
            # does not replace the combined table with a one-row partial file.
            collect_summary(list(MODEL_KEYS), run_name)
    if not args.dry_run:
        collect_summary(list(MODEL_KEYS), run_name)
        print(f"Successful models: {len(succeeded)}/{len(args.models)}")
        if failed:
            print(f"Failed model keys: {', '.join(failed)}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()

