#!/usr/bin/env python3
"""Run adaptive native non-overlap unit-wise RDM construction for active models."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA_ENVS = Path(os.environ.get("SLM_CONDA_ROOT", Path.home() / "anaconda3" / "envs"))
RUNTIME_PYTHONS = {
    "main": CONDA_ENVS / "py310_SLM" / "bin" / "python",
    "kimi": CONDA_ENVS / "py310_kimi" / "bin" / "python",
    "covo": CONDA_ENVS / "py310_covo" / "bin" / "python",
    "firered": CONDA_ENVS / "py310_fireredaudio" / "bin" / "python",
}


def _paths(directory: str, prefix: str) -> dict[str, Path]:
    model_root = PROJECT_ROOT / directory
    output_root = model_root / "02_construct_RDMs" / "adaptive_nonoverlap"
    return {
        "script": (
            model_root
            / "02_construct_RDMs"
            / f"construct_{prefix}_adaptive_nonoverlap_unit_RDMs.py"
        ),
        "input": (
            model_root
            / "01_extract_activation"
            / f"{prefix}_hidden_dimension_activations"
            / f"{prefix}_hidden_dimension_activations.npy"
        ),
        "output": output_root / f"{prefix}_adaptive_nonoverlap_unit_RDMs.mat",
        "summary": (
            output_root / f"{prefix}_adaptive_nonoverlap_unit_RDMs_summary.csv"
        ),
    }


def _model(
    key: str, model_name: str, directory: str, prefix: str, runtime: str = "main"
) -> tuple[str, dict[str, object]]:
    return key, {
        "model_name": model_name,
        "directory": directory,
        "prefix": prefix,
        "python": RUNTIME_PYTHONS[runtime],
        **_paths(directory, prefix),
    }


# This ordered mapping is the single source of truth for model, script, input,
# MAT output, and CSV output paths.
MODEL_CONFIGS = dict(
    [
        _model(
            "baichuan_audio_base",
            "Baichuan-Audio-Base",
            "code_for_Baichuan-Audio-Base",
            "Baichuan_Audio_Base",
        ),
        _model(
            "baichuan_audio_instruct",
            "Baichuan-Audio-Instruct",
            "code_for_Baichuan-Audio-Instruct",
            "Baichuan_Audio_Instruct",
        ),
        _model(
            "covo_audio_chat",
            "Covo-Audio-Chat",
            "code_for_Covo-Audio-Chat",
            "Covo_Audio_Chat",
            "covo",
        ),
        _model(
            "firered_audio",
            "FireRedAudio",
            "code_for_FireRedAudio",
            "FireRedAudio",
            "firered",
        ),
        _model(
            "freeze_omni",
            "Freeze-Omni",
            "code_for_Freeze-Omni",
            "Freeze_Omni",
        ),
        _model(
            "glm_4_voice",
            "GLM-4-Voice",
            "code_for_GLM-4-Voice",
            "GLM_4_Voice",
        ),
        _model(
            "kimi_audio_7b_instruct",
            "Kimi-Audio-7B-Instruct",
            "code_for_Kimi-Audio-7B-Instruct",
            "Kimi_Audio_7B_Instruct",
            "kimi",
        ),
        _model(
            "minicpm_o_2_6",
            "MiniCPM-o 2.6",
            "code_for_MiniCPM-o 2.6",
            "MiniCPM_o_2_6",
        ),
        _model(
            "minicpm_o_4_5",
            "MiniCPM-o 4.5",
            "code_for_MiniCPM-o 4.5",
            "MiniCPM_o_4_5",
        ),
        _model("qwen_audio", "Qwen-Audio", "code_for_Qwen-Audio", "Qwen_Audio"),
        _model(
            "qwen_audio_chat",
            "Qwen-Audio-Chat",
            "code_for_Qwen-Audio-Chat",
            "Qwen_Audio_Chat",
        ),
        _model("qwen2_audio", "Qwen2-Audio", "code_for_Qwen2-Audio", "Qwen2_Audio"),
        _model(
            "qwen2_audio_instruct",
            "Qwen2-Audio-Instruct",
            "code_for_Qwen2-Audio-Instruct",
            "Qwen2_Audio_Instruct",
        ),
        _model(
            "qwen2_5_omni",
            "Qwen2.5-Omni",
            "code_for_Qwen2.5-Omni",
            "Qwen2_5_Omni",
        ),
    ]
)


def model_config(model_key: str):
    if model_key not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model {model_key!r}; choices={list(MODEL_CONFIGS)}"
        )
    spec = MODEL_CONFIGS[model_key]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline_code.step02_construct_rdms.rdm_construction_common import ModelConfig

    return ModelConfig(
        model_name=str(spec["model_name"]),
        default_input=str(spec["input"]),
        default_output=str(spec["output"]),
    )


def run_model_entrypoint(model_key: str) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline_code.step02_construct_rdms.rdm_construction_common import run_adaptive_nonoverlap

    run_adaptive_nonoverlap(model_config(model_key))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help="One or more model keys; default: all declared models in order.",
    )
    parser.add_argument("--unit-chunk", type=int, default=128)
    parser.add_argument(
        "--compression", choices=("lzf", "gzip", "none"), default="lzf"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.unit_chunk <= 0:
        raise ValueError("--unit-chunk must be positive")
    # Make a multi-model rebuild restart-safe.  Explicit --overwrite first
    # removes every selected stale output before any model starts.  Children
    # then run in resumable mode, so a later global --resume can safely skip
    # completed models and create models that had not started yet.
    if args.overwrite:
        print("Preparing clean rebuild for selected models:", flush=True)
        for model_key in args.models:
            spec = MODEL_CONFIGS[model_key]
            for artifact_key in ("output", "summary"):
                artifact = Path(spec[artifact_key])
                if artifact.exists():
                    artifact.unlink()
                    print(f"  removed: {artifact}", flush=True)
    succeeded = []
    for model_key in args.models:
        spec = MODEL_CONFIGS[model_key]
        script = Path(spec["script"])
        input_path = Path(spec["input"])
        runtime_python = Path(spec["python"])
        if not script.is_file():
            raise FileNotFoundError(f"Missing model entry script: {script}")
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing formal activation NPY: {input_path}")
        if not runtime_python.is_file():
            raise FileNotFoundError(
                f"Missing runtime Python for {spec['model_name']}: {runtime_python}"
            )
        command = [
            str(runtime_python),
            str(script),
            "--unit-chunk",
            str(args.unit_chunk),
            "--compression",
            args.compression,
        ]
        if args.dry_run:
            command.append("--dry-run")
        elif args.resume:
            command.append("--resume")
        elif args.overwrite:
            command.append("--resume")
        print("=" * 80, flush=True)
        print(
            f"MODEL {len(succeeded) + 1}/{len(args.models)}: "
            f"{spec['model_name']} ({model_key})",
            flush=True,
        )
        print("COMMAND: " + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Model failed: {spec['model_name']} ({model_key}); "
                f"exit={result.returncode}; command={shlex.join(command)}"
            )
        if not args.dry_run:
            missing = [
                str(path)
                for path in (Path(spec["output"]), Path(spec["summary"]))
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Model command returned success but outputs are missing: {missing}"
                )
        succeeded.append(model_key)
        print(f"SUCCESS: {spec['model_name']}", flush=True)

    print("=" * 80)
    print(f"Successful models: {len(succeeded)}/{len(args.models)}")
    for model_key in succeeded:
        spec = MODEL_CONFIGS[model_key]
        if args.dry_run:
            print(f"  {spec['model_name']}: dry-run OK; output not created")
        else:
            print(f"  {spec['model_name']}: {spec['output']}")
            print(f"    summary: {spec['summary']}")


if __name__ == "__main__":
    main()
