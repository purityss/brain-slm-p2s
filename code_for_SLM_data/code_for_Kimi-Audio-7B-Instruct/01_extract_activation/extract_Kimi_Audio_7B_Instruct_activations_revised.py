#!/usr/bin/env python3
"""Run the shared Kimi extractor with the Instruct checkpoint contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = Path(__file__).resolve().parent / "kimi_audio_extractor.py"

spec = importlib.util.spec_from_file_location("kimi_shared_extractor", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load shared Kimi extractor: {BASE_SCRIPT}")
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

extractor.MODEL_NAME = "Kimi-Audio-7B-Instruct"
extractor.FILE_PREFIX = "Kimi_Audio_7B_Instruct"
extractor.MODEL_PATH = "/data/zss/SLMs/All_Models/Kimi-Audio-7B-Instruct"
extractor.OUTPUT_DIR = str(
    PROJECT_ROOT
    / "code_for_Kimi-Audio-7B-Instruct/01_extract_activation/Kimi_Audio_7B_Instruct_hidden_dimension_activations"
)
extractor.CHECKPOINT_VARIANT = "Instruct"
extractor.COMPARABILITY_NOTE = (
    "Instruction-tuned checkpoint; official deterministic text-only ASR path"
)


if __name__ == "__main__":
    extractor.main()
