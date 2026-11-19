#!/usr/bin/env python3
"""Run the shared Baichuan-Audio extractor with the Instruct checkpoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt  # noqa: E402

BASE_SCRIPT = (
    PROJECT_ROOT
    / "code_for_Baichuan-Audio-Base/01_extract_activation"
    / "extract_Baichuan_Audio_Base_activations_revised.py"
)

spec = importlib.util.spec_from_file_location(
    "baichuan_shared_extractor", BASE_SCRIPT
)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load shared Baichuan extractor: {BASE_SCRIPT}")
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

extractor.MODEL_NAME = "Baichuan-Audio-Instruct"
extractor.FILE_PREFIX = "Baichuan_Audio_Instruct"
extractor.MODEL_PATH = "/data/zss/SLMs/All_Models/Baichuan-Audio-Instruct"
extractor.OUTPUT_DIR = str(
    PROJECT_ROOT
    / "code_for_Baichuan-Audio-Instruct/01_extract_activation/Baichuan_Audio_Instruct_hidden_dimension_activations"
)
extractor.CHECKPOINT_VARIANT = "Instruct"
extractor.TASK_PROMPT = final_prompt("baichuan_audio_instruct")


if __name__ == "__main__":
    extractor.main()
