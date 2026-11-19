#!/usr/bin/env python3
"""Construct adaptive native non-overlapping unit-wise cosine RDMs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The server's physical GPU 0 is exposed as cuda:0 inside this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP_DIR = PROJECT_ROOT / "code_for_all_models_group"
for path in (str(PROJECT_ROOT), str(GROUP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from step02_run_construct_adaptive_nonoverlap_unit_RDMs import run_model_entrypoint


if __name__ == "__main__":
    run_model_entrypoint("baichuan_audio_instruct")
