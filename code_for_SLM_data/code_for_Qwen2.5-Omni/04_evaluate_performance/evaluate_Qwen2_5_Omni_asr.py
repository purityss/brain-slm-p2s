#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_code.step04_evaluate_performance.asr_benchmark_common import model_entrypoint
if __name__ == "__main__":
    model_entrypoint("qwen2_5_omni", "Qwen2.5-Omni", "code_for_Qwen2.5-Omni")
