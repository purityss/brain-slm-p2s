#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.qwen_audio_extractor import run_qwen_audio_extraction  # noqa: E402

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/zss/SLMs/All_Models/Qwen-Audio-Chat")
    parser.add_argument("--audio-dir", default=str(PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "code_for_Qwen-Audio-Chat/01_extract_activation/Qwen_Audio_Chat_hidden_dimension_activations"))
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    run_qwen_audio_extraction(model_name="Qwen-Audio-Chat", file_prefix="Qwen_Audio_Chat", chat_mode=True, **vars(args))

if __name__ == "__main__": main()
