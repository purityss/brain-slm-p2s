#!/usr/bin/env python3
# Shared Step04 manifest implementation for Mandarin and English ASR.
"""Prepare FLEURS-zh, WenetSpeech meeting, and LibriSpeech manifests."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path


DEFAULT_BENCHMARK_ROOT = Path(
    os.environ.get("SLM_BENCHMARK_ROOT", "/data/zss/SLMs/Benchmark")
).expanduser()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "code_for_all_models_group/step04_manifests"


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def fleurs_rows(root: Path):
    base = root / "Fleurs-zh"
    with (base / "data2Ftest.tsv").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 4:
                raise ValueError(f"Malformed FLEURS row {line_number}: {len(columns)} columns")
            filename = columns[1]
            yield {
                "dataset": "fleurs_zh", "subset": "test",
                "sample_id": Path(filename).stem,
                "audio_path": str((base / "test" / filename).resolve()),
                "reference": columns[2],
            }


def librispeech_rows(root: Path):
    base = root / "LibriSpeech" / "test-clean"
    transcripts = sorted(base.glob("*/*/*.trans.txt"))
    if not transcripts:
        raise FileNotFoundError(f"No LibriSpeech transcripts under {base}")
    for transcript in transcripts:
        with transcript.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                sample_id, separator, reference = line.rstrip("\n").partition(" ")
                if not separator or not reference:
                    raise ValueError(f"Malformed LibriSpeech row {transcript}:{line_number}")
                audio = transcript.parent / f"{sample_id}.flac"
                if not audio.is_file():
                    raise FileNotFoundError(audio)
                yield {
                    "dataset": "librispeech", "subset": "test-clean",
                    "sample_id": sample_id, "audio_path": str(audio.resolve()),
                    "reference": reference,
                }


def _arrow_batches(paths):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            try:
                reader = ipc.open_stream(source)
            except Exception:
                source.seek(0)
                reader = ipc.open_file(source)
            for batch_index in range(reader.num_record_batches) if hasattr(reader, "num_record_batches") else ():
                yield reader.get_batch(batch_index)
            if not hasattr(reader, "num_record_batches"):
                yield from reader


def wenet_rows(
    root: Path,
    output_root: Path,
    overwrite_audio: bool,
    subsets: set[str] | None = None,
):
    import numpy as np
    import soundfile as sf

    wenet_root = root / "WenetSpeech"
    audio_root = output_root / "wenetspeech_audio"
    for source_name, subset in (("TEST_MEETING", "test_meeting"),):
        if subsets is not None and subset not in subsets:
            continue
        arrow_paths = sorted((wenet_root / source_name / "dataset").glob("*.arrow"))
        if not arrow_paths:
            raise FileNotFoundError(f"No Arrow shards under {source_name}")
        subset_root = audio_root / subset
        subset_root.mkdir(parents=True, exist_ok=True)
        for batch in _arrow_batches(arrow_paths):
            for row in batch.to_pylist():
                sample_id = str(row["sid"])
                destination = subset_root / f"{sample_id}.wav"
                if overwrite_audio or not destination.is_file():
                    audio_record = row["audio"]
                    audio_bytes = audio_record.get("bytes") if audio_record else None
                    if not audio_bytes:
                        raise ValueError(f"WenetSpeech sample {sample_id} has no embedded audio")
                    samples, sample_rate = sf.read(
                        io.BytesIO(audio_bytes), dtype="float32", always_2d=False
                    )
                    if getattr(samples, "ndim", 1) == 2:
                        samples = np.asarray(samples).mean(axis=1)
                    sf.write(destination, samples, sample_rate, subtype="PCM_16")
                yield {
                    "dataset": "wenetspeech", "subset": subset,
                    "sample_id": sample_id, "audio_path": str(destination.resolve()),
                    "reference": str(row["text"]),
                }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=("fleurs_zh", "wenetspeech", "librispeech"), default=["fleurs_zh", "wenetspeech"])
    parser.add_argument(
        "--subsets", nargs="+",
        choices=("test_meeting",), help="Only test_meeting is retained in final Step04.",
    )
    parser.add_argument("--overwrite-audio", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    loaders = {
        "fleurs_zh": lambda: fleurs_rows(args.benchmark_root),
        "wenetspeech": lambda: wenet_rows(
            args.benchmark_root, args.output_root, args.overwrite_audio,
            set(args.subsets) if args.subsets else None,
        ),
        "librispeech": lambda: librispeech_rows(args.benchmark_root),
    }
    for dataset in args.datasets:
        count = write_jsonl(args.output_root / f"{dataset}.jsonl", loaders[dataset]())
        print(f"{dataset}: {count} samples", flush=True)


if __name__ == "__main__":
    main()



