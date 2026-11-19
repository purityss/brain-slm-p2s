#!/usr/bin/env python3
"""Shared, resumable ASR benchmark loop for Mandarin CER and English WER."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import traceback
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "code_for_all_models_group/step04_manifests"

# Step04 benchmark utterances are substantially longer than the fixed Step01
# stimuli, so every model gets the same conservative decoding ceiling.  This
# changes only the output-length safety cap; prompts, model interfaces, and
# deterministic decoding parameters remain aligned with Step01 generate pass.
DEFAULT_MAX_NEW_TOKENS = 512


@dataclass(frozen=True)
class Sample:
    dataset: str
    subset: str
    sample_id: str
    audio_path: str
    reference: str

    @property
    def key(self) -> str:
        return f"{self.dataset}\t{self.subset}\t{self.sample_id}"


def normalize_for_cer(text: str) -> str:
    """NFKC/lowercase text and remove whitespace/punctuation for Mandarin CER."""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    kept: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character.isspace() or category.startswith(("P", "S", "C")):
            continue
        kept.append(character)
    return "".join(kept)


def normalize_for_wer(text: str) -> list[str]:
    """Return lowercase English word tokens for standard LibriSpeech WER."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def edit_counts(reference, hypothesis) -> tuple[int, int, int, int]:
    """Return Levenshtein substitutions, deletions, insertions and ref length."""
    ref, hyp = list(reference), list(hypothesis)
    # Each cell stores (cost, substitutions, deletions, insertions).
    previous = [(index, 0, index, 0) for index in range(len(ref) + 1)]
    for hyp_index, hyp_char in enumerate(hyp, start=1):
        current = [(hyp_index, 0, 0, hyp_index)]
        for ref_index, ref_char in enumerate(ref, start=1):
            if ref_char == hyp_char:
                candidates = [previous[ref_index - 1]]
            else:
                cost, sub, delete, insert = previous[ref_index - 1]
                candidates = [(cost + 1, sub + 1, delete, insert)]
            cost, sub, delete, insert = current[ref_index - 1]
            candidates.append((cost + 1, sub, delete + 1, insert))
            cost, sub, delete, insert = previous[ref_index]
            candidates.append((cost + 1, sub, delete, insert + 1))
            current.append(min(candidates))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions, len(ref)


def load_samples(
    manifest_root: Path,
    datasets: Iterable[str],
    subsets: set[str] | None,
    max_samples_per_subset: int | None,
) -> list[Sample]:
    samples: list[Sample] = []
    for dataset in datasets:
        manifest_path = manifest_root / f"{dataset}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Missing manifest "
                f"{manifest_path}. Run "
                "pipeline_code/step04_evaluate_performance/prepare_asr_manifests.py first."
            )
        subset_counts: dict[str, int] = defaultdict(int)
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample = Sample(**record)
                if subsets is not None and sample.subset not in subsets:
                    continue
                if not Path(sample.audio_path).is_file():
                    raise FileNotFoundError(
                        f"Manifest audio is missing ({manifest_path}:{line_number}): "
                        f"{sample.audio_path}"
                    )
                if (
                    max_samples_per_subset is not None
                    and subset_counts[sample.subset] >= max_samples_per_subset
                ):
                    continue
                subset_counts[sample.subset] += 1
                samples.append(sample)
    return samples


def _read_latest_predictions(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[str(record["key"])] = record
            except Exception as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return latest


def _aggregate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, int | float]] = defaultdict(
        lambda: {
            "samples": 0, "failures": 0, "S": 0, "D": 0, "I": 0, "N": 0,
            "utterance_accuracy_sum": 0.0,
        }
    )
    for record in records:
        key = (str(record["dataset"]), str(record["subset"]))
        group = groups[key]
        group["samples"] += 1
        if record.get("status") != "ok":
            group["failures"] += 1
            # A failed transcription is an empty hypothesis, not a sample that
            # disappears from the metric.  Count every normalized reference
            # character as a deletion so the utterance contributes CER=1 and
            # accuracy=0.  Deriving N from ``reference`` also repairs summaries
            # produced from legacy error records that stored S=D=I=N=0.
            reference_length = int(record.get("N", 0))
            if not reference_length:
                reference_length = (
                    len(normalize_for_wer(str(record.get("reference", ""))))
                    if str(record.get("dataset")) == "librispeech"
                    else len(normalize_for_cer(str(record.get("reference", ""))))
                )
            group["D"] += reference_length
            group["N"] += reference_length
            continue
        for field in ("S", "D", "I", "N"):
            group[field] += int(record[field])
        # Calculate and clip accuracy independently for every utterance, then
        # average the utterance scores. Do not derive accuracy from corpus CER.
        errors = int(record["S"]) + int(record["D"]) + int(record["I"])
        reference_length = int(record["N"])
        utterance_cer = (
            errors / reference_length
            if reference_length else (0.0 if errors == 0 else 1.0)
        )
        group["utterance_accuracy_sum"] += max(0.0, 1.0 - utterance_cer)

    rows: list[dict[str, Any]] = []
    for (dataset, subset), values in sorted(groups.items()):
        public_values = {
            field: int(values[field])
            for field in ("samples", "failures", "S", "D", "I", "N")
        }
        errors = public_values["S"] + public_values["D"] + public_values["I"]
        cer = (
            errors / public_values["N"]
            if public_values["N"] else (0.0 if errors == 0 else 1.0)
        )
        mean_utterance_accuracy = (
            float(values["utterance_accuracy_sum"]) / public_values["samples"]
            if public_values["samples"] else 0.0
        )
        metric = "WER" if dataset == "librispeech" else "CER"
        rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                **public_values,
                "metric": metric,
                "ErrorRate": cer,
                "CER": cer if metric == "CER" else "",
                "WER": cer if metric == "WER" else "",
                "mean_utterance_ASR_accuracy": mean_utterance_accuracy,
                # Backward-compatible name; the value follows the requested
                # macro-average definition above.
                "ASR_accuracy_1_minus_CER": mean_utterance_accuracy,
            }
        )

    return rows


def write_summary(predictions_path: Path, summary_path: Path) -> list[dict[str, Any]]:
    rows = _aggregate(_read_latest_predictions(predictions_path).values())
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "subset", "samples", "failures", "S", "D", "I", "N",
        "metric", "ErrorRate", "CER", "WER", "mean_utterance_ASR_accuracy",
        "ASR_accuracy_1_minus_CER",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser(model_key: str, model_name: str, default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate {model_name} on FLEURS-zh or WenetSpeech meeting."
    )
    parser.set_defaults(model_key=model_key, model_name=model_name)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument(
        "--datasets", nargs="+",
        choices=("fleurs_zh", "wenetspeech", "librispeech"),
        default=["fleurs_zh"],
    )
    parser.add_argument(
        "--subsets", nargs="+",
        choices=("test_meeting",), help="Only WenetSpeech test_meeting is retained.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--max-new-tokens", type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Step04 output-length ceiling (default: 512 for every model).",
    )
    parser.add_argument("--max-samples-per-subset", type=int)
    parser.add_argument(
        "--cuda-visible-devices", default="0",
        help="Physical GPU list exposed to the model process; default uses GPU 0 only.",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run_evaluation(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_samples_per_subset is not None and args.max_samples_per_subset <= 0:
        raise ValueError("--max-samples-per-subset must be positive")

    # Must be set before importing torch/transformers. Within each model process
    # the selected physical GPU is consequently addressed as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "dataset_summary.csv"
    config_path = output_dir / "run_config.json"
    if args.overwrite and predictions_path.exists():
        predictions_path.unlink()

    existing = _read_latest_predictions(predictions_path)
    samples = load_samples(
        args.manifest_root.resolve(), args.datasets,
        set(args.subsets) if args.subsets else None,
        args.max_samples_per_subset,
    )
    pending = [
        sample for sample in samples
        if sample.key not in existing
        or (args.retry_failures and existing[sample.key].get("status") != "ok")
    ]

    from .asr_model_adapters import create_adapter

    adapter = create_adapter(args.model_key, args)
    config = {
        "model_key": args.model_key,
        "model_name": args.model_name,
        "checkpoint": str(adapter.model_path),
        "prompt": adapter.prompt,
        "system_prompt": getattr(adapter, "system_prompt", None),
        "prompt_source": getattr(adapter, "prompt_source", None),
        "native_interface": getattr(adapter, "native_interface", None),
        "generation": adapter.generation_description,
        "datasets": args.datasets,
        "subsets": args.subsets,
        "max_new_tokens": args.max_new_tokens,
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "seed": args.seed,
        "normalization": "Unicode NFKC + lowercase + remove whitespace/punctuation/symbol/control",
        "metric": (
            "For each utterance i: CER_i=(S_i+D_i+I_i)/N_i and "
            "accuracy_i=max(0,1-CER_i); reported ASR accuracy is the arithmetic "
            "mean of accuracy_i across all utterances (failed utterances score 0). "
            "Corpus CER is retained separately as sum(errors)/sum(reference characters)."
        ),
        "sample_count": len(samples),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if not pending:
        print(
            f"Samples={len(samples)}; already complete={len(samples)}; pending=0; "
            "skipping model load",
            flush=True,
        )
        rows = write_summary(predictions_path, summary_path)
        for row in rows:
            print(
                f"{row['dataset']}/{row['subset']}: {row['metric']}={row['ErrorRate']:.6f}; "
                f"accuracy={row['ASR_accuracy_1_minus_CER']:.6f}; "
                f"failures={row['failures']}/{row['samples']}",
                flush=True,
            )
        if any(int(row["failures"]) for row in rows):
            raise SystemExit(1)
        return
    print(f"Loading {args.model_name} from {adapter.model_path}", flush=True)
    adapter.load()
    print(
        f"Samples={len(samples)}; already complete={len(samples)-len(pending)}; "
        f"pending={len(pending)}",
        flush=True,
    )

    started = time.time()
    with predictions_path.open("a", encoding="utf-8", buffering=1) as output:
        for index, sample in enumerate(pending, start=1):
            record: dict[str, Any] = {
                "key": sample.key,
                **asdict(sample),
                "model_key": args.model_key,
                "model_name": args.model_name,
            }
            sample_started = time.time()
            try:
                hypothesis = str(adapter.transcribe(Path(sample.audio_path))).strip()
                scoring_hypothesis = adapter.hypothesis_for_scoring(hypothesis)
                is_english = sample.dataset == "librispeech"
                normalized_reference = (
                    normalize_for_wer(sample.reference)
                    if is_english else normalize_for_cer(sample.reference)
                )
                normalized_hypothesis = (
                    normalize_for_wer(scoring_hypothesis)
                    if is_english else normalize_for_cer(scoring_hypothesis)
                )
                substitutions, deletions, insertions, reference_length = edit_counts(
                    normalized_reference, normalized_hypothesis
                )
                errors = substitutions + deletions + insertions
                sample_cer = (
                    errors / reference_length
                    if reference_length else (0.0 if errors == 0 else 1.0)
                )
                sample_accuracy = max(0.0, 1.0 - sample_cer)
                record.update(
                    status="ok", hypothesis=hypothesis,
                    scoring_hypothesis=scoring_hypothesis,
                    normalized_reference=(
                        " ".join(normalized_reference) if is_english else normalized_reference
                    ),
                    normalized_hypothesis=(
                        " ".join(normalized_hypothesis) if is_english else normalized_hypothesis
                    ),
                    S=substitutions, D=deletions, I=insertions, N=reference_length,
                    CER=sample_cer,
                    ASR_accuracy_1_minus_CER=sample_accuracy,
                )
            except Exception as error:
                is_english = sample.dataset == "librispeech"
                normalized_reference = (
                    normalize_for_wer(sample.reference)
                    if is_english else normalize_for_cer(sample.reference)
                )
                reference_length = len(normalized_reference)
                record.update(
                    status="error", hypothesis="",
                    normalized_reference=(
                        " ".join(normalized_reference) if is_english else normalized_reference
                    ),
                    normalized_hypothesis="",
                    S=0, D=reference_length, I=0, N=reference_length,
                    CER=1.0 if reference_length else 0.0,
                    ASR_accuracy_1_minus_CER=0.0,
                    error_type=type(error).__name__, error=str(error),
                    traceback=traceback.format_exc(limit=8),
                )
            record["elapsed_seconds"] = time.time() - sample_started
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(pending)}] {sample.dataset}/{sample.subset}/"
                f"{sample.sample_id}: {record['status']} "
                f"({record['elapsed_seconds']:.2f}s)",
                flush=True,
            )
            if record["status"] == "error" and args.fail_fast:
                raise RuntimeError(record["error"])

    rows = write_summary(predictions_path, summary_path)
    print(f"Completed in {(time.time()-started)/3600:.2f} h", flush=True)
    for row in rows:
        print(
            f"{row['dataset']}/{row['subset']}: {row['metric']}={row['ErrorRate']:.6f}; "
            f"accuracy={row['ASR_accuracy_1_minus_CER']:.6f}; "
            f"failures={row['failures']}/{row['samples']}",
            flush=True,
        )
    if any(int(row["failures"]) for row in rows):
        raise SystemExit(1)


def model_entrypoint(model_key: str, model_name: str, directory: str) -> None:
    default_output = (
        PROJECT_ROOT / "code_for_all_models_group" / "step04_model_performance"
        / "direct" / model_key
    )
    parser = build_parser(model_key, model_name, default_output)
    run_evaluation(parser.parse_args())


