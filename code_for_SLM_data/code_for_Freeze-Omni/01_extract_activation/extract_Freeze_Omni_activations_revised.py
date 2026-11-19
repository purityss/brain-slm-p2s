#!/usr/bin/env python3
"""Extract source-audio-indexed hidden dimensions from streaming Freeze-Omni.

Each stimulus uses independent behavior and activation passes. The activation
pass ends after the final audio chunk, before chat suffix or generated tokens.
Cached overlap is contextual input and is never saved as a new position.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_code.step01_extract_activation.activation_extraction_common import (  # noqa: E402
    LayerSpec, first_tensor, save_final_outputs, selected_stimuli,
    validate_activation_set, validate_audio_files,
)
from pipeline_code.step01_extract_activation.final_asr_prompts import final_prompt  # noqa: E402

MODEL_NAME = "Freeze-Omni"
FILE_PREFIX = "Freeze_Omni"
REFERENCE_ROOT = PROJECT_ROOT / "code_for_Freeze-Omni/model_runtime"
MODEL_PATH = "/data/zss/SLMs/All_Models/Freeze-omni/checkpoints"
LLM_PATH = "/data/zss/SLMs/All_Models/qwen2-7b-instruct"
AUDIO_DIR = PROJECT_ROOT / "0_Stimuli/WordVocoder(160916)"
OUTPUT_DIR = PROJECT_ROOT / "code_for_Freeze-Omni/01_extract_activation/Freeze_Omni_hidden_dimension_activations"
TASK_PROMPT = final_prompt("freeze_omni")


class Config:
    def __init__(self, model_path: Path, llm_path: Path):
        self.model_path = str(model_path)
        self.llm_path = str(llm_path)
        self.top_k = 1
        self.top_p = 1.0
        self.temperature = 1.0


class AudioEncoderProcessor:
    """Native 16-new-frame frontend with three cached left-context frames."""

    def __init__(self):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_shift = 160
        self.frame_overlap = 240
        self.samples_per_chunk = self.frame_shift * self.chunk_size
        self.reset()

    def reset(self):
        self.input_chunk = torch.zeros(
            1, self.chunk_size + self.chunk_overlap, self.feat_dim
        )
        self.input_sample = torch.zeros(
            1, self.samples_per_chunk + self.frame_overlap, 1
        )

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape != (self.samples_per_chunk,):
            raise ValueError(f"Invalid audio chunk shape: {audio.shape}")
        sample_data = audio.reshape(1, -1, 1) * 32768
        self.input_sample[:, :self.frame_overlap] = self.input_sample[
            :, -self.frame_overlap:
        ].clone()
        self.input_sample[:, self.frame_overlap:] = sample_data
        features = kaldi.fbank(
            waveform=self.input_sample.squeeze(-1), dither=0,
            frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim,
        )
        if features.shape[0] != self.chunk_size:
            raise RuntimeError(f"Expected 16 new fbank frames, got {features.shape}")
        self.input_chunk[:, :self.chunk_overlap] = self.input_chunk[
            :, -self.chunk_overlap:
        ].clone()
        self.input_chunk[:, self.chunk_overlap:] = features.unsqueeze(0)
        return self.input_chunk.clone()


def load_waveform(path: Path, samples_per_chunk: int):
    waveform, sample_rate = sf.read(path)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    original_samples = int(waveform.numel())
    if original_samples < 1:
        raise ValueError(f"Empty waveform: {path}")
    padded_length = math.ceil(original_samples / samples_per_chunk) * samples_per_chunk
    padded = torch.zeros(padded_length, dtype=torch.float32)
    padded[:original_samples] = waveform
    return padded, original_samples


def chunk_plan(padded, original_samples, samples_per_chunk):
    for chunk_index, offset in enumerate(range(0, padded.numel(), samples_per_chunk)):
        valid_samples = max(0, min(samples_per_chunk, original_samples - offset))
        valid_fbank = max(1, math.ceil(valid_samples / 160))
        yield chunk_index, padded[offset:offset + samples_per_chunk], valid_fbank


def initialize_stream(pipeline):
    outputs = pipeline.speech_dialogue(None, stat="pre", role=TASK_PROMPT)
    if outputs["stat"] != "sl":
        raise RuntimeError(f"Unexpected state after system role: {outputs['stat']}")
    return outputs


def run_behavior_pass(pipeline, padded, original_samples, max_new_tokens):
    processor = AudioEncoderProcessor()
    outputs = initialize_stream(pipeline)
    for _, samples, _ in chunk_plan(
        padded, original_samples, processor.samples_per_chunk
    ):
        outputs = pipeline.speech_dialogue(processor.process(samples), **outputs)
        outputs["stat"] = "cl"
    processor.reset()
    outputs.update(adapter_cache=None, encoder_cache=None, pe_index=0, stat="ss")
    outputs = pipeline.speech_dialogue(None, **outputs)
    latest_text = str(outputs.get("text", ""))
    while outputs["stat"] == "cs" and len(outputs.get("past_tokens", [])) < max_new_tokens:
        outputs.pop("text", None)
        outputs.pop("hidden_state", None)
        outputs = pipeline.speech_dialogue(None, **outputs)
        if "text" in outputs:
            latest_text = str(outputs["text"])
    return latest_text.strip()


def run_activation_pass(
    pipeline, specs, conv1, conv2, audio_layers, projector, decoder_layers,
    padded, original_samples,
):
    """Capture and validate only current source-audio positions."""
    processor = AudioEncoderProcessor()
    outputs = initialize_stream(pipeline)
    captured = {spec.name: [] for spec in specs}
    phase = dict(
        active=False, chunk_index=-1, valid_fbank=0, valid_encoder=0,
        valid_projected=0, actual_projected=0,
    )
    decoder_current_lengths = {}
    audits = []

    def conv_hook(name, temporal_stride, new_start):
        def hook(_module, _inputs, output):
            if not phase["active"]:
                return
            valid_time = max(1, math.ceil(phase["valid_fbank"] / temporal_stride))
            stop = new_start + valid_time
            if output.ndim != 4 or stop > output.shape[2]:
                raise RuntimeError(f"Invalid {name} crop for {output.shape}")
            value = output[0, :, new_start:stop, :]
            value = value.permute(1, 2, 0).reshape(-1, output.shape[1])
            captured[name].append(value.detach().float().cpu())
        return hook

    def projector_hook(_module, _inputs, output):
        if not phase["active"]:
            return
        value = first_tensor(output)
        if value.ndim != 3 or value.shape[0] != 1:
            raise RuntimeError(f"Invalid projector output: {value.shape}")
        actual = int(value.shape[1])
        valid = min(max(1, math.ceil(phase["valid_fbank"] / 8)), actual)
        phase["actual_projected"] = actual
        phase["valid_projected"] = valid
        captured["multimodal_projector"].append(
            value[0, :valid].detach().float().cpu()
        )

    def decoder_hook(name):
        def hook(_module, _inputs, output):
            if not phase["active"]:
                return
            value = first_tensor(output)
            actual_audio = int(phase["actual_projected"])
            valid_audio = int(phase["valid_projected"])
            if value.ndim != 3 or actual_audio < 1 or valid_audio < 1:
                raise RuntimeError(f"Invalid {name} output or projector lengths")
            current_length = int(value.shape[1])
            audio_start = current_length - actual_audio
            if audio_start < 0:
                raise RuntimeError(f"{name}: current length is shorter than audio")
            if phase["chunk_index"] > 0 and audio_start != 0:
                raise RuntimeError(
                    f"{name}: continuation chunk contains {audio_start} non-audio positions"
                )
            selected = value[0, audio_start:audio_start + valid_audio]
            if selected.shape[0] != valid_audio:
                raise RuntimeError(f"Failed to select audio positions from {name}")
            captured[name].append(selected.detach().float().cpu())
            decoder_current_lengths[name] = current_length
        return hook

    def wrap_encoder_infer(name, original_infer):
        def wrapped(*args, **kwargs):
            result = original_infer(*args, **kwargs)
            if phase["active"]:
                value = result[0]
                valid = int(phase["valid_encoder"])
                if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < valid:
                    raise RuntimeError(
                        f"Invalid {name} output {value.shape}; valid={valid}"
                    )
                captured[name].append(value[0, :valid].detach().float().cpu())
            return result
        return wrapped

    handles = [
        # Conv1 position 0 is formed exclusively from cached overlap.
        conv1.register_forward_hook(conv_hook("audio_conv1", 2, 1)),
        conv2.register_forward_hook(conv_hook("audio_conv2", 4, 0)),
        projector.register_forward_hook(projector_hook),
    ]
    handles.extend(
        layer.register_forward_hook(decoder_hook(f"decoder_block_{index:02d}"))
        for index, layer in enumerate(decoder_layers)
    )
    original_encoder_infers = [layer.infer for layer in audio_layers]
    for index, (layer, original_infer) in enumerate(
        zip(audio_layers, original_encoder_infers)
    ):
        layer.infer = wrap_encoder_infer(
            f"audio_block_{index:02d}", original_infer
        )

    try:
        for chunk_index, samples, valid_fbank in chunk_plan(
            padded, original_samples, processor.samples_per_chunk
        ):
            counts_before = {name: len(values) for name, values in captured.items()}
            decoder_current_lengths.clear()
            phase.update(
                active=True, chunk_index=chunk_index, valid_fbank=valid_fbank,
                valid_encoder=max(1, math.ceil(valid_fbank / 4)),
                valid_projected=0, actual_projected=0,
            )
            outputs = pipeline.speech_dialogue(processor.process(samples), **outputs)
            phase["active"] = False
            outputs["stat"] = "cl"
            missing = [
                name for name, values in captured.items()
                if len(values) != counts_before[name] + 1
            ]
            if missing:
                raise RuntimeError(f"Chunk {chunk_index} missing captures: {missing}")
            decoder_lengths = set(decoder_current_lengths.values())
            if len(decoder_lengths) != 1:
                raise RuntimeError(
                    f"Decoder layer lengths disagree: {decoder_current_lengths}"
                )
            audits.append(dict(
                valid_fbank=int(valid_fbank),
                valid_encoder=int(phase["valid_encoder"]),
                actual_projected=int(phase["actual_projected"]),
                valid_projected=int(phase["valid_projected"]),
                decoder_current=decoder_lengths.pop(),
            ))
    finally:
        phase["active"] = False
        for layer, original_infer in zip(audio_layers, original_encoder_infers):
            layer.infer = original_infer
        for handle in handles:
            handle.remove()
        processor.reset()

    matrices = {}
    for spec in specs:
        if not captured[spec.name]:
            raise RuntimeError(f"No activation captured for {spec.name}")
        matrices[spec.name] = torch.cat(captured[spec.name], dim=0).numpy()
    projected_length = matrices["multimodal_projector"].shape[0]
    if projected_length != sum(item["valid_projected"] for item in audits):
        raise RuntimeError("Projector reconstruction length mismatch")
    for index in range(len(decoder_layers)):
        name = f"decoder_block_{index:02d}"
        if matrices[name].shape[0] != projected_length:
            raise RuntimeError(
                f"{name} length {matrices[name].shape[0]} != {projected_length}"
            )
    validate_activation_set(matrices, specs)
    return matrices, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--llm-path", type=Path, default=LLM_PATH)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-stimuli", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    reference_root = args.reference_root.resolve()
    model_path = args.model_path.resolve()
    llm_path = args.llm_path.resolve()
    audio_dir = args.audio_dir.resolve()
    output_dir = args.output_dir.resolve()
    for required in (reference_root, model_path, llm_path):
        if not required.exists():
            raise FileNotFoundError(required)
    stimuli = selected_stimuli(args.max_stimuli)
    validate_audio_files(audio_dir, stimuli)

    sys.path.insert(0, str(reference_root))
    from models.pipeline import inferencePipeline  # type: ignore

    pipeline = inferencePipeline(Config(model_path, llm_path))
    model = pipeline.model.eval()
    conv1 = model.encoder.enc[0].core.conv[0]
    conv2 = model.encoder.enc[0].core.conv[2]
    audio_layers = list(model.encoder.enc[1].encoders)
    projector = model.adpter.project
    decoder_layers = list(model.llm_decoder.model.layers)
    audio_dim = int(model.encoder.enc[1].attention_dim)
    decoder_dim = int(model.llm_decoder.config.hidden_size)
    specs = [
        LayerSpec("audio_conv1", "audio_native_time_frequency", int(conv1.out_channels)),
        LayerSpec("audio_conv2", "audio_native_time_frequency", int(conv2.out_channels)),
    ]
    specs.extend(
        LayerSpec(f"audio_block_{i:02d}", "audio_native", audio_dim)
        for i in range(len(audio_layers))
    )
    specs.append(
        LayerSpec("multimodal_projector", "audio_projected", int(projector.out_features))
    )
    specs.extend(
        LayerSpec(
            f"decoder_block_{i:02d}", "decoder_streaming_audio_position", decoder_dim
        )
        for i in range(len(decoder_layers))
    )

    all_activations, transcriptions, chunk_counts = [], [], []
    samples_per_chunk = AudioEncoderProcessor().samples_per_chunk
    for stimulus_index, (audio_file, _) in enumerate(stimuli, 1):
        padded, original_samples = load_waveform(
            audio_dir / audio_file, samples_per_chunk
        )
        with torch.inference_mode():
            transcription = run_behavior_pass(
                pipeline, padded, original_samples, args.max_new_tokens
            )
            activations, audits = run_activation_pass(
                pipeline, specs, conv1, conv2, audio_layers, projector,
                decoder_layers, padded, original_samples,
            )
        all_activations.append(activations)
        transcriptions.append(transcription)
        chunk_counts.append(len(audits))
        print(
            f"[{stimulus_index:02d}/{len(stimuli)}] {audio_file}: "
            f"{transcription!r}; chunks={len(audits)}; "
            f"conv={activations['audio_conv1'].shape}/{activations['audio_conv2'].shape}; "
            f"encoder={activations['audio_block_00'].shape}; "
            f"projector={activations['multimodal_projector'].shape}; "
            f"decoder={activations['decoder_block_00'].shape}",
            flush=True,
        )

    result = save_final_outputs(
        model_name=MODEL_NAME, file_prefix=FILE_PREFIX, output_dir=output_dir,
        stimuli=stimuli, specs=specs, all_activations=all_activations,
        transcriptions=transcriptions,
        summary_rows=[
            ("model_class", type(model).__name__),
            ("model_checkpoint", str(model_path)),
            ("llm_checkpoint", str(llm_path)),
            ("parameter_count", str(sum(p.numel() for p in model.parameters()))),
            ("architecture_family", "streaming 2-D CNN + speech Transformer + subsampling adapter + Qwen2 LLM"),
            ("prompt", TASK_PROMPT),
            ("audio_encoder_layer_count", str(len(audio_layers))),
            ("decoder_layer_count", str(len(decoder_layers))),
            ("generation_mode", f"independent native behavior pass; deterministic top_k=1; max_new_tokens={args.max_new_tokens}"),
            ("activation_pass", "independent streaming audio prefill only; stopped before chat suffix and generated text"),
            ("streaming_chunk_new_fbank_frames", "16"),
            ("streaming_left_context_fbank_frames", "3; context only; never duplicated"),
            ("streaming_chunk_count_range", f"min={min(chunk_counts)}; max={max(chunk_counts)}"),
            ("cnn_sequence_definition", "valid new time-frequency positions; channel is unit axis"),
            ("audio_encoder_sequence_definition", "valid new acoustic positions concatenated across chunks"),
            ("projector_sequence_definition", "valid new adapter audio positions concatenated across chunks"),
            ("decoder_sequence_definition", "Qwen2 states at source-audio positions only; system/chat prefix, suffix, and generated text excluded"),
            ("decoder_audio_axis_validation", "passed: all decoder layers and projector have identical source-audio lengths"),
            ("padding_policy", "final padded chunk cropped at every extracted stage using original waveform length"),
        ],
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
