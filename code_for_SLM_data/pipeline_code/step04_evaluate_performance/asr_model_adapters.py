#!/usr/bin/env python3
"""Lazy model adapters matching the prompts and deterministic generate paths in 01_extract_activation."""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import re
import shutil
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline_code.step01_extract_activation.final_asr_prompts import (
    QWEN2_AUDIO_INSTRUCT_SYSTEM_PROMPT,
    QWEN25_OMNI_SYSTEM_PROMPT,
    QWEN_AUDIO_CONTROL_PROMPT,
    final_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(
    os.environ.get("SLM_MODEL_ROOT", "/data/zss/SLMs/All_Models")
).expanduser().resolve()


def first_existing_path(*candidates: Path) -> Path:
    """Return the first existing portable layout, or the first candidate."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class Adapter:
    model_name = ""
    default_model_path = ""
    prompt = ""
    generation_description = ""
    native_interface = "natural-language task prompt"
    prompt_source = (
        "pipeline_code/step01_extract_activation/final_asr_prompts.py "
        "(from prompt_excel_tables_final/01_Final_prompts_14_models.xlsx)"
    )

    def __init__(self, args):
        self.args = args
        self.model_path = Path(args.model_path or self.default_model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

    def load(self) -> None:
        raise NotImplementedError

    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError

    def hypothesis_for_scoring(self, hypothesis: str) -> str:
        """Return transcript content used for ASR scoring while preserving raw output."""
        return hypothesis


class Qwen2AudioAdapter(Adapter):
    generation_description = "deterministic 5-beam generate; do_sample=False; repetition_penalty=1.0"

    def __init__(self, args, instruct: bool):
        self.instruct = instruct
        self.model_name = "Qwen2-Audio-Instruct" if instruct else "Qwen2-Audio"
        self.default_model_path = (
            str(MODEL_ROOT / "Qwen2-Audio-7B-Instruct")
            if instruct else str(MODEL_ROOT / "Qwen2-Audio-7B")
        )
        self.prompt = final_prompt(
            "qwen2_audio_instruct" if instruct else "qwen2_audio"
        )
        super().__init__(args)

    def load(self):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, device_map="auto", torch_dtype="auto",
            low_cpu_mem_usage=True,
        ).eval()

    def transcribe(self, audio_path):
        import librosa
        import torch
        sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
        if self.instruct:
            conversation = [
                {"role": "system", "content": QWEN2_AUDIO_INSTRUCT_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": self.prompt},
                ]},
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
        else:
            text = (
                "<|audio_bos|><|AUDIO|><|audio_eos|>"
                + self.prompt
            )
        inputs = self.processor(
            text=text, audio=[audio] if self.instruct else audio,
            sampling_rate=sampling_rate, return_tensors="pt", padding=True,
            return_attention_mask=True,
        )
        prompt_length = int(inputs["input_ids"].shape[1])
        inputs = inputs.to(next(self.model.parameters()).device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False,
                num_beams=5, early_stopping=True, repetition_penalty=1.0, use_cache=True,
            )
        return self.processor.batch_decode(
            output[:, prompt_length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def hypothesis_for_scoring(self, hypothesis: str) -> str:
        # The official Qwen2-Audio-Instruct example prompt commonly answers as
        # "The person says '...'.".  The quoted span is the model's transcript;
        # retain the complete response in predictions.jsonl for auditability,
        # but exclude the official response wrapper from ASR scoring.
        if self.instruct and self.prompt.strip() == "What does the person say?":
            for pattern in (r"['‘]([^'’]+)['’]", r'[“\"]([^”\"]+)[”\"]'):
                match = re.search(pattern, hypothesis)
                if match:
                    return match.group(1).strip()
        if not self.instruct:
            # Some Base completions prepend the answer-field marker "A:".
            # It is formatting rather than transcript content.
            return re.sub(r"^\s*A\s*:\s*", "", hypothesis, flags=re.IGNORECASE).strip()
        return hypothesis


class QwenAudioLegacyAdapter(Adapter):
    control_prompt = QWEN_AUDIO_CONTROL_PROMPT
    generation_description = "native deterministic generation; do_sample=False; num_beams=1"

    def __init__(self, args, chat_mode: bool):
        self.chat_mode = chat_mode
        self.model_name = "Qwen-Audio-Chat" if chat_mode else "Qwen-Audio"
        self.default_model_path = (
            str(MODEL_ROOT / "Qwen-Audio-Chat")
            if chat_mode else str(MODEL_ROOT / "Qwen-audio")
        )
        self.prompt = (
            final_prompt("qwen_audio_chat") if chat_mode else self.control_prompt
        )
        self.native_interface = (
            "Qwen-Audio chat with official task prompt"
            if chat_mode else "Qwen-Audio native ASR control tokens"
        )
        super().__init__(args)

    def load(self):
        os.environ["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        if shutil.which("ffmpeg") is None:
            # Qwen's released tokenizer shells out to ffmpeg even for ordinary
            # WAV input.  Cluster compute nodes do not necessarily expose that
            # binary, so use the project's pinned librosa decoder for these
            # local benchmark files while preserving the model preprocessing.
            import librosa

            tokenizer_module = sys.modules[self.tokenizer.__class__.__module__]

            def local_load_audio(audio_path):
                audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                return audio

            tokenizer_module.load_audio = local_load_audio
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True,
            device_map="cuda:0", bf16=True, fp16=False, fp32=False,
        ).eval()

    def transcribe(self, audio_path):
        import torch
        if self.chat_mode:
            query = self.tokenizer.from_list_format(
                [{"audio": str(audio_path)}, {"text": self.prompt}]
            )
            with torch.inference_mode():
                text, _ = self.model.chat(
                    self.tokenizer, query=query, history=None, do_sample=False,
                    num_beams=1, max_new_tokens=self.args.max_new_tokens,
                    repetition_penalty=1.0, use_cache=True,
                )
            chinese_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", str(text))
            # Preserve all clauses in long transcripts. Punctuation splits a
            # valid meeting transcript into multiple Chinese runs; selecting
            # only the final run caused large artificial deletion counts.
            return ("".join(chinese_runs) if chinese_runs else str(text)).strip()
        query = f"<audio>{audio_path}</audio>{self.control_prompt}"
        audio_info = self.tokenizer.process_audio(query)
        inputs = self.tokenizer(query, return_tensors="pt", audio_info=audio_info).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, audio_info=audio_info, do_sample=False, num_beams=1,
                max_new_tokens=self.args.max_new_tokens, repetition_penalty=1.0, use_cache=True,
            )
        return self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1]:].cpu(),
            skip_special_tokens=True, audio_info=audio_info,
        ).strip()


class BaichuanAudioAdapter(Adapter):
    generation_description = "deterministic 5-beam generate; do_sample=False; repetition_penalty=1.1"

    def __init__(self, args, instruct: bool):
        self.instruct = instruct
        self.model_name = "Baichuan-Audio-Instruct" if instruct else "Baichuan-Audio-Base"
        self.default_model_path = (
            str(MODEL_ROOT / "Baichuan-Audio-Instruct")
            if instruct else str(MODEL_ROOT / "Baichuan-Audio-Base")
        )
        self.prompt = (
            final_prompt("baichuan_audio_instruct")
            if instruct else final_prompt("baichuan_audio_base")
        )
        super().__init__(args)

    def load(self):
        import torch
        compatibility = PROJECT_ROOT / "code_for_Baichuan-Audio-Base/01_extract_activation/baichuan_runtime_compat"
        # The same project-local inference shims are used by Step01: a minimal
        # deepspeed namespace plus a PyTorch-SDPA implementation of Baichuan's
        # sole FlashAttention varlen call.  Every model runs in its own process,
        # so keeping this path active cannot shadow another model's runtime.
        sys.path.insert(0, str(compatibility))
        __import__("deepspeed")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True, local_files_only=True,
            model_max_length=128000,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda",
        ).eval()
        self.model.config.use_cache = True
        self.model.bind_processor(self.tokenizer, training=False, relative_path="/")
        self.audio_start = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_start_token_id
        )
        self.audio_end = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_end_token_id
        )

    def transcribe(self, audio_path):
        import torch
        prompt = self.prompt + self.audio_start + json.dumps(
            {"path": str(audio_path)}, ensure_ascii=False
        ) + self.audio_end
        inputs = self.model.processor([prompt])
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(),
                audios=inputs.audios.cuda(), encoder_length=inputs.encoder_length.cuda(),
                bridge_length=inputs.bridge_length.cuda(),
                max_new_tokens=self.args.max_new_tokens, num_beams=5, do_sample=False,
                repetition_penalty=1.1, use_cache=True,
            )
        return self.tokenizer.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()


class FireRedAudioAdapter(Adapter):
    model_name = "FireRedAudio"
    default_model_path = str(MODEL_ROOT / "FireRedAudio/FireRedAudio")
    prompt = final_prompt("firered_audio")
    generation_description = "official deterministic ASR generation config (4 beams)"
    source_root = PROJECT_ROOT / "code_for_FireRedAudio/model_runtime/FireRedAudio-main"

    def load(self):
        sys.path.insert(0, str(self.source_root))
        from inference import (  # type: ignore
            FEAT_TYPE_UNDERSTAND, FireRedAudioInference, UNDERSTAND_SAMPLE_RATE,
            build_understand_prompt, split_thinking,
        )
        self.FEAT_TYPE_UNDERSTAND = FEAT_TYPE_UNDERSTAND
        self.UNDERSTAND_SAMPLE_RATE = UNDERSTAND_SAMPLE_RATE
        self.build_understand_prompt = build_understand_prompt
        self.split_thinking = split_thinking
        self.engine = FireRedAudioInference(str(self.model_path), device="cuda:0")

    def transcribe(self, audio_path):
        import numpy as np
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly
        model = self.engine.model
        chatml = self.build_understand_prompt(
            self.prompt, 1, model.config.audio_special_token, enable_thinking=False
        )
        waveform, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if rate != self.UNDERSTAND_SAMPLE_RATE:
            waveform = resample_poly(waveform, self.UNDERSTAND_SAMPLE_RATE, rate).astype(np.float32)
        batch = self.engine.encoder.encode(chatml, [{
            "feat_type": self.FEAT_TYPE_UNDERSTAND,
            "audio_understand": waveform, "audio_generation": None, "role": "user",
        }])
        with torch.inference_mode():
            output = model.generate(
                input_ids=batch["input_ids"].to("cuda:0"),
                attention_mask=batch["attention_mask"].to("cuda:0"),
                audio_features=batch["audio_features"].to("cuda:0"),
                audio_feature_attention_mask=batch["audio_feature_attention_mask"].to("cuda:0"),
                generation_config=self.engine._gen_config("asr", max_new_tokens=self.args.max_new_tokens),
            )
        raw = self.engine.tokenizer.decode(output[0], skip_special_tokens=True)
        _reasoning, transcription = self.split_thinking(raw)
        return (transcription or raw).strip()


class GLM4VoiceAdapter(Adapter):
    model_name = "GLM-4-Voice"
    default_model_path = str(MODEL_ROOT / "GLM-4-Voice")
    prompt = final_prompt("glm_4_voice")
    generation_description = "deterministic greedy generate; do_sample=False; num_beams=1"
    reference_root = PROJECT_ROOT / "code_for_GLM-4-Voice/model_runtime/GLM-4-Voice-main"

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor
        sys.path.insert(0, str(self.reference_root))
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder  # type: ignore
        tokenizer_path = self.model_path / "glm-4-voice-tokenizer"
        language_path = self.model_path / "GLM-4-Voice-9B"
        self.tokenizer = AutoTokenizer.from_pretrained(language_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            language_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).eval().to("cuda:0")
        if not hasattr(self.model, "_extract_past_from_model_output"):
            # GLM-4-Voice's remote generation code targets the pre-4.51
            # GenerationMixin helper.  Transformers 4.51 removed it while the
            # model's custom _update_model_kwargs_for_generation still calls
            # it.  Restore the small compatibility method without changing the
            # generation path used by Step01.
            def _extract_past_from_model_output(model, outputs, *args, **kwargs):
                for cache_name in ("past_key_values", "mems", "past_buckets_states"):
                    cache = getattr(outputs, cache_name, None)
                    if cache is not None:
                        return cache_name, cache
                    if isinstance(outputs, dict) and outputs.get(cache_name) is not None:
                        return cache_name, outputs[cache_name]
                return "past_key_values", None

            self.model._extract_past_from_model_output = types.MethodType(
                _extract_past_from_model_output, self.model
            )
        self.speech_model = WhisperVQEncoder.from_pretrained(tokenizer_path, local_files_only=True).eval().to("cuda:0")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path, local_files_only=True)

    def transcribe(self, audio_path):
        import librosa
        import torch
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        pooling = int(self.speech_model.config.pooling_kernel_size or 1)
        stride = int(self.speech_model.conv1.stride[0] * self.speech_model.conv2.stride[0] * pooling * self.feature_extractor.hop_length)
        features = self.feature_extractor(
            [audio], sampling_rate=16000, return_attention_mask=True,
            return_tensors="pt", padding="longest", pad_to_multiple_of=stride,
        ).to("cuda:0")
        pooled_mask = features.attention_mask[:, ::2][:, ::pooling].bool()
        with torch.inference_mode():
            speech_output = self.speech_model(**features)
        tokens = speech_output.quantized_token_ids[0, pooled_mask[0]].tolist()
        audio_text = "<|begin_of_audio|>" + "".join(f"<|audio_{x}|>" for x in tokens) + "<|end_of_audio|>"
        assistant_prefix = "Transcription:" if os.environ.get("SLM_INPUT_LANGUAGE", "zh").lower().startswith("en") else "请转写："
        prompt = f"<|system|>\n{self.prompt}\n<|user|>\n{audio_text}<|assistant|>\n{assistant_prefix}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False,
                num_beams=1, use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return re.sub(r"<\|audio_\d+\|>", "", text).strip()


class MiniCPMOAdapter(Adapter):
    def __init__(self, args, version: str):
        self.version = version
        self.model_name = f"MiniCPM-o {version}"
        self.default_model_path = str(
            MODEL_ROOT / f"MiniCPM-o-{version.replace('.', '_')}"
        )
        self.generation_description = "native deterministic chat; audio generation disabled"
        self.prompt = final_prompt(f"minicpm_o_{version.replace('.', '_')}")
        super().__init__(args)

    def load(self):
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, PreTrainedModel
        if self.version == "2.6" and not hasattr(transformers, "LogitsWarper"):
            # MiniCPM-o 2.6 imports this legacy base class only for type
            # annotations.  Transformers 4.51 still provides the concrete
            # TopK/TopP warpers but no longer exports LogitsWarper.
            transformers.LogitsWarper = transformers.LogitsProcessor
        extra = {}
        if self.version == "2.6":
            # The study requests text-only ASR.  Match Step01 and disable the
            # unused synthesis branch before construction; otherwise the stock
            # remote class asserts on optional Vocos/FSQ TTS dependencies.
            config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True, local_files_only=True,
            )
            config.init_tts = False
            extra.update(config=config)
        if self.version == "4.5":
            # Transformers 5 expects this mapping during checkpoint finalization,
            # while MiniCPM-o 4.5's downloaded remote class predates the field.
            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                PreTrainedModel.all_tied_weights_keys = {}
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
            for field, value in (("top_p", 0.8), ("top_k", 20), ("repetition_penalty", 1.05)):
                if not hasattr(config.tts_config, field):
                    setattr(config.tts_config, field, value)
            # Transformers 5 finalizes low-memory meta loading via an attribute
            # absent from this checkpoint's remote model class. Normal CPU
            # loading avoids that incompatible path and is safe on this host.
            extra.update(config=config, low_cpu_mem_usage=False)
        original_import = builtins.__import__

        def import_with_legacy_logits_warper(name, globals=None, locals=None, fromlist=(), level=0):
            module = original_import(name, globals, locals, fromlist, level)
            if (
                self.version == "2.6"
                and name == "transformers"
                and "LogitsWarper" in (fromlist or ())
                and not hasattr(module, "LogitsWarper")
            ):
                module.LogitsWarper = module.LogitsProcessor
            return module

        try:
            # Transformers' dynamic-module loader can refresh its lazy top-level
            # module, discarding a one-time attribute assignment.  The import
            # hook restores the removed annotation base at the exact remote-code
            # import that requests it.
            builtins.__import__ = import_with_legacy_logits_warper
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True, local_files_only=True,
                attn_implementation="sdpa", torch_dtype=torch.bfloat16, **extra,
            ).eval().to("cuda:0")
        finally:
            builtins.__import__ = original_import
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)

    def transcribe(self, audio_path):
        import librosa
        import torch
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        messages = [{"role": "user", "content": [self.prompt, audio]}]
        kwargs = dict(
            msgs=messages, tokenizer=self.tokenizer, processor=self.processor,
            max_new_tokens=self.args.max_new_tokens, use_tts_template=True,
            generate_audio=False, stream=False,
        )
        if self.version == "4.5":
            kwargs.update(do_sample=False, enable_thinking=False)
        else:
            kwargs.update(sampling=False)
        with torch.inference_mode():
            return str(self.model.chat(**kwargs)).strip()


class Qwen25OmniAdapter(Adapter):
    model_name = "Qwen2.5-Omni"
    default_model_path = str(MODEL_ROOT / "Qwen2.5-Omni-7B")
    prompt = final_prompt("qwen2_5_omni")
    system_prompt = QWEN25_OMNI_SYSTEM_PROMPT
    generation_description = "Thinker-only deterministic greedy generate; Talker disabled"
    reference_root = PROJECT_ROOT / "code_for_Qwen2.5-Omni/model_runtime/Qwen2.5-Omni-main"

    def load(self):
        import torch
        sys.path.insert(0, str(self.reference_root / "qwen-omni-utils/src"))
        sys.path.insert(0, str(self.reference_root / "transformers/src"))
        from qwen_omni_utils import process_mm_info  # type: ignore
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.process_mm_info = process_mm_info
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map="auto",
        ).eval()
        self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, local_files_only=True)

    def transcribe(self, audio_path):
        import torch
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": self.prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = self.process_mm_info(messages, use_audio_in_video=True)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        prompt_length = int(inputs["input_ids"].shape[1])
        inputs = inputs.to(next(self.model.thinker.parameters()).device).to(self.model.thinker.dtype)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, use_audio_in_video=True, return_audio=False,
                thinker_max_new_tokens=self.args.max_new_tokens,
                thinker_do_sample=False, thinker_num_beams=1, thinker_use_cache=True,
            )
        generated = output[:, prompt_length:] if output.shape[1] > prompt_length else output
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


class CovoAudioAdapter(Adapter):
    model_name = "Covo-Audio-Chat"
    default_model_path = str(MODEL_ROOT / "Covo-Audio-Chat")
    prompt = final_prompt("covo_audio_chat")
    system_prompt = prompt
    native_interface = "native ChatML with the ASR task in the system position"
    generation_description = "Step01 ChatML deterministic generate; do_sample=False; num_beams=1"

    def load(self):
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer, LogitsProcessorList
        self.torch = torch
        self.np = np
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        self.c_audio_id = self.tokenizer.convert_tokens_to_ids("<|cAUDIO|>")
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, config=config, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda:0").eval()
        asset_path = self.model_path / "assets/mel_filters.npz"
        module = sys.modules[self.model.__class__.__module__]
        @lru_cache(maxsize=None)
        def local_mel_filters(device, n_mels):
            with np.load(asset_path, allow_pickle=False) as archive:
                return torch.from_numpy(archive[f"mel_{n_mels}"]).to(device)
        module.mel_filters = local_mel_filters
        self.logits_processor = LogitsProcessorList([
            module.WindowedRepetitionPenaltyLogitsProcessor(
                penalty=1.05, window_size=8
            )
        ])

    def transcribe(self, audio_path):
        import numpy as np
        import torch
        import torchaudio
        from scipy.io import wavfile

        # Reproduce Step01's final-prompt ChatML and 24-kHz preprocessing
        # exactly.  In particular, the ASR instruction is the system message;
        # no model-default dialog prompt is substituted here.
        try:
            sample_rate, samples = wavfile.read(str(audio_path))
        except ValueError:
            # LibriSpeech is FLAC, which scipy.io.wavfile cannot decode.
            # Keep the WAV path bit-for-bit unchanged while using torchaudio
            # for compressed formats supported by the benchmark.
            decoded, sample_rate = torchaudio.load(str(audio_path))
            samples = decoded.mean(dim=0).cpu().numpy()
        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            samples = samples.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            samples = samples.astype(np.float32, copy=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
        if sample_rate != 24000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)
        waveform = waveform.squeeze(0)
        # Covo's Whisper-derived audio encoder has a 30-second positional
        # limit.  Longer audio triggers a CUDA device-side assert and poisons
        # every subsequent sample in the process.  Transcribe deterministic,
        # non-overlapping chunks below that hard limit and concatenate them.
        max_chunk_samples = int(29.5 * 24000)
        if waveform.shape[-1] > max_chunk_samples:
            transcripts = []
            for start in range(0, waveform.shape[-1], max_chunk_samples):
                chunk = waveform[start : start + max_chunk_samples]
                if chunk.numel():
                    transcripts.append(self._transcribe_waveform(chunk))
            return "".join(part for part in transcripts if part).strip()
        return self._transcribe_waveform(waveform)

    def _transcribe_waveform(self, waveform):
        import torch

        waveform = waveform[: (waveform.shape[-1] // 240) * 240]
        if waveform.shape[-1] == 0:
            raise ValueError("Audio is shorter than one 10-ms frame")
        pad_samples = (-waveform.shape[-1]) % 480
        if pad_samples:
            waveform = torch.nn.functional.pad(waveform, (0, pad_samples))
        projected_length = waveform.shape[-1] * 100 // 24000
        for _ in range(4):
            projected_length = (projected_length + 1) // 2
        audio_placeholders = "<|cAUDIO|>" * projected_length
        prompt = (
            f"<|im_start|>system\n{self.prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"<|begofcAUDIO|>{audio_placeholders}<|endofcAUDIO|>"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to("cuda:0")
        attention_mask = encoded.attention_mask.to("cuda:0")
        if int(input_ids.eq(self.c_audio_id).sum().item()) != projected_length:
            raise RuntimeError("Covo audio placeholder count does not match projected length")
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                wavs=[waveform.to("cuda:0", dtype=torch.float32)],
                max_new_tokens=self.args.max_new_tokens, do_sample=False, num_beams=1,
                use_cache=True, repetition_penalty=1.05,
                logits_processor=self.logits_processor,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        sequences = output.sequences if hasattr(output, "sequences") else output
        ids = sequences[0, input_ids.shape[1]:]
        ids = ids[ids < len(self.tokenizer)]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


class FreezeOmniAdapter(Adapter):
    """Freeze-Omni streaming ASR path shared with Step01 activation extraction."""

    model_name = "Freeze-Omni"
    default_model_path = str(MODEL_ROOT / "Freeze-omni" / "checkpoints")
    prompt = final_prompt("freeze_omni")
    native_interface = "native streaming speech_dialogue state machine"
    generation_description = (
        "Step01 native streaming generation; deterministic top_k=1; "
        "top_p=1.0; temperature=1.0"
    )
    reference_root = PROJECT_ROOT / "code_for_Freeze-Omni" / "model_runtime"
    default_llm_path = first_existing_path(
        MODEL_ROOT / "qwen2-7b-instruct",
        MODEL_ROOT / "Qwen2-7B-Instruct",
    )

    def __init__(self, args):
        super().__init__(args)
        self.llm_path = Path(
            os.environ.get("FREEZE_OMNI_LLM_PATH", str(self.default_llm_path))
        ).expanduser().resolve()

    def load(self):
        if not self.reference_root.is_dir():
            raise FileNotFoundError(f"Freeze-Omni runtime not found: {self.reference_root}")
        if not self.llm_path.is_dir():
            raise FileNotFoundError(
                "Freeze-Omni Qwen2 backbone not found: "
                f"{self.llm_path}. Set FREEZE_OMNI_LLM_PATH if it is stored elsewhere."
            )

        # Load the exact frontend, streaming state transitions, and generation
        # loop used by Step01 rather than maintaining a second implementation.
        step01_path = (
            PROJECT_ROOT / "code_for_Freeze-Omni" / "01_extract_activation"
            / "extract_Freeze_Omni_activations_revised.py"
        )
        spec = importlib.util.spec_from_file_location(
            "freeze_omni_step01_runtime", step01_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Freeze-Omni Step01 runtime: {step01_path}")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        self.runtime = runtime

        sys.path.insert(0, str(self.reference_root))
        from models.pipeline import inferencePipeline  # type: ignore

        self.pipeline = inferencePipeline(
            runtime.Config(self.model_path, self.llm_path)
        )
        self.pipeline.model.eval()
        self.samples_per_chunk = runtime.AudioEncoderProcessor().samples_per_chunk

    def transcribe(self, audio_path):
        import torch

        padded, original_samples = self.runtime.load_waveform(
            Path(audio_path), self.samples_per_chunk
        )
        with torch.inference_mode():
            return self.runtime.run_behavior_pass(
                self.pipeline, padded, original_samples, self.args.max_new_tokens
            )

class KimiAudioAdapter(Adapter):
    model_name = "Kimi-Audio-7B-Instruct"
    default_model_path = str(MODEL_ROOT / "Kimi-Audio-7B-Instruct")
    prompt = final_prompt("kimi_audio_7b_instruct")
    generation_description = "official text generation; temperature=0; top_k=5; no audio detokenizer"
    glm_tokenizer_path = str(MODEL_ROOT / "GLM-4-Voice/glm-4-voice-tokenizer")
    source_root = PROJECT_ROOT / "code_for_Kimi-Audio-7B-Instruct/model_runtime"

    def load(self):
        sys.path.insert(0, str(self.source_root))
        # Match the compatibility layer already used by Step01 activation
        # extraction.  Kimi's checkpoint expects the pre-indexed Qwen2 rotary
        # helper contract, and its bundled Whisper safetensors file is valid but
        # was published without the optional metadata dictionary.
        import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling
        import transformers.modeling_utils as modeling_utils
        from safetensors import safe_open
        from safetensors.torch import load_file as safe_load_file

        def apply_rotary_pos_emb_preindexed(
            query, key, cosine, sine, position_ids=None, unsqueeze_dim=1,
        ):
            del position_ids
            cosine = cosine.unsqueeze(unsqueeze_dim)
            sine = sine.unsqueeze(unsqueeze_dim)
            query_embed = (query * cosine) + (qwen2_modeling.rotate_half(query) * sine)
            key_embed = (key * cosine) + (qwen2_modeling.rotate_half(key) * sine)
            return query_embed, key_embed

        qwen2_modeling.apply_rotary_pos_emb = apply_rotary_pos_emb_preindexed

        class SafeOpenWithDefaultMetadata:
            def __init__(self, *args, **kwargs):
                self._context = safe_open(*args, **kwargs)
                self._archive = None

            def __enter__(self):
                self._archive = self._context.__enter__()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return self._context.__exit__(exc_type, exc_value, traceback)

            def metadata(self):
                metadata = self._archive.metadata()
                return {"format": "pt"} if metadata is None else metadata

            def __getattr__(self, name):
                return getattr(self._archive, name)

        modeling_utils.safe_open = SafeOpenWithDefaultMetadata
        original_load_state_dict = modeling_utils.load_state_dict

        def load_state_dict_allow_missing_safetensors_metadata(
            checkpoint_file, is_quantized=False,
        ):
            checkpoint_path = str(checkpoint_file)
            if checkpoint_path.endswith(".safetensors"):
                with safe_open(checkpoint_path, framework="pt") as archive:
                    if archive.metadata() is None:
                        return safe_load_file(checkpoint_path)
            return original_load_state_dict(checkpoint_file, is_quantized=is_quantized)

        modeling_utils.load_state_dict = load_state_dict_allow_missing_safetensors_metadata
        import kimia_infer.api.prompt_manager as prompt_manager_module
        from kimia_infer.api.kimia import KimiAudio
        original = prompt_manager_module.Glm4Tokenizer
        prompt_manager_module.Glm4Tokenizer = lambda _hub_id: original(self.glm_tokenizer_path)
        try:
            self.kimi = KimiAudio(model_path=str(self.model_path), load_detokenizer=False)
        finally:
            prompt_manager_module.Glm4Tokenizer = original
        self.kimi.alm.eval()

    def transcribe(self, audio_path):
        import torch
        messages = [
            {"role": "user", "message_type": "text", "content": self.prompt},
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
        ]
        with torch.inference_mode():
            _waveform, text = self.kimi.generate(
                messages, output_type="text", text_temperature=0.0, text_top_k=5,
                audio_temperature=0.8, audio_top_k=10,
                max_new_tokens=self.args.max_new_tokens,
            )
        return str(text).strip()


def create_adapter(model_key: str, args) -> Adapter:
    factories = {
        "qwen2_audio": lambda: Qwen2AudioAdapter(args, False),
        "qwen2_audio_instruct": lambda: Qwen2AudioAdapter(args, True),
        "qwen_audio": lambda: QwenAudioLegacyAdapter(args, False),
        "qwen_audio_chat": lambda: QwenAudioLegacyAdapter(args, True),
        "firered_audio": lambda: FireRedAudioAdapter(args),
        "freeze_omni": lambda: FreezeOmniAdapter(args),
        "baichuan_audio_base": lambda: BaichuanAudioAdapter(args, False),
        "glm_4_voice": lambda: GLM4VoiceAdapter(args),
        "minicpm_o_2_6": lambda: MiniCPMOAdapter(args, "2.6"),
        "minicpm_o_4_5": lambda: MiniCPMOAdapter(args, "4.5"),
        "qwen2_5_omni": lambda: Qwen25OmniAdapter(args),
        "covo_audio_chat": lambda: CovoAudioAdapter(args),
        "kimi_audio_7b_instruct": lambda: KimiAudioAdapter(args),
        "baichuan_audio_instruct": lambda: BaichuanAudioAdapter(args, True),
    }
    try:
        return factories[model_key]()
    except KeyError as error:
        raise KeyError(f"Unknown ASR adapter {model_key!r}; choices={sorted(factories)}") from error
