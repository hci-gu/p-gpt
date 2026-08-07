from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .protocol import SpeakerInputLanguage


class EndpointASR(Protocol):
    model_id: str
    device: str

    def transcribe_pcm16(self, audio_bytes: bytes) -> str: ...

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int) -> str: ...

    def close(self) -> None: ...


def normalize_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int = 16_000,
) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError("ASR audio must be mono or samples-by-channels.")

    if source_sample_rate <= 0 or waveform.size == 0:
        raise ValueError("ASR audio must contain samples at a valid rate.")
    if source_sample_rate != target_sample_rate:
        output_length = max(
            1,
            round(waveform.size * target_sample_rate / source_sample_rate),
        )
        source_positions = np.arange(waveform.size, dtype=np.float64)
        target_positions = np.linspace(
            0,
            waveform.size - 1,
            output_length,
            dtype=np.float64,
        )
        waveform = np.interp(target_positions, source_positions, waveform).astype(
            np.float32
        )
    return np.clip(waveform, -1, 1)


def pcm16_to_float32(audio_bytes: bytes) -> np.ndarray:
    if len(audio_bytes) % 2:
        raise ValueError("PCM16 input must contain complete samples.")
    waveform = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32)
    waveform /= 32_768.0
    return waveform


@dataclass
class ParakeetASR:
    model: Any
    device: str
    model_id: str = "nvidia/parakeet-tdt-0.6b-v3"

    @classmethod
    def from_pretrained(cls, model_id: str) -> "ParakeetASR":
        import torch
        from nano_parakeet import from_pretrained

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = from_pretrained(model_name=model_id, device=device)
        return cls(model=model, device=device, model_id=model_id)

    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int = 16_000,
    ) -> np.ndarray:
        return normalize_audio(audio, source_sample_rate, target_sample_rate)

    def transcribe_pcm16(self, audio_bytes: bytes) -> str:
        waveform = pcm16_to_float32(audio_bytes)
        return self.transcribe_waveform(waveform, 16_000)

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int) -> str:
        import torch

        waveform = self.normalize_audio(audio, sample_rate)
        inference_mode = getattr(torch, "inference_mode", nullcontext)
        with inference_mode():
            return self.model.transcribe(waveform).strip()

    def close(self) -> None:
        self.model = None


@dataclass
class KBWhisperASR:
    model: Any
    processor: Any
    transcriber: Any
    device: str
    dtype: str
    model_id: str
    revision: str
    chunk_length_seconds: float = 30
    stride_length_seconds: float = 5

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str = "standard",
    ) -> "KBWhisperASR":
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            pipeline_device: str | int = 0
        else:
            device = "cpu"
            torch_dtype = torch.float32
            pipeline_device = -1

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch_dtype,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch_dtype,
            device=pipeline_device,
        )
        return cls(
            model=model,
            processor=processor,
            transcriber=transcriber,
            device=device,
            dtype=str(torch_dtype).removeprefix("torch."),
            model_id=model_id,
            revision=revision,
        )

    def transcribe_pcm16(self, audio_bytes: bytes) -> str:
        return self.transcribe_waveform(pcm16_to_float32(audio_bytes), 16_000)

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int) -> str:
        import torch

        waveform = normalize_audio(audio, sample_rate)
        inference_mode = getattr(torch, "inference_mode", nullcontext)
        with inference_mode():
            result = self.transcriber(
                {"raw": waveform, "sampling_rate": 16_000},
                chunk_length_s=self.chunk_length_seconds,
                stride_length_s=self.stride_length_seconds,
                generate_kwargs={"task": "transcribe", "language": "sv"},
            )
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str):
            raise RuntimeError("KB-Whisper returned an invalid transcription result.")
        return text.strip()

    def close(self) -> None:
        self.transcriber = None
        self.processor = None
        self.model = None


@dataclass
class SpeakerASRRouter:
    routes: dict[SpeakerInputLanguage, EndpointASR]

    def adapter_for(self, language: SpeakerInputLanguage) -> EndpointASR:
        try:
            return self.routes[language]
        except KeyError as exc:
            raise ValueError(f"Unsupported speaker input language: {language}") from exc

    def transcribe_pcm16(
        self,
        audio_bytes: bytes,
        language: SpeakerInputLanguage,
    ) -> str:
        return self.adapter_for(language).transcribe_pcm16(audio_bytes)

    def close(self) -> None:
        for adapter in self.routes.values():
            adapter.close()
