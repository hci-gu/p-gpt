from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ParakeetASR:
    model: Any
    device: str

    @classmethod
    def from_pretrained(cls, model_id: str) -> "ParakeetASR":
        import torch
        from nano_parakeet import from_pretrained

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = from_pretrained(model_name=model_id, device=device)
        return cls(model=model, device=device)

    @staticmethod
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

    def transcribe_pcm16(self, audio_bytes: bytes) -> str:
        if len(audio_bytes) % 2:
            raise ValueError("PCM16 input must contain complete samples.")
        waveform = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32)
        waveform /= 32_768.0
        return self.transcribe_waveform(waveform, 16_000)

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int) -> str:
        waveform = self.normalize_audio(audio, sample_rate)
        return self.model.transcribe(waveform).strip()

    def close(self) -> None:
        self.model = None
