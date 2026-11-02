"""OpenAI Whisper on Apple Silicon via mlx-whisper (local, no API).

Default: ``mlx-community/whisper-large-v3-turbo`` (809M params; large-v3 encoder with a
4-layer decoder), language forced to Spanish, greedy decoding (temperature 0), no
previous-text conditioning. ``condition_on_previous_text`` is irrelevant for the
short utterances here.
"""
from __future__ import annotations

import numpy as np

from .base import ASRSystem


class WhisperMLX(ASRSystem):
    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo", language: str | None = "es"):
        import mlx_whisper  # noqa: F401

        self.model = model
        self.language = language
        short = model.split("/")[-1].replace("whisper-", "")
        self.name = f"whisper-{short}" + (f"-{language}" if language else "-auto")
        self.description = f"{model} (mlx-whisper, language={language or 'auto'}, greedy)"

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        import mlx_whisper

        assert sr == 16000, "mlx-whisper expects 16 kHz"
        out = mlx_whisper.transcribe(
            np.asarray(audio, dtype=np.float32),
            path_or_hf_repo=self.model,
            language=self.language,
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=None,
        )
        return out["text"].strip()
