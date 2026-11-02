"""Common interface for ASR systems under test."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ASRSystem(ABC):
    #: short id used in result paths, e.g. "nemotron35-es-US"
    name: str = "base"
    #: human-readable description for the report
    description: str = ""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcribe mono float32 audio at `sr` Hz; return the raw (un-normalized) text."""

    def info(self) -> dict:
        return {"name": self.name, "description": self.description}
