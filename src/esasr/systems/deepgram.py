"""Deepgram Nova-3 adapter (pre-recorded REST API).

DISABLED BY DEFAULT: it only runs if DEEPGRAM_API_KEY is set, and no results in this
repository were produced with it (paid API; spend requires explicit approval).

Cost reference (check current pricing before running): Nova-3 pay-as-you-go is
billed per audio minute; the full eval grid here is ~2.5 h of audio x 5 conditions.
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf

from .base import ASRSystem

API = "https://api.deepgram.com/v1/listen"


class DeepgramNova3(ASRSystem):
    def __init__(self, language: str = "es", model: str = "nova-3", api_key: str | None = None):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set; Deepgram adapter is disabled.")
        self.language = language  # "es", "es-419", or "multi" for code-switching
        self.model = model
        self.name = f"deepgram-{model}-{language}"
        self.description = f"Deepgram {model} REST (language={language}, smart_format=false)"

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
        q = urllib.parse.urlencode(
            {"model": self.model, "language": self.language, "punctuate": "true", "smart_format": "false"}
        )
        req = urllib.request.Request(
            f"{API}?{q}",
            data=buf.getvalue(),
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.load(r)
        alts = res["results"]["channels"][0]["alternatives"]
        return alts[0]["transcript"] if alts else ""
