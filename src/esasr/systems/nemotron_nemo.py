"""Nemotron 3.5 ASR through NVIDIA NeMo (``nemo_toolkit[asr]``), the reference runtime.

Used for (a) cross-checking the transformers port on a subset and (b) as the base for
fine-tuning (finetune/). NeMo installs and runs inference on macOS/Apple Silicon (CPU),
but NVIDIA only lists Linux + NVIDIA GPUs as supported, so treat this as a dev path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import ASRSystem

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"


class NemotronNeMo(ASRSystem):
    def __init__(
        self,
        language: str = "es-US",
        att_context_size: tuple[int, int] = (56, 3),
        device: str | None = None,
        model: str = MODEL_ID,
    ):
        import nemo.collections.asr as nemo_asr
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model.endswith(".nemo"):
            self.model = nemo_asr.models.ASRModel.restore_from(model, map_location=device)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model, map_location=device)
        self.model.eval()
        if hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(list(att_context_size))
        self.language = language
        ms = (att_context_size[1] + 1) * 80
        tag = "" if model == MODEL_ID else "-" + Path(model).stem
        self.name = f"nemotron35-nemo-{language}-{ms}ms{tag}"
        self.description = f"{model} (NeMo, lang={language}, att_context={list(att_context_size)}, {device})"

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        return self.transcribe_many([audio], sr)[0]

    def transcribe_many(self, audios: list[np.ndarray], sr: int = 16000, batch_size: int = 8) -> list[str]:
        # NOTE: in nemo_toolkit 3.0.0, transcribe(list_of_paths, target_lang=...) builds a
        # manifest without a language field and fails with "Unknown prompt key: 'None'".
        # Passing our own manifest with a "lang" field (the transcribe dataloader's default
        # lang_field) works. Upstream: NVIDIA-NeMo/Speech#15820, fixed on main by
        # NVIDIA-NeMo/Speech#15940 (merged 2026-10-23) but not in a PyPI release as of 3.0.0.
        import json

        with tempfile.TemporaryDirectory() as td:
            man = Path(td) / "manifest.json"
            with man.open("w") as f:
                for i, a in enumerate(audios):
                    p = Path(td) / f"{i}.wav"
                    sf.write(p, a, sr, subtype="PCM_16")
                    f.write(json.dumps({"audio_filepath": str(p), "duration": len(a) / sr, "text": "",
                                        "lang": self.language, "target_lang": self.language}) + "\n")
            out = self.model.transcribe(str(man), batch_size=batch_size, verbose=False)
        return [getattr(o, "text", o) for o in out]
