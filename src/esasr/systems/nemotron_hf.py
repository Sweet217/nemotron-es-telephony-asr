"""NVIDIA Nemotron 3.5 ASR (nvidia/nemotron-3.5-asr-streaming-0.6b) via Hugging Face Transformers.

Transformers >= 5.13 ships a native port (``Nemotron3_5AsrForRNNT``). A full-utterance
``generate`` call applies the same chunked, cache-aware attention mask the model uses
when streaming, with right context ``num_lookahead_tokens`` (in 80 ms encoder frames):

    lookahead 0 -> 80 ms chunks, 3 -> 320 ms (default), 6 -> 560 ms, 13 -> 1.12 s

so the WER here corresponds to the model card's streaming configuration of the same
chunk size (card: Spanish FLEURS 4.39 % at 320 ms with LangID). This runs on Apple
Silicon (MPS) or CPU; no CUDA required.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import ASRSystem

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
LOOKAHEAD_TO_MS = {0: 80, 1: 160, 3: 320, 6: 560, 13: 1120}


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NemotronHF(ASRSystem):
    def __init__(
        self,
        language: str = "es-US",
        num_lookahead_tokens: int = 3,
        device: str | None = None,
        model_id: str = MODEL_ID,
    ):
        from transformers import AutoModelForRNNT, AutoProcessor

        self.language = language
        self.lookahead = num_lookahead_tokens
        self.device = pick_device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.set_num_lookahead_tokens(num_lookahead_tokens)
        self.model = AutoModelForRNNT.from_pretrained(model_id).to(self.device).eval()
        self.model_id = model_id
        ms = LOOKAHEAD_TO_MS.get(num_lookahead_tokens, num_lookahead_tokens)
        self.name = f"nemotron35-{language}-{ms}ms"
        self.description = (
            f"{model_id} (transformers, lang={language}, chunk={ms} ms, {self.device}, fp32)"
        )

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        inputs = self.processor(
            np.asarray(audio, dtype=np.float32),
            sampling_rate=sr,
            language=self.language,
            return_tensors="pt",
        )
        inputs["num_lookahead_tokens"] = self.lookahead
        inputs = inputs.to(self.device)
        out = self.model.generate(**inputs, return_dict_in_generate=True)
        # keep special tokens so the <xx-XX> language tag stays visible in raw output;
        # the normalizer strips it before scoring.
        text = self.processor.decode(out.sequences, skip_special_tokens=False)
        if isinstance(text, list):
            text = text[0]
        return text.replace("<blank>", "").strip()
