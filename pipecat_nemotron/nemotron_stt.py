"""Pipecat STT service for NVIDIA Nemotron 3.5 ASR (local inference, no NIM/Riva needed).

Nemotron 3.5 ASR ships without VAD or diarization. This service follows the same
pattern as Pipecat's local ``WhisperSTTService``: it subclasses
``SegmentedSTTService``, so an upstream ``VADProcessor(SileroVADAnalyzer())`` decides
where an utterance starts and stops, and the service transcribes each VAD segment
once it ends (pre-roll buffer + 0.5 s trailing silence are added by the base class).

Inference runs through the Hugging Face Transformers port of the model
(``Nemotron3_5AsrForRNNT``, transformers >= 5.13), on CUDA, Apple MPS or CPU, in a
worker thread so the event loop keeps moving audio.

Language handling: Nemotron is prompt-conditioned. Pipecat ``Language`` values are
mapped to the model's locale prompts (es-MX / es-419 -> ``es-US``, the Latin American
prompt; es-ES -> ``es-ES``); ``None`` means automatic language detection (``auto``).
The model's trailing ``<xx-XX>`` tag is stripped from the transcript and reported as
the frame's language.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
_TAG = re.compile(r"<([a-z]{2,3}(?:-[A-Za-z]{2,4})?)>\s*$")


def language_to_nemotron_language(language: Language | str | None) -> str:
    """Map a Pipecat language to a Nemotron prompt key (``auto`` when unknown)."""
    if language is None:
        return "auto"
    lang = str(language)
    explicit = {
        "es": "es-US",
        "es-MX": "es-US",
        "es-US": "es-US",
        "es-419": "es-US",
        "es-AR": "es-US",
        "es-CO": "es-US",
        "es-CL": "es-US",
        "es-PE": "es-US",
        "es-ES": "es-ES",
        "en": "en-US",
        "en-US": "en-US",
        "en-GB": "en-GB",
    }
    if lang in explicit:
        return explicit[lang]
    return lang  # e.g. "de-DE", "fr-FR", "auto": validated by the processor


@dataclass
class NemotronSTTSettings(STTSettings):
    """Runtime-updatable settings.

    Parameters:
        num_lookahead_tokens: right attention context in 80 ms frames
            (0 -> 80 ms, 3 -> 320 ms, 6 -> 560 ms, 13 -> 1.12 s chunks).
    """

    num_lookahead_tokens: int | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class NemotronSTTService(SegmentedSTTService):
    """Segment-level (VAD-gated) transcription with Nemotron 3.5 ASR."""

    Settings = NemotronSTTSettings
    _settings: Settings

    @property
    def wants_wav_segments(self) -> bool:
        """Receive raw 16-bit PCM: the model reads the buffer directly."""
        return False

    def __init__(
        self,
        *,
        model: str = MODEL_ID,
        device: str | None = None,
        settings: Settings | None = None,
        model_obj=None,
        processor_obj=None,
        **kwargs,
    ):
        """Create the service.

        Args:
            model: Hugging Face repo id or local path.
            device: "cuda", "mps" or "cpu"; auto-detected when None.
            settings: runtime settings (language, num_lookahead_tokens).
            model_obj / processor_obj: pre-loaded objects (lets tests share one model).
        """
        default_settings = self.Settings(model=model, language=Language.ES_MX, num_lookahead_tokens=3)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(settings=default_settings, **kwargs)
        self._device = device
        self._model = model_obj
        self._processor = processor_obj
        if self._model is None or self._processor is None:
            self._load()

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_nemotron_language(language)

    def _load(self):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        device = self._device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._device = device
        name = self._settings.model
        logger.debug(f"Loading Nemotron ASR {name} on {device}...")
        self._processor = AutoProcessor.from_pretrained(name)
        self._model = AutoModelForRNNT.from_pretrained(name).to(device).eval()
        logger.debug("Loaded Nemotron ASR")

    def _transcribe(self, audio: np.ndarray, prompt_lang: str, lookahead: int) -> str:
        import torch

        self._processor.set_num_lookahead_tokens(lookahead)
        inputs = self._processor(audio, sampling_rate=16000, language=prompt_lang, return_tensors="pt")
        inputs["num_lookahead_tokens"] = lookahead
        inputs = inputs.to(self._device or "cpu")
        with torch.inference_mode():
            out = self._model.generate(**inputs, return_dict_in_generate=True)
        text = self._processor.decode(out.sequences, skip_special_tokens=False)
        return (text[0] if isinstance(text, list) else text).replace("<blank>", "").strip()

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language | None = None):
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one VAD segment of 16-bit mono PCM at ``self.sample_rate``."""
        if self._model is None:
            yield ErrorFrame("Nemotron model not available")
            return
        await self.start_processing_metrics()
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if self.sample_rate != 16000:
            from scipy.signal import resample_poly

            g = np.gcd(self.sample_rate, 16000)
            pcm = resample_poly(pcm, 16000 // g, self.sample_rate // g).astype(np.float32)
        lang_setting = self._settings.language
        prompt_lang = "auto" if isinstance(lang_setting, NotGiven) else language_to_nemotron_language(lang_setting)
        lookahead = self._settings.num_lookahead_tokens
        lookahead = 3 if isinstance(lookahead, NotGiven) or lookahead is None else int(lookahead)
        try:
            raw = await asyncio.to_thread(self._transcribe, pcm, prompt_lang, lookahead)
        except Exception as e:  # surface as a pipeline error, don't kill the task
            await self.stop_processing_metrics()
            yield ErrorFrame(error=f"Nemotron transcription failed: {e}")
            return
        await self.stop_processing_metrics()
        m = _TAG.search(raw)
        detected = m.group(1) if m else None
        text = _TAG.sub("", raw).strip()
        if text:
            language = Language(detected) if detected in Language._value2member_map_ else None
            await self._handle_transcription(text, True, language)
            logger.debug(f"Nemotron transcription [{detected}]: {text}")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
