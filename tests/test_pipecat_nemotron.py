"""Pipecat integration: Silero VAD + NemotronSTTService on an offline WAV.

The end-to-end test needs the model weights (~2.5 GB) in the local HF cache and
macOS `say` to synthesize the input; it is skipped otherwise.
"""
import asyncio
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from pipecat_nemotron import NemotronSTTService, language_to_nemotron_language
from pipecat.transcriptions.language import Language


def test_language_mapping():
    assert language_to_nemotron_language(Language.ES_MX) == "es-US"
    assert language_to_nemotron_language(Language.ES) == "es-US"
    assert language_to_nemotron_language(Language.ES_ES) == "es-ES"
    assert language_to_nemotron_language(None) == "auto"


def test_service_is_a_segmented_stt_service():
    from pipecat.services.stt_service import SegmentedSTTService

    assert issubclass(NemotronSTTService, SegmentedSTTService)


def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("nvidia/nemotron-3.5-asr-streaming-0.6b", "model.safetensors"), str)
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("say") is None or not _model_cached(), reason="needs macOS say + cached model")
def test_vad_segments_two_utterances_at_8khz(tmp_path):
    from esasr.codec import g711_mulaw_roundtrip, resample, telephone_bandpass
    from pipecat_nemotron.transcribe_wav import transcribe

    parts = []
    for i, text in enumerate(["Buenas tardes, quiero revisar mi saldo.", "Muchas gracias, eso es todo."]):
        p = tmp_path / f"{i}.wav"
        subprocess.run(["say", "-v", "Paulina (Spanish (Mexico))", "--data-format=LEI16@16000", "-o", str(p), text],
                       check=True)
        a, _ = sf.read(p, dtype="float32")
        parts += [np.zeros(8000, np.float32), a, np.zeros(24000, np.float32)]
    audio16 = np.concatenate(parts)
    audio8 = g711_mulaw_roundtrip(telephone_bandpass(resample(audio16, 16000, 8000), 8000))

    frames = asyncio.run(transcribe(audio8, 8000))
    texts = [f.text.lower() for f in frames]
    assert len(texts) == 2, texts  # VAD split the call into two turns
    assert "saldo" in texts[0]
    assert "gracias" in texts[1]
