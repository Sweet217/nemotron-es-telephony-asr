"""Offline Pipecat demo: WAV file -> Silero VAD -> Nemotron STT -> transcripts.

    python -m pipecat_nemotron.transcribe_wav path/to/audio.wav [--language es-MX] [--telephony]

The WAV is streamed into the pipeline as 20 ms ``InputAudioRawFrame``s (the same
frames a transport would produce), so the VAD -> segmented STT path is exercised
exactly as in a live bot. ``--telephony`` first passes the audio through the
G.711 mu-law 8 kHz channel simulation from ``esasr.codec`` and feeds the pipeline
at 8 kHz, like a SIP/PSTN leg would.
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np
import soundfile as sf

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language

from .nemotron_stt import NemotronSTTService


def wav_to_frames(audio: np.ndarray, sr: int, frame_ms: int = 20, tail_silence_s: float = 1.5):
    audio = np.concatenate([audio, np.zeros(int(sr * tail_silence_s), dtype=np.float32)])
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    n = sr * frame_ms // 1000
    return [
        InputAudioRawFrame(audio=pcm[i : i + n].tobytes(), sample_rate=sr, num_channels=1)
        for i in range(0, len(pcm) - n + 1, n)
    ]


async def transcribe(audio: np.ndarray, sr: int, stt: NemotronSTTService | None = None,
                     language: str = "es-MX") -> list[TranscriptionFrame]:
    stt = stt or NemotronSTTService(settings=NemotronSTTService.Settings(language=Language(language)))
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=sr, params=VADParams(stop_secs=0.6)))
    pipeline = Pipeline([vad, stt])
    from pipecat.pipeline.worker import PipelineParams

    down, _ = await run_test(
        pipeline,
        frames_to_send=wav_to_frames(audio, sr),
        pipeline_params=PipelineParams(audio_in_sample_rate=sr),
    )
    return [f for f in down if isinstance(f, TranscriptionFrame)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--language", default="es-MX")
    ap.add_argument("--telephony", action="store_true")
    a = ap.parse_args()
    audio, sr = sf.read(a.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if a.telephony:
        from esasr.codec import g711_mulaw_roundtrip, resample, telephone_bandpass

        audio = g711_mulaw_roundtrip(telephone_bandpass(resample(audio, sr, 8000), 8000))
        sr = 8000
    for f in asyncio.run(transcribe(audio, sr, language=a.language)):
        print(f"[{f.language}] {f.text}")


if __name__ == "__main__":
    main()
