"""Telephony channel simulation: G.711 mu-law, narrowband filtering, packet loss, noise.

Everything here is pure numpy/scipy and deterministic given a seed, so a degraded
utterance can be regenerated bit-exactly on any machine.

The G.711 mu-law tables follow the ITU-T G.711 reference algorithm (the classic
Sun Microsystems ``g711.c`` implementation that ffmpeg's ``pcm_mulaw`` and sox also
use). ``tests/test_codec.py`` checks the encoder byte-for-byte against ffmpeg.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import signal

# --------------------------------------------------------------------------- G.711

_BIAS = 0x84  # 132
_CLIP14 = 8159
_SEG_UEND14 = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float audio in [-1, 1] to int16 with rounding and clipping."""
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / 32768.0


def mulaw_encode(pcm16: np.ndarray) -> np.ndarray:
    """Encode int16 linear PCM to 8-bit G.711 mu-law codes (uint8).

    Works on the 14-bit magnitude (``x >> 2``) exactly like the reference ``g711.c``
    so decision boundaries match ffmpeg/sox bit-for-bit.
    """
    pcm = np.asarray(pcm16, dtype=np.int32) >> 2  # arithmetic shift (floor)
    neg = pcm < 0
    mask = np.where(neg, 0x7F, 0xFF).astype(np.int32)
    mag = np.minimum(np.where(neg, -pcm, pcm), _CLIP14) + (_BIAS >> 2)
    seg = np.searchsorted(_SEG_UEND14, mag, side="left").astype(np.int32)
    uval = np.where(seg >= 8, 0x7F, (np.minimum(seg, 7) << 4) | ((mag >> (np.minimum(seg, 7) + 1)) & 0x0F))
    return ((uval ^ mask) & 0xFF).astype(np.uint8)


def _build_decode_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.int32)
    u = ~codes & 0xFF
    t = ((u & 0x0F) << 3) + _BIAS
    t = t << ((u & 0x70) >> 4)
    return np.where(u & 0x80, _BIAS - t, t - _BIAS).astype(np.int16)


_DECODE = _build_decode_table()


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """Decode 8-bit G.711 mu-law codes to int16 linear PCM."""
    return _DECODE[np.asarray(codes, dtype=np.uint8)]


def g711_mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """Float audio -> int16 -> mu-law -> int16 -> float (same sample rate)."""
    return int16_to_float(mulaw_decode(mulaw_encode(float_to_int16(x))))


# ------------------------------------------------------------- resampling / filtering


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling with scipy's default Kaiser anti-alias filter."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = np.gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def telephone_bandpass(x: np.ndarray, sr: int = 8000, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    """300-3400 Hz channel filter (G.712-style voice band), 4th-order Butterworth, zero-phase."""
    sos = signal.butter(4, [lo, min(hi, sr / 2 * 0.98)], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


# ------------------------------------------------------------------ impairments


def packet_loss(
    x: np.ndarray,
    sr: int,
    loss_rate: float,
    rng: np.random.Generator,
    frame_ms: int = 20,
    concealment: str = "repeat",
) -> tuple[np.ndarray, np.ndarray]:
    """Drop RTP-sized frames at random (Bernoulli loss).

    concealment="zero" inserts silence; "repeat" repeats the last good frame with
    -6 dB attenuation per consecutive loss (a crude stand-in for G.711 Appendix I PLC).
    Returns (audio, lost_mask_per_frame).
    """
    n = int(sr * frame_ms / 1000)
    y = np.array(x, dtype=np.float32, copy=True)
    n_frames = int(np.ceil(len(y) / n))
    lost = rng.random(n_frames) < loss_rate
    last_good = np.zeros(n, dtype=np.float32)
    gain = 1.0
    for i in range(n_frames):
        s, e = i * n, min((i + 1) * n, len(y))
        if lost[i]:
            if concealment == "zero":
                y[s:e] = 0.0
            else:
                gain *= 0.5
                y[s:e] = last_good[: e - s] * gain
        else:
            gain = 1.0
            frame = y[s:e]
            last_good = np.zeros(n, dtype=np.float32)
            last_good[: len(frame)] = frame
    return y, lost


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping of white Gaussian noise, unit RMS."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / np.sqrt(f)
    pn = np.fft.irfft(spec, n)
    return (pn / (np.sqrt(np.mean(pn**2)) + 1e-12)).astype(np.float32)


def active_rms(x: np.ndarray, sr: int, frame_ms: int = 20, floor_db: float = -40.0) -> float:
    """RMS over frames within `floor_db` of the loudest frame (rough speech-active level)."""
    n = max(1, int(sr * frame_ms / 1000))
    m = len(x) // n
    if m == 0:
        return float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    frames = np.asarray(x[: m * n], dtype=np.float64).reshape(m, n)
    e = np.mean(frames**2, axis=1) + 1e-12
    keep = 10 * np.log10(e) > 10 * np.log10(e.max()) + floor_db
    return float(np.sqrt(e[keep].mean()))


def add_noise(x: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = pink_noise(len(x), rng)
    sig = active_rms(x, sr)
    noise *= sig / (10 ** (snr_db / 20.0))
    return (x + noise).astype(np.float32)


def peak_normalize_if_clipping(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > peak else x.astype(np.float32)


# ------------------------------------------------------------------ conditions


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    narrowband: bool = False
    mulaw: bool = False
    loss_rate: float = 0.0
    snr_db: float | None = None


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in [
        Condition("clean", "Original audio resampled to 16 kHz"),
        Condition("nb8k", "8 kHz + 300-3400 Hz channel filter, linear PCM (bandwidth only)", narrowband=True),
        Condition("g711", "8 kHz + 300-3400 Hz + G.711 mu-law", narrowband=True, mulaw=True),
        Condition(
            "g711_loss5",
            "g711 + 5% random 20 ms packet loss, repeat-last-frame concealment",
            narrowband=True,
            mulaw=True,
            loss_rate=0.05,
        ),
        Condition(
            "g711_snr10",
            "pink noise at 10 dB SNR (caller side) then g711",
            narrowband=True,
            mulaw=True,
            snr_db=10.0,
        ),
    ]
}


def seed_for(utt_id: str, condition: str, base_seed: int = 20261101) -> int:
    h = hashlib.sha256(f"{base_seed}:{condition}:{utt_id}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def apply_condition(x16k: np.ndarray, cond: Condition | str, utt_id: str = "", sr: int = 16000) -> np.ndarray:
    """Apply a telephony condition to 16 kHz float audio; returns 16 kHz float audio.

    Order: (noise at the caller) -> resample to 8 kHz -> channel filter -> mu-law
    -> packet loss (on the decoded 8 kHz stream, as at the receiving jitter buffer)
    -> resample back to 16 kHz for the ASR front-end.
    """
    if isinstance(cond, str):
        cond = CONDITIONS[cond]
    x = np.asarray(x16k, dtype=np.float32)
    if not cond.narrowband and cond.snr_db is None and cond.loss_rate == 0 and not cond.mulaw:
        return x
    rng = np.random.default_rng(seed_for(utt_id, cond.name))
    if cond.snr_db is not None:
        x = add_noise(x, sr, cond.snr_db, rng)
    if cond.narrowband:
        x = resample(x, sr, 8000)
        x = telephone_bandpass(x, 8000)
        work_sr = 8000
    else:
        work_sr = sr
    x = peak_normalize_if_clipping(x)
    if cond.mulaw:
        x = g711_mulaw_roundtrip(x)
    if cond.loss_rate > 0:
        x, _ = packet_loss(x, work_sr, cond.loss_rate, rng)
    if work_sr != sr:
        x = resample(x, work_sr, sr)
    return x.astype(np.float32)
