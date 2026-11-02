import shutil
import subprocess

import numpy as np
import pytest

from esasr import codec


def test_decode_table_is_inverse_of_encode():
    codes = np.arange(256, dtype=np.uint8)
    re = codec.mulaw_encode(codec.mulaw_decode(codes))
    # 0x7F and 0xFF both decode to 0; encode(0) == 0xFF. Every other code round-trips.
    assert (re == codes).sum() == 255
    assert codec.mulaw_decode(np.array([0x7F, 0xFF], dtype=np.uint8)).tolist() == [0, 0]


def test_decode_known_values():
    # ITU-T G.711 mu-law extremes: 0x00 -> -32124, 0x80 -> +32124
    assert codec.mulaw_decode(np.array([0x00, 0x80], dtype=np.uint8)).tolist() == [-32124, 32124]


def test_encode_is_monotonic_in_magnitude():
    x = np.arange(0, 32768, dtype=np.int16)
    dec = codec.mulaw_decode(codec.mulaw_encode(x)).astype(int)
    assert np.all(np.diff(dec) >= 0)


def test_quantization_snr_is_g711_like():
    sr = 8000
    t = np.arange(sr) / sr
    x = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = codec.g711_mulaw_roundtrip(x)
    snr = 10 * np.log10(np.sum(x**2) / np.sum((x - y) ** 2))
    assert 33 < snr < 40  # mu-law gives ~38 dB SQNR for loud signals


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_matches_ffmpeg(tmp_path):
    x = np.arange(-32768, 32768, dtype=np.int16)
    raw = tmp_path / "ramp.s16"
    ul = tmp_path / "ramp.ul"
    dec = tmp_path / "dec.s16"
    x.tofile(raw)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", "8000", "-ac", "1",
                    "-i", str(raw), "-c:a", "pcm_mulaw", "-f", "mulaw", str(ul)], check=True)
    ff = np.fromfile(ul, dtype=np.uint8)
    ours = codec.mulaw_encode(x)
    mism = ff != ours
    # Decoder is bit-exact. Encoder follows the ITU/Sun g711.c reference; ffmpeg builds
    # its table differently and disagrees on <1% of inputs, always by one adjacent code
    # exactly at a decision boundary.
    assert mism.mean() < 0.01
    assert np.all(np.abs(ff[mism].astype(int) - ours[mism].astype(int)) == 1)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "mulaw", "-ar", "8000", "-ac", "1",
                    "-i", str(ul), "-f", "s16le", str(dec)], check=True)
    assert np.array_equal(np.fromfile(dec, dtype=np.int16), codec.mulaw_decode(ff))


def test_narrowband_removes_high_frequencies():
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    y = codec.apply_condition(x, "nb8k", utt_id="t")
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    p1k = spec[(f > 950) & (f < 1050)].max()
    p6k = spec[(f > 5950) & (f < 6050)].max()
    assert p6k < p1k * 1e-3
    assert len(y) == len(x)


def test_conditions_are_deterministic_and_seeded_per_utterance():
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(16000)).astype(np.float32)
    a = codec.apply_condition(x, "g711_snr10", utt_id="u1")
    b = codec.apply_condition(x, "g711_snr10", utt_id="u1")
    c = codec.apply_condition(x, "g711_snr10", utt_id="u2")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_clean_is_identity():
    x = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    assert np.array_equal(codec.apply_condition(x, "clean"), x)


def test_packet_loss_rate_and_concealment():
    sr = 8000
    x = np.ones(sr * 20, dtype=np.float32)  # 20 s -> 1000 frames of 20 ms
    y, lost = codec.packet_loss(x, sr, 0.05, np.random.default_rng(1), concealment="zero")
    assert 0.03 < lost.mean() < 0.07
    n = sr * 20 // 1000
    first = int(np.argmax(lost))
    assert np.all(y[first * n:(first + 1) * n] == 0)
    y2, _ = codec.packet_loss(x, sr, 0.05, np.random.default_rng(1), concealment="repeat")
    assert np.allclose(y2[first * n:(first + 1) * n], 0.5)


def test_add_noise_hits_target_snr():
    sr = 16000
    t = np.arange(sr * 2) / sr
    x = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    y = codec.add_noise(x, sr, 10.0, np.random.default_rng(0))
    snr = 10 * np.log10(np.mean(x**2) / np.mean((y - x) ** 2))
    assert abs(snr - 10.0) < 0.5
