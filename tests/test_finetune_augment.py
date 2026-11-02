import numpy as np

from finetune.g711_augment import TelephonyAugment


def _tone(sr=16000, sec=1.0):
    t = np.arange(int(sr * sec)) / sr
    return (0.2 * np.sin(2 * np.pi * 1000 * t) + 0.2 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)


def test_p0_is_identity():
    x = _tone()
    assert np.array_equal(TelephonyAugment(p=0.0, seed=0)(x), x)


def test_p1_band_limits_and_keeps_length():
    x = _tone()
    y = TelephonyAugment(p=1.0, p_noise=0.0, max_loss=0.0, seed=0)(x)
    assert len(y) == len(x) and y.dtype == np.float32
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / 16000)
    assert spec[(f > 5900) & (f < 6100)].max() < 1e-3 * spec[(f > 900) & (f < 1100)].max()


def test_apply_batch_respects_lengths():
    import torch

    x = torch.zeros(2, 16000)
    x[0] = torch.from_numpy(_tone())
    x[1, :8000] = torch.from_numpy(_tone(sec=0.5))
    lens = torch.tensor([16000, 8000])
    y = TelephonyAugment(p=1.0, seed=1).apply_batch(x, lens)
    assert y.shape == x.shape
    assert torch.all(y[1, 8000:] == 0)  # padding untouched
