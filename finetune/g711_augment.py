"""On-the-fly telephony augmentation for NeMo fine-tuning.

Wraps the Lhotse prompt-index dataset that Nemotron 3.5 ASR trains with
(``LhotseSpeechToTextBpeDatasetWithPromptIndex``) so that, with probability ``p``,
each training utterance is pushed through a randomly drawn telephony channel
before feature extraction:

    * 8 kHz resample + 300-3400 Hz channel filter + G.711 mu-law   (always, when applied)
    * packet loss: rate ~ U(0, max_loss), 20 ms frames, repeat-last-frame concealment
    * caller-side pink noise: SNR ~ U(snr_lo, snr_hi) dB, with probability p_noise

The audio stays at 16 kHz for the model (we upsample back), so the encoder sees the
telephone band-limit and codec artefacts exactly as it will in production, and the
model keeps working on wideband audio for the (1 - p) clean share.

Implementation reuses ``esasr.codec`` (numpy, unit-tested). Augmentation happens in
the dataloader workers, on CPU, per utterance.
"""
from __future__ import annotations

import numpy as np
import torch

from esasr.codec import (
    add_noise,
    g711_mulaw_roundtrip,
    packet_loss,
    peak_normalize_if_clipping,
    resample,
    telephone_bandpass,
)


class TelephonyAugment:
    def __init__(
        self,
        p: float = 0.7,
        max_loss: float = 0.08,
        p_noise: float = 0.5,
        snr_lo: float = 5.0,
        snr_hi: float = 25.0,
        sample_rate: int = 16000,
        seed: int | None = None,
    ):
        self.p, self.max_loss, self.p_noise = p, max_loss, p_noise
        self.snr_lo, self.snr_hi, self.sr = snr_lo, snr_hi, sample_rate
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return x
        y = np.asarray(x, dtype=np.float32)
        if self.rng.random() < self.p_noise:
            y = add_noise(y, self.sr, float(self.rng.uniform(self.snr_lo, self.snr_hi)), self.rng)
        y = telephone_bandpass(resample(y, self.sr, 8000), 8000)
        y = g711_mulaw_roundtrip(peak_normalize_if_clipping(y))
        loss = float(self.rng.uniform(0, self.max_loss))
        if loss > 0:
            y, _ = packet_loss(y, 8000, loss, self.rng)
        y = resample(y, 8000, self.sr)
        # keep the exact original length so Lhotse/NeMo length bookkeeping stays valid
        if len(y) >= len(x):
            y = y[: len(x)]
        else:
            y = np.pad(y, (0, len(x) - len(y)))
        return y.astype(np.float32)

    def apply_batch(self, audio: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
        out = audio.clone()
        for i in range(audio.shape[0]):
            n = int(audio_lens[i])
            out[i, :n] = torch.from_numpy(self(audio[i, :n].cpu().numpy()))
        return out


def patch_prompt_dataset(augment: TelephonyAugment) -> None:
    """Monkey-patch NeMo's prompt-index dataset to augment the waveform batch.

    __getitem__ returns (audio, audio_lens, tokens, token_lens, prompt_targets); only
    the waveform is changed, so tokenisation and prompt targets are untouched.
    Validation dataloaders are built from the same class, so call this *after*
    ``setup_validation_data`` if validation must stay clean (train.py does).
    """
    from nemo.collections.asr.data import audio_to_text_lhotse_prompt_index as mod

    cls = mod.LhotseSpeechToTextBpeDatasetWithPromptIndex
    if getattr(cls, "_telephony_patched", False):
        return
    orig = cls.__getitem__

    def __getitem__(self, cuts):
        batch = orig(self, cuts)
        if not getattr(self, "_telephony_augment", True):
            return batch
        audio, audio_lens, *rest = batch
        if not getattr(cls, "_logged", False):
            from nemo.utils import logging

            logging.info(f"Telephony augmentation active (p={augment.p}) on batch of {audio.shape[0]}")
            cls._logged = True
        return (augment.apply_batch(audio, audio_lens), audio_lens, *rest)

    cls.__getitem__ = __getitem__
    cls._telephony_patched = True
