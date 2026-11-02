"""WER/CER with per-utterance error counts and bootstrap confidence intervals.

Corpus-level WER is computed the standard way: sum(S+D+I) / sum(N_ref) over all
utterances (not the mean of per-utterance WERs). Confidence intervals come from a
percentile bootstrap that resamples *utterances* with replacement, which respects
the fact that errors are correlated within an utterance. For comparisons between two
systems or two conditions on the same utterances we use a *paired* bootstrap (the
same resampled utterance indices for both sides).
"""
from __future__ import annotations

from dataclasses import dataclass

import jiwer
import numpy as np


@dataclass
class UttCounts:
    sub: int
    dele: int
    ins: int
    hit: int

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def ref_len(self) -> int:
        return self.sub + self.dele + self.hit


def word_counts(ref: str, hyp: str) -> UttCounts:
    """Edit counts for already-normalized strings (whitespace tokenised)."""
    if not ref.split():
        # No reference words: every hypothesis word is an insertion.
        return UttCounts(0, 0, len(hyp.split()), 0)
    if not hyp.split():
        return UttCounts(0, len(ref.split()), 0, 0)
    o = jiwer.process_words(ref, hyp)
    return UttCounts(o.substitutions, o.deletions, o.insertions, o.hits)


def char_counts(ref: str, hyp: str) -> UttCounts:
    """Character edit counts (spaces kept as characters, as jiwer.cer does)."""
    if not ref:
        return UttCounts(0, 0, len(hyp), 0)
    if not hyp:
        return UttCounts(0, len(ref), 0, 0)
    o = jiwer.process_characters(ref, hyp)
    return UttCounts(o.substitutions, o.deletions, o.insertions, o.hits)


def corpus_rate(errors: np.ndarray, ref_lens: np.ndarray) -> float:
    n = float(np.sum(ref_lens))
    return float(np.sum(errors)) / n if n > 0 else float("nan")


def bootstrap_ci(
    errors: np.ndarray,
    ref_lens: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for corpus error rate with a percentile bootstrap."""
    errors = np.asarray(errors, dtype=np.float64)
    ref_lens = np.asarray(ref_lens, dtype=np.float64)
    point = corpus_rate(errors, ref_lens)
    n = len(errors)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    e = errors[idx].sum(axis=1)
    r = ref_lens[idx].sum(axis=1)
    rates = np.divide(e, r, out=np.full_like(e, np.nan), where=r > 0)
    lo, hi = np.nanpercentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_delta(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    ref_lens: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Delta = rate(b) - rate(a) on the same utterances.

    Returns (delta, lo, hi, p_b_not_worse) where the last value is the fraction of
    bootstrap replicates in which b is not worse than a (a one-sided p-value for
    "b is worse than a").
    """
    ea = np.asarray(errors_a, dtype=np.float64)
    eb = np.asarray(errors_b, dtype=np.float64)
    r = np.asarray(ref_lens, dtype=np.float64)
    assert ea.shape == eb.shape == r.shape
    delta = corpus_rate(eb, r) - corpus_rate(ea, r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    rs = r[idx].sum(axis=1)
    d = (eb[idx].sum(axis=1) - ea[idx].sum(axis=1)) / rs
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(delta), float(lo), float(hi), float(np.mean(d <= 0))


def aligned_word_hits(ref: str, hyp: str) -> list[bool]:
    """For each reference word, whether it was recognised exactly (aligned 'equal')."""
    words = ref.split()
    if not words:
        return []
    if not hyp.split():
        return [False] * len(words)
    out = [False] * len(words)
    o = jiwer.process_words(ref, hyp)
    for chunk in o.alignments[0]:
        if chunk.type == "equal":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                out[i] = True
    return out
