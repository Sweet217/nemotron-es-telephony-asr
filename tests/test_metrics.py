import numpy as np
import pytest

from esasr.metrics import (
    aligned_word_hits,
    bootstrap_ci,
    char_counts,
    corpus_rate,
    paired_bootstrap_delta,
    word_counts,
)


def test_word_counts_basic():
    c = word_counts("el gato negro", "el perro negro grande")
    assert (c.sub, c.dele, c.ins, c.hit) == (1, 0, 1, 2)
    assert c.errors == 2 and c.ref_len == 3


def test_empty_hyp_and_ref():
    c = word_counts("hola mundo", "")
    assert c.dele == 2 and c.errors == 2
    c = word_counts("", "ruido")
    assert c.ins == 1 and c.ref_len == 0


def test_char_counts():
    c = char_counts("casa", "cosa")
    assert c.sub == 1 and c.ref_len == 4


def test_corpus_rate_is_pooled_not_mean():
    # utt1: 1 error / 1 word, utt2: 0 errors / 9 words -> pooled 10%, mean-of-utts 50%
    assert corpus_rate(np.array([1, 0]), np.array([1, 9])) == pytest.approx(0.1)


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    refs = rng.integers(5, 20, 300)
    errs = rng.binomial(refs, 0.1)
    p, lo, hi = bootstrap_ci(errs, refs, n_boot=500, seed=0)
    assert lo < p < hi
    assert 0.07 < lo and hi < 0.13
    assert bootstrap_ci(errs, refs, n_boot=500, seed=0) == (p, lo, hi)


def test_paired_delta_detects_degradation():
    rng = np.random.default_rng(1)
    refs = rng.integers(5, 20, 300)
    a = rng.binomial(refs, 0.05)
    b = a + rng.binomial(refs, 0.05)
    d, lo, hi, p_not_worse = paired_bootstrap_delta(a, b, refs, n_boot=500)
    assert d > 0 and lo > 0
    assert p_not_worse < 0.01


def test_aligned_word_hits():
    assert aligned_word_hits("quiero un refund ya", "quiero un refun ya") == [True, True, False, True]
    assert aligned_word_hits("hola", "") == [False]
