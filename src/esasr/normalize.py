"""Spanish (and Spanish-English code-switched) text normalization for WER scoring.

Design goals:
  * symmetric: the same function is applied to references and hypotheses;
  * conservative: only removes differences that are formatting, not recognition
    (case, punctuation, digits vs. words, optional diacritics);
  * transparent: every rule is listed here and tested in tests/test_normalize.py.

Rules, in order:
  1. Unicode NFKC; strip ASR language tags such as ``<es-US>`` and bracketed
     annotations such as ``[ruido]`` / ``<sil>``.
  2. Lowercase.
  3. Expand numbers with num2words (Spanish): integers ("1300" -> "mil trescientos"),
     thousands separators ("1.300" / "1,300" -> 1300), decimals ("3,5" -> "tres coma
     cinco"), percentages ("%" -> "por ciento"), ordinals "1º/1ª" -> "primero/primera".
  4. Map a few symbols to words ("&" -> "y", "+" -> "más").
  5. Remove punctuation (all Unicode P* categories, plus symbols), keeping
     word-internal apostrophes so English contractions ("don't") survive.
  6. Optionally fold diacritics (á->a, ü->u) while KEEPING "ñ", which is a distinct
     letter in Spanish. Default: fold (references across corpora are inconsistent
     about accents; see README for the accent-sensitive numbers).
  7. Collapse whitespace.
"""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

_LANG_TAG = re.compile(r"<\s*[a-z]{2,3}(?:-[A-Za-z]{2,4})?\s*>", re.IGNORECASE)
_ANNOT = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_PERCENT = re.compile(r"(\d)\s*%")
_ORD = re.compile(r"\b(\d+)\s*([ºª°])")
_THOUSANDS = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+\b")
_DECIMAL = re.compile(r"\b(\d+)[.,](\d+)\b")
_INT = re.compile(r"\d+")
_APOS = re.compile(r"[’`´]")

_SYMBOLS = {"&": " y ", "+": " más ", "=": " igual ", "@": " arroba "}


def _num(n: str) -> str:
    try:
        return " " + num2words(int(n), lang="es") + " "
    except Exception:  # pragma: no cover - absurdly long digit strings
        return " " + " ".join(num2words(int(d), lang="es") for d in n) + " "


def expand_numbers(text: str) -> str:
    text = _PERCENT.sub(r"\1 por ciento", text)

    def ordinal(m: re.Match) -> str:
        n = int(m.group(1))
        word = num2words(n, lang="es", to="ordinal")
        if m.group(2) == "ª" and word.endswith("o"):
            word = word[:-1] + "a"
        return " " + word + " "

    text = _ORD.sub(ordinal, text)
    text = _THOUSANDS.sub(lambda m: re.sub(r"[.,]", "", m.group(0)), text)
    text = _DECIMAL.sub(lambda m: f"{_num(m.group(1))} coma {_num(m.group(2))}", text)
    text = _INT.sub(lambda m: _num(m.group(0)), text)
    return text


def fold_diacritics(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) == "Mn":
            # keep the tilde of ñ (n + U+0303)
            if ch == "̃" and out and out[-1] in ("n", "N"):
                out.append(ch)
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _strip_punct(text: str) -> str:
    text = _APOS.sub("'", text)
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == "'":
            chars.append(ch)
        elif cat.startswith("P") or cat.startswith("S"):
            chars.append(" ")
        else:
            chars.append(ch)
    text = "".join(chars)
    # keep apostrophes only when word-internal (don't, y'all)
    text = re.sub(r"(?<![^\W\d_])'|'(?![^\W\d_])", " ", text)
    return text


def normalize(text: str, *, fold_accents: bool = True) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _LANG_TAG.sub(" ", t)
    t = _ANNOT.sub(" ", t)
    t = t.lower()
    t = expand_numbers(t)
    for k, v in _SYMBOLS.items():
        t = t.replace(k, v)
    t = _strip_punct(t)
    if fold_accents:
        t = fold_diacritics(t)
    return " ".join(t.split())


def strip_lang_tags(text: str) -> str:
    """Remove Nemotron's trailing ``<xx-XX>`` language tag (for display)."""
    return " ".join(_LANG_TAG.sub(" ", text or "").split())


def detected_lang_tag(text: str) -> str | None:
    m = _LANG_TAG.search(text or "")
    return m.group(0).strip("<> ").strip() if m else None
