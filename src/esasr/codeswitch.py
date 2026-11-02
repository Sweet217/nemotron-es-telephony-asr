"""Build the SYNTHETIC Spanish-English code-switching set with macOS `say`.

No open, permissively licensed Spanish-English code-switching speech corpus with
transcripts was usable without registration (Bangor Miami is CC-BY-NC-SA and
distributed via TalkBank login; LDC sets are paid), so this set is synthesized:
hand-written contact-center sentences (codeswitch/*.tsv) are spoken by the macOS
es_MX system voices. Every manifest row carries ``"synthetic": true``.

Two sets are written:
  * codeswitch_synth       - Spanish sentences with English insertions ({...} spans)
  * spanish_synth_control  - monolingual Spanish sentences, same voices/register
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SRC = ROOT / "codeswitch"

VOICES = {
    "paulina": "Paulina (Spanish (Mexico))",
    "eddy": "Eddy (Spanish (Mexico))",
    "flo": "Flo (Spanish (Mexico))",
    "reed": "Reed (Spanish (Mexico))",
}
_SPAN = re.compile(r"\{([^}]*)\}")


def read_prompts(path: Path) -> list[tuple[str, str, list[int]]]:
    """Return (id, plain_text, english_word_indices) per prompt.

    English word indices refer to whitespace tokens of the *plain* text.
    """
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        pid, marked = line.split("\t", 1)
        plain_tokens: list[str] = []
        en_idx: list[int] = []
        pos = 0
        for m in _SPAN.finditer(marked):
            before = marked[pos : m.start()].split()
            plain_tokens += before
            for w in m.group(1).split():
                en_idx.append(len(plain_tokens))
                plain_tokens.append(w)
            pos = m.end()
        plain_tokens += marked[pos:].split()
        # glue punctuation-only tokens (e.g. the "," in "{Okay}, entonces") to the previous token
        merged: list[str] = []
        idx_map: dict[int, int] = {}
        for i, tok in enumerate(plain_tokens):
            if merged and not any(ch.isalnum() for ch in tok):
                merged[-1] += tok
            else:
                merged.append(tok)
            idx_map[i] = len(merged) - 1
        en_idx = sorted({idx_map[i] for i in en_idx})
        out.append((pid, " ".join(merged), en_idx))
    return out


def available_voices() -> set[str]:
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    names = set()
    for line in out.splitlines():
        # name is everything before the locale column, e.g. "Eddy (Spanish (Mexico)) es_MX # ..."
        m = re.match(r"^(.*?)\s+[a-z]{2}_[A-Z]{2}\s+#", line)
        if m:
            names.add(m.group(1).strip())
    return names


def synth(text: str, voice: str, out_wav: Path) -> float:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "x.wav"
        subprocess.run(
            ["say", "-v", voice, "--data-format=LEI16@16000", "-o", str(tmp), text],
            check=True,
        )
        audio, sr = sf.read(tmp, dtype="float32")
    sf.write(out_wav, audio, sr, subtype="PCM_16")
    return len(audio) / sr


def _build(name: str, tsv: Path) -> None:
    have = available_voices()
    voices = {k: v for k, v in VOICES.items() if v in have}
    if not voices:
        raise RuntimeError("No macOS es_MX voices found (this set can only be built on macOS).")
    rows = []
    for pid, text, en_idx in read_prompts(tsv):
        for vkey, vname in voices.items():
            rid = f"{pid}_{vkey}"
            wav = DATA / name / "wav" / f"{rid}.wav"
            dur = synth(text, vname, wav) if not wav.exists() else sf.info(wav).duration
            rows.append(
                dict(
                    id=rid,
                    audio=str(wav.relative_to(ROOT)),
                    text=text,
                    set=name,
                    accent="es-MX (TTS)",
                    speaker=f"say:{vname}",
                    gender="u",
                    duration=round(dur, 3),
                    source="synthetic: macOS say es_MX voice, prompts in codeswitch/",
                    synthetic=True,
                    english_word_idx=en_idx,
                )
            )
    out = DATA / name / "manifest.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{name}] {len(rows)} synthetic utterances ({len(voices)} voices) -> {out.relative_to(ROOT)}")


def build_synthetic() -> None:
    _build("codeswitch_synth", SRC / "sentences_codeswitch.tsv")
    _build("spanish_synth_control", SRC / "sentences_spanish_control.tsv")


if __name__ == "__main__":
    build_synthetic()
