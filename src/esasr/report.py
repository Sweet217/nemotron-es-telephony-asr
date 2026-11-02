"""Score stored hypotheses and write result tables.

Reads data/<set>/manifest.jsonl and results/hyps/<system>/<set>/<condition>.jsonl,
normalizes both sides with esasr.normalize, and writes:

  results/scores_utt.csv        per (system, set, condition, utterance) edit counts
  results/summary.csv           corpus WER/CER + 95% bootstrap CI per cell
  results/deltas.csv            paired-bootstrap degradation vs clean, per system/set
  results/codeswitch.csv        English-word vs Spanish-word error on the Spanglish set
  results/RESULTS.md            all of the above as Markdown tables

Only cells where *every* utterance of the set has a hypothesis are reported, so a
partially finished run never leaks into the tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .codec import CONDITIONS
from .metrics import aligned_word_hits, bootstrap_ci, char_counts, paired_bootstrap_delta, word_counts
from .normalize import detected_lang_tag, normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

SET_ORDER = [
    "fleurs_es419",
    "ciempiess_mx",
    "openslr_ar",
    "openslr_cl",
    "openslr_co",
    "openslr_pe",
    "openslr_pr",
    "openslr_ve",
    "spanish_synth_control",
    "codeswitch_synth",
]
COND_ORDER = list(CONDITIONS)
N_BOOT = 1000


def _manifest(name: str) -> dict[str, dict]:
    p = DATA / name / "manifest.jsonl"
    return {r["id"]: r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip())}


def _hyps(p: Path) -> dict[str, dict]:
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def score(fold_accents: bool = True) -> pd.DataFrame:
    rows = []
    for sysdir in sorted((RESULTS / "hyps").glob("*")):
        if not sysdir.is_dir():
            continue
        for setdir in sorted(sysdir.glob("*")):
            if not setdir.is_dir() or not (DATA / setdir.name / "manifest.jsonl").exists():
                continue
            man = _manifest(setdir.name)
            for cfile in sorted(setdir.glob("*.jsonl")):
                hyps = _hyps(cfile)
                if set(man) - set(hyps):
                    print(f"incomplete, skipped: {sysdir.name}/{setdir.name}/{cfile.stem} "
                          f"({len(hyps)}/{len(man)})")
                    continue
                for uid, m in man.items():
                    h = hyps[uid]
                    ref = normalize(m["text"], fold_accents=fold_accents)
                    hyp = normalize(h["hyp"], fold_accents=fold_accents)
                    w = word_counts(ref, hyp)
                    c = char_counts(ref, hyp)
                    rows.append(
                        dict(
                            system=sysdir.name,
                            set=setdir.name,
                            condition=cfile.stem,
                            id=uid,
                            accent=m["accent"],
                            speaker=m.get("speaker", ""),
                            w_err=w.errors,
                            w_sub=w.sub,
                            w_del=w.dele,
                            w_ins=w.ins,
                            w_ref=w.ref_len,
                            c_err=c.errors,
                            c_ref=c.ref_len,
                            empty_hyp=int(not hyp),
                            lang_tag=detected_lang_tag(h["hyp"]) or "",
                            audio_sec=h.get("audio_sec", 0.0),
                            proc_sec=h.get("proc_sec", 0.0),
                        )
                    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (s, st, c), g in df.groupby(["system", "set", "condition"], sort=False):
        wer, wlo, whi = bootstrap_ci(g.w_err.values, g.w_ref.values, N_BOOT, seed=1)
        cer, clo, chi = bootstrap_ci(g.c_err.values, g.c_ref.values, N_BOOT, seed=2)
        out.append(
            dict(
                system=s,
                set=st,
                condition=c,
                accent=g.accent.iloc[0],
                n_utt=len(g),
                n_words=int(g.w_ref.sum()),
                wer=100 * wer,
                wer_lo=100 * wlo,
                wer_hi=100 * whi,
                cer=100 * cer,
                cer_lo=100 * clo,
                cer_hi=100 * chi,
                sub=100 * g.w_sub.sum() / g.w_ref.sum(),
                dele=100 * g.w_del.sum() / g.w_ref.sum(),
                ins=100 * g.w_ins.sum() / g.w_ref.sum(),
                empty_hyps=int(g.empty_hyp.sum()),
                rtfx=g.audio_sec.sum() / max(g.proc_sec.sum(), 1e-9),
            )
        )
    return pd.DataFrame(out)


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (s, st), g in df.groupby(["system", "set"], sort=False):
        base = g[g.condition == "clean"].set_index("id")
        if base.empty:
            continue
        for c in COND_ORDER[1:]:
            o = g[g.condition == c].set_index("id")
            if o.empty:
                continue
            o = o.loc[base.index]
            d, lo, hi, p = paired_bootstrap_delta(base.w_err.values, o.w_err.values, base.w_ref.values, N_BOOT, seed=3)
            out.append(dict(system=s, set=st, condition=c, delta_wer=100 * d, lo=100 * lo, hi=100 * hi,
                            p_not_worse=p))
    return pd.DataFrame(out)


def codeswitch_breakdown(fold_accents: bool = True) -> pd.DataFrame:
    """Per-language recognition on the Spanglish set: error on English vs Spanish ref words.

    'word error' here = 1 - exact aligned hit rate for that subset of reference words
    (insertions can't be attributed to a language, so this is a recall-style measure).
    """
    name = "codeswitch_synth"
    if not (DATA / name / "manifest.jsonl").exists():
        return pd.DataFrame()
    man = _manifest(name)
    out = []
    for sysdir in sorted((RESULTS / "hyps").glob("*")):
        for cfile in sorted((sysdir / name).glob("*.jsonl")):
            hyps = _hyps(cfile)
            if set(man) - set(hyps):
                continue
            for group, keep in (("all 4 voices", lambda m: True),
                                ("Paulina only", lambda m: "Paulina" in m.get("speaker", ""))):
                en_hit = en_n = es_hit = es_n = 0
                for uid, m in man.items():
                    if not keep(m):
                        continue
                    toks = m["text"].split()
                    en = set(m.get("english_word_idx", []))
                    # normalise token by token so the English tags stay aligned
                    ref_toks, is_en = [], []
                    for i, t in enumerate(toks):
                        for nt in normalize(t, fold_accents=fold_accents).split():
                            ref_toks.append(nt)
                            is_en.append(i in en)
                    hits = aligned_word_hits(" ".join(ref_toks),
                                             normalize(hyps[uid]["hyp"], fold_accents=fold_accents))
                    for h, e in zip(hits, is_en):
                        if e:
                            en_n += 1
                            en_hit += h
                        else:
                            es_n += 1
                            es_hit += h
                out.append(dict(system=sysdir.name, condition=cfile.stem, voices=group,
                                en_words=en_n, en_miss_pct=100 * (1 - en_hit / en_n),
                                es_words=es_n, es_miss_pct=100 * (1 - es_hit / es_n)))
    return pd.DataFrame(out)


def by_voice(df: pd.DataFrame) -> pd.DataFrame:
    """WER per TTS voice on the synthetic sets (voices differ a lot in quality)."""
    syn = df[df.set.isin(["codeswitch_synth", "spanish_synth_control"])]
    out = []
    for (s, st, c, v), g in syn.groupby(["system", "set", "condition", "speaker"], sort=False):
        out.append(dict(system=s, set=st, condition=c, voice=v.replace("say:", ""),
                        wer=100 * g.w_err.sum() / g.w_ref.sum(), empty_hyps=int(g.empty_hyp.sum()), n_utt=len(g)))
    return pd.DataFrame(out)


def runtime_crosscheck() -> pd.DataFrame:
    """Agreement between the Transformers port and the NeMo reference runtime."""
    a_dir, b_dir = RESULTS / "hyps" / "nemotron35-es-US-320ms", RESULTS / "hyps" / "nemotron35-nemo-es-US-320ms"
    out = []
    if not b_dir.exists():
        return pd.DataFrame()
    for f in sorted(b_dir.glob("*/*.jsonl")):
        g = a_dir / f.parent.name / f.name
        if not g.exists():
            continue
        man = _manifest(f.parent.name)
        ha, hb = _hyps(g), _hyps(f)
        if set(man) - set(ha) or set(man) - set(hb):
            continue
        same = sum(normalize(ha[u]["hyp"]) == normalize(hb[u]["hyp"]) for u in man)
        e = sum(word_counts(normalize(ha[u]["hyp"]), normalize(hb[u]["hyp"])).errors for u in man)
        n = sum(len(normalize(ha[u]["hyp"]).split()) for u in man)
        out.append(dict(set=f.parent.name, condition=f.stem, n_utt=len(man),
                        identical_pct=100 * same / len(man), cross_wer=100 * e / max(n, 1)))
    return pd.DataFrame(out)


def _fmt(v: float, lo: float, hi: float) -> str:
    return f"{v:.2f} [{lo:.2f}, {hi:.2f}]"


def to_markdown(summ: pd.DataFrame, dl: pd.DataFrame, cs: pd.DataFrame, summ_acc: pd.DataFrame | None,
                bv: pd.DataFrame | None = None) -> str:
    lines = ["# Results", "",
             "Generated by `make report` from `results/hyps/`. WER/CER in %, 95% bootstrap CI "
             f"({N_BOOT} utterance resamples). Normalization: lowercase, no punctuation, numbers "
             "as Spanish words, diacritics folded (ñ kept). Every number here was produced on the "
             "machine listed in `results/hyps/<system>/system.json`.", ""]
    systems = list(dict.fromkeys(summ.system))
    for s in systems:
        g = summ[summ.system == s]
        lines += [f"## {s}", "", "### WER by set and condition", ""]
        conds = [c for c in COND_ORDER if c in set(g.condition)]
        lines.append("| set | accent | n utt | " + " | ".join(conds) + " |")
        lines.append("|---|---|---|" + "---|" * len(conds))
        for st in [x for x in SET_ORDER if x in set(g.set)]:
            gg = g[g.set == st].set_index("condition")
            cells = [(_fmt(gg.loc[c, "wer"], gg.loc[c, "wer_lo"], gg.loc[c, "wer_hi"]) if c in gg.index else "pending")
                     for c in conds]
            lines.append(f"| {st} | {gg.accent.iloc[0]} | {gg.n_utt.iloc[0]} | " + " | ".join(cells) + " |")
        lines += ["", "### CER by set and condition", ""]
        lines.append("| set | " + " | ".join(conds) + " |")
        lines.append("|---|" + "---|" * len(conds))
        for st in [x for x in SET_ORDER if x in set(g.set)]:
            gg = g[g.set == st].set_index("condition")
            cells = [(_fmt(gg.loc[c, "cer"], gg.loc[c, "cer_lo"], gg.loc[c, "cer_hi"]) if c in gg.index else "pending")
                     for c in conds]
            lines.append(f"| {st} | " + " | ".join(cells) + " |")
        d = dl[dl.system == s] if not dl.empty else dl
        if not d.empty:
            lines += ["", "### Degradation vs clean (paired bootstrap, WER points)", ""]
            lines.append("| set | " + " | ".join(conds[1:]) + " |")
            lines.append("|---|" + "---|" * (len(conds) - 1))
            for st in [x for x in SET_ORDER if x in set(d.set)]:
                dd = d[d.set == st].set_index("condition")
                cells = [(f"{dd.loc[c,'delta_wer']:+.2f} [{dd.loc[c,'lo']:+.2f}, {dd.loc[c,'hi']:+.2f}]"
                          if c in dd.index else "pending") for c in conds[1:]]
                lines.append(f"| {st} | " + " | ".join(cells) + " |")
        e = g[g.empty_hyps > 0]
        if not e.empty:
            lines += ["", "Empty hypotheses (counted as full deletions): " +
                      ", ".join(f"{r.set}/{r.condition}: {r.empty_hyps}" for r in e.itertuples())]
        lines.append("")
    if summ_acc is not None and not summ_acc.empty:
        lines += ["## Accent-sensitive WER (diacritics kept), clean vs g711", "",
                  "| system | set | clean | g711 |", "|---|---|---|---|"]
        for (s, st), g in summ_acc.groupby(["system", "set"], sort=False):
            gg = g.set_index("condition")
            c1 = f"{gg.loc['clean','wer']:.2f}" if "clean" in gg.index else "pending"
            c2 = f"{gg.loc['g711','wer']:.2f}" if "g711" in gg.index else "pending"
            lines.append(f"| {s} | {st} | {c1} | {c2} |")
        lines.append("")
    if bv is not None and not bv.empty:
        lines += ["## Synthetic sets: WER by TTS voice (clean / g711)", "",
                  "| system | set | voice | clean WER (empty hyps) | g711 WER (empty hyps) |", "|---|---|---|---|---|"]
        for (s, st, v), g in bv.groupby(["system", "set", "voice"], sort=False):
            gg = g.set_index("condition")
            cell = lambda c: (f"{gg.loc[c,'wer']:.1f} ({gg.loc[c,'empty_hyps']})" if c in gg.index else "pending")
            lines.append(f"| {s} | {st} | {v} | {cell('clean')} | {cell('g711')} |")
        lines.append("")
    xc = runtime_crosscheck()
    if not xc.empty:
        lines += ["## Runtime cross-check: Transformers (MPS) vs NeMo 3.0.0 (CPU), same model, 320 ms", "",
                  "| set | condition | n | identical normalized hypotheses % | WER of NeMo output vs Transformers output |",
                  "|---|---|---|---|---|"]
        for r in xc.itertuples():
            lines.append(f"| {r.set} | {r.condition} | {r.n_utt} | {r.identical_pct:.1f} | {r.cross_wer:.2f} |")
        lines.append("")
    if not cs.empty:
        lines += ["## Code-switching breakdown (synthetic set)", "",
                  "Share of reference words NOT recognised exactly, split by language of the reference word.", "",
                  "| system | voices | condition | English words missed % (n) | Spanish words missed % (n) |",
                  "|---|---|---|---|---|"]
        for r in cs.itertuples():
            lines.append(f"| {r.system} | {r.voices} | {r.condition} | {r.en_miss_pct:.1f} ({r.en_words}) | "
                         f"{r.es_miss_pct:.1f} ({r.es_words}) |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    df = score(fold_accents=True)
    if df.empty:
        print("no complete results yet")
        return
    df.to_csv(RESULTS / "scores_utt.csv", index=False)
    summ = summarize(df)
    summ.to_csv(RESULTS / "summary.csv", index=False, float_format="%.3f")
    dl = deltas(df)
    dl.to_csv(RESULTS / "deltas.csv", index=False, float_format="%.3f")
    df_acc = score(fold_accents=False)
    summ_acc = summarize(df_acc[df_acc.condition.isin(["clean", "g711"])])
    summ_acc.to_csv(RESULTS / "summary_accent_sensitive.csv", index=False, float_format="%.3f")
    cs = codeswitch_breakdown()
    cs.to_csv(RESULTS / "codeswitch.csv", index=False, float_format="%.3f")
    bv = by_voice(df)
    bv.to_csv(RESULTS / "synthetic_by_voice.csv", index=False, float_format="%.3f")
    (RESULTS / "RESULTS.md").write_text(to_markdown(summ, dl, cs, summ_acc, bv))
    print(f"wrote {RESULTS/'RESULTS.md'}")


if __name__ == "__main__":
    main()
