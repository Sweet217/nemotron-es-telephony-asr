"""Print the headline tables for README.md from results/summary.csv (no hand-typed numbers)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
s = pd.read_csv(R / "summary.csv")
d = pd.read_csv(R / "deltas.csv")
cs = pd.read_csv(R / "codeswitch.csv")

REAL = ["fleurs_es419", "ciempiess_mx", "openslr_ar", "openslr_cl", "openslr_co", "openslr_pe", "openslr_pr", "openslr_ve"]
SYS = [
    ("nemotron35-es-US-320ms", "Nemotron 3.5 ASR, 320 ms"),
    ("nemotron35-es-US-1120ms", "Nemotron 3.5 ASR, 1.12 s"),
    ("whisper-large-v3-turbo-es", "Whisper large-v3-turbo"),
]
CONDS = ["clean", "nb8k", "g711", "g711_loss5", "g711_snr10"]


def cell(sys, st, c, ci=True):
    r = s[(s.system == sys) & (s.set == st) & (s.condition == c)]
    if r.empty:
        return "pending"
    r = r.iloc[0]
    return f"{r.wer:.2f} [{r.wer_lo:.2f}-{r.wer_hi:.2f}]" if ci else f"{r.wer:.2f}"


def pooled(sys, c, sets):
    g = s[(s.system == sys) & (s.condition == c) & (s.set.isin(sets))]
    if len(g) < len(sets):
        return "pending"
    # pooled over sets: weight by words
    return f"{(g.wer * g.n_words).sum() / g.n_words.sum():.2f}"


print("### Headline: WER % on real speech (8 sets, 1,600 utterances), pooled over sets\n")
print("| system | " + " | ".join(CONDS) + " |")
print("|---|" + "---|" * len(CONDS))
for k, name in SYS:
    print(f"| {name} | " + " | ".join(pooled(k, c, REAL) for c in CONDS) + " |")
print()
print("### WER % by accent, clean vs G.711 (95% bootstrap CI)\n")
print("| set | accent | " + " | ".join(f"{n} clean | {n} g711" for _, n in SYS) + " |")
print("|---|---|" + "---|---|" * len(SYS))
for st in REAL:
    acc = s[s.set == st].accent.iloc[0]
    cells = []
    for k, _ in SYS:
        cells += [cell(k, st, "clean"), cell(k, st, "g711")]
    print(f"| {st} | {acc} | " + " | ".join(cells) + " |")
print()
print("### Degradation vs clean, WER points (paired bootstrap 95% CI), Nemotron 320 ms\n")
print("| set | " + " | ".join(CONDS[1:]) + " |")
print("|---|" + "---|" * 4)
for st in REAL:
    row = []
    for c in CONDS[1:]:
        r = d[(d.system == "nemotron35-es-US-320ms") & (d.set == st) & (d.condition == c)]
        row.append("pending" if r.empty else f"{r.iloc[0].delta_wer:+.2f} [{r.iloc[0].lo:+.2f}, {r.iloc[0].hi:+.2f}]")
    print(f"| {st} | " + " | ".join(row) + " |")
print()
print("### Synthetic sets (macOS TTS, es_MX voices): WER %, clean / g711\n")
print("| system | Spanish control | Spanglish |")
print("|---|---|---|")
for k, name in [("nemotron35-es-US-320ms", "Nemotron 320 ms, es-US prompt"),
                ("nemotron35-auto-320ms", "Nemotron 320 ms, auto LID"),
                ("whisper-large-v3-turbo-es", "Whisper large-v3-turbo")]:
    print(f"| {name} | {cell(k,'spanish_synth_control','clean',False)} / {cell(k,'spanish_synth_control','g711',False)} | "
          f"{cell(k,'codeswitch_synth','clean',False)} / {cell(k,'codeswitch_synth','g711',False)} |")
print()
print("### Spanglish: share of English vs Spanish reference words missed (Paulina voice only, clean)\n")
print("| system | English words missed % | Spanish words missed % |")
print("|---|---|---|")
for k, name in [("nemotron35-es-US-320ms", "Nemotron 320 ms, es-US"), ("nemotron35-auto-320ms", "Nemotron 320 ms, auto"),
                ("whisper-large-v3-turbo-es", "Whisper large-v3-turbo")]:
    r = cs[(cs.system == k) & (cs.voices == "Paulina only") & (cs.condition == "clean")]
    if not r.empty:
        r = r.iloc[0]
        print(f"| {name} | {r.en_miss_pct:.1f} (n={r.en_words}) | {r.es_miss_pct:.1f} (n={r.es_words}) |")
print()
print("### Speed on this machine (RTFx = audio seconds / processing seconds, clean condition, all real sets)\n")
for k, name in SYS:
    g = s[(s.system == k) & (s.condition == "clean") & (s.set.isin(REAL))]
    if len(g):
        print(f"- {name}: median RTFx {g.rtfx.median():.0f}")
