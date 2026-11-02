"""Run one ASR system over (sets x telephony conditions) and store raw hypotheses.

Output: results/hyps/<system>/<set>/<condition>.jsonl, one line per utterance:
    {"id", "hyp", "audio_sec", "proc_sec"}
Runs are resumable: utterances already present in the output file are skipped.
Scoring happens separately in esasr.report, so changing the normalizer never
requires re-running a model.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .codec import CONDITIONS, apply_condition, resample
from .systems import make_system

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DEFAULT_SETS = [
    "fleurs_es419",
    "ciempiess_mx",
    "openslr_ar",
    "openslr_cl",
    "openslr_co",
    "openslr_pe",
    "openslr_pr",
    "openslr_ve",
    "codeswitch_synth",
    "spanish_synth_control",
]


def load_manifest(name: str) -> list[dict]:
    p = DATA / name / "manifest.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `make data` first")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# Every utterance is padded with 0.5 s of silence on both sides before the channel is
# simulated (so noise also covers the pad). This mirrors what a VAD front-end hands to
# a segmented STT service (Pipecat's SegmentedSTTService keeps pre-roll and appends
# 0.5 s of trailing silence) and avoids penalising models for clipped onsets in
# tightly trimmed corpora. Applied identically to every system.
PAD_SEC = 0.5


def pad(x: np.ndarray, sec: float = PAD_SEC, sr: int = 16000) -> np.ndarray:
    z = np.zeros(int(sr * sec), dtype=np.float32)
    return np.concatenate([z, x.astype(np.float32), z])


def load_audio(row: dict) -> np.ndarray:
    a, sr = sf.read(ROOT / row["audio"], dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return resample(a, sr, 16000) if sr != 16000 else a


def run(system_spec: str, sets: list[str], conditions: list[str], limit: int | None = None) -> None:
    system = make_system(system_spec)
    print(f"system: {system.description}", flush=True)
    meta_dir = RESULTS / "hyps" / system.name
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "system.json").write_text(
        json.dumps(
            {
                **system.info(),
                "spec": system_spec,
                "machine": f"{platform.platform()} / {platform.processor()} / py{sys.version.split()[0]}",
            },
            indent=2,
        )
    )
    for s in sets:
        rows = load_manifest(s)
        if limit:
            rows = rows[:limit]
        for c in conditions:
            out = meta_dir / s / f"{c}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = set()
            if out.exists():
                done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
            todo = [r for r in rows if r["id"] not in done]
            if not todo:
                print(f"[{system.name}] {s}/{c}: done ({len(done)})", flush=True)
                continue
            t_audio = t_proc = 0.0
            with out.open("a") as f:
                for r in todo:
                    x = apply_condition(pad(load_audio(r)), CONDITIONS[c], utt_id=r["id"])
                    t0 = time.perf_counter()
                    hyp = system.transcribe(x, 16000)
                    dt = time.perf_counter() - t0
                    t_audio += len(x) / 16000
                    t_proc += dt
                    f.write(
                        json.dumps(
                            {"id": r["id"], "hyp": hyp, "audio_sec": round(len(x) / 16000, 3), "proc_sec": round(dt, 4)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()
            rtfx = t_audio / t_proc if t_proc else float("nan")
            print(f"[{system.name}] {s}/{c}: {len(todo)} utts, {t_audio/60:.1f} min audio, RTFx {rtfx:.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, help="see esasr.systems.make_system")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None, help="first N utterances per set (smoke tests)")
    a = ap.parse_args()
    sets = [s for s in a.sets if (DATA / s / "manifest.jsonl").exists() or print(f"skip {s}: no data")]
    run(a.system, sets, a.conditions, a.limit)


if __name__ == "__main__":
    main()
