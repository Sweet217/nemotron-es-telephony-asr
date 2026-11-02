"""Download fixed, documented evaluation subsets of public Spanish speech corpora.

Every set is written as ``data/<set>/manifest.jsonl`` + ``data/<set>/wav/*.wav``
(16 kHz mono PCM16). Each manifest line:
    {"id", "audio", "text", "set", "accent", "speaker", "gender", "duration", "source"}

The chosen utterance IDs are stored in ``subsets/<set>.txt`` (committed). If that file
exists, it is used verbatim, so every machine evaluates exactly the same utterances.
If not, a subset is drawn with a fixed seed and the file is written.

Download footprint (approximate):
  * FLEURS es_419 test: one 0.70 GB parquet shard (cached in data/raw/)
  * CIEMPIESS test: one 0.51 GB parquet shard (cached in data/raw/)
  * OpenSLR 61/71-75: only the selected WAVs are fetched, using HTTP range requests
    into the remote zip (remotezip); ~0.1 GB per accent.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .codec import resample

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SUBSETS = ROOT / "subsets"
SEED = 1234

OPENSLR = {
    # set name: (SLR id, accent tag, [zip files])
    "openslr_ar": ("61", "es-AR", ["es_ar_female.zip", "es_ar_male.zip"]),
    "openslr_cl": ("71", "es-CL", ["es_cl_female.zip", "es_cl_male.zip"]),
    "openslr_co": ("72", "es-CO", ["es_co_female.zip", "es_co_male.zip"]),
    "openslr_pe": ("73", "es-PE", ["es_pe_female.zip", "es_pe_male.zip"]),
    "openslr_pr": ("74", "es-PR", ["es_pr_female.zip"]),
    "openslr_ve": ("75", "es-VE", ["es_ve_female.zip", "es_ve_male.zip"]),
}
OPENSLR_MIRROR = "https://www.openslr.org/resources"

HF_SETS = {
    "fleurs_es419": dict(
        repo="google/fleurs",
        path="es_419/test/0000.parquet",
        accent="es-419",
        source="google/fleurs es_419 test (CC-BY-4.0)",
    ),
    "ciempiess_mx": dict(
        repo="ciempiess/ciempiess_test",
        path="ciempiess_test/test/0000.parquet",
        accent="es-MX",
        source="ciempiess/ciempiess_test shard 0 (CC-BY-SA-4.0)",
    ),
}

ALL_SETS = list(HF_SETS) + list(OPENSLR)

# The CIEMPIESS test card: "Central Mexico [accent] except for the speaker M_09 that
# comes from El Salvador" - excluded so ciempiess_mx is Mexican Spanish only.
CIEMPIESS_EXCLUDE_SPEAKERS = {"M_09"}


# ---------------------------------------------------------------------- helpers


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample(audio.astype(np.float32), sr, 16000)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 16000, subtype="PCM_16")
    return len(audio) / 16000.0


def _load_subset(name: str) -> list[str] | None:
    p = SUBSETS / f"{name}.txt"
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return None


def _save_subset(name: str, ids: list[str]) -> None:
    SUBSETS.mkdir(exist_ok=True)
    (SUBSETS / f"{name}.txt").write_text("\n".join(ids) + "\n")


def _pick(items: list[dict], n: int, max_per_speaker: int, rng: random.Random) -> list[dict]:
    """Seeded shuffle, then greedy pick with a per-speaker cap, balanced by gender."""
    items = sorted(items, key=lambda r: r["id"])
    rng.shuffle(items)
    by_gender: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_gender[it.get("gender") or "u"].append(it)
    genders = sorted(by_gender)
    per_gender = {g: n // len(genders) for g in genders}
    per_gender[genders[0]] += n - sum(per_gender.values())
    chosen: list[dict] = []
    for g in genders:
        spk_count: dict[str, int] = defaultdict(int)
        seen_text: set[str] = set()
        got = 0
        for it in by_gender[g]:
            if got >= per_gender[g]:
                break
            if spk_count[it["speaker"]] >= max_per_speaker or it["text"] in seen_text:
                continue
            spk_count[it["speaker"]] += 1
            seen_text.add(it["text"])
            chosen.append(it)
            got += 1
        if got < per_gender[g]:
            # Few speakers (e.g. SLR74 Puerto Rico): relax the per-speaker cap.
            ids = {c["id"] for c in chosen}
            for it in by_gender[g]:
                if got >= per_gender[g]:
                    break
                if it["id"] in ids or it["text"] in seen_text:
                    continue
                seen_text.add(it["text"])
                chosen.append(it)
                got += 1
    return sorted(chosen, key=lambda r: r["id"])


def _write_manifest(name: str, rows: list[dict]) -> None:
    out = DATA / name / "manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"[{name}] {len(rows)} utterances, {hours:.2f} h -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------- HF parquet


def _hf_parquet(repo: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=path,
            repo_type="dataset",
            revision="refs/convert/parquet",
            cache_dir=str(DATA / "raw" / "hf"),
        )
    )


def build_hf(name: str, n: int) -> None:
    import pyarrow.parquet as pq

    cfg = HF_SETS[name]
    local = _hf_parquet(cfg["repo"], cfg["path"])
    table = pq.read_table(local)
    cols = table.column_names
    rows_meta = []
    for i in range(table.num_rows):
        if name == "fleurs_es419":
            rid = f"fleurs_{table['id'][i].as_py()}_{Path(table['path'][i].as_py() or str(i)).stem}"
            text = table["raw_transcription"][i].as_py()
            gender = {0: "m", 1: "f"}.get(table["gender"][i].as_py(), "u")
            # FLEURS does not publish speaker ids; the sentence id is the closest grouping.
            speaker = f"sent{table['id'][i].as_py()}"
        else:
            rid = table["audio_id"][i].as_py()
            text = table["normalized_text"][i].as_py()
            g = (table["gender"][i].as_py() or "u").lower()
            gender = "f" if g.startswith("f") else "m" if g.startswith("m") else "u"
            speaker = table["speaker_id"][i].as_py()
            if speaker in CIEMPIESS_EXCLUDE_SPEAKERS:
                continue
        rows_meta.append(dict(id=rid, text=text, gender=gender, speaker=speaker, _row=i))
    fixed = _load_subset(name)
    if fixed is None:
        picked = _pick(rows_meta, n, max_per_speaker=1 if name == "fleurs_es419" else 8, rng=random.Random(SEED))
        _save_subset(name, [r["id"] for r in picked])
    else:
        want = set(fixed)
        picked = [r for r in rows_meta if r["id"] in want]
        missing = want - {r["id"] for r in picked}
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} subset ids not found, e.g. {sorted(missing)[:3]}")
    audio_col = table.column("audio")
    out_rows = []
    for r in picked:
        a = audio_col[r["_row"]].as_py()
        audio, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        wav = DATA / name / "wav" / f"{r['id']}.wav"
        dur = _write_wav(wav, audio, sr)
        out_rows.append(
            dict(
                id=r["id"],
                audio=str(wav.relative_to(ROOT)),
                text=r["text"],
                set=name,
                accent=cfg["accent"],
                speaker=r["speaker"],
                gender=r["gender"],
                duration=round(dur, 3),
                source=cfg["source"],
            )
        )
    _write_manifest(name, out_rows)


# ---------------------------------------------------------------------- OpenSLR


def build_openslr(name: str, n: int) -> None:
    from remotezip import RemoteZip

    slr, accent, zips = OPENSLR[name]
    fixed = _load_subset(name)
    want = set(fixed) if fixed else None
    candidates = []
    zip_of = {}
    for zname in zips:
        url = f"{OPENSLR_MIRROR}/{slr}/{zname}"
        with RemoteZip(url) as z:
            names = z.namelist()
            tsv = [x for x in names if x.endswith("line_index.tsv")][0]
            prefix = tsv[: -len("line_index.tsv")]
            for line in z.read(tsv).decode("utf-8").splitlines():
                if "\t" not in line:
                    continue
                fid, text = line.split("\t", 1)
                fid = fid.strip()
                rid = f"{accent}_{fid}"
                spk = "_".join(fid.split("_")[:2])
                gender = "f" if fid[2] == "f" else "m" if fid[2] == "m" else "u"
                candidates.append(dict(id=rid, fid=fid, text=text.strip(), speaker=spk, gender=gender))
                zip_of[rid] = (url, f"{prefix}{fid}.wav")
    if want is None:
        picked = _pick(candidates, n, max_per_speaker=8, rng=random.Random(SEED))
        _save_subset(name, [r["id"] for r in picked])
    else:
        picked = [r for r in candidates if r["id"] in want]
    by_zip: dict[str, list[dict]] = defaultdict(list)
    for r in picked:
        by_zip[zip_of[r["id"]][0]].append(r)
    out_rows = []
    for url, rows in by_zip.items():
        with RemoteZip(url) as z:
            for r in rows:
                wav = DATA / name / "wav" / f"{r['id']}.wav"
                if not wav.exists():
                    audio, sr = sf.read(io.BytesIO(z.read(zip_of[r["id"]][1])), dtype="float32")
                    _write_wav(wav, audio, sr)
                dur = sf.info(wav).duration
                out_rows.append(
                    dict(
                        id=r["id"],
                        audio=str(wav.relative_to(ROOT)),
                        text=r["text"],
                        set=name,
                        accent=accent,
                        speaker=r["speaker"],
                        gender=r["gender"],
                        duration=round(dur, 3),
                        source=f"OpenSLR SLR{slr} (CC-BY-SA-4.0)",
                    )
                )
    _write_manifest(name, sorted(out_rows, key=lambda r: r["id"]))


def build(name: str, n: int) -> None:
    if (DATA / name / "manifest.jsonl").exists():
        print(f"[{name}] exists, skipping (delete data/{name} to rebuild)")
        return
    if name in HF_SETS:
        build_hf(name, n)
    elif name in OPENSLR:
        build_openslr(name, n)
    elif name in ("codeswitch_synth", "spanish_synth_control"):
        from .codeswitch import build_synthetic

        build_synthetic()
    else:
        raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=ALL_SETS + ["codeswitch_synth"])
    ap.add_argument("--n", type=int, default=200, help="utterances per set when drawing a new subset")
    ap.add_argument("--clean-raw", action="store_true", help="delete cached parquet shards afterwards")
    args = ap.parse_args()
    for s in args.sets:
        build(s, args.n)
    if args.clean_raw:
        shutil.rmtree(DATA / "raw", ignore_errors=True)


if __name__ == "__main__":
    main()
