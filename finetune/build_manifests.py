"""Build NeMo training/validation manifests for Latin American Spanish fine-tuning.

Sources (all public, CC-BY / CC-BY-SA; see DATA.md):
  * OpenSLR SLR61 (AR), SLR71 (CL), SLR72 (CO), SLR73 (PE), SLR74 (PR), SLR75 (VE) - full zips
  * CIEMPIESS Light (Mexican Spanish, radio) - ciempiess/ciempiess_light on HF
  * FLEURS es_419 train split

Leakage rules (enforced, not just documented):
  * every utterance id in subsets/*.txt (the evaluation subsets) is excluded;
  * every *speaker* that appears in an evaluation subset is excluded from training,
    so the fine-tuned model is always scored on unseen voices;
  * FLEURS: sentences that appear in the es_419 test subset are excluded from train.

Manifest line (NeMo / Lhotse "nemo" format):
  {"audio_filepath": ..., "duration": ..., "text": ..., "target_lang": "es-US", "lang": "es-US"}
Nemotron's training config reads the language prompt from ``target_lang``
(``lang_field: target_lang`` in the checkpoint's train_ds); ``lang`` is added for
NeMo's transcribe() path, which reads ``lang``.

Intended to run on the cloud GPU box (needs ~25 GB of disk for the OpenSLR zips).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ROOT / "subsets"
OPENSLR = {
    "61": ["es_ar_female.zip", "es_ar_male.zip"],
    "71": ["es_cl_female.zip", "es_cl_male.zip"],
    "72": ["es_co_female.zip", "es_co_male.zip"],
    "73": ["es_pe_female.zip", "es_pe_male.zip"],
    "74": ["es_pr_female.zip"],
    "75": ["es_ve_female.zip", "es_ve_male.zip"],
}
LANG = "es-US"  # Nemotron's Latin American Spanish prompt


def eval_exclusions() -> tuple[set[str], set[str], set[str]]:
    ids, speakers, texts = set(), set(), set()
    for f in SUBSETS.glob("*.txt"):
        ids |= {x.strip() for x in f.read_text().splitlines() if x.strip()}
    data = ROOT / "data"
    for man in data.glob("*/manifest.jsonl"):
        for line in man.read_text().splitlines():
            r = json.loads(line)
            speakers.add(r["speaker"])
            texts.add(r["text"].strip().lower())
    return ids, speakers, texts


def _row(path: Path, text: str) -> dict:
    return {"audio_filepath": str(path), "duration": round(sf.info(path).duration, 3), "text": text,
            "target_lang": LANG, "lang": LANG}


def openslr(raw: Path, out_audio: Path, excl_ids, excl_spk) -> list[dict]:
    rows = []
    for slr, zips in OPENSLR.items():
        for z in zips:
            zp = raw / z
            if not zp.exists():
                raise FileNotFoundError(f"{zp} (download https://www.openslr.org/resources/{slr}/{z})")
            with zipfile.ZipFile(zp) as zf:
                tsv = [n for n in zf.namelist() if n.endswith("line_index.tsv")][0]
                prefix = tsv[: -len("line_index.tsv")]
                for line in zf.read(tsv).decode().splitlines():
                    fid, text = line.split("\t", 1)
                    spk = "_".join(fid.split("_")[:2])
                    if spk in excl_spk or any(i.endswith(fid) for i in excl_ids):
                        continue
                    dst = out_audio / f"slr{slr}" / f"{fid}.wav"
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        a, sr = sf.read(io.BytesIO(zf.read(f"{prefix}{fid}.wav")), dtype="float32")
                        from esasr.codec import resample

                        sf.write(dst, resample(a, sr, 16000), 16000, subtype="PCM_16")
                    rows.append(_row(dst, text.strip()))
    return rows


def hf_parquet_rows(repo: str, config: str, split: str, text_col: str, id_col: str, spk_col: str | None,
                    out_audio: Path, excl_ids, excl_spk, excl_text) -> list[dict]:
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, config, split=split, revision="refs/convert/parquet")
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = []
    for r in ds:
        rid = str(r[id_col])
        if rid in excl_ids or (spk_col and r[spk_col] in excl_spk):
            continue
        if r[text_col].strip().lower() in excl_text:
            continue
        dst = out_audio / repo.replace("/", "_") / f"{rid}.wav"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            from esasr.codec import resample

            sf.write(dst, resample(a, sr, 16000), 16000, subtype="PCM_16")
        rows.append(_row(dst, r[text_col].strip()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("/data/raw"), help="dir with the OpenSLR zips")
    ap.add_argument("--out", type=Path, default=Path("/data/ft"))
    ap.add_argument("--val-frac", type=float, default=0.02)
    a = ap.parse_args()
    excl_ids, excl_spk, excl_text = eval_exclusions()
    audio = a.out / "audio"
    rows = openslr(a.raw, audio, excl_ids, excl_spk)
    rows += hf_parquet_rows("ciempiess/ciempiess_light", "ciempiess_light", "train", "normalized_text",
                            "audio_id", "speaker_id", audio, excl_ids, excl_spk, excl_text)
    rows += hf_parquet_rows("google/fleurs", "es_419", "train", "raw_transcription", "id", None,
                            audio, excl_ids, excl_spk, excl_text)
    rows = [r for r in rows if 0.5 <= r["duration"] <= 25.0 and r["text"]]
    random.Random(0).shuffle(rows)
    n_val = max(200, int(len(rows) * a.val_frac))
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        p = a.out / f"{name}_manifest.json"
        with p.open("w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{p}: {len(part)} utts, {sum(r['duration'] for r in part)/3600:.1f} h")


if __name__ == "__main__":
    main()
