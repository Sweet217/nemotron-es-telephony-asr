# nemotron-es-telephony-asr

**How does NVIDIA Nemotron 3.5 ASR hold up on Latin American Spanish over a phone line?**

Nemotron 3.5 ASR (`nvidia/nemotron-3.5-asr-streaming-0.6b`) reports **4.11% Spanish WER**,
measured on FLEURS: clean, wideband, read Wikipedia sentences. Contact-center audio is
8 kHz G.711 μ-law, often lossy and noisy. Callers are spontaneous, regionally accented
and frequently code-switch into English. The model also ships **without VAD or
diarization**, and NVIDIA's guidance for new accents and domains is to fine-tune with
NeMo.

This repo gives that question a reproducible answer:

1. **Evaluation harness**: public Latin American Spanish (8 accents/sets) and a
   synthetic Spanish-English set, each degraded through a simulated phone channel
   (8 kHz + 300-3400 Hz + G.711 μ-law, ± packet loss, ± noise). Scoring uses a
   documented Spanish normalizer, WER/CER via `jiwer`, 95% bootstrap CIs, and
   paired-bootstrap degradation deltas.
2. **Systems**: Nemotron 3.5 ASR run locally (Transformers on Apple-Silicon MPS, with
   the NeMo runtime as a cross-check), Whisper large-v3-turbo locally (MLX), and a
   Deepgram Nova-3 adapter that is **not run** (paid API).
3. **Fine-tuning recipe**: NeMo full fine-tune with on-the-fly G.711 augmentation,
   speaker-disjoint manifests, and a cloud runbook with a cost estimate.
   **Not run: results pending.**
4. **Pipecat integration**: a `SegmentedSTTService` for Nemotron behind Silero VAD,
   which supplies the VAD the model lacks. It includes an offline WAV → VAD → STT test.

> Every number below was produced on one Apple M1 Max (32 GB, no NVIDIA GPU) by the
> code in this repo. Anything not run is marked **pending**.

## Results

Measured 2026-11-01/02 on an Apple M1 Max (32 GB). Nemotron ran through Transformers on
MPS (fp32); Whisper ran through mlx-whisper. Each real-speech set has 200 utterances
(1,600 in total, 3.05 h of audio per condition before padding). WER is in %. The tables below
are produced by `scripts/readme_tables.py` from `results/summary.csv`. The full
per-system tables (CER, S/D/I, accent-sensitive WER, per-voice results, runtime
cross-check) are in [results/RESULTS.md](results/RESULTS.md).

#### Headline: WER % on real speech (8 sets, 1,600 utterances), pooled over sets

| system | clean | nb8k | g711 | g711_loss5 | g711_snr10 |
|---|---|---|---|---|---|
| Nemotron 3.5 ASR, 320 ms | 6.63 | 7.22 | 7.35 | 7.92 | 13.71 |
| Nemotron 3.5 ASR, 1.12 s | 5.88 | 6.35 | 6.56 | 6.81 | 10.87 |
| Whisper large-v3-turbo | 4.31 | 4.64 | 4.63 | 4.97 | 7.53 |

#### WER % by accent, clean vs G.711 (95% bootstrap CI)

| set | accent | Nemotron 3.5 ASR, 320 ms clean | Nemotron 3.5 ASR, 320 ms g711 | Nemotron 3.5 ASR, 1.12 s clean | Nemotron 3.5 ASR, 1.12 s g711 | Whisper large-v3-turbo clean | Whisper large-v3-turbo g711 |
|---|---|---|---|---|---|---|---|
| fleurs_es419 | es-419 | 4.72 [3.89-5.54] | 5.43 [4.55-6.31] | 4.53 [3.71-5.31] | 4.57 [3.80-5.34] | 2.80 [2.13-3.44] | 2.88 [2.27-3.54] |
| ciempiess_mx | es-MX | 12.97 [11.86-14.21] | 13.17 [12.03-14.38] | 12.80 [11.56-14.12] | 12.58 [11.44-13.90] | 12.64 [11.44-13.99] | 13.34 [12.06-14.75] |
| openslr_ar | es-AR | 4.99 [3.79-6.38] | 5.14 [3.80-6.66] | 3.29 [2.32-4.41] | 3.96 [2.91-5.10] | 1.28 [0.70-1.93] | 1.75 [1.00-2.56] |
| openslr_cl | es-CL | 4.32 [3.12-5.80] | 6.20 [4.69-7.93] | 3.15 [2.19-4.24] | 4.98 [3.68-6.39] | 1.52 [0.92-2.20] | 1.68 [1.09-2.39] |
| openslr_co | es-CO | 4.92 [3.66-6.15] | 5.54 [4.11-6.91] | 3.83 [2.86-4.86] | 4.97 [3.76-6.29] | 1.50 [0.89-2.23] | 1.86 [1.17-2.63] |
| openslr_pe | es-PE | 3.18 [2.23-4.16] | 3.95 [2.96-5.12] | 2.82 [2.04-3.69] | 3.54 [2.53-4.78] | 1.38 [0.82-2.00] | 1.28 [0.80-1.86] |
| openslr_pr | es-PR | 5.64 [4.39-7.10] | 6.66 [5.09-8.27] | 3.76 [2.82-4.74] | 6.80 [4.79-9.12] | 1.21 [0.61-1.92] | 1.64 [1.04-2.32] |
| openslr_ve | es-VE | 6.07 [4.74-7.48] | 7.22 [5.63-8.84] | 4.87 [3.85-5.94] | 5.39 [4.21-6.65] | 2.35 [1.59-3.15] | 2.62 [1.90-3.42] |

#### Degradation vs clean, WER points (paired bootstrap 95% CI), Nemotron 320 ms

| set | nb8k | g711 | g711_loss5 | g711_snr10 |
|---|---|---|---|---|
| fleurs_es419 | +0.48 [+0.11, +0.86] | +0.70 [+0.28, +1.13] | +1.09 [+0.62, +1.53] | +5.01 [+4.06, +6.09] |
| ciempiess_mx | -0.18 [-0.84, +0.48] | +0.20 [-0.55, +0.99] | +0.61 [-0.10, +1.33] | +7.72 [+6.48, +9.02] |
| openslr_ar | +0.82 [-0.40, +2.25] | +0.15 [-0.79, +1.12] | +0.57 [-0.41, +1.60] | +5.76 [+4.17, +7.31] |
| openslr_cl | +1.88 [+0.99, +2.79] | +1.88 [+0.91, +2.84] | +2.24 [+1.25, +3.26] | +11.34 [+8.79, +14.00] |
| openslr_co | +0.67 [-0.37, +1.74] | +0.62 [-0.40, +1.72] | +2.02 [+0.89, +3.24] | +8.38 [+6.30, +10.53] |
| openslr_pe | +0.56 [-0.10, +1.30] | +0.77 [+0.00, +1.55] | +1.84 [+0.87, +2.91] | +7.33 [+5.11, +9.55] |
| openslr_pr | +0.87 [-0.34, +2.02] | +1.01 [-0.23, +2.31] | +1.93 [+0.62, +3.41] | +6.71 [+4.31, +9.25] |
| openslr_ve | +0.94 [-0.11, +2.10] | +1.15 [-0.10, +2.65] | +1.31 [+0.05, +2.66] | +6.86 [+4.91, +8.96] |

#### Synthetic sets (macOS TTS, es_MX voices): WER %, clean / g711

| system | Spanish control | Spanglish |
|---|---|---|
| Nemotron 320 ms, es-US prompt | 20.28 / 37.03 | 40.01 / 52.54 |
| Nemotron 320 ms, auto LID | 27.48 / 40.57 | 45.60 / 56.89 |
| Whisper large-v3-turbo | 0.47 / 0.82 | 11.66 / 15.76 |

#### Spanglish: share of English vs Spanish reference words missed (Paulina voice only, clean)

| system | English words missed % | Spanish words missed % |
|---|---|---|
| Nemotron 320 ms, es-US | 39.0 (n=82) | 3.1 (n=321) |
| Nemotron 320 ms, auto | 37.8 (n=82) | 3.1 (n=321) |
| Whisper large-v3-turbo | 24.4 (n=82) | 0.6 (n=321) |

#### Speed on this machine (RTFx = audio seconds / processing seconds, clean condition, all real sets)

- Nemotron 3.5 ASR, 320 ms: median RTFx 33
- Nemotron 3.5 ASR, 1.12 s: median RTFx 36
- Whisper large-v3-turbo: median RTFx 14

### What the numbers say

1. **The FLEURS anchor reproduces.** On a 200-utterance es_419 subset, Nemotron at
   320 ms scores 4.72% (CI 3.89-5.54) against the card's 4.39%, and 4.53% at 1.12 s
   against the card's 4.11%. Both are within the CIs, using a different normalizer.
   The local setup matches NVIDIA's.
2. **G.711 by itself costs Nemotron little.** Pooled WER goes 6.63 → 7.35 (320 ms).
   Most of that comes from the 8 kHz band-limit (`nb8k` 7.22) rather than μ-law
   quantization. Per set, the paired deltas are significant on FLEURS (+0.70) and
   Chile (+1.88). For most other accents they are not distinguishable from zero at
   n = 200.
3. **Noise is the real problem.** Pink noise at 10 dB SNR before the codec roughly
   doubles Nemotron's WER (6.63 → 13.71 pooled; Chile +11.3 points). The 1.12 s chunk
   setting recovers part of it (10.87). 5% packet loss adds another 0.3-0.6 points on
   top of G.711.
4. **The accent spread is real but moderate on read speech.** On clean OpenSLR,
   Nemotron at 320 ms ranges from 3.18% (Peru) to 6.07% (Venezuela).
5. **Spontaneous Mexican speech is about 13% WER for every system** (CIEMPIESS radio,
   clean). The errors are mostly deletions, because the corpus transcribes fillers,
   repetitions and phonetic spellings that the systems drop. Part of that 13% measures
   transcription convention, not recognition (see caveats).
6. **Latency vs accuracy.** Going from 320 ms to 1.12 s chunks lowers pooled WER from
   6.63 to 5.88 (clean) and from 7.35 to 6.56 (G.711).
7. **Whisper large-v3-turbo is more accurate offline but is not a streaming model.** It
   decodes each utterance with full context, so this is not a like-for-like latency
   comparison. It is also slower here (median RTFx 14 vs 33), and it has a
   hallucination failure mode: on one Puerto Rico utterance at 10 dB SNR it emitted
   "no, no, no, …" (108 inserted words for a 4-word reference). That single utterance
   explains the wide CI in that cell.
8. **Code-switching is Nemotron's weakest spot.** With the natural-sounding TTS voice
   (Paulina), Nemotron misses 39% of the English words embedded in Spanish sentences,
   against 3.1% of the Spanish words. Switching the prompt to `auto` does not help
   (37.8%). Whisper misses 24.4%. The robotic TTS voices (Eddy/Flo/Reed) push Nemotron
   into many **empty outputs**, while Whisper stays near 1% on the same Spanish control
   audio. That says more about out-of-distribution synthetic audio than about callers,
   and it is why the synthetic results are reported per voice.
9. **The Transformers port matches NeMo.** On FLEURS, NeMo 3.0.0 (CPU) and
   Transformers 5.22.1 (MPS) give identical normalized hypotheses on 100% of clean
   utterances and 98.5% of G.711 utterances.

### Caveats

- **Sample size**: 200 utterances per set, so per-accent CIs are about ±1-1.5 WER
  points. Read the per-accent ranking with care. The pooled numbers are tighter.
- **Simulated channel**: G.711 μ-law, band-pass, Bernoulli packet loss with naive
  concealment, and synthetic pink noise. It leaves out real network jitter, AMR-NB/Opus
  legs, transcoding chains, echo, AGC, and babble or street noise.
- **Read speech dominates**: the OpenSLR sets are crowd-sourced read prompts recorded
  for TTS. CIEMPIESS is the only spontaneous set, and its references include
  disfluencies and non-standard spellings (e.g. *libertá*), which inflates WER for
  every system the same way.
- **Streaming emulation**: Nemotron is scored with a full-utterance pass under the
  model's chunked attention mask (the configuration of the card's streaming rows), not
  through the incremental cache-aware loop. Every utterance gets 0.5 s of silence
  padding on both sides, for all systems.
- **Whisper choice**: large-v3-turbo (809M, 4-layer decoder) was used because it runs
  at full speed on the M1 Max. large-v3 was not run.
- **Contamination**: FLEURS (train) is in Nemotron's training data. The FLEURS test
  split is NVIDIA's own benchmark. The proprietary training data can't be audited.
- **Synthetic sets** are TTS, not callers. They stress-test code-switching and
  out-of-domain audio, and are labelled as synthetic everywhere.
- The RTFx figures are indicative. They come from single-stream runs on a laptop GPU,
  with the machine also doing other work.

### Pending (not run, no numbers claimed)

| item | status |
|---|---|
| Deepgram Nova-3 | adapter written (`src/esasr/systems/deepgram.py`, `make eval-deepgram`). **Not run**: paid API, no key on this machine, spend needs approval |
| Nemotron fine-tuned on telephony-augmented LatAm Spanish | recipe written and CPU smoke-tested (2 steps). **No GPU run**: see [docs/CLOUD_FINETUNE.md](docs/CLOUD_FINETUNE.md) (~$8-18 estimate) |
| Nemotron at 80/160/560 ms | supported by the runner (`--system nemotron:es-US:0`), not run |
| NeMo cross-check on the remaining sets | only FLEURS clean + G.711 were run on NeMo |
| Whisper large-v3 (non-turbo) | not run |


## Model facts (verified 2026-11-01)

| | |
|---|---|
| model id | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (HF release 2026-06-04) |
| size / arch | 600M params, cache-aware FastConformer encoder (24 layers) + RNN-T decoder + language-ID prompt (NeMo class `EncDecRNNTBPEModelWithPrompt`) |
| license | OpenMDW-1.1 (commercial use allowed) |
| languages | 40 locales: 19 "transcription-ready" (incl. `es-US`, `es-ES`), 13 broad-coverage, 8 adaptation-ready |
| streaming | `att_context_size` [56, R], R ∈ {0,1,3,6,13} → 80 / 160 / 320 / 560 / 1120 ms chunks |
| card WER, Spanish FLEURS, LangID | 4.87 / 4.64 / 4.39 / 4.26 / 4.11% (80 ms → 1.12 s) |
| runtimes | NeMo (`nemo_toolkit[asr]`), 🤗 Transformers ≥ 5.13 (`Nemotron3_5AsrForRNNT`), NeMo-Speech.cpp (GGUF) |
| missing | VAD, diarization (also noted by independent coverage) |

How it runs here: Transformers 5.22.1 on `mps`, fp32, `language="es-US"` (the model's
Latin American Spanish prompt), `num_lookahead_tokens=3`. The full-utterance `generate`
applies the model's chunked attention mask, so the configuration matches the
card's **320 ms** streaming row (card: 4.39%). NeMo 3.0.0 also installs and runs on
macOS CPU. The `nemotron-nemo` runner is used as a cross-check (see below).

## Telephony conditions

All conditions are applied on the fly, deterministically, with a seed derived from the
utterance id (`src/esasr/codec.py`):

| condition | pipeline |
|---|---|
| `clean` | original, 16 kHz |
| `nb8k` | → 8 kHz → 300-3400 Hz band-pass, linear PCM (isolates **bandwidth**) |
| `g711` | `nb8k` + G.711 μ-law encode/decode (**the standard PSTN/SIP leg**) |
| `g711_loss5` | `g711` + 5% random 20 ms packet loss, repeat-last-frame concealment |
| `g711_snr10` | pink noise at 10 dB SNR (caller side) → `g711` |

Every utterance gets 0.5 s of silence on each side before the channel. A VAD front-end
does the same thing (Pipecat's `SegmentedSTTService` keeps pre-roll and appends 0.5 s of
trailing silence). Without the padding Nemotron sometimes drops the first word of tightly
trimmed clips. It is applied identically to every system.

The μ-law codec is pure numpy, following the ITU/Sun `g711.c` reference. Tests check
it against ffmpeg's `pcm_mulaw`: decoding is bit-exact. Encoding agrees on 99.2% of all
65,536 inputs and is off by exactly one code at decision boundaries on the rest, a
known table-construction difference.

## Normalization and statistics

`src/esasr/normalize.py` applies the same normalization to references and hypotheses:
NFKC, Nemotron `<xx-XX>` language tags and `[annotations]` removed, lowercase,
numbers → Spanish words (`num2words`, incl. `1.300`, `3,5`, `%`, ordinals), punctuation
removed (word-internal apostrophes kept for English), diacritics folded **except ñ**
(an accent-sensitive table is reported too). Corpus WER is pooled
(ΣS+D+I / ΣN). The 95% CIs come from a 1,000-replicate utterance-level bootstrap.
Clean-vs-condition deltas use a **paired** bootstrap on the same utterances.

## Reproduce

```bash
make setup     # python3.11 venv in .venv + pinned deps (+ NeMo) + editable install
make data      # fixed subsets from subsets/*.txt (~1.3 GB download, ~1.5 GB on disk) + synthetic sets (macOS `say`)
make eval      # Nemotron (320 ms, 1.12 s) + Whisper over sets x 5 conditions, auto-LID, NeMo cross-check (~3-4 h on M1 Max)
make report    # -> results/RESULTS.md, results/*.csv
make test      # 34 tests: codec vs ffmpeg, normalizer, WER/bootstrap, fine-tune aug, Pipecat VAD+STT
```

`make eval LIMIT=20` gives a quick smoke run. `make eval-deepgram` refuses to run without
`DEEPGRAM_API_KEY` and is not part of `make eval`.

## Pipecat: giving Nemotron a VAD

`pipecat_nemotron/nemotron_stt.py` implements `NemotronSTTService`. It follows the same
contract as Pipecat's local `WhisperSTTService`: a `SegmentedSTTService` subclass with
`wants_wav_segments = False` (raw PCM), inference in `asyncio.to_thread`, and
`TranscriptionFrame` output with the language the model detected. Pipecat `Language`
values map to Nemotron prompts (`es-MX`/`es-419` → `es-US`, `None` → `auto`), and the
segment is resampled to 16 kHz when the transport runs at 8 kHz.

```python
pipeline = Pipeline([
    transport.input(),                                   # e.g. Twilio/Telnyx media stream at 8 kHz
    VADProcessor(vad_analyzer=SileroVADAnalyzer()),      # the VAD Nemotron doesn't ship with
    NemotronSTTService(settings=NemotronSTTService.Settings(language=Language.ES_MX)),
    ...                                                  # LLM, TTS, transport.output()
])
```

Offline demo: `python -m pipecat_nemotron.transcribe_wav call.wav --telephony` streams
a WAV as 20 ms `InputAudioRawFrame`s through Silero VAD and Nemotron. The test
`tests/test_pipecat_nemotron.py` synthesizes two Spanish turns, passes them through
G.711 at 8 kHz, and asserts that the VAD produces exactly two transcripts with the right
content.

Segment-level transcription is the simplest correct integration. A true incremental
service would feed the cache-aware encoder chunk by chunk (Transformers exposes this
through `TextIteratorStreamer` + `input_features` generators) and emit interim
transcripts. That is the natural next step.

## Fine-tuning (pending)

`finetune/` contains a NeMo recipe:

- `build_manifests.py`: OpenSLR 61/71-75 + CIEMPIESS Light + FLEURS es_419 train →
  NeMo manifests with `target_lang: es-US`. **Evaluation speakers are excluded.**
- `g711_augment.py`: patches the model's Lhotse prompt-index dataset so 70% of training
  waveforms pass through the phone channel, with random loss and SNR. Validation stays
  clean.
- `train.py` + `conf/nemotron35_es_telephony.yaml`: full fine-tune from the pretrained
  checkpoint, reusing its prompt dictionary and multi-lookahead streaming config.

The recipe was **smoke-tested on CPU** for 2 steps (loads, augments, trains, validates,
writes a `.nemo`), at about 11 s/step. No real training was run. The runbook and a ~$8-18
H100 cost estimate are in [docs/CLOUD_FINETUNE.md](docs/CLOUD_FINETUNE.md).

## Repository layout

```
src/esasr/
  codec.py          G.711 mu-law, 8 kHz band-limit, packet loss, noise, condition registry
  normalize.py      Spanish / Spanglish text normalization
  metrics.py        WER/CER counts, pooled rate, bootstrap + paired bootstrap
  datasets.py       fixed public subsets (FLEURS, CIEMPIESS, OpenSLR via HTTP range reads)
  codeswitch.py     synthetic Spanglish + Spanish control sets (macOS `say`)
  run_eval.py       system x set x condition runner (resumable, stores raw hypotheses)
  report.py         scoring -> results/RESULTS.md + CSVs
  systems/          nemotron_hf.py, nemotron_nemo.py, whisper_mlx.py, deepgram.py
pipecat_nemotron/   NemotronSTTService + offline VAD demo
finetune/           NeMo fine-tuning recipe (manifests, augmentation, config, trainer)
codeswitch/         hand-written prompt lists (synthetic sets)
subsets/            committed utterance IDs of every evaluation subset
results/            raw hypotheses per system/set/condition + scored tables
docs/CLOUD_FINETUNE.md
DATA.md             sources, licenses, sizes, exclusions, contamination notes
```

## Author and how this was built

Gabriel Velázquez Berrueta, software engineer (B.Eng., Universidad de Colima)
working on LLM and voice systems, with HCI research presented at MexIHC 2026.
[LinkedIn](https://www.linkedin.com/in/gabriel-velazquez-berrueta-53a3b925a/)

This project was built with Claude Code as a coding agent. I chose the problem,
set the ground rules (public or synthetic data only, no number in this README
that did not come from a run, no paid APIs or cloud GPUs without a cost estimate
first) and reviewed the results; Claude Code wrote most of the implementation and
ran the evaluations. Commits carry a `Co-Authored-By: Claude` trailer.

## License

Code and docs: MIT, © 2026 Gabriel Velázquez. Datasets and model weights are not
included and keep their own licenses (see [DATA.md](DATA.md)). Nemotron 3.5 ASR is
OpenMDW-1.1. Whisper is MIT.
