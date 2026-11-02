# Data

The repository ships **no audio**. `make data` downloads fixed subsets of public corpora
into `data/` (git-ignored). The utterance IDs of every subset are committed in
`subsets/<set>.txt`, so any machine evaluates exactly the same audio. Subsets were
drawn once with seed 1234: a seeded shuffle, gender-balanced, at most 8 utterances per
speaker (relaxed only where a corpus has too few speakers), and no repeated prompt text
inside a subset.

## Evaluation sets

| set | accent / domain | source | license | subset | download |
|---|---|---|---|---|---|
| `fleurs_es419` | Latin American Spanish (es-419), read Wikipedia sentences. The **clean reference point**: this is the benchmark NVIDIA reports (4.11-4.87% WER depending on chunk size) | [google/fleurs](https://huggingface.co/datasets/google/fleurs) `es_419` test | CC-BY-4.0 | 200 of 908 test utts, 1 per sentence | one 0.70 GB parquet shard |
| `ciempiess_mx` | Mexican Spanish (Central Mexico), **spontaneous broadcast conversation** (UNAM radio) | [ciempiess/ciempiess_test](https://huggingface.co/datasets/ciempiess/ciempiess_test), shard 0 | CC-BY-SA-4.0 | 200 of 3,200 utts in shard 0 (speaker M_09 excluded) | one 0.51 GB parquet shard |
| `openslr_ar` | Argentina | [OpenSLR SLR61](https://www.openslr.org/61/) | CC-BY-SA-4.0 | 200 (100 f / 100 m) | selected WAVs only (HTTP range reads into the remote zip) |
| `openslr_cl` | Chile | [OpenSLR SLR71](https://www.openslr.org/71/) | CC-BY-SA-4.0 | 200 (100 f / 100 m) | same |
| `openslr_co` | Colombia | [OpenSLR SLR72](https://www.openslr.org/72/) | CC-BY-SA-4.0 | 200 (100 f / 100 m) | same |
| `openslr_pe` | Peru | [OpenSLR SLR73](https://www.openslr.org/73/) | CC-BY-SA-4.0 | 200 (100 f / 100 m) | same |
| `openslr_pr` | Puerto Rico (female speakers only in the corpus) | [OpenSLR SLR74](https://www.openslr.org/74/) | CC-BY-SA-4.0 | 200 f | same |
| `openslr_ve` | Venezuela | [OpenSLR SLR75](https://www.openslr.org/75/) | CC-BY-SA-4.0 | 200 (100 f / 100 m) | same |
| `spanish_synth_control` | **SYNTHETIC** monolingual Spanish, 4 macOS es_MX voices | prompts in `codeswitch/sentences_spanish_control.tsv` (written for this repo, MIT) | MIT (prompts); audio generated locally | 20 prompts x 4 voices = 80 | none (macOS `say`) |
| `codeswitch_synth` | **SYNTHETIC** Spanish-English code-switching, 4 macOS es_MX voices | prompts in `codeswitch/sentences_codeswitch.tsv` (written for this repo, MIT) | MIT (prompts); audio generated locally | 40 prompts x 4 voices = 160 | none (macOS `say`) |

The OpenSLR sets (Guevara-Rukoz et al., LREC 2020, "Crowdsourcing Latin American Spanish
for Low-Resource Text-to-Speech") are crowd-sourced **read** speech recorded at 48 kHz
for TTS. They are fairly clean, and the same prompts recur across speakers. They show
accent effects but not spontaneous-speech effects. CIEMPIESS covers spontaneous speech.

Measured subset sizes (audio as written to `data/`, before the 0.5 s padding):

| set | utterances | hours |
|---|---|---|
| fleurs_es419 | 200 | 0.68 |
| ciempiess_mx | 200 | 0.46 |
| openslr_ar | 200 | 0.28 |
| openslr_cl | 200 | 0.33 |
| openslr_co | 200 | 0.31 |
| openslr_pe | 200 | 0.36 |
| openslr_pr | 200 | 0.34 |
| openslr_ve | 200 | 0.28 |
| **real speech total** | **1,600** | **3.05** |
| spanish_synth_control | 80 | 0.08 |
| codeswitch_synth | 160 | 0.14 |

CIEMPIESS speaker `M_09`, who comes from El Salvador according to the corpus card, is
excluded so that `ciempiess_mx` contains only Mexican Spanish. Disk use after
`make data`: about 1.5 GB, mostly the two cached parquet shards in `data/raw/`
(`--clean-raw` deletes them).

### Attribution (CC-BY / CC-BY-SA)

- FLEURS: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech", 2022. CC-BY-4.0.
- OpenSLR 61, 71-75: Guevara-Rukoz, Demirsahin, He, Chu, Sarin, Pipatsrisawat, Gutkin,
  Butryna, Kjartansson, "Crowdsourcing Latin American Spanish for Low-Resource
  Text-to-Speech", LREC 2020. Copyright Google LLC. CC-BY-SA-4.0.
- CIEMPIESS Test: Hernández-Mena et al., Universidad Nacional Autónoma de México
  (UNAM). CC-BY-SA-4.0.

None of these files are redistributed here. Only IDs, derived scores and model
hypotheses (short transcripts) are committed, under `results/`.

## Candidates considered and not used

| corpus | why not |
|---|---|
| Multilingual LibriSpeech (MLS) Spanish | Read audiobooks with no per-speaker accent labels, mostly not Latin American. It is also **listed in Nemotron 3.5's training data**, so it would measure memorization more than robustness |
| Mozilla Common Voice Spanish | Has accent self-reports, but downloads now go through the Mozilla Data Collective (account + terms acceptance). Also listed in Nemotron's training data |
| CIEMPIESS Light / Balance / Fem | Mexican Spanish and openly downloadable, so used as **fine-tuning training data** instead (`finetune/build_manifests.py`), keeping the test corpus separate |
| Bangor Miami (Spanish-English code-switching) | CC-BY-NC-SA, behind a TalkBank login. LDC code-switching corpora are paid. That is why the code-switching set is synthetic |
| DataoceanAI Mexican Spanish sets on HF | Commercial samples, unclear license |

## Overlap with model training data

According to the model card, Nemotron 3.5 ASR was trained on NVIDIA's proprietary Riva
set, Granary, MLS, Common Voice, **FLEURS** and VoxPopuli/Europarl. The `fleurs_es419`
**test** split is NVIDIA's own published benchmark, which is why it serves as the clean
anchor here. OpenSLR 61/71-75 and CIEMPIESS are not listed, but the proprietary
portion can't be checked. Whisper's training data is undisclosed.

## Synthetic data, clearly labelled

`codeswitch_synth` and `spanish_synth_control` are TTS audio. Every manifest row carries
`"synthetic": true`. The voices are `Paulina (Spanish (Mexico))`, a natural-sounding
voice, and `Eddy`, `Flo`, `Reed (Spanish (Mexico))`, robotic formant-style voices.
Results are reported per voice because the voices behave very differently.
Synthetic-set numbers are a stress test, **not** an estimate of real-caller WER.

No customer, employer or client recordings or text are used anywhere in this repository.
