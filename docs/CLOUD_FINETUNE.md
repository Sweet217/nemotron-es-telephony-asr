# Cloud fine-tuning runbook (not executed)

**Status: PENDING.** No cloud GPU has been rented and no fine-tuned model exists.
Everything below is a plan. The only training that ran is a 2-step CPU smoke test on
the M1 Max, which checked that the recipe loads, augments, trains, validates and saves
a `.nemo` file. It produced no results.

## Goal

Fine-tune `nvidia/nemotron-3.5-asr-streaming-0.6b` on Latin American Spanish, with
on-the-fly G.711 telephony augmentation, so that `g711` / `g711_loss5` / `g711_snr10`
WER gets closer to `clean` WER without hurting wideband accuracy. Success is measured
with the same harness (`make eval report`) on the same fixed subsets. The training
manifests exclude every evaluation speaker (`finetune/build_manifests.py`).

## Recipe

| item | value |
|---|---|
| base | `nvidia/nemotron-3.5-asr-streaming-0.6b` (`EncDecRNNTBPEModelWithPrompt`, OpenMDW-1.1) |
| method | full fine-tune (NVIDIA's own Nemotron 3.5 fine-tuning post: LoRA is not supported yet) |
| data | OpenSLR 61/71-75 (AR, CL, CO, PE, PR, VE; about 23 h estimated from zip sizes at 48 kHz/16-bit), CIEMPIESS Light (MX, 18 h 25 min per its card), FLEURS es_419 train. About 45-50 h total, eval speakers removed |
| language prompt | `target_lang = es-US` (the checkpoint's `train_ds.lang_field` is `target_lang`) |
| augmentation | 70% of utterances: optional pink noise at 5-25 dB SNR, 8 kHz, 300-3400 Hz, G.711 μ-law, 0-8% packet loss, back to 16 kHz. SpecAugment from the checkpoint is kept |
| streaming | the checkpoint trains with multi-lookahead `att_context_size` [[56,3],[56,0],[56,6],[56,13]], left as is so every latency mode stays usable |
| optimizer | AdamW, lr 3e-5, cosine, 250 warmup steps, wd 1e-3, grad clip 1.0 |
| batching | Lhotse dynamic buckets, 600 s of audio per batch, bf16-mixed |
| steps | 2,500 (about 8 passes over about 50 h); validation every 250 steps on a clean held-out 2% split |

The files are `finetune/build_manifests.py`, `finetune/g711_augment.py`,
`finetune/train.py` and `finetune/conf/nemotron35_es_telephony.yaml`.

## Runbook (1x H100 80 GB, Ubuntu 22.04, CUDA 12.x)

```bash
# 0. box: any GPU cloud with a PyTorch/CUDA image. Use >= 200 GB of disk.
git clone <this repo> && cd nemotron-es-telephony-asr
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-nemo.txt && pip install -e .

# 1. eval subsets first: the manifest builder excludes their speakers
make data SETS="fleurs_es419 ciempiess_mx openslr_ar openslr_cl openslr_co openslr_pe openslr_pr openslr_ve"

# 2. raw training data (about 8 GB of OpenSLR zips)
mkdir -p /data/raw && cd /data/raw
for f in 61/es_ar_female 61/es_ar_male 71/es_cl_female 71/es_cl_male 72/es_co_female 72/es_co_male \
         73/es_pe_female 73/es_pe_male 74/es_pr_female 75/es_ve_female 75/es_ve_male; do
  wget -c https://www.openslr.org/resources/${f}.zip; done
cd -

# 3. manifests (speaker-disjoint from the eval subsets)
python finetune/build_manifests.py --raw /data/raw --out /data/ft

# 4. baseline on the box: Nemotron eval on CUDA (sanity check against the Mac numbers)
make eval-nemotron

# 5. train
python finetune/train.py exp_manager.exp_dir=/runs
#   short dry run first: trainer.max_steps=50 trainer.val_check_interval=50

# 6. evaluate base and fine-tuned checkpoints with the same harness and runtime (NeMo)
python -m esasr.run_eval --system nemotron-nemo:es-US
python -m esasr.run_eval --system nemotron-nemo:es-US:/runs/nemotron35_es_telephony_final.nemo
make report
```

Pull back `results/hyps/`, the `.nemo` file and the TensorBoard logs, then destroy the
instance.

## Cost estimate (assumptions, not measurements)

| phase | time on 1x H100 | notes |
|---|---|---|
| setup, pip, downloads, manifest build | ~1.0 h | mostly OpenSLR download and resampling (CPU) |
| baseline + final eval (2 runs x 10 sets x 5 conditions) | ~0.5 h | Nemotron reached RTFx ~20-25 on M1 Max MPS; a GPU should be much faster |
| training, 2,500 steps | ~1-2 h | assumes ~1.5-2.5 s per 600 s-audio step for a 0.6B RNNT in bf16. **Unmeasured**: do the 50-step dry run first and rescale |
| slack (OOM retries, batch tuning) | ~1 h | |
| **total** | **~3.5-4.5 GPU-hours** | |

At typical on-demand marketplace prices of roughly $2-4 per H100-hour (check current
pricing; it varies by provider and region), that comes to **about $8-$18**. Set a hard
cap of **$25**. An A100 80 GB at roughly $1.2-2/h should also work at 1.5-2x the
wall-clock time for about the same total. A 24 GB card (L4 / A10G) needs
`data.batch_duration≈150` and gradient accumulation, and takes roughly 4x longer.

The only throughput measured on this machine: the CPU smoke test (M1 Max, fp32,
12 s-audio batches) ran at **~11 s/step**. That rules out CPU training at real scale.

## Risks and checks

- **Catastrophic forgetting**: watch clean FLEURS es_419 and an English sanity set.
  Also try a mixed run with a 10-20% replay of the base languages, or freeze the first
  N encoder layers (`model.freeze_encoder_layers`).
- **Over-fitting to read speech**: OpenSLR is read sentences with heavily repeated
  prompts. CIEMPIESS Light adds spontaneous radio speech. Deduplicate prompts if
  validation WER drops while CIEMPIESS test WER rises.
- **Prompt drift**: train with `es-US` only. Then check that `auto` still detects
  Spanish (`make eval-extra`).
- **Licensing**: the training data is CC-BY / CC-BY-SA. If you distribute weights,
  check whether ShareAlike applies to them (open legal question) and keep OpenMDW-1.1
  obligations for the base model.
