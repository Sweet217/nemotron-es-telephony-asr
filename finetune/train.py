"""Fine-tune Nemotron 3.5 ASR on Latin American Spanish with telephony augmentation.

    python finetune/train.py --config finetune/conf/nemotron35_es_telephony.yaml \
        [key=value overrides, e.g. trainer.max_steps=100 data.batch_duration=300]

Mirrors NeMo's examples/asr/speech_to_text_finetune.py (init from the pretrained
checkpoint, keep its tokenizer, swap the data, new optimizer) with two additions:
  1. the checkpoint's own train_ds/validation_ds configs (prompt dictionary, lang_field,
     subsampling factor) are reused and only the manifest/batching keys are overridden;
  2. the training waveforms go through finetune.g711_augment.TelephonyAugment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_ds_cfg(base, manifest: str, d, *, train: bool):
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    with open_dict(cfg):
        cfg.manifest_filepath = manifest
        cfg.use_lhotse = True
        cfg.lang_field = d.lang_field
        cfg.batch_duration = d.batch_duration if train else None
        cfg.batch_size = None if train else 16
        cfg.use_bucketing = bool(train)
        cfg.num_buckets = d.num_buckets
        cfg.max_duration = d.max_duration
        cfg.min_duration = d.min_duration
        cfg.num_workers = d.num_workers
        cfg.shuffle = bool(train)
        cfg.defer_setup = False
        if not train:
            cfg.prompt_mode = "langID"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "conf" / "nemotron35_es_telephony.yaml"))
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.overrides))

    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from nemo.utils.exp_manager import exp_manager

    from finetune.g711_augment import TelephonyAugment, patch_prompt_dataset

    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
    if cfg.get("init_from_nemo_model"):
        model = nemo_asr.models.ASRModel.restore_from(cfg.init_from_nemo_model, map_location="cpu")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(cfg.init_from_pretrained_model, map_location="cpu")
    model.set_trainer(trainer)

    # validation stays clean: build it before the training dataset is patched...
    model.setup_validation_data(build_ds_cfg(model.cfg.validation_ds, cfg.data.val_manifest, cfg.data, train=False))
    aug = cfg.augment
    patch_prompt_dataset(TelephonyAugment(p=aug.telephony_p, max_loss=aug.max_packet_loss, p_noise=aug.noise_p,
                                          snr_lo=aug.snr_db[0], snr_hi=aug.snr_db[1]))
    # ...and mark the already-built validation dataset as un-augmented.
    for dl in (model._validation_dl if isinstance(model._validation_dl, list) else [model._validation_dl]):
        ds = getattr(dl, "dataset", None)
        if ds is not None:
            ds._telephony_augment = False
    model.setup_training_data(build_ds_cfg(model.cfg.train_ds, cfg.data.train_manifest, cfg.data, train=True))

    n_freeze = int(cfg.model.get("freeze_encoder_layers", 0))
    if n_freeze:
        for layer in model.encoder.layers[:n_freeze]:
            layer.requires_grad_(False)

    with open_dict(model.cfg):
        model.cfg.optim = cfg.model.optim
    model.setup_optimization(cfg.model.optim)

    if cfg.get("exp_manager") and cfg.trainer.get("logger") is False:
        exp_manager(trainer, cfg.exp_manager)
    trainer.fit(model)
    out = Path(cfg.exp_manager.exp_dir) / f"{cfg.exp_manager.name}_final.nemo"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
