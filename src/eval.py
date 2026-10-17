"""Evaluate a checkpoint on PlantDoc test (field) and PlantVillage test (lab), shared classes only.

`python -m src.eval --run runs/e0/resnet50/seed0` -> writes metrics.json next to the checkpoint.
Headline gap = PlantVillage-test acc - PlantDoc-test acc.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.datasets import ManifestDataset, read_manifest
from src.data.transforms import eval_transform
from src.metrics import classification_metrics
from src.models.classifier import build_classifier, norm_stats
from src.utils import progress, save_json


@torch.no_grad()
def predict(model, loader, device, amp_dtype=None, desc: str = "predict") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in progress(loader, desc, total=len(loader), unit="batch"):
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def make_loader(cfg, name, transform, batch_size=None):
    ds = ManifestDataset(read_manifest(cfg, name), transform)
    return DataLoader(ds, batch_size=batch_size or cfg.eval.batch_size, shuffle=False,
                      num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available())


def load_checkpoint(path: str | Path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_classifier(cfg.models[cfg.model].timm_name, len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg, ckpt


def evaluate_run(run: str | Path, splits=("pd_test", "pv_test"), device=None, ckpt: str = "best.pt",
                 out_name: str = "metrics.json") -> dict:
    """Evaluate one checkpoint of a run. Also saves per-image PlantDoc predictions (preds_<tag>.npz) for paired tests
    and for re-scoring after the manual test-label review, without re-running the model."""
    run = Path(run)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, state = load_checkpoint(run / ckpt, device)
    classes = state["classes"]
    tf = eval_transform(cfg.data.img_size, *norm_stats(model))
    amp = torch.bfloat16 if (device.type == "cuda" and cfg.train.precision == "bf16") else None
    splits = list(splits)
    if cfg.eval.get("secondary_pool", False) and "pd_pool" not in splits:
        splits.append("pd_pool")  # PlantDoc train split as a bigger, secondary field test (models that never saw it)

    val = state.get("val", {"pd_val": state["val_acc"]})
    out = {"exp_id": cfg.exp_id, "model": cfg.model, "seed": int(cfg.seed), "best_epoch": state["epoch"],
           "selected_on": state.get("selected_on", "pd_val"), "val_acc": val, "pd_val_acc": val.get("pd_val"),
           "n_classes": len(classes), "checkpoint": ckpt}
    tag = Path(out_name).stem.replace("metrics", "").strip("_") or "primary"
    for split in splits:
        manifest = "pd_train" if split == "pd_pool" else split
        loader = make_loader(cfg, manifest, tf)
        probs, labels = predict(model, loader, device, amp, desc=f"test {split}")
        out[split] = classification_metrics(probs, labels, classes, cfg.eval.ece_bins,
                                            cfg.eval.bootstrap if split in ("pd_test", "pd_pool") else 0)
        if split in ("pd_test", "pd_pool"):
            np.savez_compressed(run / f"preds_{tag}_{split}.npz", probs=probs.astype(np.float32), labels=labels,
                                paths=np.array(loader.dataset.paths, dtype=str))
    if "pd_test" in out and "pv_test" in out:
        out["gap"] = out["pv_test"]["acc"] - out["pd_test"]["acc"]
    save_json(out, run / out_name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--splits", nargs="+", default=["pd_test", "pv_test"])
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()
    m = evaluate_run(args.run, args.splits, ckpt=args.ckpt, out_name=args.out)
    for s in args.splits:
        print(f"{s}: acc={m[s]['acc']:.4f} macro_f1={m[s]['macro_f1']:.4f} ece={m[s]['ece']:.4f}")
    if "gap" in m:
        print(f"gap (PV - PD) = {m['gap']:.4f}")


if __name__ == "__main__":
    main()
