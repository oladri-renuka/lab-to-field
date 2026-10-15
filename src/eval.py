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
from src.utils import save_json


@torch.no_grad()
def predict(model, loader, device, amp_dtype=None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
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


def evaluate_run(run: str | Path, splits=("pd_test", "pv_test"), device=None) -> dict:
    run = Path(run)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, ckpt = load_checkpoint(run / "best.pt", device)
    classes = ckpt["classes"]
    tf = eval_transform(cfg.data.img_size, *norm_stats(model))
    amp = torch.bfloat16 if (device.type == "cuda" and cfg.train.precision == "bf16") else None

    out = {"exp_id": cfg.exp_id, "model": cfg.model, "seed": int(cfg.seed), "best_epoch": ckpt["epoch"],
           "pd_val_acc": ckpt["val_acc"], "n_classes": len(classes)}
    for split in splits:
        probs, labels = predict(model, make_loader(cfg, split, tf), device, amp)
        out[split] = classification_metrics(probs, labels, classes, cfg.eval.ece_bins,
                                            cfg.eval.bootstrap if split == "pd_test" else 0)
    if "pd_test" in out and "pv_test" in out:
        out["gap"] = out["pv_test"]["acc"] - out["pd_test"]["acc"]
    save_json(out, run / "metrics.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--splits", nargs="+", default=["pd_test", "pv_test"])
    args = ap.parse_args()
    m = evaluate_run(args.run, args.splits)
    for s in args.splits:
        print(f"{s}: acc={m[s]['acc']:.4f} macro_f1={m[s]['macro_f1']:.4f} ece={m[s]['ece']:.4f}")
    if "gap" in m:
        print(f"gap (PV - PD) = {m['gap']:.4f}")


if __name__ == "__main__":
    main()
