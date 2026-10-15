"""Background-only shortcut test on PlantVillage.

    python -m src.shortcut --config configs/audit/shortcut.yaml seed=0

Each image becomes a shuffled grid of patches cut only from leaf-free regions (leaf mask from the segmented
config, dilated). If a classifier trained on these beats chance by a wide margin on PlantVillage *test*,
the background alone predicts the label: the shortcut the project is about.
Writes runs/shortcut/<model>/seed<k>/metrics.json and reports/figures/shortcut_examples.png.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.data.composite import leaf_mask
from src.data.datasets import read_manifest
from src.data.io import load_image
from src.data.prepare import leaf_key, split_list
from src.eval import predict
from src.metrics import classification_metrics
from src.models.classifier import build_classifier, norm_stats
from src.train import lr_lambda
from src.utils import RunLogger, load_classes, load_config, run_dir, save_json, seed_everything, seed_worker


def attach_segmented(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Add `seg_path` (segmented counterpart, any split) to color manifest rows; drop rows without one."""
    seg_index = {leaf_key(r): r for s in ("train", "test") for r in split_list(cfg, "segmented", s)}
    out = []
    for r in df.itertuples():
        base, rel = r.path.split("/raw/", 1)
        seg_rel = seg_index.get(leaf_key("raw/" + rel))
        if seg_rel and Path(base, seg_rel).exists():
            out.append({**r._asdict(), "seg_path": str(Path(base, seg_rel))})
    return pd.DataFrame(out).drop(columns="Index")


def background_tile(color: Image.Image, seg: Image.Image, sc, rng: random.Random) -> Image.Image:
    color = color.resize((256, 256))
    mask = leaf_mask(seg.resize((256, 256))).filter(ImageFilter.MaxFilter(sc.dilate | 1))
    m = np.asarray(mask) > 0
    x = np.asarray(color).copy()
    c, st = sc.cell, sc.stride
    cells = [(i, j, m[i : i + c, j : j + c].mean()) for i in range(0, 256 - c + 1, st) for j in range(0, 256 - c + 1, st)]
    clean = [(i, j) for i, j, f in cells if f == 0]
    if len(clean) < sc.min_clean:  # leaf fills the frame: least-leafy cells, leaf pixels greyed out
        clean = [(i, j) for i, j, _ in sorted(cells, key=lambda t: t[2])[: sc.min_clean]]
        x[m] = 128
    picks = [rng.choice(clean) for _ in range(sc.grid * sc.grid)]
    tile = np.zeros((sc.grid * c, sc.grid * c, 3), np.uint8)
    for k, (i, j) in enumerate(picks):
        r, q = divmod(k, sc.grid)
        tile[r * c : (r + 1) * c, q * c : (q + 1) * c] = x[i : i + c, j : j + c]
    return Image.fromarray(tile)


class BackgroundDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sc, transform, deterministic: bool):
        self.df, self.sc, self.t, self.det = df.reset_index(drop=True), sc, transform, deterministic

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rng = random.Random(i) if self.det else random
        img = background_tile(load_image(r.path), load_image(r.seg_path), self.sc, rng)
        return self.t(img), int(r.label)


def example_figure(df: pd.DataFrame, sc, classes: list[str], out: Path, n: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = df.groupby("label").head(1).sample(n=min(n, df.label.nunique()), random_state=0)
    fig, axes = plt.subplots(2, len(rows), figsize=(2.2 * len(rows), 4.6))
    for k, r in enumerate(rows.itertuples()):
        axes[0, k].imshow(load_image(r.path).resize((256, 256)))
        axes[0, k].set_title(classes[r.label].replace("___", "\n"), fontsize=7)
        axes[1, k].imshow(background_tile(load_image(r.path), load_image(r.seg_path), sc, random.Random(0)))
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].text(-20, 128, "original", rotation=90, va="center", ha="right", fontsize=8)
    axes[1, 0].text(-10, 64, "background only", rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/audit/shortcut.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = load_classes(cfg.paths.processed_dir)

    train_df = attach_segmented(cfg, read_manifest(cfg, "pv_train"))
    test_df = attach_segmented(cfg, read_manifest(cfg, "pv_test"))
    print(f"[shortcut] train={len(train_df)} test={len(test_df)} (with segmented masks)")
    example_figure(test_df, cfg.shortcut, classes, Path("reports/figures/shortcut_examples.png"))

    model = build_classifier(cfg.models[cfg.model].timm_name, len(classes), pretrained=cfg.train.pretrained).to(dev)
    mean, std = norm_stats(model)
    size = cfg.data.img_size
    train_tf = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])
    loader = DataLoader(BackgroundDataset(train_df, cfg.shortcut, train_tf, False), batch_size=cfg.train.batch_size,
                        shuffle=True, drop_last=True, num_workers=cfg.data.num_workers, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(cfg.seed), pin_memory=dev.type == "cuda")
    test_loader = DataLoader(BackgroundDataset(test_df, cfg.shortcut, test_tf, True), batch_size=cfg.eval.batch_size,
                             num_workers=cfg.data.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.models[cfg.model].lr, weight_decay=cfg.train.weight_decay)
    steps = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg.train.warmup_epochs * steps, cfg.train.epochs * steps, 0.0))
    crit = nn.CrossEntropyLoss()
    amp = torch.bfloat16 if dev.type == "cuda" and cfg.train.precision == "bf16" else None
    logger = RunLogger(cfg, f"shortcut/{cfg.model}/seed{cfg.seed}")
    for epoch in range(cfg.train.epochs):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=amp or torch.float32, enabled=amp is not None):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        logger.log({"train_loss": tot / steps}, step=epoch)
        print(f"[shortcut {epoch:02d}] loss={tot / steps:.4f}")

    probs, labels = predict(model, test_loader, dev, amp)
    m = classification_metrics(probs, labels, classes, cfg.eval.ece_bins, cfg.eval.bootstrap)
    majority = float(pd.Series(labels).value_counts(normalize=True).iloc[0])
    result = {"seed": int(cfg.seed), "model": cfg.model, "test_acc": m["acc"], "test_acc_ci95": m["acc_ci95"],
              "macro_f1": m["macro_f1"], "chance_uniform": 1 / len(classes), "chance_majority": majority,
              "n_train": len(train_df), "n_test": len(test_df), "per_class_recall": m["per_class_recall"]}
    save_json(result, out_dir / "metrics.json")
    logger.log({"test_acc": m["acc"], "macro_f1": m["macro_f1"]}, step=cfg.train.epochs)
    logger.finish()
    print(f"[shortcut] background-only PlantVillage test acc={m['acc']:.3f} (95% CI {m['acc_ci95'][0]:.3f}–"
          f"{m['acc_ci95'][1]:.3f}), macro-F1={m['macro_f1']:.3f}; chance: uniform={1 / len(classes):.3f}, "
          f"majority={majority:.3f}")


if __name__ == "__main__":
    main()
