"""What the model looks at: Grad-CAM (CNNs) and attention rollout (ViT) on 20 fixed PlantDoc-test images.

    python -m src.explain --runs runs/e0/resnet50/seed0 runs/e2/resnet50/seed0 runs/e3/resnet50/seed0

Images are a fixed seed-0 draw from PlantDoc test; they are used only for visualisation, never selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.datasets import read_manifest
from src.data.transforms import eval_transform
from src.eval import load_checkpoint
from src.models.classifier import is_vit, norm_stats


def grad_cam(model, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int]:
    """Grad-CAM on the final feature map of a timm CNN (forward_features -> forward_head)."""
    model.eval()
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    target = int(logits.argmax(1)) if target is None else target
    model.zero_grad()
    logits[0, target].backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    return _to_map(cam, x.shape[-2:]), target


def attention_rollout(model, x: torch.Tensor, discard_ratio: float = 0.0) -> tuple[np.ndarray, int]:
    """Abnar & Zuidema (2020): multiply (A + I) row-normalised, head-averaged attention across layers."""
    attns = []
    hooks = []
    for blk in model.blocks:
        blk.attn.fused_attn = False  # materialise the attention matrix
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(o.detach())))
    with torch.no_grad():
        target = int(model(x).argmax(1))
    for h in hooks:
        h.remove()
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    result = torch.eye(attns[0].size(-1), device=x.device)
    for a in attns:
        a = a[0].mean(0)
        if discard_ratio > 0:
            flat = a.flatten()
            flat[flat.argsort()[: int(flat.numel() * discard_ratio)]] = 0
        a = a + torch.eye(a.size(-1), device=x.device)
        result = (a / a.sum(-1, keepdim=True)) @ result
    mask = result[0, n_prefix:]
    side = int(mask.numel() ** 0.5)
    return _to_map(mask.reshape(1, 1, side, side), x.shape[-2:]), target


def _to_map(m: torch.Tensor, size) -> np.ndarray:
    m = F.interpolate(m.float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    m = m - m.min()
    return (m / (m.max() + 1e-8)).detach().cpu().numpy()


def saliency(model, x):
    return attention_rollout(model, x) if is_vit(model) else grad_cam(model, x)


def overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(img.resize(heat.shape[::-1])).astype(float) / 255
    return (1 - alpha) * base + alpha * plt.cm.jet(heat)[..., :3]


def fixed_images(cfg, n: int = 20) -> list[tuple[str, int]]:
    df = read_manifest(cfg, "pd_test").sample(n=n, random_state=0)
    return list(zip(df.path, df.label))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = [load_checkpoint(Path(r) / "best.pt", dev) for r in args.runs]
    cfg, classes = loaded[0][1], loaded[0][2]["classes"]
    images = fixed_images(cfg, args.n)
    fig, axes = plt.subplots(len(images), len(loaded) + 1, figsize=(2.4 * (len(loaded) + 1), 2.4 * len(images)))
    for r, (path, label) in enumerate(images):
        img = Image.open(path).convert("RGB")
        axes[r, 0].imshow(img.resize((224, 224)))
        axes[r, 0].set_title(classes[label].replace("___", "\n"), fontsize=6)
        for c, (model, mcfg, _) in enumerate(loaded, start=1):
            x = eval_transform(224, *norm_stats(model))(img)[None].to(dev)
            heat, pred = saliency(model, x)
            axes[r, c].imshow(overlay(img, heat))
            ok = "✓" if pred == label else "✗"
            axes[r, c].set_title(f"{mcfg.exp_id} {ok}\n{classes[pred].split('___')[-1]}", fontsize=6,
                                 color="green" if pred == label else "red")
    for ax in axes.flat:
        ax.axis("off")
    out = Path(args.out or f"reports/figures/saliency_{loaded[0][1].model}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
