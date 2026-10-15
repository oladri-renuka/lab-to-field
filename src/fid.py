"""FID (torchmetrics, Inception-v3 pool features) of generated images vs. PlantDoc train, plus sample grids.

Reference rows: raw PlantVillage vs PlantDoc (the starting gap) so the CycleGAN number has context.
    python -m src.fid --n 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

from src.data.datasets import read_manifest
from src.utils import load_config, save_json, seed_everything


def load_batch(paths, size=299):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


@torch.no_grad()
def fid(real_paths, fake_paths, dev, batch=64) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    m = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    for paths, real in ((real_paths, True), (fake_paths, False)):
        for s in range(0, len(paths), batch):
            m.update(load_batch(paths[s : s + batch]).to(dev), real=real)
    return float(m.compute())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(df: pd.DataFrame) -> list[str]:
        return df.sample(n=min(args.n, len(df)), random_state=0).path.tolist()

    real = sample(read_manifest(cfg, "pd_train"))
    results = {"reference_plantvillage": fid(real, sample(read_manifest(cfg, "pv_train")), dev)}
    figs = Path("reports/figures")
    figs.mkdir(parents=True, exist_ok=True)
    for kind in ("cyclegan", "cvae"):
        try:
            gen = read_manifest(cfg, kind)
        except FileNotFoundError:
            continue
        results[kind] = fid(real, sample(gen), dev)
        grid = gen.sample(n=min(16, len(gen)), random_state=0).path.tolist()
        save_image(load_batch(grid, 128), figs / f"{kind}_grid.png", nrow=4, padding=2)
    save_json({**results, "n": args.n, "real": "pd_train"}, "reports/fid.json")
    print(results)


if __name__ == "__main__":
    main()
