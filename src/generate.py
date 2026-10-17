"""Cache generated training images + manifests.

    python -m src.generate cyclegan   # translate every PlantVillage train image once with G_AB
    python -m src.generate cvae       # sample N field-domain images per class
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms as T
from torchvision.utils import save_image

from src.data.io import load_image
from src.data.class_map import normalize
from src.data.datasets import read_manifest
from src.models.cvae import ConditionalVAE
from src.models.cyclegan import ResnetGenerator
from src.utils import load_classes, load_config, progress, seed_everything


@torch.no_grad()
def gen_cyclegan(cfg, dev, batch: int = 32) -> pd.DataFrame:
    ckpt = torch.load(Path(cfg.output_dir) / "latest.pt", map_location=dev, weights_only=False)
    G = ResnetGenerator(ngf=cfg.model.ngf, n_blocks=cfg.model.n_blocks).to(dev).eval()
    G.load_state_dict(ckpt["G_AB"])
    tf = T.Compose([T.Resize((cfg.data.crop_size, cfg.data.crop_size)), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    src = read_manifest(cfg, "pv_train")
    out_dir = Path(cfg.paths.generated_dir) / "cyclegan"
    rows = []
    for s in progress(range(0, len(src), batch), "translate PlantVillage -> field", unit="batch"):
        chunk = src.iloc[s : s + batch]
        x = torch.stack([tf(load_image(p)) for p in chunk.path]).to(dev)
        y = G(x) * 0.5 + 0.5
        for img, r in zip(y, chunk.itertuples()):
            path = out_dir / normalize(r.class_name) / f"{Path(r.path).stem}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(img, path)
            rows.append({"path": str(path), "label": r.label, "class_name": r.class_name, "source": "cyclegan",
                         "source_path": r.path})
    return pd.DataFrame(rows)


@torch.no_grad()
def gen_cvae(cfg, dev) -> pd.DataFrame:
    classes = load_classes(cfg.paths.processed_dir)
    idx = {c: i for i, c in enumerate(classes)}
    out_dir = Path(cfg.paths.generated_dir) / "cvae"
    rows = []
    ckpts = sorted(Path(cfg.output_dir).glob("*/cvae.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no */cvae.pt under {cfg.output_dir}; run src.train_cvae first")
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        members = ckpt["members"]
        model = ConditionalVAE(2 * len(members), cfg.model.latent_dim, tuple(cfg.model.channels), cfg.data.img_size).to(dev).eval()
        model.load_state_dict(ckpt["model"])
        for local, cls in enumerate(progress(members, f"sample {ckpt_path.parent.name}", unit="class")):
            y = torch.full((cfg.sample.per_class,), local * 2 + 1, device=dev)  # field domain
            for j, img in enumerate(model.sample(y)):
                path = out_dir / normalize(cls) / f"{j:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(img, path)
                rows.append({"path": str(path), "label": idx[cls], "class_name": cls, "source": "cvae"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["cyclegan", "cvae"])
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config or f"configs/{args.kind}.yaml", args.overrides)
    seed_everything(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate/{args.kind}] start", flush=True)
    df = gen_cyclegan(cfg, dev) if args.kind == "cyclegan" else gen_cvae(cfg, dev)
    path = Path(cfg.paths.generated_dir) / args.kind / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[generate/{args.kind}] wrote {len(df)} images -> {path}")


if __name__ == "__main__":
    main()
