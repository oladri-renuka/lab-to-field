"""Train conditional VAEs (one per crop group by default) on PlantVillage + PlantDoc train.

The condition is (class, domain), so we can later sample *field-domain* images for each class.
Domains are balanced with a weighted sampler; otherwise PlantVillage (~95% of images) dominates.
β = 1 with linear KL warm-up over the first `kl_warmup_epochs`.

    python -m src.train_cvae --config configs/cvae.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.utils import save_image

from src.data.class_map import crop_of
from src.data.datasets import read_manifest
from src.models.cvae import ConditionalVAE, vae_loss
from src.utils import RunLogger, load_classes, load_config, seed_everything, seed_worker

DOMAINS = {"pv_train": 0, "pd_train": 1}  # 0 = lab, 1 = field


class CondDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int):
        self.df = df.reset_index(drop=True)
        self.t = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.t(Image.open(r.path).convert("RGB")), int(r.cond)


def groups(classes: list[str], group_by: str) -> dict[str, list[str]]:
    if group_by == "none":
        return {"all": classes}
    out: dict[str, list[str]] = {}
    for c in classes:
        out.setdefault(crop_of(c), []).append(c)
    return out


def group_frame(cfg, members: list[str]) -> pd.DataFrame:
    local = {c: i for i, c in enumerate(members)}
    parts = []
    for name, dom in DOMAINS.items():
        df = read_manifest(cfg, name)
        df = df[df.class_name.isin(local)].copy()
        df["domain"] = dom
        df["cond"] = df.class_name.map(local) * 2 + dom
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cvae.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(cfg.paths.processed_dir)
    out = Path(cfg.output_dir)

    for gname, members in groups(classes, cfg.group_by).items():
        seed_everything(cfg.seed)
        df = group_frame(cfg, members)
        dom_counts = df.domain.value_counts()
        weights = df.domain.map(lambda d: 1.0 / dom_counts[d]).values
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(df), replacement=True)
        loader = DataLoader(CondDataset(df, cfg.data.img_size), batch_size=cfg.train.batch_size, sampler=sampler,
                            num_workers=cfg.data.num_workers, worker_init_fn=seed_worker, drop_last=len(df) > cfg.train.batch_size)
        model = ConditionalVAE(2 * len(members), cfg.model.latent_dim, tuple(cfg.model.channels), cfg.data.img_size).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        logger = RunLogger(cfg, f"cvae/{gname}")
        print(f"[cvae/{gname}] classes={len(members)} images={len(df)} (field={int(dom_counts.get(1, 0))})")
        for epoch in range(cfg.train.epochs):
            beta = cfg.train.beta * min(1.0, (epoch + 1) / max(1, cfg.train.kl_warmup_epochs))
            tot = rec_s = kl_s = 0.0
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                recon, mu, logvar = model(x, y)
                loss, rec, kl = vae_loss(recon, x, mu, logvar, beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot, rec_s, kl_s = tot + loss.item(), rec_s + rec.item(), kl_s + kl.item()
            n = len(loader)
            logger.log({"loss": tot / n, "recon": rec_s / n, "kl": kl_s / n, "beta": beta}, step=epoch)
            print(f"[cvae/{gname} {epoch:03d}] loss={tot/n:.1f} recon={rec_s/n:.1f} kl={kl_s/n:.1f} beta={beta:.2f}")
        gdir = out / gname
        gdir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "members": members, "cfg": OmegaConf.to_container(cfg)}, gdir / "cvae.pt")
        with torch.no_grad():
            y = torch.arange(len(members), device=dev).repeat_interleave(8) * 2 + 1  # field domain
            save_image(model.eval().sample(y), gdir / "samples_field.jpg", nrow=8)
        logger.finish()


if __name__ == "__main__":
    main()
