"""Train CycleGAN lab (A = PlantVillage shared-class train) -> field (B = PlantDoc train split).

    python -m src.train_cyclegan --config configs/cyclegan.yaml

One "epoch" = len(B) iterations with A sampled at random (PlantVillage is ~20x larger than PlantDoc).
"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import save_image

from src.data.io import load_image
from src.data.datasets import read_manifest
from src.models.cyclegan import ImagePool, PatchDiscriminator, ResnetGenerator, init_weights
from src.utils import RunLogger, Steps, load_config, progress, seed_everything, seed_worker


def gan_transform(load: int, crop: int) -> T.Compose:
    return T.Compose([T.Resize((load, load), Image.BICUBIC), T.RandomCrop(crop), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])


class UnalignedDataset(Dataset):
    def __init__(self, a_paths, b_paths, transform):
        self.a, self.b, self.t = a_paths, b_paths, transform

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        a = load_image(random.choice(self.a))
        b = load_image(self.b[i])
        return self.t(a), self.t(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cyclegan.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    ds = UnalignedDataset(read_manifest(cfg, "pv_train")["path"].tolist(), read_manifest(cfg, "pd_train")["path"].tolist(),
                          gan_transform(cfg.data.load_size, cfg.data.crop_size))
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.data.num_workers, worker_init_fn=seed_worker, pin_memory=True)

    m = cfg.model
    G_AB, G_BA = ResnetGenerator(ngf=m.ngf, n_blocks=m.n_blocks).to(dev), ResnetGenerator(ngf=m.ngf, n_blocks=m.n_blocks).to(dev)
    D_A, D_B = PatchDiscriminator(ndf=m.ndf, n_layers=m.n_layers_d).to(dev), PatchDiscriminator(ndf=m.ndf, n_layers=m.n_layers_d).to(dev)
    for net in (G_AB, G_BA, D_A, D_B):
        net.apply(init_weights)

    t = cfg.train
    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=t.lr, betas=(t.beta1, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=t.lr, betas=(t.beta1, 0.999))
    decay = lambda e: 1.0 - max(0, e - t.decay_start) / float(t.epochs - t.decay_start + 1)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, decay)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, decay)
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    pool_A, pool_B = ImagePool(t.pool_size), ImagePool(t.pool_size)
    logger = RunLogger(cfg, "cyclegan")

    def gan_loss(pred, real: bool):
        return mse(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))

    fixed_a = torch.stack([ds[i][0] for i in range(8)]).to(dev)
    step = Steps("cyclegan", 2)
    step(f"training lab->field CycleGAN: {t.epochs} epochs x {len(loader)} iters (A=PlantVillage, B=PlantDoc train)")
    for epoch in range(t.epochs):
        sums = {"G": 0.0, "D": 0.0, "cyc": 0.0, "idt": 0.0}
        bar = progress(loader, f"epoch {epoch + 1}/{t.epochs}", total=len(loader), unit="it")
        for it, (real_A, real_B) in enumerate(bar):
            real_A, real_B = real_A.to(dev, non_blocking=True), real_B.to(dev, non_blocking=True)
            # Generators
            for d in (D_A, D_B):
                d.requires_grad_(False)
            fake_B, fake_A = G_AB(real_A), G_BA(real_B)
            loss_idt = (l1(G_AB(real_B), real_B) + l1(G_BA(real_A), real_A)) * t.lambda_identity
            loss_cyc = (l1(G_BA(fake_B), real_A) + l1(G_AB(fake_A), real_B)) * t.lambda_cycle
            loss_G = gan_loss(D_B(fake_B), True) + gan_loss(D_A(fake_A), True) + loss_cyc + loss_idt
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()
            # Discriminators
            for d in (D_A, D_B):
                d.requires_grad_(True)
            fb, fa = pool_B.query(fake_B), pool_A.query(fake_A)
            loss_D = 0.5 * (gan_loss(D_B(real_B), True) + gan_loss(D_B(fb), False)) \
                + 0.5 * (gan_loss(D_A(real_A), True) + gan_loss(D_A(fa), False))
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()
            for k, v in (("G", loss_G), ("D", loss_D), ("cyc", loss_cyc), ("idt", loss_idt)):
                sums[k] += v.item()
            if it % 50 == 0:
                bar.set_postfix(G=f"{sums['G'] / (it + 1):.2f}", D=f"{sums['D'] / (it + 1):.2f}")
        sched_G.step()
        sched_D.step()
        avg = {f"loss_{k}": v / len(loader) for k, v in sums.items()}
        logger.log(avg, step=epoch)
        print(f"[cyclegan epoch {epoch + 1}/{t.epochs}] " + " ".join(f"{k}={v:.3f}" for k, v in avg.items())
              + f" lr={sched_G.get_last_lr()[0]:.1e} (t+{step.elapsed()})", flush=True)
        if epoch % t.sample_every == 0:
            G_AB.eval()
            with torch.no_grad():
                save_image(torch.cat([fixed_a, G_AB(fixed_a)]) * 0.5 + 0.5, out / "samples" / f"epoch{epoch:03d}.jpg", nrow=8)
            G_AB.train()
        if (epoch + 1) % t.save_every == 0 or epoch + 1 == t.epochs:
            torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "epoch": epoch, "cfg": OmegaConf.to_container(cfg)},
                       out / "latest.pt")
    logger.finish()
    step(f"saved {out / 'latest.pt'} and samples in {out / 'samples'}")
    step.done()


if __name__ == "__main__":
    main()
