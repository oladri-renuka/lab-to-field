"""Train a classifier from a config. Single GPU or DDP (torchrun).

    python -m src.train --config configs/experiments/e0_baseline.yaml model=resnet50 seed=0
    torchrun --nproc_per_node=4 -m src.train --config configs/compute/ddp.yaml exp_id=ddp_ws4

Early stopping and checkpoint selection use PlantDoc-val only. After training, the best checkpoint is
evaluated once on the held-out test sets (src.eval). Logged per epoch: loss, val acc, lr, images/sec,
peak GPU memory, fraction of step time spent waiting on the data loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.data.datasets import build_train_dataset, manifest_path
from src.data.transforms import eval_transform, train_transform
from src.eval import evaluate_run, make_loader, predict
from src.models.classifier import build_classifier, norm_stats
from src.utils import (RunLock, RunLogger, Steps, load_classes, load_config, progress, run_dir, save_json, seed_everything,
                       seed_worker)


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, dev
    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)


def lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def train_one_epoch(model, loader, opt, sched, criterion, mixup, device, amp_dtype, cfg, prof=None, desc="train",
                    show=True):
    model.train()
    n_img, data_wait, loss_sum = 0, 0.0, torch.zeros((), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_epoch = t0 = time.perf_counter()
    bar = progress(loader, desc, total=len(loader), enabled=show, unit="batch")
    for step, (x, y) in enumerate(bar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        data_wait += time.perf_counter() - t0
        if mixup is not None:
            x, y = mixup(x, y)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()
        loss_sum += loss.detach()
        n_img += x.size(0)
        if prof is not None:
            prof.step()
        if show and step % 25 == 0:  # .item() syncs the GPU, so only occasionally
            bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t_epoch
    steps = max(1, step + 1)
    return {
        "train_loss": (loss_sum / steps).item(),
        "images_per_sec": n_img / elapsed,  # per rank
        "data_wait_frac": data_wait / elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
        "epoch_time_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    done = run_dir(cfg) / "metrics.json"
    if os.environ.get("SKIP_FINISHED") == "1" and done.exists() and not cfg.get("overwrite", False):
        print(f"[train] already finished, skipping: {done.parent}", flush=True)
        return
    if int(os.environ.get("RANK", 0)) == 0:
        with RunLock(run_dir(cfg)):
            run(cfg)
    else:
        run(cfg)


def config_hash(cfg) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)
    c.pop("overwrite", None)
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]


def claim_run_dir(cfg, out_dir: Path) -> None:
    """Fail loudly instead of silently mixing two runs in one directory."""
    h, overwrite = config_hash(cfg), bool(cfg.get("overwrite", False))
    hash_file = out_dir / "config_hash.txt"
    if not overwrite:
        if (out_dir / "metrics.json").exists():
            raise RuntimeError(f"{out_dir} already holds a finished run. Move it away, or pass overwrite=true to replace it.")
        if hash_file.exists() and hash_file.read_text().strip() != h:
            raise RuntimeError(f"{out_dir} holds an unfinished run with a different config "
                               f"({hash_file.read_text().strip()} vs {h}). Move it away, or pass overwrite=true.")
    else:
        for f in list(out_dir.glob("metrics*.json")) + list(out_dir.glob("best*.pt")) + list(out_dir.glob("preds_*.npz")):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(h)
    OmegaConf.save(cfg, out_dir / "config.yaml")


def run(cfg) -> None:
    rank, world, device = setup_distributed()
    is_main = rank == 0
    steps_log = Steps("train", 5, enabled=is_main)
    out_dir = run_dir(cfg)
    if is_main:
        claim_run_dir(cfg, out_dir)

    seed_everything(cfg.seed, cfg.train.deterministic)  # identical model init on every rank
    classes = load_classes(cfg.paths.processed_dir)
    mcfg = cfg.models[cfg.model]
    steps_log(f"building {cfg.model} ({mcfg.timm_name}, {len(classes)} classes)"
              + (f", init from {cfg.train.init_from}" if cfg.train.init_from else ""))
    model = build_classifier(mcfg.timm_name, len(classes), pretrained=cfg.train.pretrained,
                             grad_checkpointing=cfg.train.grad_checkpointing)
    if cfg.train.init_from:
        state = torch.load(cfg.train.init_from, map_location="cpu", weights_only=False)
        if state["classes"] != classes:
            raise ValueError("init_from checkpoint has a different label space")
        model.load_state_dict(state["model"])
    model.to(device)
    mean, std = norm_stats(model)

    seed_everything(cfg.seed + rank)  # different augmentation stream per rank
    steps_log(f"building datasets: train sources {list(cfg.data.train_sources)}, aug={cfg.aug.kind}")
    train_ds = build_train_dataset(cfg, train_transform(cfg.aug.kind, cfg.data.img_size, mean, std), cfg.seed)
    sampler = DistributedSampler(train_ds, seed=cfg.seed, drop_last=True) if world > 1 else None
    g = torch.Generator().manual_seed(cfg.seed)
    bs = int(cfg.train.batch_size)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=sampler is None, sampler=sampler, drop_last=True,
                        num_workers=cfg.data.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=cfg.data.num_workers > 0, worker_init_fn=seed_worker, generator=g)
    if len(loader) == 0:
        raise ValueError(f"train set ({len(train_ds)}) smaller than one batch ({bs})")
    # Validation: PlantDoc val always; PlantVillage (lab) val when available. `train.select_on` picks the one that
    # drives best.pt + early stopping; the other's best checkpoint is kept as best_<name>.pt for a secondary report.
    val_names = ["pd_val"] + (["pv_val"] if manifest_path(cfg, "pv_val").exists() else [])
    select_on = str(cfg.train.select_on)
    if select_on not in val_names:
        raise ValueError(f"train.select_on={select_on} but available val sets are {val_names} (re-run src.data.prepare)")
    etf = eval_transform(cfg.data.img_size, mean, std)
    val_loaders = {n: make_loader(cfg, n, etf) for n in val_names}

    global_batch = bs * world
    lr = mcfg.lr * cfg.train.lr_mult * (global_batch / cfg.train.base_batch if cfg.train.scale_lr else 1.0)
    opt = create_optimizer_v2(model, "adamw", lr=lr, weight_decay=cfg.train.weight_decay)  # no wd on norm/bias
    steps = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(int(cfg.train.warmup_epochs * steps), int(cfg.train.epochs * steps), cfg.train.min_lr_ratio))

    mixup = None
    if cfg.aug.mixup:
        mixup = Mixup(mixup_alpha=cfg.aug.mixup_alpha, cutmix_alpha=cfg.aug.cutmix_alpha,
                      switch_prob=cfg.aug.mixup_switch_prob, label_smoothing=cfg.train.label_smoothing,
                      num_classes=len(classes))
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    amp_dtype = torch.bfloat16 if cfg.train.precision == "bf16" and device.type == "cuda" else None
    if world > 1:
        model = DDP(model, device_ids=[device.index])
    core = model.module if world > 1 else model

    logger = RunLogger(cfg, f"{cfg.exp_id}/{cfg.model}/seed{cfg.seed}", enabled=is_main)
    steps_log(f"training {cfg.exp_id} | {cfg.model} | seed {cfg.seed} | train={len(train_ds)} imgs, "
              f"{len(loader)} batches/epoch | val: " + ", ".join(f"{n}={len(v.dataset)}" for n, v in val_loaders.items())
              + f" | selecting on {select_on} | up to {cfg.train.epochs} epochs, "
              f"early stop after {cfg.train.early_stop_patience} without {select_on} gain | global batch {global_batch}, "
              f"lr {lr:.2e}, {cfg.train.precision} | output {out_dir}")

    best_acc, best_epoch, bad_epochs, history = -1.0, -1, 0, []
    best_other = {n: -1.0 for n in val_names if n != select_on}
    for epoch in range(cfg.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        prof = None
        if cfg.profile.enabled and epoch == 0 and is_main:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
                schedule=torch.profiler.schedule(wait=cfg.profile.wait, warmup=cfg.profile.warmup, active=cfg.profile.active),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir / "profile")),
            )
            prof.start()
        stats = train_one_epoch(model, loader, opt, sched, criterion, mixup, device, amp_dtype, cfg, prof,
                                desc=f"epoch {epoch + 1}/{cfg.train.epochs} train", show=is_main)
        if prof is not None:
            prof.stop()
            key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
            table = prof.key_averages().table(sort_by=key, row_limit=5)
            (out_dir / "profile_top5.txt").write_text(table + f"\ndata_wait_frac={stats['data_wait_frac']:.3f}\n")

        stop = torch.zeros(1, device=device)
        if is_main:
            accs = {}
            for n, vl in val_loaders.items():
                probs, labels = predict(core, vl, device, amp_dtype, desc=f"epoch {epoch + 1} {n}")
                accs[n] = float((probs.argmax(1) == labels).mean())
            val_acc = accs[select_on]
            stats.update(epoch=epoch, val_acc=val_acc, lr=sched.get_last_lr()[0],
                         images_per_sec_global=stats["images_per_sec"] * world, **{f"val_{n}": a for n, a in accs.items()})
            history.append(stats)
            logger.log(stats, step=epoch)
            ckpt = {"model": core.state_dict(), "cfg": OmegaConf.to_container(cfg), "classes": classes,
                    "epoch": epoch, "val_acc": val_acc, "val": accs, "selected_on": select_on}
            for n in best_other:
                if accs[n] > best_other[n]:
                    best_other[n] = accs[n]
                    torch.save({**ckpt, "val_acc": accs[n], "selected_on": n}, out_dir / f"best_{n}.pt")
            if val_acc > best_acc:
                best_acc, best_epoch, bad_epochs = val_acc, epoch, 0
                torch.save(ckpt, out_dir / "best.pt")
                note = "new best, saved"
            else:
                bad_epochs += 1
                note = f"no gain {bad_epochs}/{cfg.train.early_stop_patience} (best {best_acc:.4f} @ {best_epoch + 1})"
            others = " ".join(f"{n}={a:.4f}" for n, a in accs.items() if n != select_on)
            print(f"[epoch {epoch + 1:>2}/{cfg.train.epochs}] loss={stats['train_loss']:.4f} {select_on}={val_acc:.4f}"
                  + (f" ({others})" if others else "") + " "
                  f"| {note} | {stats['epoch_time_s']:.0f}s, {stats['images_per_sec_global']:.0f} img/s, "
                  f"data_wait={stats['data_wait_frac']:.2f}, mem={stats['peak_mem_gb']:.1f}GB | t+{steps_log.elapsed()}",
                  flush=True)
            if bad_epochs >= cfg.train.early_stop_patience:
                print(f"[train] early stop after epoch {epoch + 1}: best val_acc {best_acc:.4f} at epoch {best_epoch + 1}")
                stop += 1
        if world > 1:
            dist.broadcast(stop, 0)
        if stop.item():
            break

    if is_main:
        save_json({"history": history, "best_epoch": best_epoch, "best_val_acc": best_acc, "selected_on": select_on,
                   "world_size": world, "global_batch": global_batch, "lr": lr}, out_dir / "train_log.json")
        steps_log(f"evaluating best checkpoint (epoch {best_epoch + 1}, {select_on} {best_acc:.4f}) on PlantDoc test + "
                  f"PlantVillage test" + (f"; also the best-{', '.join(best_other)} checkpoint (secondary)" if best_other else ""))
        m = evaluate_run(out_dir, device=str(device))
        for n in best_other:
            m2 = evaluate_run(out_dir, device=str(device), ckpt=f"best_{n}.pt", out_name=f"metrics_{n}_selected.json")
            print(f"[test, secondary: selected on {n}] PD acc={m2['pd_test']['acc']:.4f} | PV acc={m2['pv_test']['acc']:.4f}")
        logger.log({"pd_test_acc": m["pd_test"]["acc"], "pv_test_acc": m["pv_test"]["acc"], "gap": m["gap"],
                    "pd_test_macro_f1": m["pd_test"]["macro_f1"], "pd_test_ece": m["pd_test"]["ece"]}, step=epoch + 1)
        lo, hi = m["pd_test"]["acc_ci95"]
        print(f"[test, selected on {select_on}] PD acc={m['pd_test']['acc']:.4f} (95% CI {lo:.3f}-{hi:.3f}) macro-F1={m['pd_test']['macro_f1']:.4f} "
              f"| PV acc={m['pv_test']['acc']:.4f} | gap={m['gap']:.4f}")
        logger.finish()
        steps_log(f"saved best.pt, metrics.json, train_log.json in {out_dir}")
        steps_log.done("finished")
    if dist.is_initialized():  # also under torchrun with a single process
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
