"""Manifest-backed datasets and the per-experiment training-set builder.

Every split lives as a CSV manifest (path,label,class_name,source) written by `src.data.prepare`
or `src.generate`, so each experiment is just a list of sources in its config.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from src.data.composite import composite, random_background
from src.data.splits import few_shot

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def manifest_path(cfg: DictConfig, name: str) -> Path:
    generated = {"cyclegan": "cyclegan", "cvae": "cvae"}
    if name in generated:
        return Path(cfg.paths.generated_dir) / generated[name] / "manifest.csv"
    return Path(cfg.paths.processed_dir) / "manifests" / f"{name}.csv"


def read_manifest(cfg: DictConfig, name: str) -> pd.DataFrame:
    path = manifest_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"manifest {path} missing (see README 'How to run')")
    return pd.read_csv(path)


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return (self.transform(img) if self.transform else img), self.labels[i]


class CompositeDataset(Dataset):
    """A fresh random composite of each segmented leaf every time it is drawn."""

    def __init__(self, seg_df: pd.DataFrame, background_paths: list[str], ccfg: DictConfig, transform=None):
        if not background_paths:
            raise ValueError("no background images for compositing")
        self.leaves = ManifestDataset(seg_df)
        self.backgrounds = background_paths
        self.c = ccfg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.leaves)

    def __getitem__(self, i):
        rng = random  # seeded per worker by src.utils.seed_worker
        leaf, label = self.leaves[i]
        with Image.open(rng.choice(self.backgrounds)) as bg_src:
            bg = random_background(bg_src.convert("RGB"), self.c.canvas, tuple(self.c.bg_crop_scale), rng)
        img = composite(
            leaf, bg, rng,
            leaf_scale=tuple(self.c.leaf_scale), rotation=self.c.rotation,
            leaf_blur=tuple(self.c.leaf_blur), bg_blur=tuple(self.c.bg_blur),
            mask_threshold=self.c.mask_threshold,
        )
        return (self.transform(img) if self.transform else img), label


def background_paths(cfg: DictConfig) -> list[str]:
    if cfg.paths.backgrounds_dir:
        return sorted(str(p) for p in Path(cfg.paths.backgrounds_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    return read_manifest(cfg, "pd_train")["path"].tolist()  # train split only; val/test never touched


def source_frame(cfg: DictConfig, source: str, seed: int) -> pd.DataFrame:
    if source == "pv_color":
        return read_manifest(cfg, "pv_train")
    if source == "pv_composite":
        return read_manifest(cfg, "pv_seg_train")
    if source in ("cyclegan", "cvae", "pd_train"):
        return read_manifest(cfg, source)
    if source == "pd_fewshot":
        if not cfg.data.fewshot_k:
            raise ValueError("pd_fewshot needs data.fewshot_k")
        return few_shot(read_manifest(cfg, "pd_train"), int(cfg.data.fewshot_k), seed)
    raise ValueError(f"unknown train source: {source}")


def build_train_dataset(cfg: DictConfig, transform, seed: int) -> Dataset:
    parts = []
    for source in cfg.data.train_sources:
        df = source_frame(cfg, source, seed)
        if cfg.data.max_train_samples:
            df = df.sample(n=min(len(df), int(cfg.data.max_train_samples)), random_state=seed)
        if source == "pv_composite":
            parts.append(CompositeDataset(df, background_paths(cfg), cfg.composite, transform))
        else:
            parts.append(ManifestDataset(df, transform))
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
