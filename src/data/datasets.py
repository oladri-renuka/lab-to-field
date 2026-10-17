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
from src.data.composite_cache import cached_backgrounds, cached_masks
from src.data.io import load_image
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
        img = load_image(self.paths[i])
        return (self.transform(img) if self.transform else img), self.labels[i]


class CompositeDataset(Dataset):
    """A fresh random composite of each leaf every time it is drawn.

    Mask: repaired segmentation mask (lesions the segmentation cut away are filled back in), precomputed by
    src.data.composite_cache. Pixels: the matching color image when the manifest has `color_path`.
    """

    def __init__(self, seg_df: pd.DataFrame, mask_paths: list[str], background_paths: list[str], ccfg: DictConfig,
                 transform=None):
        if not background_paths:
            raise ValueError("no background images for compositing")
        self.color = seg_df["color_path"].tolist() if "color_path" in seg_df else seg_df["path"].tolist()
        self.masks = mask_paths
        self.labels = seg_df["label"].astype(int).tolist()
        self.backgrounds = background_paths
        self.c = ccfg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.color)

    def __getitem__(self, i):
        rng = random  # seeded per worker by src.utils.seed_worker
        leaf = load_image(self.color[i])
        with Image.open(self.masks[i]) as m:
            mask = m.convert("L").resize(leaf.size)
        bg = random_background(load_image(rng.choice(self.backgrounds)), self.c.canvas, tuple(self.c.bg_crop_scale), rng)
        img = composite(
            leaf, bg, rng,
            leaf_scale=tuple(self.c.leaf_scale), rotation=self.c.rotation,
            leaf_blur=tuple(self.c.leaf_blur), bg_blur=tuple(self.c.bg_blur), mask=mask,
        )
        return (self.transform(img) if self.transform else img), self.labels[i]


def background_paths(cfg: DictConfig) -> list[str]:
    if cfg.paths.backgrounds_dir:
        return sorted(str(p) for p in Path(cfg.paths.backgrounds_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    return read_manifest(cfg, "pd_train")["path"].tolist()  # train split only; val/test never touched


def source_frame(cfg: DictConfig, source: str, seed: int) -> pd.DataFrame:
    if source == "pv_color":
        return read_manifest(cfg, "pv_train")
    if source == "pv_composite":
        return read_manifest(cfg, "pv_seg_train")
    if source == "cyclegan":
        # Translations exist for every image that was in PlantVillage train when they were generated; keep only those
        # whose source is still in the current train split (never a lab-val/test leaf).
        gen = read_manifest(cfg, "cyclegan")
        train = read_manifest(cfg, "pv_train")
        keep = set(zip(train.class_name, train.path.map(lambda x: Path(x).stem)))
        mask = [k in keep for k in zip(gen.class_name, gen.path.map(lambda x: Path(x).stem))]
        return gen[mask].reset_index(drop=True)
    if source in ("cvae", "pd_train"):
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
            c, cache = cfg.composite, Path(cfg.paths.processed_dir) / "composite_cache"
            workers = max(1, int(cfg.data.num_workers))
            masks = cached_masks(df["path"].tolist(), cache, int(c.mask_threshold), int(c.mask_repair), workers)
            bgs = cached_backgrounds(background_paths(cfg), cache, int(c.bg_max_side), workers)
            parts.append(CompositeDataset(df, masks, bgs, c, transform))
        else:
            parts.append(ManifestDataset(df, transform))
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
