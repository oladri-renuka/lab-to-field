"""One-time cache for background compositing (E2), so per-sample work is just paste + augment.

- repaired leaf masks (from the segmented images) -> small PNGs, keyed by threshold/repair settings
- background source images (PlantDoc train, ~800 px) -> downsized to `bg_max_side` JPEGs
Files are written atomically and reused across runs; only missing ones are computed.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from src.data.composite import leaf_mask
from src.data.io import load_image
from src.utils import progress


def _key(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def _atomic_save(img: Image.Image, out: Path, **kw) -> None:
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    img.save(tmp, format=kw.pop("format"), **kw)
    os.replace(tmp, out)


def _make_mask(args) -> None:
    seg_path, out, threshold, repair = args
    _atomic_save(leaf_mask(load_image(seg_path), threshold, repair=repair), Path(out), format="PNG")


def _make_bg(args) -> None:
    src, out, max_side = args
    img = load_image(src)
    img.thumbnail((max_side, max_side), Image.BILINEAR)
    _atomic_save(img, Path(out), format="JPEG", quality=92)


def _run(fn, jobs, desc: str, workers: int) -> None:
    if not jobs:
        return
    if workers <= 1:
        for j in progress(jobs, desc, unit="img"):
            fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        for _ in progress(ex.map(fn, jobs, chunksize=64), desc, total=len(jobs), unit="img"):
            pass


def cached_masks(seg_paths: list[str], cache_root: Path, threshold: int, repair: int, workers: int) -> list[str]:
    d = cache_root / f"masks_t{threshold}_r{repair}"
    d.mkdir(parents=True, exist_ok=True)
    outs = [str(d / f"{_key(p)}.png") for p in seg_paths]
    todo = [(p, o, threshold, repair) for p, o in zip(seg_paths, outs) if not Path(o).exists()]
    if todo:
        print(f"[composite-cache] computing {len(todo)} repaired leaf masks -> {d}", flush=True)
    _run(_make_mask, todo, "cache leaf masks", workers)
    return outs


def cached_backgrounds(paths: list[str], cache_root: Path, max_side: int, workers: int) -> list[str]:
    d = cache_root / f"backgrounds_{max_side}"
    d.mkdir(parents=True, exist_ok=True)
    outs = [str(d / f"{_key(p)}.jpg") for p in paths]
    todo = [(p, o, max_side) for p, o in zip(paths, outs) if not Path(o).exists()]
    if todo:
        print(f"[composite-cache] downsizing {len(todo)} background images -> {d}", flush=True)
    _run(_make_bg, todo, "cache backgrounds", workers)
    return outs
