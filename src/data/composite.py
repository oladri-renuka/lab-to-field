"""Background compositing: paste segmented PlantVillage leaves onto natural backgrounds.

Breaks the lab's uniform-background shortcut. Segmented PlantVillage images are leaves on (near-)black,
so the alpha mask is a threshold on max channel, cleaned with a morphological open and feathered.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter


def leaf_mask(leaf: Image.Image, threshold: int = 12) -> Image.Image:
    arr = np.asarray(leaf.convert("RGB"))
    mask = Image.fromarray(((arr.max(-1) > threshold) * 255).astype(np.uint8), mode="L")
    return mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.0))


def random_background(img: Image.Image, size: int, crop_scale: tuple[float, float], rng: random.Random) -> Image.Image:
    w, h = img.size
    side = int(min(w, h) * rng.uniform(*crop_scale))
    side = max(side, 16)
    x, y = rng.randint(0, w - side), rng.randint(0, h - side)
    return img.crop((x, y, x + side, y + side)).resize((size, size), Image.BILINEAR)


def composite(
    leaf: Image.Image,
    background: Image.Image,
    rng: random.Random,
    leaf_scale=(0.45, 0.95),
    rotation: float = 45,
    leaf_blur=(0.0, 1.0),
    bg_blur=(0.0, 2.0),
    mask_threshold: int = 12,
) -> Image.Image:
    canvas = background.convert("RGB").copy()
    size = canvas.size[0]
    if (r := rng.uniform(*bg_blur)) > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(r))

    leaf = leaf.convert("RGB")
    mask = leaf_mask(leaf, mask_threshold)
    bbox = mask.getbbox()
    if bbox is None:  # empty segmentation -> background only would carry a wrong label; return the leaf as is
        return leaf.resize((size, size))
    leaf, mask = leaf.crop(bbox), mask.crop(bbox)

    angle = rng.uniform(-rotation, rotation)
    leaf = leaf.rotate(angle, resample=Image.BILINEAR, expand=True)
    mask = mask.rotate(angle, resample=Image.BILINEAR, expand=True)

    target = rng.uniform(*leaf_scale) * size
    s = target / max(leaf.size)
    new = (max(1, int(leaf.size[0] * s)), max(1, int(leaf.size[1] * s)))
    leaf, mask = leaf.resize(new, Image.BILINEAR), mask.resize(new, Image.BILINEAR)
    if (r := rng.uniform(*leaf_blur)) > 0:
        leaf = leaf.filter(ImageFilter.GaussianBlur(r))

    x = rng.randint(0, max(0, size - new[0]))
    y = rng.randint(0, max(0, size - new[1]))
    canvas.paste(leaf, (x, y), mask)
    return canvas
