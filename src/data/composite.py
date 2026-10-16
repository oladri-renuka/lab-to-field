"""Background compositing: paste PlantVillage leaves onto natural backgrounds.

Breaks the lab's uniform-background shortcut. The alpha mask comes from the segmented config (leaf on
near-black), repaired because that segmentation drops dark lesions; pixels come from the *color* image so
lesions are pasted intact.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter


def binary_leaf_mask(seg: Image.Image, threshold: int = 12, repair: int = 0, hull: bool = False) -> np.ndarray:
    """Boolean leaf mask from a segmented PlantVillage image (leaf on near-black).

    The PlantVillage segmentation often cuts away dark necrotic lesions as if they were background.
    repair > 0: morphological closing (that many 3x3 iterations) + hole filling, restoring interior lesions.
    hull: convex hull of the mask; a conservative "everything that could be leaf" region.
    """
    from scipy import ndimage

    m = np.asarray(seg.convert("RGB")).max(-1) > threshold
    m = ndimage.binary_opening(m, iterations=1)
    if repair > 0:
        pad = np.pad(m, repair)  # avoid eroding leaves that touch the border
        m = ndimage.binary_closing(pad, iterations=repair)[repair:-repair, repair:-repair]
        m = ndimage.binary_fill_holes(m)
    if hull and m.sum() >= 3:
        from PIL import ImageDraw
        from scipy.spatial import ConvexHull, QhullError

        pts = np.argwhere(m)
        try:
            poly = pts[ConvexHull(pts).vertices][:, ::-1]
            img = Image.new("L", (m.shape[1], m.shape[0]), 0)
            ImageDraw.Draw(img).polygon([tuple(map(int, xy)) for xy in poly], fill=255)
            m = m | (np.asarray(img) > 0)
        except QhullError:  # degenerate (collinear) mask
            pass
    return m


def leaf_mask(seg: Image.Image, threshold: int = 12, repair: int = 0, hull: bool = False) -> Image.Image:
    """Feathered 'L' alpha mask, see binary_leaf_mask."""
    m = binary_leaf_mask(seg, threshold, repair, hull)
    return Image.fromarray((m * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(1.0))


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
    mask: Image.Image | None = None,
) -> Image.Image:
    """Paste `leaf` onto `background`. `mask` (L) defaults to a threshold mask of `leaf` itself;
    pass a repaired mask from the segmented image to paste the original color pixels instead."""
    canvas = background.convert("RGB").copy()
    size = canvas.size[0]
    if (r := rng.uniform(*bg_blur)) > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(r))

    leaf = leaf.convert("RGB")
    mask = leaf_mask(leaf, mask_threshold) if mask is None else mask
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
