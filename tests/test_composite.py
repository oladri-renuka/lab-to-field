import random

import numpy as np
from PIL import Image

from src.data.composite import composite, leaf_mask


def leaf_on_black():
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[30:70, 20:80] = (40, 160, 40)
    return Image.fromarray(arr)


def test_mask_covers_leaf_only():
    m = np.asarray(leaf_mask(leaf_on_black()))
    assert m[50, 50] > 200 and m[5, 5] == 0


def test_composite_size_and_background_visible():
    bg = Image.new("RGB", (128, 128), (200, 120, 60))
    out = composite(leaf_on_black(), bg, random.Random(0), leaf_scale=(0.4, 0.4), bg_blur=(0, 0))
    arr = np.asarray(out)
    assert out.size == (128, 128)
    assert (np.abs(arr.astype(int) - (200, 120, 60)).sum(-1) < 10).mean() > 0.5  # background retained
    assert (np.abs(arr.astype(int) - (40, 160, 40)).sum(-1) < 30).any()  # leaf pasted


def leaf_with_cut_lesion():
    """Color image with a red lesion; the 'segmented' version has the lesion blacked out (as PlantVillage does)."""
    color = np.full((128, 128, 3), (150, 150, 150), np.uint8)
    color[24:104, 24:104] = (40, 160, 40)
    color[54:74, 54:74] = (200, 30, 30)  # lesion
    seg = np.zeros_like(color)
    seg[24:104, 24:104] = (40, 160, 40)
    seg[54:74, 54:74] = 0  # segmentation cut the lesion away
    return Image.fromarray(color), Image.fromarray(seg)


def test_repair_restores_cut_lesion():
    from src.data.composite import binary_leaf_mask

    _, seg = leaf_with_cut_lesion()
    assert not binary_leaf_mask(seg)[64, 64]
    assert binary_leaf_mask(seg, repair=7)[64, 64]


def test_hull_covers_concave_gap():
    from src.data.composite import binary_leaf_mask

    arr = np.zeros((100, 100, 3), np.uint8)
    arr[10:90, 10:30] = arr[10:90, 70:90] = arr[10:30, 10:90] = 200  # U shape
    m = binary_leaf_mask(Image.fromarray(arr), hull=True)
    assert m[60, 50]  # inside the U's opening, covered by the hull


def test_composite_pastes_lesion_from_color_image():
    from src.data.composite import leaf_mask

    color, seg = leaf_with_cut_lesion()
    bg = Image.new("RGB", (128, 128), (0, 0, 255))
    out = np.asarray(composite(color, bg, random.Random(0), leaf_scale=(0.9, 0.9), rotation=0,
                               leaf_blur=(0, 0), bg_blur=(0, 0), mask=leaf_mask(seg, repair=7)))
    assert ((out[..., 0] > 150) & (out[..., 1] < 80)).any()  # red lesion present
    assert not ((out[..., 0] > 140) & (out[..., 1] > 140) & (out[..., 2] > 140)).sum() > 50  # lab grey not pasted
