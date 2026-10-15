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
