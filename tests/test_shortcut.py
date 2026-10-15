import random

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from src.data.prepare import leaf_key
from src.shortcut import background_tile

SC = OmegaConf.create({"cell": 32, "grid": 4, "stride": 16, "dilate": 9, "min_clean": 4})


def test_leaf_key_pairs_color_and_segmented():
    color = "raw/color/Tomato___healthy/abc___RS_HL 1.JPG"
    seg = "raw/segmented/Tomato___healthy/abc___RS_HL 1_final_masked.jpg"
    assert leaf_key(color) == leaf_key(seg)


def test_background_tile_contains_no_leaf_pixels():
    color = np.full((256, 256, 3), (90, 60, 30), np.uint8)  # brown background
    color[64:192, 64:192] = (0, 255, 0)  # bright green leaf
    seg = np.zeros_like(color)
    seg[64:192, 64:192] = (0, 255, 0)
    tile = np.asarray(background_tile(Image.fromarray(color), Image.fromarray(seg), SC, random.Random(0)))
    assert tile.shape == (128, 128, 3)
    assert not (tile[..., 1] > 200).any()


def test_background_tile_when_leaf_fills_frame():
    color = np.full((256, 256, 3), (0, 255, 0), np.uint8)
    tile = np.asarray(background_tile(Image.fromarray(color), Image.fromarray(color), SC, random.Random(0)))
    assert (tile == 128).all()  # leaf greyed out, never leaked
