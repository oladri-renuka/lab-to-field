import random

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from src.data.prepare import leaf_key
from src.shortcut import background_tile

SC = OmegaConf.create({"cell": 32, "grid": 4, "stride": 16, "dilate": 9, "min_clean": 4, "hull": True})


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


def test_pair_index_falls_back_to_unique_suffix():
    from src.data.prepare import PairIndex

    seg = ["raw/segmented/Corn_(maize)___Common_rust_/00b29f2a-uuid___RS_Rust 1566_final_masked.jpg",
           "raw/segmented/Corn_(maize)___Common_rust_/aaaa___RS_Rust 1567_final_masked.jpg",
           "raw/segmented/Corn_(maize)___Common_rust_/bbbb___RS_Rust 1567_final_masked.jpg"]
    idx = PairIndex(seg)
    assert idx.get("raw/color/Corn_(maize)___Common_rust_/RS_Rust 1566.JPG") == seg[0]
    assert idx.get("raw/color/Corn_(maize)___Common_rust_/RS_Rust 1567.JPG") is None  # ambiguous -> no pairing


def test_background_tile_never_samples_cut_away_lesion():
    color = np.full((256, 256, 3), (90, 60, 30), np.uint8)
    color[40:216, 40:216] = (0, 200, 0)
    color[100:156, 100:156] = (255, 0, 0)  # lesion
    seg = np.zeros_like(color)
    seg[40:216, 40:216] = (0, 200, 0)
    seg[100:156, 100:156] = 0  # segmentation removed the lesion -> a hole inside the leaf
    for k in range(5):
        tile = np.asarray(background_tile(Image.fromarray(color), Image.fromarray(seg), SC, random.Random(k)))
        assert not ((tile[..., 0] > 200) & (tile[..., 1] < 50)).any()
