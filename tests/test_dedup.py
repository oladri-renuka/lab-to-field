import numpy as np
import pytest
from PIL import Image

pytest.importorskip("imagehash")
from src.data.dedup import matches, nearest, phash  # noqa: E402


def test_near_duplicate_is_found(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    b = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(a).resize((80, 80)).save(tmp_path / "a_resized.jpg", quality=80)
    Image.fromarray(b).save(tmp_path / "b.png")
    train = phash([str(tmp_path / "a_resized.jpg"), str(tmp_path / "b.png")], workers=1)
    test = phash([str(tmp_path / "a.png")], workers=1)
    assert [m[0] for m in matches(train, test, max_dist=4)] == [0]


def test_nearest_hamming():
    a = np.array([0b1011, 0xFFFF], dtype=np.uint64)
    b = np.array([0b0011, 0x0], dtype=np.uint64)
    idx, dist = nearest(a, b)
    assert idx.tolist() == [0, 0] and dist.tolist() == [1, 14]
