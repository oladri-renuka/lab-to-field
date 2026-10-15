import numpy as np
import pytest
from PIL import Image

pytest.importorskip("imagehash")
from src.data.dedup import cross_split_duplicates, phash_bits  # noqa: E402


def test_near_duplicate_is_found(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    b = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(a).resize((80, 80)).save(tmp_path / "a_resized.jpg", quality=80)
    Image.fromarray(b).save(tmp_path / "b.png")
    train = phash_bits([str(tmp_path / "a_resized.jpg"), str(tmp_path / "b.png")])
    test = phash_bits([str(tmp_path / "a.png")])
    dups = cross_split_duplicates(train, test, max_dist=4)
    assert [d[0] for d in dups] == [0]
