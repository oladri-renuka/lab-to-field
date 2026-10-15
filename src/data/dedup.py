"""Near-duplicate detection with perceptual hashing (internet-scraped data leaks across splits)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def phash_bits(paths: list[str], hash_size: int = 8) -> np.ndarray:
    import imagehash

    bits = np.zeros((len(paths), hash_size * hash_size), dtype=bool)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            bits[i] = imagehash.phash(im.convert("RGB"), hash_size=hash_size).hash.flatten()
    return bits


def hamming_matrix(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty((len(a), len(b)), dtype=np.int16)
    for s in range(0, len(a), chunk):
        out[s : s + chunk] = (a[s : s + chunk, None, :] != b[None, :, :]).sum(-1)
    return out


def cross_split_duplicates(train_bits: np.ndarray, test_bits: np.ndarray, max_dist: int) -> list[tuple[int, int, int]]:
    """(train_idx, closest_test_idx, distance) for every train image within `max_dist` of some test image."""
    if len(train_bits) == 0 or len(test_bits) == 0:
        return []
    d = hamming_matrix(train_bits, test_bits)
    nearest = d.argmin(1)
    dist = d[np.arange(len(d)), nearest]
    return [(int(i), int(nearest[i]), int(dist[i])) for i in np.flatnonzero(dist <= max_dist)]
