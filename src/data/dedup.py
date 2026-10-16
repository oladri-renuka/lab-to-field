"""Near-duplicate detection with perceptual hashing (internet-scraped data leaks across splits).

Hashes are packed into uint64 so nearest-neighbour search over ~50k x 2.5k pairs stays fast.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.data.io import load_image

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _phash_one(path: str) -> int:
    import imagehash

    bits = imagehash.phash(load_image(path), hash_size=8).hash.flatten()
    return int(np.packbits(bits).view(">u8")[0])


def phash(paths: list[str], workers: int = 8, desc: str = "phash") -> np.ndarray:
    """64-bit pHash per image as uint64."""
    from src.utils import progress

    if workers <= 1 or len(paths) < 200:
        return np.array([_phash_one(p) for p in progress(paths, desc, unit="img")], dtype=np.uint64)
    with ProcessPoolExecutor(workers) as ex:
        it = progress(ex.map(_phash_one, paths, chunksize=256), desc, total=len(paths), unit="img")
        return np.fromiter(it, dtype=np.uint64, count=len(paths))


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POPCOUNT8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def nearest(a: np.ndarray, b: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each hash in `a`, index of and Hamming distance to its nearest hash in `b`."""
    idx = np.empty(len(a), dtype=np.int64)
    dist = np.empty(len(a), dtype=np.int64)
    for s in range(0, len(a), chunk):
        d = _popcount(a[s : s + chunk, None] ^ b[None, :])
        idx[s : s + chunk] = d.argmin(1)
        dist[s : s + chunk] = d[np.arange(len(d)), idx[s : s + chunk]]
    return idx, dist


def matches(a: np.ndarray, b: np.ndarray, max_dist: int) -> list[tuple[int, int, int]]:
    """(i, j, distance) for every a[i] within `max_dist` of its nearest b[j]."""
    if len(a) == 0 or len(b) == 0:
        return []
    idx, dist = nearest(a, b)
    return [(int(i), int(idx[i]), int(dist[i])) for i in np.flatnonzero(dist <= max_dist)]
