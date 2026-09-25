"""Image encodings from a vocabulary: hard, soft (kernel codebook), VLAD.

All encoders take the ragged `LocalFeatures` and return one row per image.
Post-processing follows common practice: signed square-root (power) then L2.

  hard : K-dim term-frequency histogram
  soft : K-dim kernel-codebook histogram over the `knn` nearest words
         (van Gemert et al. 2010); sigma = sigma_scale x the median distance
         to the nearest word
  vlad : K*d residual aggregation with intra-normalisation (Arandjelovic &
         Zisserman 2013). Size grows as K*d, so keep K small (<= 256).
"""

import numpy as np

from .extract import LocalFeatures
from .vocab import Vocabulary, l2n


def _sqdist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = (x * x).sum(1, keepdims=True) - 2.0 * x @ c.T + (c * c).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _chunks(feats: LocalFeatures, vocab: Vocabulary, max_rows: int = 65_536):
    """Yield (image_ids, transformed descriptors) in chunks to bound memory."""
    n = feats.n_images
    i = 0
    while i < n:
        j, rows = i, 0
        while j < n and (rows == 0 or rows + (feats.offsets[j + 1] - feats.offsets[j]) <= max_rows):
            rows += feats.offsets[j + 1] - feats.offsets[j]
            j += 1
        lo, hi = feats.offsets[i], feats.offsets[j]
        img_ids = np.repeat(np.arange(i, j), np.diff(feats.offsets[i : j + 1]))
        yield img_ids, vocab.transform(np.asarray(feats.desc[lo:hi], dtype=np.float32))
        i = j


def postprocess(x: np.ndarray, power: float = 0.5, l2: bool = True) -> np.ndarray:
    if power and power != 1.0:
        x = np.sign(x) * np.abs(x) ** power
    return l2n(x) if l2 else x


def hard(feats: LocalFeatures, vocab: Vocabulary, **pp) -> np.ndarray:
    out = np.zeros((feats.n_images, vocab.k), np.float32)
    for ids, x in _chunks(feats, vocab):
        a = _sqdist(x, vocab.centers).argmin(1)
        np.add.at(out, (ids, a), 1.0)
    out /= np.maximum(out.sum(1, keepdims=True), 1.0)
    return postprocess(out, **pp)


def estimate_sigma(feats: LocalFeatures, vocab: Vocabulary, max_n: int = 20_000, seed: int = 0) -> float:
    n = feats.desc.shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n, max_n), replace=False))
    x = vocab.transform(np.asarray(feats.desc[idx], dtype=np.float32))
    return float(np.sqrt(np.median(_sqdist(x, vocab.centers).min(1))) + 1e-8)


def soft(feats: LocalFeatures, vocab: Vocabulary, knn: int = 5, sigma: float | None = None,
         sigma_scale: float = 1.0, **pp) -> np.ndarray:
    """sigma_scale multiplies the median nearest-word distance; smaller = closer to hard."""
    sigma = (sigma or estimate_sigma(feats, vocab)) * sigma_scale
    knn = min(knn, vocab.k)
    out = np.zeros((feats.n_images, vocab.k), np.float32)
    for ids, x in _chunks(feats, vocab):
        d = _sqdist(x, vocab.centers)
        nn = np.argpartition(d, knn - 1, axis=1)[:, :knn]
        dn = np.take_along_axis(d, nn, 1)
        w = np.exp(-(dn - dn.min(1, keepdims=True)) / (2.0 * sigma**2))  # shift for stability
        w /= w.sum(1, keepdims=True)
        np.add.at(out, (np.repeat(ids, knn), nn.ravel()), w.ravel())
    out /= np.maximum(out.sum(1, keepdims=True), 1e-12)
    return postprocess(out, **pp)


def vlad(feats: LocalFeatures, vocab: Vocabulary, intra_norm: bool = True, **pp) -> np.ndarray:
    k, d = vocab.centers.shape
    out = np.zeros((feats.n_images, k, d), np.float32)
    for ids, x in _chunks(feats, vocab):
        a = _sqdist(x, vocab.centers).argmin(1)
        np.add.at(out, (ids, a), x - vocab.centers[a])
    if intra_norm:
        out /= np.linalg.norm(out, axis=2, keepdims=True) + 1e-12
    return postprocess(out.reshape(feats.n_images, k * d), **pp)


ENCODERS = {"hard": hard, "soft": soft, "vlad": vlad}


def encode(method: str, feats: LocalFeatures, vocab: Vocabulary, **kwargs) -> np.ndarray:
    if method not in ENCODERS:
        raise KeyError(f"Unknown assignment '{method}'. Known: {sorted(ENCODERS)}")
    return ENCODERS[method](feats, vocab, **kwargs)
