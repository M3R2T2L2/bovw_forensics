"""Visual vocabulary: optional PCA-whitening, then (spherical) k-means.

The vocabulary is fit on a random subsample of descriptors so K up to a few
thousand stays tractable on Colab.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


@dataclass
class Vocabulary:
    centers: np.ndarray                      # (K, d) in the transformed space
    pca: Optional[PCA] = None
    spherical: bool = False
    info: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.centers.shape[0]

    def transform(self, desc: np.ndarray) -> np.ndarray:
        x = np.asarray(desc, dtype=np.float32)
        if self.pca is not None:
            x = self.pca.transform(x).astype(np.float32)
        return l2n(x) if self.spherical else x


def sample_descriptors(desc: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    n = desc.shape[0]
    if n <= max_n:
        return np.asarray(desc, dtype=np.float32)
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max_n, replace=False))
    return np.asarray(desc[idx], dtype=np.float32)


def build_vocabulary(desc: np.ndarray, k: int, *, max_descriptors: int = 200_000,
                     pca_dim: Optional[int] = None, whiten: bool = False,
                     spherical: bool = False, seed: int = 0, batch_size: int = 4096,
                     n_init: int = 3) -> Vocabulary:
    x = sample_descriptors(desc, max_descriptors, seed)
    pca = None
    if pca_dim and pca_dim < x.shape[1]:
        pca = PCA(n_components=pca_dim, whiten=whiten, random_state=seed).fit(x)
        x = pca.transform(x).astype(np.float32)
    if spherical:
        x = l2n(x)
    km = MiniBatchKMeans(n_clusters=k, batch_size=max(batch_size, 3 * k), n_init=n_init,
                         random_state=seed, max_no_improvement=20).fit(x)
    centers = km.cluster_centers_.astype(np.float32)
    if spherical:
        centers = l2n(centers)
    counts = np.bincount(km.labels_, minlength=k)
    return Vocabulary(centers=centers, pca=pca, spherical=spherical,
                      info={"k": k, "n_fit": int(x.shape[0]), "empty_words": int((counts == 0).sum()),
                            "inertia": float(km.inertia_)})


def build_vocabulary_torch(desc: np.ndarray, k: int, *, max_descriptors: int = 200_000, spherical: bool = True,
                           seed: int = 0, iters: int = 50, chunk: int | None = None,
                           max_block: int = 50_000_000, **_ignored) -> Vocabulary:
    """Lloyd k-means on the GPU (CPU torch fallback) for large K, where sklearn is too slow.

    Same sample as build_vocabulary (sample_descriptors with this seed). Initialisation:
    k distinct random samples (seeded); empty clusters are re-seeded with the samples
    farthest from their centre. Spherical: L2-normalised points and centres
    (cosine k-means). No PCA. A different algorithm from sklearn's MiniBatchKMeans
    with k-means++ and 3 inits, so results are labelled separately (codebook_gpu).
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x_np = sample_descriptors(desc, max_descriptors, seed)
    if spherical:
        x_np = l2n(x_np)
    n = len(x_np)
    k = min(k, n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32)).to(dev)
    c = x[torch.randperm(n, generator=g)[:k].to(dev)].clone()
    if chunk is None:  # rows per distance block: block = chunk x k floats, at most max_block (200 MB)
        chunk = max(256, min(65536, max_block // max(k, 1)))
    x_sq = (x * x).sum(1)
    assign = torch.empty(n, dtype=torch.long, device=dev)
    best = torch.empty(n, device=dev)
    inertia = float("nan")
    for _ in range(iters):
        c_sq = (c * c).sum(1)
        for i in range(0, n, chunk):
            xi = x[i:i + chunk]
            d = torch.addmm(c_sq[None, :], xi, c.T, alpha=-2.0).add_(x_sq[i:i + chunk, None])
            v, a = d.min(1)
            del d
            best[i:i + chunk], assign[i:i + chunk] = v.clamp_min(0), a
        counts = torch.bincount(assign, minlength=k)
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        new = sums / counts.clamp_min(1)[:, None].float()
        empty = (counts == 0).nonzero().flatten()
        if len(empty):
            far = best.topk(len(empty)).indices
            new[empty] = x[far]
        if spherical:
            new = new / new.norm(dim=1, keepdim=True).clamp_min(1e-12)
        shift = float((new - c).abs().max())
        c = new
        inertia = float(best.sum())
        if shift < 1e-6:
            break
    counts = torch.bincount(assign, minlength=k)
    centers = c.cpu().numpy().astype(np.float32)
    del x, c, x_sq, assign, best
    if dev == "cuda":
        torch.cuda.empty_cache()
    return Vocabulary(centers=centers, pca=None, spherical=spherical,
                      info={"k": k, "n_fit": n, "empty_words": int((counts == 0).sum()), "inertia": inertia,
                            "backend": "torch", "device": dev})
