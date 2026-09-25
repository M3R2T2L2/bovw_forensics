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
