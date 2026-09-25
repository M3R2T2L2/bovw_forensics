"""Unsupervised clustering evaluation: NMI, ARI and Hungarian accuracy.

Image vectors are optionally PCA-reduced, then clustered with k-means into
as many clusters as there are classes. Several seeds give mean +/- std.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Best one-to-one mapping of clusters to classes (Hungarian)."""
    t = np.unique(y_true, return_inverse=True)[1]
    p = np.unique(y_pred, return_inverse=True)[1]
    m = np.zeros((p.max() + 1, t.max() + 1), dtype=np.int64)
    np.add.at(m, (p, t), 1)
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(y_true))


def evaluate_clustering(x: np.ndarray, y: np.ndarray, *, n_clusters: int | None = None,
                        pca_dim: int | None = 128, seeds=(0, 1, 2)) -> dict:
    x = np.asarray(x, dtype=np.float32)
    n_clusters = n_clusters or len(np.unique(y))
    if pca_dim and x.shape[1] > pca_dim and x.shape[0] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=0).fit_transform(x).astype(np.float32)
    rows = []
    for s in seeds:
        pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=s).fit_predict(x)
        rows.append((normalized_mutual_info_score(y, pred), adjusted_rand_score(y, pred), cluster_accuracy(y, pred)))
    a = np.array(rows)
    mean, std = a.mean(0), a.std(0)
    return {"nmi": mean[0], "nmi_std": std[0], "ari": mean[1], "ari_std": std[1],
            "acc": mean[2], "acc_std": std[2], "n_seeds": len(seeds)}
