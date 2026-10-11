"""Unsupervised anomaly detection with visual vocabularies (P0 task 3).

Everything is fit on normal training images only. Per test image we produce
patch-level scores (-> pixel map) and an image score, then report image- and
pixel-level AUROC.

Methods
  patch_knn        PatchCore-style baseline: exact nearest-neighbour distance from
                   each test patch to *all* normal training patches (no coreset,
                   so it is the accuracy ceiling for that family).
  global_knn       Image-level baseline: nearest-neighbour distance between CLS
                   embeddings (foundation backbones only).
  codebook_dist    Patch score = distance to the nearest visual word. K words
                   replace the full patch bank (K << N): the BoVW analogue.
  codebook_norm    Patch score = distance to the nearest word divided by that word's
                   radius (95th percentile of normal-patch distances to it), so broad
                   words tolerate more deviation than tight ones. (A pure word-rarity
                   score, -log p(word), was tried and dropped: with max-pooling its
                   discrete values tie across images and it scores AUROC 0.5.)
  hist_knn         Image-level: nearest-neighbour distance between bag-of-words
                   histograms (signed-sqrt + L2, i.e. Hellinger-like).
  coreset_knn      Matched-memory baseline: PatchCore's greedy k-center coreset of M
                   normal patches (M = K), then nearest-neighbour distance to it.
  random_knn       Matched-memory baseline: M normal patches sampled uniformly.
  hybrid_wNN       One memory of M vectors: NN% visual words + the rest real patches,
                   chosen greedily as those the words cover worst (residual coreset).

Pixel maps: patch scores on the feature grid, bilinear upsampling to the mask
size, Gaussian smoothing (sigma=4, as in PatchCore). Image score: max patch score.
"""

import gc
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import ad_data, cache
from .encode import _sqdist, hard
from .extract import LocalFeatures, get_extractor
from .sweep import RUNTIME_KEYS, _append, _env, _run_id, load_results
from .timing import budget
from .vocab import build_vocabulary, build_vocabulary_torch, l2n

PATCH_METHODS = ("patch_knn", "codebook_dist", "codebook_norm")
IMAGE_METHODS = ("global_knn", "hist_knn")
CODEBOOK_METHODS = ("codebook_dist", "codebook_norm", "hist_knn")
SUBSAMPLE_METHODS = ("coreset_knn", "random_knn", "coreset_trim1", "coreset_trim5")
TRIM_FRACTION = {"coreset_trim1": 0.01, "coreset_trim5": 0.05}


def hybrid_name(word_fraction: float) -> str:
    return f"hybrid_w{int(round(word_fraction * 100))}"
HANDCRAFTED = {"sift", "dense_sift", "orb"}  # no global embedding


# --------------------------------------------------------------------------- nearest neighbours

def _torch_cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def _to_cuda_f32(torch, x: np.ndarray, chunk: int = 262144):
    """Copy rows to a float32 CUDA tensor chunk by chunk (no full float32 copy in host RAM)."""
    out = torch.empty((len(x), x.shape[1]), dtype=torch.float32, device="cuda")
    for i in range(0, len(x), chunk):
        out[i:i + chunk] = torch.from_numpy(np.ascontiguousarray(x[i:i + chunk], dtype=np.float32)).cuda()
    return out


def nn_search(queries: np.ndarray, bank: np.ndarray, q_chunk: int = 4096, b_chunk: int = 32768,
              exclude_zero: bool = False):
    """Exact nearest bank row for each query: (Euclidean distance, index). GPU when available.

    exclude_zero: ignore bank rows at distance 0 (the query itself when it is in the bank).
    """
    torch = _torch_cuda()
    nq = len(queries)
    dist = np.empty(nq, np.float32)
    idx = np.empty(nq, np.int64)
    if torch is not None:
        b_all = _to_cuda_f32(torch, bank)
        b_sq = (b_all * b_all).sum(1)
        for i in range(0, nq, q_chunk):
            q = torch.from_numpy(np.ascontiguousarray(queries[i:i + q_chunk], dtype=np.float32)).cuda()
            q_sq = (q * q).sum(1, keepdim=True)
            best = torch.full((len(q),), float("inf"), device="cuda")
            arg = torch.zeros(len(q), dtype=torch.long, device="cuda")
            for j in range(0, len(b_all), b_chunk):
                d = (q_sq - 2.0 * q @ b_all[j:j + b_chunk].T + b_sq[j:j + b_chunk][None, :]).clamp_min(0)
                if exclude_zero:  # identical rows, up to float32 rounding
                    tol = 1e-6 * (q_sq + b_sq[j:j + b_chunk][None, :])
                    d = torch.where(d <= tol, torch.full_like(d, float("inf")), d)
                v, a = d.min(1)
                better = v < best
                best = torch.where(better, v, best)
                arg = torch.where(better, a + j, arg)
            dist[i:i + q_chunk] = best.sqrt().cpu().numpy()
            idx[i:i + q_chunk] = arg.cpu().numpy()
        del b_all, b_sq
        torch.cuda.empty_cache()
        return dist, idx
    bc = b_chunk // 4
    b_sq = np.concatenate([(np.asarray(bank[j:j + bc], np.float32) ** 2).sum(1) for j in range(0, len(bank), bc)])
    for i in range(0, nq, 1024):
        q = np.asarray(queries[i:i + 1024], np.float32)
        q_sq = (q * q).sum(1, keepdims=True)
        best = np.full(len(q), np.inf, np.float32)
        arg = np.zeros(len(q), np.int64)
        for j in range(0, len(bank), bc):
            d = np.maximum(q_sq - 2.0 * q @ np.asarray(bank[j:j + bc], np.float32).T + b_sq[None, j:j + bc], 0)
            if exclude_zero:  # identical rows, up to float32 rounding
                d[d <= 1e-6 * (q_sq + b_sq[None, j:j + bc])] = np.inf
            a = d.argmin(1)
            v = d[np.arange(len(a)), a]
            better = v < best
            best[better], arg[better] = v[better], a[better] + j
        dist[i:i + 1024], idx[i:i + 1024] = np.sqrt(best), arg
    return dist, idx


def min_dist(queries: np.ndarray, bank: np.ndarray, q_chunk: int = 4096, b_chunk: int = 65536) -> np.ndarray:
    """Euclidean distance from each query row to its nearest bank row (exact).

    Uses the GPU when available (Colab T4: ~10 s for 160k x 400k x 384); falls
    back to chunked NumPy. Both loops are chunked so memory stays bounded.
    """
    torch = _torch_cuda()
    out = np.empty(len(queries), np.float32)
    if torch is not None:
        b_all = _to_cuda_f32(torch, bank)
        b_sq = (b_all * b_all).sum(1)
        for i in range(0, len(queries), q_chunk):
            q = torch.from_numpy(np.ascontiguousarray(queries[i:i + q_chunk], dtype=np.float32)).cuda()
            q_sq = (q * q).sum(1, keepdim=True)
            best = torch.full((len(q),), float("inf"), device="cuda")
            for j in range(0, len(b_all), b_chunk):
                d = q_sq - 2.0 * q @ b_all[j:j + b_chunk].T + b_sq[j:j + b_chunk][None, :]
                best = torch.minimum(best, d.min(1).values)
            out[i:i + q_chunk] = best.clamp_min(0).sqrt().cpu().numpy()
        del b_all, b_sq
        torch.cuda.empty_cache()
        return out
    bc = b_chunk // 4
    b_sq = np.concatenate([(np.asarray(bank[j:j + bc], np.float32) ** 2).sum(1) for j in range(0, len(bank), bc)])
    q_chunk = min(q_chunk, 1024)
    for i in range(0, len(queries), q_chunk):
        q = np.asarray(queries[i:i + q_chunk], np.float32)
        q_sq = (q * q).sum(1, keepdims=True)
        best = np.full(len(q), np.inf, np.float32)
        for j in range(0, len(bank), bc):  # bank chunks converted on the fly: no full float32 copy
            d = q_sq - 2.0 * q @ np.asarray(bank[j:j + bc], np.float32).T + b_sq[None, j:j + bc]
            np.minimum(best, d.min(1), out=best)
        out[i:i + q_chunk] = np.sqrt(np.maximum(best, 0))
    return out


def greedy_coreset(x: np.ndarray, m: int, seed: int = 0, proj_dim: int = 128, chunk: int = 262144,
                   init_centers: np.ndarray | None = None) -> np.ndarray:
    """Indices of a greedy k-center coreset (PatchCore, Roth et al. 2022).

    Features are first mapped by a Gaussian random projection to `proj_dim`
    dimensions (as in PatchCore) to speed up the distance updates. Greedy
    selection is nested: the first m' < m indices are the size-m' coreset.

    init_centers (e.g. visual words): selection starts as if these were already
    chosen, so it picks the patches they cover worst (a residual coreset).
    """
    n = len(x)
    m = min(m, n)
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    if proj_dim and proj_dim < d:
        proj = (rng.standard_normal((d, proj_dim)) / np.sqrt(proj_dim)).astype(np.float32)
    else:
        proj = None
    first = int(rng.integers(n))
    if m <= 0:
        return np.zeros(0, np.int64)
    cz = None
    if init_centers is not None and len(init_centers):
        cz = np.asarray(init_centers, np.float32)
        cz = cz @ proj if proj is not None else cz
    torch = _torch_cuda()
    if torch is not None:
        if proj is not None:
            p_t = torch.from_numpy(proj).cuda()
            z = torch.empty((n, proj.shape[1]), dtype=torch.float32, device="cuda")
            for i in range(0, n, chunk):
                z[i:i + chunk] = _to_cuda_f32(torch, x[i:i + chunk]) @ p_t
        else:
            z = _to_cuda_f32(torch, x)
        if cz is None:
            sel = [first]
            mind = ((z - z[first]) ** 2).sum(1)
        else:
            c_t = torch.from_numpy(cz).cuda()
            c_sq = (c_t * c_t).sum(1)
            mind = torch.empty(n, device="cuda")
            for i in range(0, n, chunk):
                zc = z[i:i + chunk]
                dd = (zc * zc).sum(1, keepdim=True) - 2.0 * zc @ c_t.T + c_sq[None, :]
                mind[i:i + chunk] = dd.min(1).values.clamp_min(0)
            sel = []
        while len(sel) < m:
            nxt = int(mind.argmax())
            sel.append(nxt)
            mind = torch.minimum(mind, ((z - z[nxt]) ** 2).sum(1))
        del z, mind
        torch.cuda.empty_cache()
        return np.array(sel, np.int64)
    z = np.empty((n, proj_dim if proj is not None else d), np.float32)
    for i in range(0, n, chunk):
        blk = np.asarray(x[i:i + chunk], np.float32)
        z[i:i + chunk] = blk @ proj if proj is not None else blk
    if cz is None:
        sel = [first]
        mind = ((z - z[first]) ** 2).sum(1)
    else:
        c_sq = (cz * cz).sum(1)
        mind = np.empty(n, np.float32)
        for i in range(0, n, chunk):
            zc = z[i:i + chunk]
            mind[i:i + chunk] = np.maximum(((zc * zc).sum(1, keepdims=True) - 2.0 * zc @ cz.T + c_sq[None]).min(1), 0)
        sel = []
    while len(sel) < m:
        nxt = int(mind.argmax())
        sel.append(nxt)
        np.minimum(mind, ((z - z[nxt]) ** 2).sum(1), out=mind)
    return np.array(sel, np.int64)


def outlier_scores(x: np.ndarray, n_ref: int = 50000, seed: int = 0) -> np.ndarray:
    """Distance from each row to its nearest neighbour in a random reference sample of rows
    (itself excluded): a cheap kNN outlier score for trimming before k-center."""
    rng = np.random.default_rng(seed + 7919)
    ref = np.sort(rng.choice(len(x), size=min(n_ref, len(x)), replace=False))
    d, _ = nn_search(x, np.asarray(x[ref], np.float32), exclude_zero=True)
    return d


def trimmed_coreset(x: np.ndarray, m: int, trim: float, seed: int = 0) -> np.ndarray:
    """Greedy k-center on the rows left after dropping the `trim` fraction with the largest outlier score."""
    s = outlier_scores(x, seed=seed)
    keep = np.sort(np.argsort(s)[:int(round(len(x) * (1 - trim)))])
    sel = greedy_coreset(x[keep], m, seed=seed)
    return keep[sel]


# --------------------------------------------------------------------------- scoring helpers

def grid_side(feats: LocalFeatures) -> int:
    counts = feats.counts()
    if counts.min() != counts.max():
        raise ValueError("Pixel maps need the same number of patches per image (a dense grid).")
    side = math.isqrt(int(counts[0]))
    if side * side != counts[0]:
        raise ValueError(f"{counts[0]} patches per image is not a square grid.")
    return side


def patch_maps(patch_scores: np.ndarray, n_images: int, side: int, mask_size: int, sigma: float = 4.0,
               dtype=np.float32):
    """(n_images*side*side,) -> smoothed (n_images, mask_size, mask_size) maps and max-patch image scores."""
    grid = patch_scores.reshape(n_images, side, side).astype(np.float32)
    maps = np.empty((n_images, mask_size, mask_size), dtype)
    for i in range(n_images):
        m = cv2.resize(grid[i], (mask_size, mask_size), interpolation=cv2.INTER_LINEAR)
        maps[i] = cv2.GaussianBlur(m, (0, 0), sigma) if sigma else m
    return maps, grid.reshape(n_images, -1).max(1)


def _auroc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, steps: int = 301) -> float:
    """Area under the per-region-overlap curve up to `fpr_limit`, normalised to [0, 1].

    MVTec AD's region metric (Bergmann et al. 2021): every connected defect region
    (8-connectivity) counts equally, however large. For each false-positive rate f
    on a grid over [0, fpr_limit], the threshold is the (1 - f) quantile of normal
    pixel scores; PRO(f) is the mean fraction of each region above it.
    """
    masks = np.asarray(masks).astype(bool)
    if not masks.any():
        return float("nan")
    neg = np.asarray(maps, np.float32)[~masks]
    fprs = np.linspace(0.0, fpr_limit, steps)
    thr = np.quantile(neg, 1.0 - fprs)
    del neg
    regions = []
    for m, s in zip(masks, maps):
        if not m.any():
            continue
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        for r in range(1, n):
            regions.append(np.sort(s[lab == r].astype(np.float32)))
    pro = np.zeros(steps)
    for reg in regions:  # fraction of region pixels with score >= threshold
        pro += 1.0 - np.searchsorted(reg, thr, side="left") / len(reg)
    pro /= len(regions)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(pro, fprs) / fpr_limit)


def pixel_dprime(masks: np.ndarray, maps: np.ndarray) -> float:
    """(mean defect-pixel score - mean normal-pixel score) / SD of normal-pixel scores.

    Scale-free separation; a low-noise normal background raises it.
    """
    m = np.asarray(masks).astype(bool)
    if not m.any():
        return float("nan")
    s = np.asarray(maps, np.float32)
    neg = s[~m]
    return float((s[m].mean() - neg.mean()) / (neg.std() + 1e-12))


def topk_image_scores(patch_scores: np.ndarray, n_images: int, k: int) -> np.ndarray:
    """Image score = mean of the k highest patch scores (k = 1 is the max)."""
    g = np.asarray(patch_scores, np.float32).reshape(n_images, -1)
    k = min(k, g.shape[1])
    return np.partition(g, g.shape[1] - k, axis=1)[:, -k:].mean(1)


LARGE_PIXELS = 100_000_000  # above this many pixels, pixel metrics use a 65,536-bin histogram (3CAD only)
_BINS = 65536


def _score_range(maps: np.ndarray, chunk: int = 256) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for i in range(0, len(maps), chunk):
        b = maps[i:i + chunk]
        lo, hi = min(lo, float(b.min())), max(hi, float(b.max()))
    return lo, hi if hi > lo else lo + 1e-6


def _hist(maps, masks, lo, hi, want_pos: bool, chunk: int = 256) -> np.ndarray:
    """Histogram of scores of pixels where mask == want_pos, streamed over images."""
    h = np.zeros(_BINS, np.int64)
    scale = (_BINS - 1) / (hi - lo)
    for i in range(0, len(maps), chunk):
        m = masks[i:i + chunk].astype(bool)
        sel = m if want_pos else ~m
        v = maps[i:i + chunk][sel].astype(np.float32)
        h += np.bincount(np.clip(((v - lo) * scale).astype(np.int64), 0, _BINS - 1), minlength=_BINS)
    return h


def binned_pixel_metrics(masks: np.ndarray, maps: np.ndarray, region_metrics: bool,
                         fpr_limit: float = 0.3, steps: int = 301) -> dict:
    """Pixel AUROC, AUPRO and d' from streamed 65,536-bin histograms (memory-bounded).

    Used only when there are more than LARGE_PIXELS pixels; the binning error in
    AUROC is below 1e-4 (tests compare with the exact values).
    """
    lo, hi = _score_range(maps)
    hp, hn = _hist(maps, masks, lo, hi, True), _hist(maps, masks, lo, hi, False)
    npos, nneg = hp.sum(), hn.sum()
    out = {}
    if npos == 0 or nneg == 0:
        return {"pixel_auroc": float("nan")}
    # AUROC = P(pos > neg) + 0.5 P(tie), ties = same bin
    neg_below = np.concatenate([[0], np.cumsum(hn)[:-1]])
    out["pixel_auroc"] = float((hp * (neg_below + 0.5 * hn)).sum() / (npos * nneg))
    if region_metrics:
        centers = lo + (np.arange(_BINS) + 0.5) * (hi - lo) / (_BINS - 1)
        neg_above = nneg - np.cumsum(hn) + hn          # negatives with score in bin >= b
        fpr_at_bin = neg_above / nneg                  # decreasing in b
        fprs = np.linspace(0.0, fpr_limit, steps)
        # threshold for each target FPR: first bin whose FPR <= f
        idx = np.searchsorted(-fpr_at_bin, -fprs, side="left").clip(0, _BINS - 1)
        thr = lo + idx * (hi - lo) / (_BINS - 1)
        pro, n_reg = np.zeros(steps), 0
        for m, sc in zip(masks, maps):
            if not m.any():
                continue
            n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
            for r in range(1, n):
                reg = np.sort(sc[lab == r].astype(np.float32))
                pro += 1.0 - np.searchsorted(reg, thr, side="left") / len(reg)
                n_reg += 1
        trap = getattr(np, "trapezoid", None) or np.trapz
        out["aupro"] = float(trap(pro / max(n_reg, 1), fprs) / fpr_limit)
        mu_p = (hp * centers).sum() / npos
        mu_n = (hn * centers).sum() / nneg
        sd_n = np.sqrt((hn * (centers - mu_n) ** 2).sum() / nneg)
        out["pixel_dprime"] = float((mu_p - mu_n) / (sd_n + 1e-12))
    return out


def metrics(labels, image_scores, types, masks=None, maps=None, region_metrics: bool = False) -> dict:
    out = {"image_auroc": _auroc(labels, image_scores)}
    good = np.array([t == "good" for t in types])
    by_type = {}
    for t in sorted(set(types) - {"good"}):
        sel = good | np.array([x == t for x in types])
        by_type[t] = _auroc(np.asarray(labels)[sel], np.asarray(image_scores)[sel])
    out["image_auroc_by_type"] = json.dumps(by_type)
    if maps is not None and maps.size > LARGE_PIXELS:
        out.update(binned_pixel_metrics(masks, maps, region_metrics))
        out["pixel_metrics_binned"] = True
    elif maps is not None:
        out["pixel_auroc"] = _auroc(masks.reshape(-1), maps.reshape(-1))
        if region_metrics:
            out["aupro"] = aupro(masks, maps)
            out["pixel_dprime"] = pixel_dprime(masks, maps)
    return out


def _norm_inplace(x: np.ndarray, normalize: bool, chunk: int = 262144) -> np.ndarray:
    """L2-normalise rows in place, chunk by chunk (no full float32 copy); keeps x's dtype."""
    if not normalize:
        return x
    for i in range(0, len(x), chunk):
        b = np.asarray(x[i:i + chunk], np.float32)
        x[i:i + chunk] = l2n(b)
    return x


def _norm(x: np.ndarray, normalize: bool) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return l2n(x) if normalize else x


def nearest_word(desc: np.ndarray, vocab, chunk: int = 65536):
    """Transformed-space distance to, and index of, the nearest visual word (GPU when available)."""
    if _torch_cuda() is not None:
        dist = np.empty(len(desc), np.float32)
        idx = np.empty(len(desc), np.int64)
        for i in range(0, len(desc), chunk * 4):
            x = vocab.transform(np.asarray(desc[i:i + chunk * 4], np.float32))
            dist[i:i + chunk * 4], idx[i:i + chunk * 4] = nn_search(x, vocab.centers)
        return dist, idx
    dist = np.empty(len(desc), np.float32)
    idx = np.empty(len(desc), np.int64)
    for i in range(0, len(desc), chunk):
        x = vocab.transform(np.asarray(desc[i:i + chunk], np.float32))
        d = _sqdist(x, vocab.centers)
        a = d.argmin(1)
        idx[i:i + chunk] = a
        dist[i:i + chunk] = np.sqrt(d[np.arange(len(a)), a])
    return dist, idx


def word_radius(dist: np.ndarray, words: np.ndarray, k: int, q: float = 0.95, min_members: int = 5) -> np.ndarray:
    """Per-word q-quantile of member distances; sparse words fall back to the global median radius."""
    order = np.argsort(words, kind="stable")
    counts = np.bincount(words, minlength=k)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    r = np.empty(k, np.float32)
    for w in range(k):
        m = dist[order[starts[w]:starts[w] + counts[w]]]
        r[w] = np.quantile(m, q) if counts[w] >= min_members else np.nan
    fill = np.nanmedian(r) if np.isfinite(r).any() else 1.0
    r = np.where(np.isfinite(r), r, fill)
    return np.maximum(r, 1e-6)


def cached_vocabulary(cache_dir: Path | None, key: dict, build):
    """Build a vocabulary once and pickle it (with its build timing) under cache_dir/vocab."""
    import hashlib
    import pickle

    if cache_dir is None:
        t: dict = {}
        with budget(t, "vocab"):
            v = build()
        return v, t
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = Path(cache_dir) / "vocab" / f"{h}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            v, t = pickle.load(f)
        return v, {**t, "vocab_cache_hit": True}
    t = {}
    with budget(t, "vocab"):
        v = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump((v, t), f)
    tmp.replace(path)
    return v, t


# --------------------------------------------------------------------------- dataset with lazy images

@dataclass
class _Meta:
    labels: np.ndarray
    masks: np.ndarray
    types: list
    n_train: int


class _LazyAD:
    """Loads images only when a feature cache misses; labels/masks are cached."""

    def __init__(self, cfg_data: dict, category: str, cache_dir: Path, tag: str):
        self.cfg, self.category = cfg_data, category
        self.meta_path = cache_dir / f"{tag}__meta.npz"
        self._split = None
        self.meta = self._load_meta() if self.meta_path.exists() else None

    def _load_meta(self):
        z = np.load(self.meta_path, allow_pickle=False)
        return _Meta(z["labels"], z["masks"], list(z["types"]), int(z["n_train"]))

    @property
    def split(self) -> ad_data.ADSplit:
        if self._split is None:
            name = self.cfg.get("name", "mvtec")
            if name == "mvtec":
                self._split = ad_data.load_mvtec(self.cfg["root"], self.category, self.cfg.get("size", 448),
                                                 self.cfg.get("mask_size", 256))
            elif name == "3cad":
                self._split = ad_data.load_mvtec_style(self.cfg["root"], self.category, self.cfg.get("size", 448),
                                                       self.cfg.get("mask_size", 256), lazy=True)
            elif name == "visa":
                self._split = ad_data.load_visa(self.cfg["root"], self.category, self.cfg.get("size", 448),
                                                self.cfg.get("mask_size", 256))
            elif name == "synthetic":
                kw = {k: v for k, v in self.cfg.items() if k not in ("name", "categories", "root")}
                self._split = ad_data.load_synthetic_ad(seed=zlib.crc32(self.category.encode()) % 1000, **kw)
            else:
                raise KeyError(f"Unknown anomaly dataset '{name}'")
            if self.meta is None:
                s = self._split
                self.meta_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self.meta_path, labels=s.test_labels, masks=s.test_masks,
                                    types=np.array(s.test_types), n_train=len(s.train_images))
                self.meta = _Meta(s.test_labels, s.test_masks, list(s.test_types), len(s.train_images))
        return self._split

    def get_meta(self) -> _Meta:
        if self.meta is None:
            _ = self.split
        return self.meta


# --------------------------------------------------------------------------- runner

def run(cfg: dict, progress: bool = True) -> pd.DataFrame:
    data_cfg = dict(cfg["data"])
    cats = data_cfg.get("categories", "all")
    cats = ad_data.categories(data_cfg.get("name", "mvtec")) if cats == "all" else list(cats)
    cache_dir, out_dir = Path(cfg["cache_dir"]), Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)
    jsonl = out_dir / f"{cfg['name']}.jsonl"
    prev = load_results(jsonl)
    done = set(prev["run_id"]) if "run_id" in prev else set()

    methods = cfg["methods"]
    ks = methods.get("codebook", {}).get("k", [])
    cb_methods = [m for m in CODEBOOK_METHODS + ("medoid_knn",)
                  if m in methods.get("codebook", {}).get("scores", CODEBOOK_METHODS)]
    gpu_ks = sorted(methods.get("codebook_gpu", {}).get("k", []))
    xm = cfg.get("extra_metrics", {})
    xm_methods, xm_ks = set(xm.get("methods", [])), set(xm.get("k", []))
    xm_sigmas, xm_fprs = [float(v) for v in xm.get("sigmas", [])], [float(v) for v in xm.get("aupro_fpr", [])]
    vocab_cfg = dict(cfg.get("vocab", {}))
    seeds = vocab_cfg.pop("seeds", None) or [vocab_cfg.get("seed", 0)]
    vocab_cfg.pop("seed", None)
    sub_cfg = methods.get("subsample", {})
    sub_sizes = sorted(sub_cfg.get("sizes", ks)) if sub_cfg else []
    sub_methods = ([m for m in SUBSAMPLE_METHODS if m in sub_cfg.get("methods", ("coreset_knn", "random_knn"))]
                   if sub_cfg else [])
    hyb_cfg = methods.get("hybrid", {})
    hyb_sizes = sorted(hyb_cfg.get("sizes", [])) if hyb_cfg else []
    hyb_fracs = [float(f) for f in hyb_cfg.get("word_fractions", [0.5])] if hyb_cfg else []
    normalize = cfg.get("normalize", True)
    topks = [int(k) for k in cfg.get("image_topk", [])]
    region = bool(cfg.get("region_metrics", False))
    save_maps_for = set(cfg.get("save_maps_for", []))
    cache_vocab = bool(cfg.get("cache_vocab", False))
    low_mem = bool(cfg.get("low_memory", False))
    if cfg.get("require_gpu") and _torch_cuda() is None:
        raise RuntimeError("No GPU found. This run needs a GPU runtime (Colab: Runtime > Change runtime type > "
                           "T4 GPU). On CPU, DINOv2 extraction alone takes hours per category.")
    map_dtype = np.float16 if low_mem else np.float32
    (out_dir / "maps").mkdir(exist_ok=True)
    sigma = cfg.get("smoothing_sigma", 4.0)
    mask_size = data_cfg.get("mask_size", 256)
    env = _env()

    for cat in cats:
        tag = f"{data_cfg.get('name', 'mvtec')}_{cat}_s{data_cfg.get('size', 448)}_m{mask_size}"
        ds = _LazyAD(data_cfg, cat, cache_dir, tag)
        for ex in cfg["extractors"]:
            ex_name, ex_params = ex["name"], ex.get("params", {})
            cache_params = {k: v for k, v in ex_params.items() if k not in RUNTIME_KEYS}
            baselines = [m for m in ("patch_knn", "global_knn") if m in methods
                         and not (m == "global_knn" and ex_name in HANDCRAFTED)]
            todo_ids = ([_run_id(cfg["name"], cat, ex_name, m, 0, 0) for m in baselines]
                        + [_run_id(cfg["name"], cat, ex_name, m, k, s) for m in cb_methods for k in ks for s in seeds]
                        + [_run_id(cfg["name"], cat, ex_name, m, k, s)
                           for m in sub_methods for k in sub_sizes for s in seeds]
                        + [_run_id(cfg["name"], cat, ex_name, hybrid_name(f), k, s)
                           for f in hyb_fracs for k in hyb_sizes for s in seeds]
                        + [_run_id(cfg["name"], cat, ex_name, "codebook_gpu", k, s) for k in gpu_ks for s in seeds])
            if all(r in done for r in todo_ids):
                continue

            extractor = None

            def extract(images):
                nonlocal extractor
                extractor = extractor or get_extractor(ex_name, **ex_params)
                return extractor(images, progress=progress)

            tr, tr_info = cache.get_or_compute(cache_dir, f"{tag}_train", ex_name, cache_params,
                                              lambda: extract(ds.split.train_images))
            if low_mem and ds._split is not None:
                ds._split.train_images = []   # free decoded training images before the test pass
            te, te_info = cache.get_or_compute(cache_dir, f"{tag}_test", ex_name, cache_params,
                                              lambda: extract(ds.split.test_images))
            del extractor
            meta = ds.get_meta()
            if low_mem:
                ds._split = None              # labels and masks live on in meta
                gc.collect()
            side = grid_side(te)
            base = {"config": cfg["name"], "category": cat, "extractor": ex_name, "n_train": tr.n_images,
                    "n_test": te.n_images, "n_anomalous": int(meta.labels.sum()), "patches_per_image": side * side,
                    "extract_seconds": (tr_info.get("extract_seconds") or 0) + (te_info.get("extract_seconds") or 0),
                    **env}

            def record(rid, method, k, seed, image_scores, maps=None, extra=None, timing=None, patch_scores=None):
                row = {"run_id": rid, **base, "method": method, "k": k, "vocab_seed": seed, **(timing or {}),
                       **(extra or {}),
                       **metrics(meta.labels, image_scores, meta.types, meta.masks, maps, region_metrics=region)}
                if maps is not None and method in xm_methods and k in xm_ks:
                    for f in xm_fprs:
                        row[f"aupro_f{int(round(f * 100))}"] = (
                            binned_pixel_metrics(meta.masks, maps, True, fpr_limit=f)["aupro"]
                            if maps.size > LARGE_PIXELS else aupro(meta.masks, maps, fpr_limit=f))
                    for sg in xm_sigmas:
                        m2, _ = patch_maps(patch_scores, te.n_images, side, mask_size, sg, map_dtype)
                        r2 = (binned_pixel_metrics(meta.masks, m2, True) if m2.size > LARGE_PIXELS else
                              {"pixel_auroc": _auroc(meta.masks.reshape(-1), m2.reshape(-1)),
                               "aupro": aupro(meta.masks, m2)})
                        tag_s = f"s{sg:g}"
                        row[f"pixel_auroc_{tag_s}"], row[f"aupro_{tag_s}"] = r2["pixel_auroc"], r2["aupro"]
                        del m2
                if patch_scores is not None:
                    for tk in topks:
                        row[f"image_auroc_top{tk}"] = _auroc(meta.labels, topk_image_scores(
                            patch_scores, te.n_images, tk))
                if maps is not None and cat in save_maps_for:
                    np.save(out_dir / "maps" / (rid.replace("|", "__") + ".npy"), maps.astype(np.float16))
                _append(jsonl, row)
                np.save(out_dir / "scores" / (rid.replace("|", "__") + ".npy"), image_scores.astype(np.float32))
                if progress:
                    pa = row.get("pixel_auroc", float("nan"))
                    print(f"[{cat:>10}|{ex_name:>10}] {method:<15} K={k:<5} s={seed}  "
                          f"image={row['image_auroc']:.3f}  pixel={pa:.3f}")

            if low_mem:  # normalise the loaded float16 arrays in place: no float32 copies of the bank
                tr_desc = _norm_inplace(np.asarray(tr.desc), normalize)
                te_desc = _norm_inplace(np.asarray(te.desc), normalize)
            else:
                tr_desc = _norm(tr.desc, normalize)
                te_desc = _norm(te.desc, normalize)

            if "patch_knn" in methods:
                rid = _run_id(cfg["name"], cat, ex_name, "patch_knn", 0, 0)
                if rid not in done:
                    t: dict = {}
                    with budget(t, "score"):
                        ps = min_dist(te_desc, tr_desc)
                        maps, img = patch_maps(ps, te.n_images, side, mask_size, sigma, map_dtype)
                    record(rid, "patch_knn", 0, 0, img, maps, {"bank_size": len(tr_desc)}, t, ps)

            if "global_knn" in methods and tr.global_ is not None:
                rid = _run_id(cfg["name"], cat, ex_name, "global_knn", 0, 0)
                if rid not in done:
                    t = {}
                    with budget(t, "score"):
                        img = min_dist(_norm(te.global_, normalize), _norm(tr.global_, normalize))
                    record(rid, "global_knn", 0, 0, img, None, None, t)

            for seed in seeds:
                ids = {(m, k): _run_id(cfg["name"], cat, ex_name, m, k, seed) for m in sub_methods for k in sub_sizes}
                if all(i in done for i in ids.values()):
                    continue
                picks = {}
                if "coreset_knn" in sub_methods:
                    ct: dict = {}
                    with budget(ct, "vocab"):  # selection cost, reported like vocabulary building
                        order = greedy_coreset(tr_desc, max(sub_sizes), seed=seed)
                    picks["coreset_knn"] = (order, ct)
                if "random_knn" in sub_methods:
                    order = np.random.default_rng(seed).permutation(len(tr_desc))[:max(sub_sizes)]
                    picks["random_knn"] = (order, {})
                for tm in ("coreset_trim1", "coreset_trim5"):
                    if tm in sub_methods:
                        ct = {}
                        with budget(ct, "vocab"):
                            order = trimmed_coreset(tr_desc, max(sub_sizes), TRIM_FRACTION[tm], seed=seed)
                        picks[tm] = (order, ct)
                for m, (order, sel_t) in picks.items():
                    for k in sub_sizes:  # nested: the first k picks are the size-k subset
                        rid = ids[(m, k)]
                        if rid in done:
                            continue
                        t = dict(sel_t)
                        with budget(t, "score"):
                            ps = min_dist(te_desc, tr_desc[np.sort(order[:k])])
                            maps, img = patch_maps(ps, te.n_images, side, mask_size, sigma, map_dtype)
                        record(rid, m, k, seed, img, maps, {"bank_size": int(min(k, len(tr_desc)))}, t, ps)

            tr_n = LocalFeatures(desc=tr_desc, offsets=tr.offsets)
            te_n = LocalFeatures(desc=te_desc, offsets=te.offsets)
            for k in ks:
                for seed in seeds:
                    ids = {m: _run_id(cfg["name"], cat, ex_name, m, k, seed) for m in cb_methods}
                    if all(i in done for i in ids.values()):
                        continue
                    vkey = {"tag": tag, "extractor": ex_name, "params": cache_params, "normalize": normalize,
                            "k": k, "seed": seed, "vocab": vocab_cfg}
                    vocab, vt = cached_vocabulary(cache_dir if cache_vocab else None, vkey,
                                                  lambda: build_vocabulary(tr_n.desc, k, seed=seed, **vocab_cfg))
                    extra = {"empty_words": vocab.info["empty_words"]}
                    if "codebook_dist" in ids or "codebook_norm" in ids:
                        t = dict(vt)
                        with budget(t, "score"):
                            d_te, w_te = nearest_word(te_n.desc, vocab)
                        if ids.get("codebook_dist") and ids["codebook_dist"] not in done:
                            maps, img = patch_maps(d_te, te.n_images, side, mask_size, sigma, map_dtype)
                            record(ids["codebook_dist"], "codebook_dist", k, seed, img, maps, extra, t, d_te)
                        if ids.get("codebook_norm") and ids["codebook_norm"] not in done:
                            t2 = dict(vt)
                            with budget(t2, "score"):
                                radius = word_radius(*nearest_word(tr_n.desc, vocab), k)
                                ns = d_te / radius[w_te]
                                maps, img = patch_maps(ns, te.n_images, side, mask_size, sigma, map_dtype)
                            record(ids["codebook_norm"], "codebook_norm", k, seed, img, maps, extra, t2, ns)
                    if ids.get("medoid_knn") and ids["medoid_knn"] not in done:
                        t = dict(vt)
                        with budget(t, "select"):
                            _, mi = nn_search(vocab.centers, tr_n.desc)  # real patch nearest each word
                            med = np.unique(mi)
                        with budget(t, "score"):
                            ps = min_dist(te_n.desc, tr_n.desc[med])
                            maps, img = patch_maps(ps, te.n_images, side, mask_size, sigma, map_dtype)
                        record(ids["medoid_knn"], "medoid_knn", k, seed, img, maps,
                               {**extra, "bank_size": int(len(med))}, t, ps)
                    if ids.get("hist_knn") and ids["hist_knn"] not in done:
                        t = dict(vt)
                        with budget(t, "score"):
                            img = min_dist(hard(te_n, vocab), hard(tr_n, vocab))
                        record(ids["hist_knn"], "hist_knn", k, seed, img, None, extra, t)
                    del vocab

            # codebook_gpu: Lloyd k-means on the GPU (for large K); same scoring as codebook_dist
            for k in gpu_ks:
                for seed in seeds:
                    rid = _run_id(cfg["name"], cat, ex_name, "codebook_gpu", k, seed)
                    if rid in done:
                        continue
                    vkey = {"tag": tag, "extractor": ex_name, "params": cache_params, "normalize": normalize,
                            "k": k, "seed": seed, "vocab": vocab_cfg, "backend": "torch"}
                    vocab, vt = cached_vocabulary(cache_dir if cache_vocab else None, vkey,
                                                  lambda: build_vocabulary_torch(tr_desc, k, seed=seed, **vocab_cfg))
                    t = dict(vt)
                    with budget(t, "score"):
                        d_te, _ = nearest_word(te_desc, vocab)
                        maps, img = patch_maps(d_te, te.n_images, side, mask_size, sigma, map_dtype)
                    record(rid, "codebook_gpu", k, seed, img, maps,
                           {"empty_words": vocab.info["empty_words"], "bank_size": k}, t, d_te)
                    del vocab

            # hybrid: K = f*M visual words + (1-f)*M real patches the words cover worst, one memory
            for f in hyb_fracs:
                for M in hyb_sizes:
                    for seed in seeds:
                        rid = _run_id(cfg["name"], cat, ex_name, hybrid_name(f), M, seed)
                        if rid in done:
                            continue
                        kw = max(1, min(M, int(round(M * f))))
                        kp = M - kw
                        vkey = {"tag": tag, "extractor": ex_name, "params": cache_params, "normalize": normalize,
                                "k": kw, "seed": seed, "vocab": vocab_cfg}
                        vocab, vt = cached_vocabulary(cache_dir if cache_vocab else None, vkey,
                                                      lambda: build_vocabulary(tr_desc, kw, seed=seed, **vocab_cfg))
                        if vocab.pca is not None:
                            raise ValueError("hybrid needs words in the patch space (no PCA)")
                        t = dict(vt)
                        with budget(t, "select"):
                            picks = greedy_coreset(tr_desc, kp, seed=seed, init_centers=vocab.centers)
                        bank = np.concatenate([vocab.centers.astype(np.float32),
                                               np.asarray(tr_desc[np.sort(picks)], np.float32)])
                        with budget(t, "score"):
                            ps = min_dist(te_desc, bank)
                            maps, img = patch_maps(ps, te.n_images, side, mask_size, sigma, map_dtype)
                        record(rid, hybrid_name(f), M, seed, img, maps,
                               {"n_words": kw, "n_patches": kp, "bank_size": int(len(bank))}, t, ps)
                        del vocab, bank
            del tr, te, tr_desc, te_desc, tr_n, te_n
            gc.collect()

    df = load_results(jsonl)
    df.to_csv(out_dir / f"{cfg['name']}.csv", index=False)
    return df


def summary(df: pd.DataFrame, metric: str = "image_auroc") -> pd.DataFrame:
    """Method (x K) rows, category columns + mean; vocabulary seeds averaged."""
    d = df.copy()
    d["setting"] = d.apply(lambda r: r["method"] if r["method"] in ("patch_knn", "global_knn")
                           else f"{r['method']} K={int(r['k'])}", axis=1)
    t = d.pivot_table(index=["extractor", "setting"], columns="category", values=metric, aggfunc="mean")
    t.insert(0, "mean", t.mean(1))
    return t.sort_values(["extractor", "mean"], ascending=[True, False])


def followup_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over categories and seeds of every image/pixel metric, plus hybrid rows.

    Hybrid K: image score from the coreset of M = K patches, map from the K-word
    codebook (same seed). Both parts are read off existing rows, not re-run.
    """
    cols = [c for c in ["image_auroc", "image_auroc_top3", "image_auroc_top10", "image_auroc_top30",
                        "pixel_auroc", "aupro", "pixel_dprime"] if c in df]
    d = df.copy()
    d["k"] = d["k"].fillna(0).astype(int)
    per_seed = d.groupby(["method", "k", "vocab_seed", "category"])[cols].mean().reset_index()
    cs = per_seed[per_seed.method == "coreset_knn"].set_index(["k", "vocab_seed", "category"])
    cb = per_seed[per_seed.method == "codebook_dist"].set_index(["k", "vocab_seed", "category"])
    if len(cs) and len(cb):
        img_cols = [c for c in cols if c.startswith("image")]
        pix_cols = [c for c in cols if not c.startswith("image")]
        hy = pd.concat([cs[img_cols], cb[pix_cols]], axis=1, join="inner").reset_index()
        hy["method"] = "hybrid"
        per_seed = pd.concat([per_seed, hy], ignore_index=True)
    m = per_seed.groupby(["method", "k", "vocab_seed"])[cols].mean()
    out = m.groupby(level=[0, 1]).mean()
    sd = m.groupby(level=[0, 1]).std()
    out.columns = [c.replace("image_auroc", "img").replace("pixel_auroc", "pix") for c in out.columns]
    sd.columns = [c + "_sd" for c in out.columns]
    return pd.concat([out, sd[["img_sd", "pix_sd"] if "pix_sd" in sd else ["img_sd"]]], axis=1)


def normal_pixel_spread(results_dir, cfg_name: str, category: str, masks: np.ndarray,
                        methods=(("codebook_dist", 1024), ("coreset_knn", 1024), ("patch_knn", 0)),
                        extractor: str = "dinov2_s", seed: int = 0) -> pd.DataFrame:
    """Normal-pixel score spread per method on saved maps, in units of that method's own scale.

    `rel_iqr` = IQR / median of normal-pixel scores; `p99_over_median` shows how
    far the noisiest normal pixels reach. Lower = quieter background.
    """
    rows = []
    m = np.asarray(masks).astype(bool)
    for meth, k in methods:
        s_ = 0 if meth == "patch_knn" else seed
        rid = _run_id(cfg_name, category, extractor, meth, k, s_).replace("|", "__")
        path = Path(results_dir) / "maps" / f"{rid}.npy"
        if not path.exists():
            continue
        mp = np.load(path).astype(np.float32)
        neg, pos = mp[~m], mp[m]
        q1, med, q3, p99 = np.quantile(neg, [0.25, 0.5, 0.75, 0.99])
        rows.append({"method": meth, "k": k, "rel_iqr": (q3 - q1) / med, "p99_over_median": p99 / med,
                     "defect_median_over_normal_median": float(np.median(pos) / med) if len(pos) else np.nan})
    return pd.DataFrame(rows)


def prereg_checks(df: pd.DataFrame, tol_image: float = 0.005, tol_full: float = 0.01) -> pd.DataFrame:
    """The four preregistered VisA predictions (docs/preregistration_visa.md), pass/fail."""
    t = followup_table(df)

    def v(method, k, col):
        return float(t.loc[(method, k), col])

    rows = []
    aup = {k: (v("codebook_dist", k, "aupro"), v("coreset_knn", k, "aupro")) for k in (64, 256, 1024)}
    rows.append(("P1", "codebook AUPRO > coreset at M = 64, 256, 1024",
                 "; ".join(f"M={k}: {a:.3f} vs {b:.3f}" for k, (a, b) in aup.items()),
                 all(a > b for a, b in aup.values())))
    a, b = v("codebook_dist", 1024, "img_top10"), v("coreset_knn", 1024, "img_top10")
    rows.append(("P2", f"codebook top-10 image AUROC >= coreset - {tol_image} (M = 1024)",
                 f"{a:.3f} vs {b:.3f}", a >= b - tol_image))
    a, b = v("codebook_dist", 1024, "pix"), v("coreset_knn", 1024, "pix")
    rows.append(("P3", "codebook pixel AUROC >= coreset (M = 1024)", f"{a:.3f} vs {b:.3f}", a >= b))
    gaps = {c: v("patch_knn", 0, c) - v("codebook_dist", 1024, c) for c in ("img_top10", "pix", "aupro")}
    rows.append(("P4", f"codebook within {tol_full} of full bank (K = 1024)",
                 "; ".join(f"{c} gap {g:+.3f}" for c, g in gaps.items()), all(g <= tol_full for g in gaps.values())))
    return pd.DataFrame(rows, columns=["id", "prediction", "observed", "pass"])


def hybrid_compare(base: pd.DataFrame, hyb: pd.DataFrame) -> pd.DataFrame:
    """Codebook, coreset and hybrid side by side at each memory size (means over categories, then seeds)."""
    keep = ["codebook_dist", "coreset_knn", "patch_knn"]
    d = pd.concat([base[base.method.isin(keep)], hyb], ignore_index=True)
    cols = [c for c in ["image_auroc", "image_auroc_top10", "pixel_auroc", "aupro"] if c in d]
    d["k"] = d["k"].fillna(0).astype(int)
    per_seed = d.groupby(["method", "k", "vocab_seed"])[cols].mean()
    t = per_seed.groupby(level=[0, 1]).mean()
    t.columns = [c.replace("image_auroc", "img").replace("pixel_auroc", "pix") for c in t.columns]
    return t.sort_index(level=[1, 0])


def prereg_checks_hybrid(df: pd.DataFrame, method: str = "hybrid_w75", tol_aupro: float = 0.02) -> pd.DataFrame:
    """The four preregistered hybrid predictions (docs/preregistration_hybrid_3cad.md), pass/fail.

    Detection uses the maximum patch score (image_auroc), localization uses AUPRO.
    """
    d = df.copy()
    d["k"] = d["k"].fillna(0).astype(int)
    m = d.groupby(["method", "k", "vocab_seed"])[["image_auroc", "aupro"]].mean().groupby(level=[0, 1]).mean()

    def v(meth, k, col):
        return float(m.loc[(meth, k), col])

    sizes = (256, 1024)
    rows = []
    obs = {k: (v(method, k, "image_auroc"), v("coreset_knn", k, "image_auroc")) for k in sizes}
    rows.append(("H1", "hybrid max-score image AUROC >= coreset at M = 256, 1024",
                 "; ".join(f"M={k}: {a:.3f} vs {b:.3f}" for k, (a, b) in obs.items()),
                 all(a >= b for a, b in obs.values())))
    a, b = v(method, 1024, "image_auroc"), v("codebook_dist", 1024, "image_auroc")
    rows.append(("H2", "hybrid max-score image AUROC > codebook at M = 1024", f"{a:.3f} vs {b:.3f}", a > b))
    obs = {k: (v(method, k, "aupro"), v("codebook_dist", k, "aupro")) for k in sizes}
    rows.append(("H3", f"hybrid AUPRO >= codebook - {tol_aupro} at M = 256, 1024",
                 "; ".join(f"M={k}: {a:.3f} vs {b:.3f}" for k, (a, b) in obs.items()),
                 all(a >= b - tol_aupro for a, b in obs.values())))
    obs = {k: (v(method, k, "aupro"), v("coreset_knn", k, "aupro")) for k in sizes}
    rows.append(("H4", "hybrid AUPRO > coreset at M = 256, 1024",
                 "; ".join(f"M={k}: {a:.3f} vs {b:.3f}" for k, (a, b) in obs.items()),
                 all(a > b for a, b in obs.values())))
    return pd.DataFrame(rows, columns=["id", "prediction", "observed", "pass"])
