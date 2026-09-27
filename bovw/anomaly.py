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
from .vocab import build_vocabulary, l2n

PATCH_METHODS = ("patch_knn", "codebook_dist", "codebook_norm")
IMAGE_METHODS = ("global_knn", "hist_knn")
CODEBOOK_METHODS = ("codebook_dist", "codebook_norm", "hist_knn")
SUBSAMPLE_METHODS = ("coreset_knn", "random_knn")
HANDCRAFTED = {"sift", "dense_sift", "orb"}  # no global embedding


# --------------------------------------------------------------------------- nearest neighbours

def _torch_cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def min_dist(queries: np.ndarray, bank: np.ndarray, q_chunk: int = 4096, b_chunk: int = 65536) -> np.ndarray:
    """Euclidean distance from each query row to its nearest bank row (exact).

    Uses the GPU when available (Colab T4: ~10 s for 160k x 400k x 384); falls
    back to chunked NumPy. Both loops are chunked so memory stays bounded.
    """
    torch = _torch_cuda()
    out = np.empty(len(queries), np.float32)
    if torch is not None:
        b_all = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32)).cuda()
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
    bank = np.asarray(bank, np.float32)
    b_sq = (bank * bank).sum(1)
    q_chunk = min(q_chunk, 1024)
    for i in range(0, len(queries), q_chunk):
        q = np.asarray(queries[i:i + q_chunk], np.float32)
        q_sq = (q * q).sum(1, keepdims=True)
        best = np.full(len(q), np.inf, np.float32)
        for j in range(0, len(bank), b_chunk // 4):
            d = q_sq - 2.0 * q @ bank[j:j + b_chunk // 4].T + b_sq[None, j:j + b_chunk // 4]
            np.minimum(best, d.min(1), out=best)
        out[i:i + q_chunk] = np.sqrt(np.maximum(best, 0))
    return out


def greedy_coreset(x: np.ndarray, m: int, seed: int = 0, proj_dim: int = 128, chunk: int = 262144) -> np.ndarray:
    """Indices of a greedy k-center coreset (PatchCore, Roth et al. 2022).

    Features are first mapped by a Gaussian random projection to `proj_dim`
    dimensions (as in PatchCore) to speed up the distance updates. Greedy
    selection is nested: the first m' < m indices are the size-m' coreset.
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
    torch = _torch_cuda()
    if torch is not None:
        z = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()
        if proj is not None:
            z = z @ torch.from_numpy(proj).cuda()
        mind = torch.full((n,), float("inf"), device="cuda")
        sel = [first]
        for _ in range(m - 1):
            c = z[sel[-1]]
            mind = torch.minimum(mind, ((z - c) ** 2).sum(1))
            sel.append(int(mind.argmax()))
        del z, mind
        torch.cuda.empty_cache()
        return np.array(sel, np.int64)
    z = np.empty((n, proj_dim if proj is not None else d), np.float32)
    for i in range(0, n, chunk):
        blk = np.asarray(x[i:i + chunk], np.float32)
        z[i:i + chunk] = blk @ proj if proj is not None else blk
    mind = np.full(n, np.inf, np.float32)
    sel = [first]
    for _ in range(m - 1):
        np.minimum(mind, ((z - z[sel[-1]]) ** 2).sum(1), out=mind)
        sel.append(int(mind.argmax()))
    return np.array(sel, np.int64)


# --------------------------------------------------------------------------- scoring helpers

def grid_side(feats: LocalFeatures) -> int:
    counts = feats.counts()
    if counts.min() != counts.max():
        raise ValueError("Pixel maps need the same number of patches per image (a dense grid).")
    side = math.isqrt(int(counts[0]))
    if side * side != counts[0]:
        raise ValueError(f"{counts[0]} patches per image is not a square grid.")
    return side


def patch_maps(patch_scores: np.ndarray, n_images: int, side: int, mask_size: int, sigma: float = 4.0):
    """(n_images*side*side,) -> smoothed (n_images, mask_size, mask_size) maps and max-patch image scores."""
    grid = patch_scores.reshape(n_images, side, side).astype(np.float32)
    maps = np.empty((n_images, mask_size, mask_size), np.float32)
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


def metrics(labels, image_scores, types, masks=None, maps=None, region_metrics: bool = False) -> dict:
    out = {"image_auroc": _auroc(labels, image_scores)}
    good = np.array([t == "good" for t in types])
    by_type = {}
    for t in sorted(set(types) - {"good"}):
        sel = good | np.array([x == t for x in types])
        by_type[t] = _auroc(np.asarray(labels)[sel], np.asarray(image_scores)[sel])
    out["image_auroc_by_type"] = json.dumps(by_type)
    if maps is not None:
        out["pixel_auroc"] = _auroc(masks.reshape(-1), maps.reshape(-1))
        if region_metrics:
            out["aupro"] = aupro(masks, maps)
            out["pixel_dprime"] = pixel_dprime(masks, maps)
    return out


def _norm(x: np.ndarray, normalize: bool) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return l2n(x) if normalize else x


def nearest_word(desc: np.ndarray, vocab, chunk: int = 65536):
    """Transformed-space distance to, and index of, the nearest visual word."""
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
    cats = ad_data.MVTEC_CATEGORIES if cats == "all" else list(cats)
    cache_dir, out_dir = Path(cfg["cache_dir"]), Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)
    jsonl = out_dir / f"{cfg['name']}.jsonl"
    prev = load_results(jsonl)
    done = set(prev["run_id"]) if "run_id" in prev else set()

    methods = cfg["methods"]
    ks = methods.get("codebook", {}).get("k", [])
    cb_methods = [m for m in CODEBOOK_METHODS if m in methods.get("codebook", {}).get("scores", CODEBOOK_METHODS)]
    vocab_cfg = dict(cfg.get("vocab", {}))
    seeds = vocab_cfg.pop("seeds", None) or [vocab_cfg.get("seed", 0)]
    vocab_cfg.pop("seed", None)
    sub_cfg = methods.get("subsample", {})
    sub_sizes = sorted(sub_cfg.get("sizes", ks)) if sub_cfg else []
    sub_methods = [m for m in SUBSAMPLE_METHODS if m in sub_cfg.get("methods", SUBSAMPLE_METHODS)] if sub_cfg else []
    normalize = cfg.get("normalize", True)
    topks = [int(k) for k in cfg.get("image_topk", [])]
    region = bool(cfg.get("region_metrics", False))
    save_maps_for = set(cfg.get("save_maps_for", []))
    cache_vocab = bool(cfg.get("cache_vocab", False))
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
                           for m in sub_methods for k in sub_sizes for s in seeds])
            if all(r in done for r in todo_ids):
                continue

            extractor = None

            def extract(images):
                nonlocal extractor
                extractor = extractor or get_extractor(ex_name, **ex_params)
                return extractor(images, progress=progress)

            tr, tr_info = cache.get_or_compute(cache_dir, f"{tag}_train", ex_name, cache_params,
                                              lambda: extract(ds.split.train_images))
            te, te_info = cache.get_or_compute(cache_dir, f"{tag}_test", ex_name, cache_params,
                                              lambda: extract(ds.split.test_images))
            del extractor
            meta = ds.get_meta()
            side = grid_side(te)
            base = {"config": cfg["name"], "category": cat, "extractor": ex_name, "n_train": tr.n_images,
                    "n_test": te.n_images, "n_anomalous": int(meta.labels.sum()), "patches_per_image": side * side,
                    "extract_seconds": (tr_info.get("extract_seconds") or 0) + (te_info.get("extract_seconds") or 0),
                    **env}

            def record(rid, method, k, seed, image_scores, maps=None, extra=None, timing=None, patch_scores=None):
                row = {"run_id": rid, **base, "method": method, "k": k, "vocab_seed": seed, **(timing or {}),
                       **(extra or {}),
                       **metrics(meta.labels, image_scores, meta.types, meta.masks, maps, region_metrics=region)}
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

            tr_desc = _norm(tr.desc, normalize)

            if "patch_knn" in methods:
                rid = _run_id(cfg["name"], cat, ex_name, "patch_knn", 0, 0)
                if rid not in done:
                    t: dict = {}
                    with budget(t, "score"):
                        ps = min_dist(_norm(te.desc, normalize), tr_desc)
                        maps, img = patch_maps(ps, te.n_images, side, mask_size, sigma)
                    record(rid, "patch_knn", 0, 0, img, maps, {"bank_size": len(tr_desc)}, t, ps)

            if "global_knn" in methods and tr.global_ is not None:
                rid = _run_id(cfg["name"], cat, ex_name, "global_knn", 0, 0)
                if rid not in done:
                    t = {}
                    with budget(t, "score"):
                        img = min_dist(_norm(te.global_, normalize), _norm(tr.global_, normalize))
                    record(rid, "global_knn", 0, 0, img, None, None, t)

            te_desc = _norm(te.desc, normalize)
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
                for m, (order, sel_t) in picks.items():
                    for k in sub_sizes:  # nested: the first k picks are the size-k subset
                        rid = ids[(m, k)]
                        if rid in done:
                            continue
                        t = dict(sel_t)
                        with budget(t, "score"):
                            ps = min_dist(te_desc, tr_desc[np.sort(order[:k])])
                            maps, img = patch_maps(ps, te.n_images, side, mask_size, sigma)
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
                            maps, img = patch_maps(d_te, te.n_images, side, mask_size, sigma)
                            record(ids["codebook_dist"], "codebook_dist", k, seed, img, maps, extra, t, d_te)
                        if ids.get("codebook_norm") and ids["codebook_norm"] not in done:
                            t2 = dict(vt)
                            with budget(t2, "score"):
                                radius = word_radius(*nearest_word(tr_n.desc, vocab), k)
                                ns = d_te / radius[w_te]
                                maps, img = patch_maps(ns, te.n_images, side, mask_size, sigma)
                            record(ids["codebook_norm"], "codebook_norm", k, seed, img, maps, extra, t2, ns)
                    if ids.get("hist_knn") and ids["hist_knn"] not in done:
                        t = dict(vt)
                        with budget(t, "score"):
                            img = min_dist(hard(te_n, vocab), hard(tr_n, vocab))
                        record(ids["hist_knn"], "hist_knn", k, seed, img, None, extra, t)
                    del vocab
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
