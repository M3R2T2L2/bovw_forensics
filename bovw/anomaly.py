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


def metrics(labels, image_scores, types, masks=None, maps=None) -> dict:
    out = {"image_auroc": _auroc(labels, image_scores)}
    good = np.array([t == "good" for t in types])
    by_type = {}
    for t in sorted(set(types) - {"good"}):
        sel = good | np.array([x == t for x in types])
        by_type[t] = _auroc(np.asarray(labels)[sel], np.asarray(image_scores)[sel])
    out["image_auroc_by_type"] = json.dumps(by_type)
    if maps is not None:
        out["pixel_auroc"] = _auroc(masks.reshape(-1), maps.reshape(-1))
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
    normalize = cfg.get("normalize", True)
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
                        + [_run_id(cfg["name"], cat, ex_name, m, k, s) for m in cb_methods for k in ks for s in seeds])
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

            def record(rid, method, k, seed, image_scores, maps=None, extra=None, timing=None):
                row = {"run_id": rid, **base, "method": method, "k": k, "vocab_seed": seed, **(timing or {}),
                       **(extra or {}), **metrics(meta.labels, image_scores, meta.types, meta.masks, maps)}
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
                    record(rid, "patch_knn", 0, 0, img, maps, {"bank_size": len(tr_desc)}, t)

            if "global_knn" in methods and tr.global_ is not None:
                rid = _run_id(cfg["name"], cat, ex_name, "global_knn", 0, 0)
                if rid not in done:
                    t = {}
                    with budget(t, "score"):
                        img = min_dist(_norm(te.global_, normalize), _norm(tr.global_, normalize))
                    record(rid, "global_knn", 0, 0, img, None, None, t)

            tr_n = LocalFeatures(desc=tr_desc, offsets=tr.offsets)
            te_n = LocalFeatures(desc=_norm(te.desc, normalize), offsets=te.offsets)
            for k in ks:
                for seed in seeds:
                    ids = {m: _run_id(cfg["name"], cat, ex_name, m, k, seed) for m in cb_methods}
                    if all(i in done for i in ids.values()):
                        continue
                    vt: dict = {}
                    with budget(vt, "vocab"):
                        vocab = build_vocabulary(tr_n.desc, k, seed=seed, **vocab_cfg)
                    extra = {"empty_words": vocab.info["empty_words"]}
                    if "codebook_dist" in ids or "codebook_norm" in ids:
                        t = dict(vt)
                        with budget(t, "score"):
                            d_te, w_te = nearest_word(te_n.desc, vocab)
                        if ids.get("codebook_dist") and ids["codebook_dist"] not in done:
                            maps, img = patch_maps(d_te, te.n_images, side, mask_size, sigma)
                            record(ids["codebook_dist"], "codebook_dist", k, seed, img, maps, extra, t)
                        if ids.get("codebook_norm") and ids["codebook_norm"] not in done:
                            t2 = dict(vt)
                            with budget(t2, "score"):
                                radius = word_radius(*nearest_word(tr_n.desc, vocab), k)
                                maps, img = patch_maps(d_te / radius[w_te], te.n_images, side, mask_size, sigma)
                            record(ids["codebook_norm"], "codebook_norm", k, seed, img, maps, extra, t2)
                    if ids.get("hist_knn") and ids["hist_knn"] not in done:
                        t = dict(vt)
                        with budget(t, "score"):
                            img = min_dist(hard(te_n, vocab), hard(tr_n, vocab))
                        record(ids["hist_knn"], "hist_knn", k, seed, img, None, extra, t)
                    del vocab
            del tr, te, tr_desc, tr_n, te_n
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
