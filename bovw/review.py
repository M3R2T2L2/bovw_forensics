"""Analyses added in response to the referee report.

image_level_paired  hierarchical bootstrap of image-AUROC differences: resample categories,
                    then test images within each category (good and defective separately),
                    using the per-image scores every run saves to results/scores.
benchmark           same-hardware timings of vocabulary/coreset construction and scoring.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .sweep import _run_id


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via the rank-sum (Mann-Whitney) formula; ties get average ranks."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _scores(results_dir: Path, cfg_name: str, cat: str, extractor: str, method: str, k: int, seed: int):
    rid = _run_id(cfg_name, cat, extractor, method, k, seed).replace("|", "__")
    p = Path(results_dir) / "scores" / f"{rid}.npy"
    return np.load(p) if p.exists() else None


def image_level_paired(results_dir, cache_dir, cfg_name: str, dataset: str, categories, a: tuple, b: tuple,
                       seeds=(0, 1, 2), extractor: str = "dinov2_s", size: int = 448, mask_size: int = 256,
                       n_boot: int = 2000, level: float = 0.95, rng_seed: int = 0, cfg_name_b: str | None = None) -> dict:
    """Mean over categories of [AUROC(a) - AUROC(b)] (seed-averaged), with a two-level bootstrap CI.

    a, b = (method, k). Deterministic methods (patch_knn, k = 0) use seed 0.
    """
    rng = np.random.default_rng(rng_seed)
    cats_used, obs, draws = [], [], []
    for cat in categories:
        meta = Path(cache_dir) / f"{dataset}_{cat}_s{size}_m{mask_size}__meta.npz"
        if not meta.exists():
            continue
        y = np.load(meta, allow_pickle=False)["labels"].astype(bool)
        sa, sb = [], []
        for s in seeds:
            va = _scores(results_dir, cfg_name, cat, extractor, a[0], a[1], 0 if a[1] == 0 else s)
            vb = _scores(results_dir, cfg_name_b or cfg_name, cat, extractor, b[0], b[1], 0 if b[1] == 0 else s)
            if va is not None and vb is not None and len(va) == len(y) == len(vb):
                sa.append(va)
                sb.append(vb)
        if not sa:
            continue
        sa, sb = np.stack(sa), np.stack(sb)
        obs.append(np.mean([fast_auroc(y, x) for x in sa]) - np.mean([fast_auroc(y, x) for x in sb]))
        pos, neg = np.where(y)[0], np.where(~y)[0]
        d = np.empty(n_boot)
        for i in range(n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            yy = y[idx]
            d[i] = np.mean([fast_auroc(yy, x[idx]) for x in sa]) - np.mean([fast_auroc(yy, x[idx]) for x in sb])
        draws.append(d)
        cats_used.append(cat)
    if not cats_used:
        return {"n_categories": 0}
    draws = np.stack(draws)                                   # (categories, n_boot)
    pick = rng.integers(0, len(cats_used), (n_boot, len(cats_used)))
    boot = draws[pick, np.arange(n_boot)[:, None]].mean(1)    # resample categories, then images
    img_only = draws.mean(0)                                  # images resampled, categories fixed
    q = (1 - level) / 2
    return {"n_categories": len(cats_used), "mean_diff": float(np.mean(obs)),
            "ci_low": float(np.quantile(boot, q)), "ci_high": float(np.quantile(boot, 1 - q)),
            "ci_low_images_only": float(np.quantile(img_only, q)),
            "ci_high_images_only": float(np.quantile(img_only, 1 - q)),
            "per_category": dict(zip(cats_used, map(float, obs)))}


def image_level_table(specs: list, **common) -> pd.DataFrame:
    """specs: dicts with keys label, cfg_name, dataset, categories, a, b (plus optional overrides)."""
    rows = []
    for sp in specs:
        sp = {**common, **sp}
        label = sp.pop("label")
        r = image_level_paired(**sp)
        r.pop("per_category", None)
        rows.append({"comparison": label, "a": f"{sp['a'][0]}@{sp['a'][1]}", "b": f"{sp['b'][0]}@{sp['b'][1]}", **r})
    return pd.DataFrame(rows)


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def benchmark(tr_desc: np.ndarray, te_desc: np.ndarray, k: int = 1024, seed: int = 0,
              vocab_cfg: dict | None = None) -> dict:
    """Wall-clock seconds for each step on the current hardware (GPU when available)."""
    from .anomaly import greedy_coreset, min_dist, nearest_word
    from .vocab import build_vocabulary, build_vocabulary_torch

    vocab_cfg = vocab_cfg or {"max_descriptors": 200000, "spherical": True}
    out = {"k": k, "n_train_patches": len(tr_desc), "n_test_patches": len(te_desc)}

    def timed(name, fn):
        _sync()
        t0 = time.perf_counter()
        r = fn()
        _sync()
        out[name] = time.perf_counter() - t0
        return r

    v_cpu = timed("kmeans_sklearn_cpu_s", lambda: build_vocabulary(tr_desc, k, seed=seed, **vocab_cfg))
    timed("kmeans_torch_s", lambda: build_vocabulary_torch(tr_desc, k, seed=seed, **vocab_cfg))
    sel = timed("coreset_select_s", lambda: greedy_coreset(tr_desc, k, seed=seed))
    timed("score_codebook_s", lambda: nearest_word(te_desc, v_cpu))
    bank = np.asarray(tr_desc[np.sort(sel)], np.float32)
    timed("score_coreset_s", lambda: min_dist(te_desc, bank))
    timed("score_codebook_as_bank_s", lambda: min_dist(te_desc, v_cpu.centers))
    return out


CONTROL_COLS = ["image_auroc", "image_auroc_top10", "pixel_auroc", "aupro", "aupro_f5", "aupro_f10",
                "pixel_auroc_s0", "aupro_s0", "pixel_auroc_s2", "aupro_s2", "pixel_auroc_s8", "aupro_s8"]


def control_table(df: pd.DataFrame) -> pd.DataFrame:
    """Method x M: mean over categories, then over seeds, for every metric present."""
    d = df.copy()
    d["k"] = d["k"].fillna(0).astype(int)
    cols = [c for c in CONTROL_COLS if c in d]
    t = d.groupby(["method", "k", "vocab_seed"])[cols].mean().groupby(level=[0, 1]).mean()
    return t.sort_index(level=[1, 0])


def stl10_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Per encoding: NMI and accuracy over ALL runs (vocabulary seeds x clustering seeds), with SD and minimum."""
    import json

    rows = []
    d = df[df["extractor"] == "dinov2_s"] if "extractor" in df else df
    for (asg, k), g in d.groupby(["assignment", "k"]):
        acc = [a for s in g["acc_runs"].dropna() for a in json.loads(s)]
        nmi = [a for s in g["nmi_runs"].dropna() for a in json.loads(s)]
        if not acc:
            continue
        rows.append({"encoding": asg, "k": int(k), "n_runs": len(acc), "nmi_mean": float(np.mean(nmi)),
                     "acc_mean": float(np.mean(acc)), "acc_sd": float(np.std(acc)), "acc_min": float(np.min(acc))})
    return pd.DataFrame(rows).sort_values("acc_mean", ascending=False)


def inspect_train_folder(root, category: str, n_thumbs: int = 48, seed: int = 0, thumb: int = 96):
    """Check an MVTec-layout training folder: counts, image sizes, exact duplicates (within train and
    between train and test, by MD5), and a thumbnail sheet of random training images to eyeball defects.

    Returns (summary dict, matplotlib figure or None).
    """
    import hashlib
    from collections import Counter

    import cv2

    from .ad_data import _images_in

    base = Path(root) / category
    train = _images_in(base / "train" / "good")
    test = [p for d in sorted(x for x in (base / "test").iterdir() if x.is_dir()) for p in _images_in(d)]
    other_train = sorted(d.name for d in (base / "train").iterdir() if d.is_dir() and d.name != "good")

    def md5(p):
        return hashlib.md5(Path(p).read_bytes()).hexdigest()

    h_tr = [md5(p) for p in train]
    h_te = set(md5(p) for p in test)
    sizes = Counter()
    for p in train[:: max(1, len(train) // 200)]:     # sample up to ~200 headers
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            sizes[f"{img.shape[1]}x{img.shape[0]}"] += 1
    out = {"category": category, "n_train": len(train), "n_test": len(test),
           "other_train_folders": ",".join(other_train),
           "dup_within_train": len(h_tr) - len(set(h_tr)),
           "train_also_in_test": sum(h in h_te for h in set(h_tr)),
           "train_sizes": "; ".join(f"{k} ({v})" for k, v in sizes.most_common(4))}
    fig = None
    if n_thumbs and train:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(seed)
        pick = rng.choice(len(train), min(n_thumbs, len(train)), replace=False)
        cols = 8
        rows = int(np.ceil(len(pick) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), pick):
            img = cv2.imread(str(train[i]), cv2.IMREAD_COLOR)
            if img is not None:
                ax.imshow(cv2.cvtColor(cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB))
            ax.set_title(train[i].stem[-10:], fontsize=5)
        fig.suptitle(f"{category}: {len(pick)} random training images", fontsize=8)
        fig.tight_layout()
    return out, fig


def dhash(path, size: int = 8) -> int:
    """64-bit difference hash (grayscale, (size+1) x size, horizontal gradient signs)."""
    import cv2

    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicates(root, category: str, thresholds=(0, 4, 8)) -> dict:
    """Perceptual near-duplicate check of an MVTec-layout category (dHash Hamming distance).

    For each threshold t: training images with another training image within t bits, and test images with
    a training image within t bits (possible leaks). t = 0 also catches re-encoded exact copies.
    """
    from .ad_data import _images_in

    base = Path(root) / category
    train = _images_in(base / "train" / "good")
    test = [p for d in sorted(x for x in (base / "test").iterdir() if x.is_dir()) for p in _images_in(d)]
    ht = np.array([dhash(p) for p in train], dtype=np.uint64)
    hs = np.array([dhash(p) for p in test], dtype=np.uint64)

    def ham(a, b):  # (len(a), len(b)) Hamming distances, chunked
        out = np.empty((len(a), len(b)), np.uint8)
        for i in range(0, len(a), 512):
            x = a[i:i + 512, None] ^ b[None, :]
            out[i:i + 512] = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)
        return out

    tt = ham(ht, ht)
    np.fill_diagonal(tt, 255)
    ts = ham(hs, ht)
    out = {"category": category, "n_train": len(train), "n_test": len(test)}
    for t in thresholds:
        out[f"train_near_train_le{t}"] = int((tt.min(1) <= t).sum()) if len(train) > 1 else 0
        out[f"test_near_train_le{t}"] = int((ts.min(1) <= t).sum()) if len(train) else 0
    return out


def token_norm_check(cache_dir, tag: str, extractor: str, k: int = 64, seed: int = 0, size_params=None,
                     vocab_cfg=None) -> dict:
    """Do the highest-scoring patches of normal test images have unusually large token norms?

    Uses the raw cached features (only valid for caches written without in-place normalisation, i.e. runs
    without low_memory). For each normal test image, takes its single highest-scoring patch under the
    K-word codebook and under the same-size coreset, and reports where that patch's token norm falls in the
    distribution of all test-patch norms (percentile), plus the Spearman correlation between codebook score
    and norm over all test patches.
    """
    from scipy.stats import spearmanr

    from . import cache
    from .anomaly import cached_vocabulary, greedy_coreset, min_dist, nearest_word
    from .vocab import build_vocabulary, l2n

    params = size_params or {"image_size": 448}
    vocab_cfg = vocab_cfg or {"max_descriptors": 200000, "spherical": True}
    tr, _ = cache.get_or_compute(cache_dir, f"{tag}_train", extractor, params, lambda: None)
    te, _ = cache.get_or_compute(cache_dir, f"{tag}_test", extractor, params, lambda: None)
    meta = np.load(Path(cache_dir) / f"{tag}__meta.npz", allow_pickle=False)
    labels = meta["labels"].astype(bool)
    norms = np.linalg.norm(np.asarray(te.desc, np.float32), axis=1)
    if np.allclose(np.median(norms), 1.0, atol=1e-2):
        raise ValueError("cached features are already L2-normalised; token norms are not recoverable")
    trd, ted = l2n(np.asarray(tr.desc, np.float32)), l2n(np.asarray(te.desc, np.float32))
    vkey = {"tag": tag, "extractor": extractor, "params": params, "normalize": True, "k": k, "seed": seed,
            "vocab": vocab_cfg}
    vocab, _ = cached_vocabulary(cache_dir, vkey, lambda: build_vocabulary(trd, k, seed=seed, **vocab_cfg))
    cb, _ = nearest_word(ted, vocab)
    cs = min_dist(ted, trd[np.sort(greedy_coreset(trd, k, seed=seed))])
    n_img = len(labels)
    pct = rankdata(norms) / len(norms)              # percentile of each patch's norm, 0..1
    out = {"tag": tag, "extractor": extractor, "k": k, "seed": seed,
           "spearman_codebook_score_vs_norm": float(spearmanr(cb, norms).correlation),
           "spearman_coreset_score_vs_norm": float(spearmanr(cs, norms).correlation)}
    for name, s in (("codebook", cb), ("coreset", cs)):
        top = s.reshape(n_img, -1).argmax(1) + np.arange(n_img) * (len(s) // n_img)
        p = pct[top][~labels]
        out[f"{name}_top_patch_norm_pct_median"] = float(np.median(p))
        out[f"{name}_top_patch_in_top1pct_norm"] = float((p >= 0.99).mean())
    return out
