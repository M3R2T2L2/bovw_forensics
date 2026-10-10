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
