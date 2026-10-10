"""Paired comparisons with uncertainty, as asked for in review.

Unit of analysis = a category. Seeds are averaged first (they only vary the
k-means initialisation and the coreset start/projection), then method A minus
method B is taken per category, and the mean difference gets a percentile
bootstrap CI over categories. Win/loss counts are given per metric, never
pooled across metrics. A two-sided sign test and a Wilcoxon signed-rank test
are reported as rough guides; with 8-15 categories they have little power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

METRICS = ("image_auroc", "image_auroc_top10", "pixel_auroc", "aupro")


def category_means(df: pd.DataFrame, metrics=METRICS) -> pd.DataFrame:
    """(method, k, category) -> metric means over seeds."""
    d = df.copy()
    d["k"] = d["k"].fillna(0).astype(int)
    cols = [m for m in metrics if m in d]
    return d.groupby(["method", "k", "category"])[cols].mean().sort_index()


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10000, level: float = 0.95, seed: int = 0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired(cm: pd.DataFrame, a: tuple, b: tuple, metric: str, tie: float = 0.001, **kw) -> dict:
    """a, b = (method, k). Per-category paired difference a - b for one metric."""
    sa, sb = cm.loc[a][metric], cm.loc[b][metric]
    cats = sa.index.intersection(sb.index)
    dlt = (sa[cats] - sb[cats]).dropna()
    mean, lo, hi = bootstrap_mean_ci(dlt.values, **kw)
    wins, losses = int((dlt > tie).sum()), int((dlt < -tie).sum())
    n_dec = wins + losses
    p_sign = float(binomtest(wins, n_dec, 0.5).pvalue) if n_dec else np.nan
    try:
        p_wil = float(wilcoxon(dlt.values).pvalue) if (dlt.abs() > 0).sum() >= 5 else np.nan
    except ValueError:
        p_wil = np.nan
    return {"metric": metric, "n_categories": int(len(dlt)), "mean_diff": mean, "ci_low": lo, "ci_high": hi,
            "wins": wins, "ties": int(len(dlt)) - n_dec, "losses": losses, "p_sign": p_sign, "p_wilcoxon": p_wil}


def compare(datasets: dict, a_method: str, b_method: str, ks=(64, 256, 1024), metrics=METRICS,
            pooled: bool = True, **kw) -> pd.DataFrame:
    """Per-dataset (and pooled over all categories) paired comparisons of a_method vs b_method.

    datasets: {name: results DataFrame}; categories from different datasets are kept distinct.
    """
    rows, cms = [], {}
    for name, df in datasets.items():
        cm = category_means(df, metrics)
        cms[name] = cm
        for k in ks:
            if (a_method, k) not in cm.index.droplevel(2) or (b_method, k) not in cm.index.droplevel(2):
                continue
            for m in metrics:
                if m in cm:
                    rows.append({"dataset": name, "k": k, **paired(cm, (a_method, k), (b_method, k), m, **kw)})
    if pooled and len(datasets) > 1:
        allc = pd.concat({n: c for n, c in cms.items()}, names=["dataset"]).reset_index()
        allc["category"] = allc["dataset"] + "/" + allc["category"]
        cm = allc.drop(columns="dataset").set_index(["method", "k", "category"]).sort_index()
        for k in ks:
            have = cm.index.droplevel(2)
            if (a_method, k) not in have or (b_method, k) not in have:
                continue
            for m in metrics:
                if m in cm:
                    rows.append({"dataset": "pooled", "k": k, **paired(cm, (a_method, k), (b_method, k), m, **kw)})
    out = pd.DataFrame(rows)
    out.insert(0, "comparison", f"{a_method} - {b_method}")
    return out
