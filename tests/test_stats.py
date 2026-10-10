import numpy as np
import pandas as pd

from bovw import stats


def _df(delta, cats=10, seeds=3):
    rng = np.random.default_rng(0)
    rows = []
    for c in range(cats):
        base = rng.uniform(0.7, 0.9)
        for s in range(seeds):
            rows.append({"method": "B", "k": 64, "vocab_seed": s, "category": f"c{c}", "aupro": base})
            rows.append({"method": "A", "k": 64, "vocab_seed": s, "category": f"c{c}", "aupro": base + delta})
    return pd.DataFrame(rows)


def test_paired_bootstrap_detects_consistent_gain():
    t = stats.compare({"d": _df(0.05)}, "A", "B", ks=(64,), metrics=("aupro",))
    r = t.iloc[0]
    assert abs(r.mean_diff - 0.05) < 1e-9 and r.ci_low > 0 and r.wins == 10 and r.losses == 0
    assert r.p_sign < 0.01


def test_ties_and_pooling():
    t = stats.compare({"d1": _df(0.0005), "d2": _df(0.0005, cats=5)}, "A", "B", ks=(64,), metrics=("aupro",))
    pooled = t[t.dataset == "pooled"].iloc[0]
    assert pooled.n_categories == 15 and pooled.ties == 15 and np.isnan(pooled.p_sign)


def test_bootstrap_ci_contains_mean():
    m, lo, hi = stats.bootstrap_mean_ci(np.array([0.1, 0.2, 0.3, 0.4]), n_boot=2000)
    assert lo <= m <= hi and abs(m - 0.25) < 1e-12
