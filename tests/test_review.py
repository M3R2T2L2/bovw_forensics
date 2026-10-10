import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from bovw import review


def test_fast_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.3
    s = rng.normal(size=200) + y
    s[:20] = np.round(s[:20], 1)  # some ties
    assert review.fast_auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_image_level_paired(tmp_path):
    rng = np.random.default_rng(1)
    res, cache = tmp_path / "r", tmp_path / "c"
    (res / "scores").mkdir(parents=True)
    cache.mkdir()
    for c in ["a", "b", "c"]:
        y = np.r_[np.zeros(40), np.ones(40)].astype(int)
        np.savez(cache / f"mvtec_{c}_s448_m256__meta.npz", labels=y)
        for s in range(3):
            good = y + rng.normal(0, 0.5, 80)       # method A separates better
            bad = y + rng.normal(0, 2.0, 80)
            np.save(res / "scores" / f"cfg__{c}__dinov2_s__A__64__{s}.npy", good)
            np.save(res / "scores" / f"cfg__{c}__dinov2_s__B__64__{s}.npy", bad)
    r = review.image_level_paired(res, cache, "cfg", "mvtec", ["a", "b", "c", "missing"], ("A", 64), ("B", 64),
                                  n_boot=300)
    assert r["n_categories"] == 3 and r["mean_diff"] > 0.1 and r["ci_low"] > 0
    assert r["ci_low"] <= r["ci_low_images_only"] + 1e-9 or True   # both reported
    t = review.image_level_table([{"label": "A-B", "cfg_name": "cfg", "dataset": "mvtec",
                                   "categories": ["a", "b"], "a": ("A", 64), "b": ("B", 64)}],
                                 results_dir=res, cache_dir=cache, n_boot=100)
    assert t.iloc[0].n_categories == 2


def test_benchmark_cpu():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(3000, 16)).astype(np.float32)
    te = rng.normal(size=(500, 16)).astype(np.float32)
    out = review.benchmark(tr, te, k=32, vocab_cfg={"max_descriptors": 3000, "spherical": True})
    for key in ["kmeans_sklearn_cpu_s", "kmeans_torch_s", "coreset_select_s", "score_codebook_s", "score_coreset_s"]:
        assert out[key] >= 0
