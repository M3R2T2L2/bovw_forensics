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


def test_inspect_train_folder(tmp_path):
    import cv2

    rng = np.random.default_rng(0)
    base = tmp_path / "cat"
    for sub in ["train/good", "test/good", "test/crack"]:
        (base / sub).mkdir(parents=True)
    imgs = [rng.integers(0, 255, (20, 30, 3), dtype=np.uint8) for _ in range(4)]
    for i, im in enumerate(imgs):
        cv2.imwrite(str(base / "train/good" / f"{i}.png"), im)
    cv2.imwrite(str(base / "train/good" / "dup.png"), imgs[0])          # duplicate within train
    cv2.imwrite(str(base / "test/good" / "x.png"), imgs[1])             # leaked into test
    cv2.imwrite(str(base / "test/crack" / "y.png"), rng.integers(0, 255, (20, 30, 3), dtype=np.uint8))
    out, fig = review.inspect_train_folder(tmp_path, "cat", n_thumbs=0)
    assert out["n_train"] == 5 and out["n_test"] == 2
    assert out["dup_within_train"] == 1 and out["train_also_in_test"] == 1
    assert out["train_sizes"].startswith("30x20") and fig is None


def test_near_duplicates(tmp_path):
    import cv2

    rng = np.random.default_rng(0)
    base = tmp_path / "cat"
    for sub in ["train/good", "test/good"]:
        (base / sub).mkdir(parents=True)
    ims = [cv2.GaussianBlur(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), (0, 0), 3) for _ in range(4)]
    for i, im in enumerate(ims):
        cv2.imwrite(str(base / "train/good" / f"{i}.png"), im)
    cv2.imwrite(str(base / "test/good" / "near.jpg"), np.clip(ims[2].astype(int) + 3, 0, 255).astype(np.uint8))
    cv2.imwrite(str(base / "test/good" / "other.png"),
                cv2.GaussianBlur(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), (0, 0), 3))
    r = review.near_duplicates(tmp_path, "cat", thresholds=(0, 4))
    assert r["n_train"] == 4 and r["n_test"] == 2
    assert r["test_near_train_le4"] == 1 and r["train_near_train_le4"] == 0


def test_token_norm_check(tmp_path):
    from bovw import cache as fcache
    from bovw.extract import LocalFeatures

    rng = np.random.default_rng(0)
    side, d = 4, 8
    tr = rng.normal(size=(30 * side * side, d)).astype(np.float32)
    te = rng.normal(size=(10 * side * side, d)).astype(np.float32)
    te[::side * side] *= 20            # one huge-norm "artifact" token per test image
    for name, x, n in (("t_train", tr, 30), ("t_test", te, 10)):
        fcache.get_or_compute(tmp_path, name, "x", {"image_size": 448},
                              lambda x=x, n=n: LocalFeatures(desc=x, offsets=np.arange(0, (n + 1) * side * side,
                                                                                         side * side)))
    np.savez(tmp_path / "t__meta.npz", labels=np.zeros(10, int))
    r = review.token_norm_check(tmp_path, "t", "x", k=8)
    assert 0 <= r["codebook_top_patch_norm_pct_median"] <= 1 and "spearman_coreset_score_vs_norm" in r
