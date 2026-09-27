import numpy as np
import pytest

from bovw import ad_data, anomaly
from bovw.extract import LocalFeatures


def test_min_dist_matches_brute_force():
    rng = np.random.default_rng(0)
    q, b = rng.normal(size=(300, 16)), rng.normal(size=(5000, 16))
    ref = np.sqrt(((q[:, None] - b[None]) ** 2).sum(-1)).min(1)
    got = anomaly.min_dist(q, b, q_chunk=64, b_chunk=4096)  # forces several chunks
    assert np.allclose(got, ref, atol=1e-4)


def test_grid_side_and_maps():
    f = LocalFeatures.from_list([np.zeros((49, 3))] * 2)
    assert anomaly.grid_side(f) == 7
    with pytest.raises(ValueError):
        anomaly.grid_side(LocalFeatures.from_list([np.zeros((50, 3))]))
    scores = np.zeros(2 * 49)
    scores[49 + 24] = 5.0  # hot centre patch in image 1
    maps, img = anomaly.patch_maps(scores, 2, 7, 28, sigma=1.0)
    assert maps.shape == (2, 28, 28) and img.tolist() == [0.0, 5.0]
    assert np.unravel_index(maps[1].argmax(), (28, 28)) in {(13, 13), (13, 14), (14, 13), (14, 14)}


def test_word_radius_quantiles_and_fallback():
    d = np.array([1, 2, 3, 4, 5, 10, 0.5], np.float32)
    w = np.array([0, 0, 0, 0, 0, 1, 2])
    r = anomaly.word_radius(d, w, k=3, q=1.0, min_members=5)
    assert r[0] == 5 and r[1] == r[2] == 5  # sparse words take the median radius


def test_metrics_by_type_and_pixel():
    labels = np.array([0, 0, 1, 1])
    types = ["good", "good", "a", "b"]
    m = anomaly.metrics(labels, np.array([0.1, 0.2, 0.9, 0.15]), types,
                        masks=np.array([[[0]], [[0]], [[1]], [[1]]]), maps=np.array([[[0]], [[0.1]], [[1]], [[0.2]]]))
    assert m["image_auroc"] == pytest.approx(0.75)
    assert '"a": 1.0' in m["image_auroc_by_type"] and '"b": 0.5' in m["image_auroc_by_type"]
    assert m["pixel_auroc"] == 1.0


def test_synthetic_end_to_end(tmp_path):
    cfg = {"name": "t", "data": {"name": "synthetic", "categories": ["tex"], "size": 128, "mask_size": 64,
                                 "n_train": 20, "n_good": 10, "n_bad": 10},
           "cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r"),
           "extractors": [{"name": "dense_sift", "params": {"resize": 128, "step": 8, "size": 16, "n_jobs": 2}}],
           "methods": {"patch_knn": {}, "global_knn": {}, "codebook": {"k": [16]}},
           "vocab": {"seeds": [0], "max_descriptors": 20000, "spherical": True}}
    df = anomaly.run(cfg, progress=False)
    assert set(df["method"]) == {"patch_knn", "codebook_dist", "codebook_norm", "hist_knn"}  # no CLS for SIFT
    best = df.set_index("method")["image_auroc"]
    assert best["codebook_dist"] > 0.9 and best["patch_knn"] > 0.9
    assert df.loc[df.method == "codebook_dist", "pixel_auroc"].iloc[0] > 0.8
    assert (tmp_path / "r" / "scores").is_dir()
    assert len(anomaly.run(cfg, progress=False)) == len(df)  # resume: nothing recomputed


def test_greedy_coreset_is_nested_kcenter():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(c, 0.01, (200, 8)) for c in (0, 5, 10, 15)]).astype(np.float32)
    idx = anomaly.greedy_coreset(x, 4, seed=1, proj_dim=0)
    assert len(set(idx)) == 4
    assert sorted(np.round(x[idx, 0] / 5).astype(int)) == [0, 1, 2, 3]  # one pick per cluster
    assert list(anomaly.greedy_coreset(x, 2, seed=1, proj_dim=0)) == list(idx[:2])  # nested
    assert len(anomaly.greedy_coreset(x[:3], 10, seed=0)) == 3  # m capped at n


def test_subsample_baselines(tmp_path):
    cfg = {"name": "t", "data": {"name": "synthetic", "categories": ["tex"], "size": 128, "mask_size": 64,
                                 "n_train": 20, "n_good": 10, "n_bad": 10},
           "cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r"),
           "extractors": [{"name": "dense_sift", "params": {"resize": 128, "step": 8, "size": 16, "n_jobs": 2}}],
           "methods": {"subsample": {"sizes": [8, 32], "methods": ["coreset_knn", "random_knn"]}},
           "vocab": {"seeds": [0, 1]}}
    df = anomaly.run(cfg, progress=False)
    assert len(df) == 2 * 2 * 2
    assert set(df["bank_size"]) == {8, 32}
    assert df["pixel_auroc"].notna().all()
    assert len(anomaly.run(cfg, progress=False)) == len(df)  # resume


def test_mvtec_loader_layout(tmp_path):
    import cv2

    cat = tmp_path / "bottle"
    for sub in ["train/good", "test/good", "test/broken", "ground_truth/broken"]:
        (cat / sub).mkdir(parents=True)
    img = np.full((40, 40, 3), 128, np.uint8)
    for p in ["train/good/000.png", "test/good/000.png", "test/broken/000.png"]:
        cv2.imwrite(str(cat / p), img)
    mask = np.zeros((40, 40), np.uint8)
    mask[10:20, 10:20] = 255
    cv2.imwrite(str(cat / "ground_truth/broken/000_mask.png"), mask)
    s = ad_data.load_mvtec(str(tmp_path), "bottle", size=32, mask_size=16)
    assert len(s.train_images) == 1 and sorted(s.test_labels.tolist()) == [0, 1]
    assert s.test_masks.shape == (2, 16, 16) and s.test_masks.sum() > 0
    assert set(s.test_types) == {"good", "broken"}


def test_ensure_archive_verifies_and_replaces_stale(tmp_path):
    import hashlib

    src = tmp_path / "src.bin"
    src.write_bytes(b"mvtec-archive-bytes")
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    dst = tmp_path / "drive" / "mvtec.tar.xz"
    dst.parent.mkdir()
    dst.write_bytes(b"")  # empty file left by a failed wget
    ad_data.ensure_archive(str(dst), src.as_uri(), good)
    assert dst.read_bytes() == b"mvtec-archive-bytes" and not (tmp_path / "drive" / "mvtec.tar.xz.part").exists()
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        ad_data.ensure_archive(str(tmp_path / "other.tar.xz"), src.as_uri(), "0" * 64)
    assert not (tmp_path / "other.tar.xz").exists() and not (tmp_path / "other.tar.xz.part").exists()


def test_ensure_archive_keeps_real_sized_mismatch(tmp_path):
    f = tmp_path / "upload.tar.xz"
    f.write_bytes(b"x" * 2000)
    with pytest.raises(RuntimeError, match="does not match"):
        ad_data.ensure_archive(str(f), "file:///nonexistent", "0" * 64, stub_bytes=1000)
    assert f.exists()  # the user's file is left alone
    assert ad_data.ensure_archive(str(f), "file:///nonexistent", None) == str(f)
