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


def test_aupro_and_dprime():
    masks = np.zeros((2, 32, 32), np.uint8)
    masks[1, 4:8, 4:8] = 1          # small region
    masks[1, 16:30, 16:30] = 1      # large region
    perfect = masks.astype(np.float32) + 0.01 * np.random.default_rng(0).random(masks.shape)
    assert anomaly.aupro(masks, perfect) > 0.99
    half = perfect.copy()
    half[1, 4:8, 4:8] = 0.0          # miss the small region entirely
    # pixel AUROC barely notices (16 of 212 defect pixels), AUPRO halves
    assert anomaly.aupro(masks, half) == pytest.approx(0.5, abs=0.02)
    assert anomaly.pixel_dprime(masks, perfect) > 10
    assert np.isnan(anomaly.aupro(np.zeros((1, 8, 8)), np.zeros((1, 8, 8))))


def test_topk_image_scores():
    ps = np.array([[1, 2, 3, 10], [4, 4, 4, 4]], np.float32).reshape(-1)
    assert list(anomaly.topk_image_scores(ps, 2, 1)) == [10, 4]
    assert list(anomaly.topk_image_scores(ps, 2, 2)) == [6.5, 4]


def test_v2_run_metrics_and_vocab_cache(tmp_path):
    cfg = {"name": "t2", "data": {"name": "synthetic", "categories": ["tex"], "size": 128, "mask_size": 64,
                                  "n_train": 20, "n_good": 10, "n_bad": 10},
           "cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r"),
           "extractors": [{"name": "dense_sift", "params": {"resize": 128, "step": 8, "size": 16, "n_jobs": 2}}],
           "region_metrics": True, "image_topk": [1, 3], "save_maps_for": ["tex"], "cache_vocab": True,
           "methods": {"patch_knn": {}, "codebook": {"k": [16], "scores": ["codebook_dist"]},
                       "subsample": {"sizes": [16], "methods": ["coreset_knn"]}},
           "vocab": {"seeds": [0], "max_descriptors": 20000, "spherical": True}}
    df = anomaly.run(cfg, progress=False)
    assert set(df.method) == {"patch_knn", "codebook_dist", "coreset_knn"}
    for c in ["aupro", "pixel_dprime", "image_auroc_top1", "image_auroc_top3"]:
        assert df[c].notna().all(), c
    assert (df["image_auroc_top1"] == df["image_auroc"]).all()  # top-1 mean = max
    assert len(list((tmp_path / "r" / "maps").glob("*.npy"))) == 3
    assert len(list((tmp_path / "c" / "vocab").glob("*.pkl"))) == 1
    # a fresh results dir reuses the cached vocabulary
    cfg2 = {**cfg, "name": "t3"}
    df2 = anomaly.run(cfg2, progress=False)
    assert df2.loc[df2.method == "codebook_dist", "vocab_cache_hit"].iloc[0] == True  # noqa: E712
    assert np.allclose(df2.sort_values("method").image_auroc, df.sort_values("method").image_auroc)

    t = anomaly.followup_table(df)
    assert ("hybrid", 16) in t.index
    assert t.loc[("hybrid", 16), "img"] == pytest.approx(t.loc[("coreset_knn", 16), "img"])
    assert t.loc[("hybrid", 16), "aupro"] == pytest.approx(t.loc[("codebook_dist", 16), "aupro"])

    from bovw import plots
    labels = np.array([0] * 10 + [1] * 10)
    masks = np.zeros((20, 64, 64), np.uint8)
    masks[10:, 10:20, 10:20] = 1
    sp = anomaly.normal_pixel_spread(tmp_path / "r", "t2", "tex", masks,
                                     methods=(("codebook_dist", 16), ("coreset_knn", 16), ("patch_knn", 0)),
                                     extractor="dense_sift")
    assert len(sp) == 3 and (sp.rel_iqr > 0).all()
    imgs = [np.zeros((128, 128, 3), np.uint8)] * 20
    fig = plots.false_alarms(tmp_path / "r", "t2", "tex", imgs, labels,
                             methods=(("codebook_dist", 16), ("coreset_knn", 16)), extractor="dense_sift", n=3)
    assert len(fig.axes) == 9


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


def _mock_visa(tmp_path):
    import cv2

    root = tmp_path / "VisA_20220922"
    (root / "candle/Data/Images/Normal").mkdir(parents=True)
    (root / "candle/Data/Images/Anomaly").mkdir(parents=True)
    (root / "candle/Data/Masks/Anomaly").mkdir(parents=True)
    rows = ["object,split,label,image,mask"]
    for i in range(3):
        cv2.imwrite(str(root / f"candle/Data/Images/Normal/{i:04d}.JPG"), np.full((120, 100, 3), 90, np.uint8))
        rows.append(f"candle,{'train' if i < 2 else 'test'},normal,candle/Data/Images/Normal/{i:04d}.JPG,")
    cv2.imwrite(str(root / "candle/Data/Images/Anomaly/000.JPG"), np.full((120, 100, 3), 200, np.uint8))
    m = np.zeros((120, 100), np.uint8)
    m[10:40, 10:40] = 1  # VisA masks may use small non-zero values
    cv2.imwrite(str(root / "candle/Data/Masks/Anomaly/000.png"), m)
    rows.append("candle,test,anomaly,candle/Data/Images/Anomaly/000.JPG,candle/Data/Masks/Anomaly/000.png")
    rows.append("cashew,train,normal,cashew/x.JPG,")
    csv_path = tmp_path / "split.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    return tmp_path, root, csv_path


def test_visa_loader(tmp_path):
    top, root, csv_path = _mock_visa(tmp_path)
    assert ad_data.find_visa_root(str(top)) == str(root)
    s = ad_data.load_visa(str(root), "candle", size=64, mask_size=32, split_csv=csv_path)
    assert len(s.train_images) == 2 and s.train_images[0].shape == (64, 64, 3)
    assert list(s.test_labels) == [0, 1] and s.test_types == ["good", "anomaly"]
    assert s.test_masks[1].max() == 1 and 0.05 < s.test_masks[1].mean() < 0.15
    assert s.test_masks[0].sum() == 0


def test_visa_official_split_counts():
    import pandas as pd

    d = pd.read_csv(ad_data.VISA_SPLIT_CSV)
    assert sorted(d.object.unique()) == sorted(ad_data.VISA_CATEGORIES)
    assert len(d) == 10821 and (d.label == "anomaly").sum() == 1200
    assert (d[d.split == "train"].label == "normal").all()


def test_prereg_checks():
    import pandas as pd

    rows = []
    for seed in (0, 1, 2):
        for k in (64, 256, 1024):
            for m, aup, top10, pix in (("codebook_dist", 0.93, 0.985, 0.978), ("coreset_knn", 0.90, 0.986, 0.970)):
                rows.append({"method": m, "k": k, "vocab_seed": seed, "category": "a", "image_auroc": 0.97,
                             "image_auroc_top10": top10, "pixel_auroc": pix, "aupro": aup})
    rows.append({"method": "patch_knn", "k": 0, "vocab_seed": 0, "category": "a", "image_auroc": 0.98,
                 "image_auroc_top10": 0.99, "pixel_auroc": 0.98, "aupro": 0.94})
    r = anomaly.prereg_checks(pd.DataFrame(rows)).set_index("id")["pass"]
    assert r.to_dict() == {"P1": True, "P2": True, "P3": True, "P4": True}


def test_defect_discovery_synthetic(tmp_path):
    from bovw import discovery

    cfg = {"name": "d", "data": {"name": "synthetic", "categories": ["tex"], "size": 128, "mask_size": 64,
                                 "n_train": 20, "n_good": 10, "n_bad": 20},
           "cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r"),
           "extractor": {"name": "dense_sift", "params": {"resize": 128, "step": 8, "size": 16, "n_jobs": 2}},
           "k": [16], "topk": 5, "vocab": {"seeds": [0], "max_descriptors": 20000, "spherical": True}}
    df = discovery.run(cfg, progress=False)
    assert set(df.setting) == {"oracle", "detected"}
    assert set(df.descriptor) == {"topk_resid", "topk_feat", "mask_feat"}  # no CLS for SIFT
    o = df[(df.setting == "oracle")]
    assert (o.n_types == 2).all() and (o.n_images == 20).all()
    # noise squares vs rotated stripes: the defect region separates them better than chance
    assert o[o.descriptor == "mask_feat"].nmi_above_chance.max() > 0.1
    assert (df.nmi_chance < 0.2).all()
    assert len(discovery.run(cfg, progress=False)) == len(df)  # resume
    assert len(discovery.summary(df)) == len(df.groupby(["setting", "algorithm", "descriptor"]))


def test_image_descriptors_shapes():
    from bovw import discovery

    rng = np.random.default_rng(0)
    n, side, d = 3, 4, 8
    desc = rng.normal(size=(n * side * side, d)).astype(np.float32)
    dist = rng.random(n * side * side).astype(np.float32)
    masks = np.zeros((n, 16, 16), np.uint8)
    masks[0, :8, :8] = 1
    out = discovery.image_descriptors(desc, n, dist, desc * 0.5, 2, masks, side, rng.normal(size=(n, 5)))
    assert {k: v.shape for k, v in out.items()} == {"topk_feat": (3, 8), "topk_resid": (3, 8), "cls": (3, 5),
                                                     "mask_feat": (3, 8)}
    assert np.allclose(np.linalg.norm(out["topk_feat"], axis=1), 1, atol=1e-5)


def test_binned_pixel_metrics_match_exact():
    rng = np.random.default_rng(0)
    masks = np.zeros((40, 64, 64), np.uint8)
    for i in range(20):
        y, x = rng.integers(0, 48, 2)
        masks[i, y:y + rng.integers(3, 16), x:x + rng.integers(3, 16)] = 1
    maps = (rng.normal(size=masks.shape) + 1.5 * masks).astype(np.float32)
    b = anomaly.binned_pixel_metrics(masks, maps.astype(np.float16), True)
    assert b["pixel_auroc"] == pytest.approx(anomaly._auroc(masks.reshape(-1), maps.reshape(-1)), abs=2e-3)
    assert b["aupro"] == pytest.approx(anomaly.aupro(masks, maps), abs=5e-3)
    assert b["pixel_dprime"] == pytest.approx(anomaly.pixel_dprime(masks, maps), abs=0.02)


def test_low_memory_run_matches(tmp_path):
    base = {"data": {"name": "synthetic", "categories": ["tex"], "size": 128, "mask_size": 64,
                     "n_train": 20, "n_good": 10, "n_bad": 10},
            "extractors": [{"name": "dense_sift", "params": {"resize": 128, "step": 8, "size": 16, "n_jobs": 2}}],
            "region_metrics": True, "image_topk": [3],
            "methods": {"patch_knn": {}, "codebook": {"k": [16], "scores": ["codebook_dist"]},
                        "subsample": {"sizes": [16], "methods": ["coreset_knn"]}},
            "vocab": {"seeds": [0], "max_descriptors": 20000, "spherical": True}}
    a = anomaly.run({**base, "name": "a", "cache_dir": str(tmp_path / "c1"), "results_dir": str(tmp_path / "r1")},
                    progress=False).sort_values("method")
    b = anomaly.run({**base, "name": "b", "low_memory": True, "cache_dir": str(tmp_path / "c2"),
                     "results_dir": str(tmp_path / "r2")}, progress=False).sort_values("method")
    for c in ["image_auroc", "image_auroc_top3", "pixel_auroc", "aupro"]:
        assert np.allclose(a[c].values, b[c].values, atol=5e-3), c


def test_mvtec_style_loader_3cad_layout(tmp_path):
    import cv2

    cat = tmp_path / "3CAD" / "Copper_Stator"
    for sub in ["train/good", "test/good", "test/scratch", "ground_truth/scratch"]:
        (cat / sub).mkdir(parents=True)
    for p in ["train/good/a.jpg", "train/good/b.jpg", "test/good/c.jpg", "test/scratch/d.jpg"]:
        cv2.imwrite(str(cat / p), np.full((90, 70, 3), 100, np.uint8))
    m = np.zeros((90, 70), np.uint8)
    m[:30, :30] = 1
    cv2.imwrite(str(cat / "ground_truth/scratch/d.png"), m)   # no "_mask" suffix
    root = ad_data.find_root(str(tmp_path), "Copper_Stator")
    assert root == str(tmp_path / "3CAD")
    s = ad_data.load_mvtec_style(root, "Copper_Stator", size=64, mask_size=32)
    assert len(s.train_images) == 2 and s.test_types == ["good", "scratch"]
    assert list(s.test_labels) == [0, 1] and s.test_masks[1].sum() > 0
    (cat / "ground_truth/scratch/d.png").unlink()
    with pytest.raises(FileNotFoundError, match="no mask"):
        ad_data.load_mvtec_style(root, "Copper_Stator", size=64, mask_size=32)


def test_lazy_images_match_eager(tmp_path):
    import cv2

    cat = tmp_path / "Copper_Stator"
    for sub in ["train/good", "test/good", "test/scratch", "ground_truth/scratch"]:
        (cat / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for p in ["train/good/a.png", "train/good/b.png", "test/good/c.png", "test/scratch/d.png"]:
        cv2.imwrite(str(cat / p), rng.integers(0, 255, (90, 70, 3), dtype=np.uint8))
    cv2.imwrite(str(cat / "ground_truth/scratch/d_mask.png"), np.full((90, 70), 255, np.uint8))
    e = ad_data.load_mvtec_style(str(tmp_path), "Copper_Stator", size=32, mask_size=16)
    z = ad_data.load_mvtec_style(str(tmp_path), "Copper_Stator", size=32, mask_size=16, lazy=True)
    assert isinstance(z.train_images, ad_data.LazyImages) and len(z.test_images) == 2
    for a, b in zip(e.train_images + e.test_images, list(z.train_images) + z.test_images.get_batch([0, 1])):
        assert np.array_equal(a, b)
    assert np.array_equal(e.test_masks, z.test_masks) and z.test_images[0:1][0].shape == (32, 32, 3)


def test_foundation_extractor_preallocated_output():
    torch = pytest.importorskip("torch")
    from bovw.extract import foundation

    class Out:
        def __init__(self, h):
            self.last_hidden_state = h

    class Fake(torch.nn.Module):  # 1 CLS + 4 patch tokens, dim 3, value = image mean + token index
        def forward(self, pixel_values, output_hidden_states=False):
            b = pixel_values.shape[0]
            m = pixel_values.mean(dim=(1, 2, 3)).view(b, 1, 1)
            return Out(m + torch.arange(5.0).view(1, 5, 1).expand(b, 5, 3))

    ex = foundation.FoundationExtractor.__new__(foundation.FoundationExtractor)
    ex.spec, ex.image_size, ex.batch_size, ex.layer = foundation.BACKBONES["dinov2_s"], 16, 3, -1
    ex.device, ex.dtype, ex.model = "cpu", torch.float32, Fake()
    ex.mean = torch.zeros(1, 3, 1, 1)
    ex.std = torch.ones(1, 3, 1, 1)
    imgs = [np.full((16, 16, 3), 25 * i, np.uint8) for i in range(7)]   # 7 = 3 + 3 + 1: uneven last batch
    f = ex(imgs, progress=False)
    assert f.desc.shape == (28, 3) and f.desc.dtype == np.float16 and list(f.offsets) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert f.global_.shape == (7, 3)
    for i in range(7):
        mean = 25 * i / 255.0
        assert np.allclose(f.desc[4 * i:4 * i + 4, 0], mean + np.arange(1, 5), atol=1e-2)
        assert np.isclose(f.global_[i, 0], mean, atol=1e-5)


def test_require_gpu_stops_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(anomaly, "_torch_cuda", lambda: None)
    cfg = {"name": "g", "require_gpu": True, "data": {"name": "synthetic", "categories": ["tex"]},
           "cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r"), "extractors": [], "methods": {}}
    with pytest.raises(RuntimeError, match="No GPU"):
        anomaly.run(cfg, progress=False)


def test_min_dist_cpu_chunked_matches_bruteforce():
    rng = np.random.default_rng(1)
    q, b = rng.normal(size=(300, 16)).astype(np.float16), rng.normal(size=(5000, 16)).astype(np.float16)
    got = anomaly.min_dist(q, b, q_chunk=64, b_chunk=4096)
    want = np.sqrt(((q.astype(np.float32)[:, None] - b.astype(np.float32)[None]) ** 2).sum(-1)).min(1)
    assert np.allclose(got, want, atol=1e-3)
