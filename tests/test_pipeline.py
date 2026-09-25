import numpy as np
import pytest

from bovw import cache, data
from bovw.encode import encode, hard, soft, vlad
from bovw.eval import cluster_accuracy, evaluate_clustering
from bovw.extract import LocalFeatures, get_extractor
from bovw.vocab import build_vocabulary


def _blobs(n_img=40, per_img=50, d=8, k=4, seed=0):
    """Images whose descriptors come from class-specific cluster centres."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, (k, d))
    per, labels = [], []
    for i in range(n_img):
        c = i % k
        per.append(centers[c] + rng.normal(0, 0.5, (per_img, d)))
        labels.append(c)
    return LocalFeatures.from_list(per), np.array(labels)


def test_localfeatures_ragged():
    f = LocalFeatures.from_list([np.ones((3, 4)), np.zeros((0, 4)), np.ones((2, 4))])
    assert f.n_images == 3 and f.dim == 4
    assert list(f.counts()) == [3, 0, 2]
    assert f.image(1).shape == (0, 4)


def test_vocab_shapes_and_spherical():
    f, _ = _blobs()
    v = build_vocabulary(f.desc, 8, spherical=True, pca_dim=4)
    assert v.centers.shape == (8, 4)
    assert np.allclose(np.linalg.norm(v.centers, axis=1), 1, atol=1e-5)


@pytest.mark.parametrize("fn,dim", [(hard, lambda k, d: k), (soft, lambda k, d: k), (vlad, lambda k, d: k * d)])
def test_encoder_dims_and_norm(fn, dim):
    f, _ = _blobs()
    v = build_vocabulary(f.desc, 6)
    x = fn(f, v)
    assert x.shape == (f.n_images, dim(6, f.dim))
    assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
    assert np.isfinite(x).all()


def test_hard_histogram_counts_before_postprocess():
    f, _ = _blobs(n_img=4, per_img=10)
    v = build_vocabulary(f.desc, 4)
    x = hard(f, v, power=1.0, l2=False)
    assert np.allclose(x.sum(1), 1.0)


def test_empty_image_is_handled():
    f = LocalFeatures.from_list([np.random.rand(20, 5), np.zeros((0, 5)), np.random.rand(20, 5)])
    v = build_vocabulary(f.desc, 3)
    for m in ("hard", "soft", "vlad"):
        x = encode(m, f, v)
        assert np.isfinite(x).all()
        assert np.allclose(x[1], 0)


def test_chunking_matches_unchunked(monkeypatch):
    import bovw.encode as E

    f, _ = _blobs(n_img=30, per_img=40)
    v = build_vocabulary(f.desc, 5)
    full = E.vlad(f, v)
    orig = E._chunks
    monkeypatch.setattr(E, "_chunks", lambda feats, vocab: orig(feats, vocab, max_rows=37))
    assert np.allclose(full, E.vlad(f, v), atol=1e-5)


def test_histograms_recover_classes():
    # Word-occupancy encodings should separate classes whose descriptors live in
    # different regions. (VLAD encodes residuals, which are pure noise in this toy
    # data, so it is checked against a reference implementation instead.)
    f, y = _blobs()
    v = build_vocabulary(f.desc, 8)
    for m in ("hard", "soft"):
        r = evaluate_clustering(encode(m, f, v), y, pca_dim=None, seeds=(0,))
        assert r["nmi"] > 0.9, (m, r)


def test_vlad_matches_naive_reference():
    f, _ = _blobs(n_img=6, per_img=15)
    v = build_vocabulary(f.desc, 4)
    got = vlad(f, v, intra_norm=True, power=0.5)
    ref = []
    for i in range(f.n_images):
        x = f.image(i)
        a = np.argmin(((x[:, None, :] - v.centers[None]) ** 2).sum(-1), axis=1)
        blocks = np.zeros_like(v.centers)
        for j in range(v.k):
            if (a == j).any():
                r = (x[a == j] - v.centers[j]).sum(0)
                blocks[j] = r / (np.linalg.norm(r) + 1e-12)
        z = blocks.ravel()
        z = np.sign(z) * np.sqrt(np.abs(z))
        ref.append(z / np.linalg.norm(z))
    assert np.allclose(got, np.array(ref), atol=1e-4)


def test_cluster_accuracy_permutation_invariant():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert cluster_accuracy(y, np.array([2, 2, 0, 0, 1, 1])) == 1.0
    assert cluster_accuracy(y, np.array([0, 1, 0, 1, 0, 1])) == pytest.approx(2 / 6 + 0, abs=0.34)


def test_cache_roundtrip(tmp_path):
    f, _ = _blobs(n_img=5)
    f.global_ = np.random.rand(5, 3).astype(np.float32)
    calls = []
    compute = lambda: (calls.append(1), f)[1]
    a, info_a = cache.get_or_compute(tmp_path, "ds", "ex", {"p": 1}, compute)
    b, info_b = cache.get_or_compute(tmp_path, "ds", "ex", {"p": 1}, compute)
    assert len(calls) == 1 and not info_a["cache_hit"] and info_b["cache_hit"]
    assert np.allclose(np.asarray(b.desc, np.float32), f.desc, atol=1e-2)
    assert (b.offsets == f.offsets).all() and b.global_.shape == (5, 3)


@pytest.mark.parametrize("name,dim", [("dense_sift", 128), ("sift", 128), ("orb", 256)])
def test_handcrafted_extractors(name, dim):
    imgs, _ = data.load_synthetic(n_per_class=2, n_classes=2)
    f = get_extractor(name)(imgs, progress=False)
    assert f.n_images == 4 and f.dim == dim
    assert f.counts().sum() > 0


def test_sweep_mixed_rows_resume_and_lazy_images(tmp_path, monkeypatch):
    """Global-baseline rows and codebook rows have different columns (the Colab
    ParserError). Also: a rerun with a warm cache must not load images."""
    import bovw.sweep as S

    rng = np.random.default_rng(0)

    def fake_extractor(name, **kw):
        def call(images, progress=True):
            per = [rng.normal(size=(20, 8)) for _ in images]
            return LocalFeatures.from_list(per, global_=rng.normal(size=(len(images), 6)).astype(np.float32))
        return call

    monkeypatch.setattr(S, "get_extractor", fake_extractor)
    cfg = {"name": "t", "dataset": {"name": "synthetic", "n_per_class": 5, "n_classes": 3},
           "cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r"),
           "extractors": [{"name": "fake"}], "vocab": {"k": [4, 8]},
           "assignments": ["hard", "soft", "vlad"], "eval": {"clustering": {"pca_dim": None, "seeds": [0]}}}
    df = S.run(cfg, progress=False)
    assert len(df) == 1 + 2 * 3
    assert set(df["assignment"]) == {"global", "hard", "soft", "vlad"}
    assert (tmp_path / "r" / "t.csv").exists()
    import pandas as pd
    assert len(pd.read_csv(tmp_path / "r" / "t.csv")) == 7

    loads = []
    real = S.data.load
    monkeypatch.setattr(S.data, "load", lambda *a, **k: (loads.append(1), real(*a, **k))[1])
    df2 = S.run(cfg, progress=False)
    assert len(df2) == 7 and loads == []


def test_load_results_skips_truncated_line(tmp_path):
    from bovw.sweep import load_results

    p = tmp_path / "x.jsonl"
    p.write_text('{"run_id": "a", "nmi": 1}\n{"run_id": "b", "n')
    assert list(load_results(p)["run_id"]) == ["a"]
