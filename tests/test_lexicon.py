import numpy as np
import pytest

from bovw import lexicon


def test_position_nmi_detects_part_words():
    side, n_img = 4, 50
    pos_words = np.tile(np.arange(side * side) // 4, n_img)       # word = row of the grid
    nmi, shuf = lexicon.position_nmi(pos_words, side)
    assert nmi > 0.5 and shuf < 0.05


def test_cohesion_is_mean_pairwise_cosine():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    w = np.repeat(np.arange(3), 20)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    brute = []
    for k in range(3):
        m = xn[w == k]
        s = m @ m.T
        brute.append((s.sum() - len(m)) / (len(m) * (len(m) - 1)))
    assert lexicon.cohesion(x, w, 3) == pytest.approx(np.mean(brute), abs=1e-5)


def test_histograms_knn_and_signatures():
    n, per, k = 6, 9, 5
    words = np.zeros(n * per, int)
    scores = np.zeros(n * per)
    for i in range(n):           # images 0-2 flag word 1, images 3-5 flag word 3
        words[i * per:(i + 1) * per] = 0
        words[i * per + 4] = 1 if i < 3 else 3
        scores[i * per + 4] = 1.0
    h = lexicon.word_histograms(words, scores, n, k, topk=1)
    assert h.sum() == n and h[0, 1] == 1 and h[5, 3] == 1
    y = np.array(["a"] * 3 + ["b"] * 3)
    assert lexicon.loo_knn(lexicon.tfidf(h), y, 1) == 1.0
    sig = lexicon.signature_words(h, y)
    assert sig["sig_cover"] == 1.0 and sig["sig_false"] == 0.0
    assert sig["per_type"]["a"]["word"] == 1 and sig["per_type"]["b"]["word"] == 3
    assert lexicon.chance_accuracy(y) == pytest.approx(0.5)


def test_lexicon_synthetic(tmp_path):
    cfg = {"name": "lx", "data": {"name": "synthetic", "categories": ["tex"], "size": 128, "mask_size": 64,
                                  "n_train": 20, "n_good": 10, "n_bad": 20},
           "cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r"),
           "extractor": {"name": "dense_sift", "params": {"resize": 128, "step": 8, "size": 16, "n_jobs": 2}},
           "k": [16], "topk": 5, "vocab": {"seeds": [0], "max_descriptors": 20000, "spherical": True}}
    df = lexicon.run(cfg, progress=False)
    assert set(df.memory) == {"codebook", "coreset"} and len(df) == 2
    assert (df.n_types == 2).all() and np.allclose(df.chance, 0.5)
    assert df.knn1.between(0, 1).all() and df.cohesion.between(-1, 1).all()
    assert len(lexicon.run(cfg, progress=False)) == len(df)   # resume
    assert len(lexicon.summary(df)) == 2
    assert (tmp_path / "r" / "lx.csv").exists()


def test_defect_vocabulary_figure():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    side, k = 4, 6
    tr_imgs = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
    te_imgs = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(6)]
    tr_w = rng.integers(0, k, 5 * side * side)
    te_w = rng.integers(0, k, 6 * side * side)
    te_d = rng.random(6 * side * side)
    types = np.array(["good", "good", "a", "a", "b", "b"])
    fig = lexicon.defect_vocabulary_figure(tr_imgs, te_imgs, tr_w, te_w, te_d, types, types != "good", side, k,
                                           topk=2, n_word_examples=3, title="t")
    assert len(fig.axes) == 2 * 5
