"""Is the vocabulary a usable defect language? (P0 follow-up to the referee report)

The discovery rerun showed any memory can pick an image's suspicious patches. What a
k-means vocabulary adds is a fixed set of K prototypes that people can name, count and
report. This module tests that claim against the same-size coreset, whose K selected
patches are used as "words" in exactly the same way (each patch is assigned to its
nearest selected patch).

Word quality, on normal training patches:
  position_nmi  NMI between word id and grid position (aligned objects: does a word
                mean "this part of the object"?), minus the same with shuffled words
  cohesion      mean over words (weighted by size) of ||mean member||^2 for unit
                features ~ average cosine between two members of the same word

Defect language, on truly anomalous test images (MVTec defect-type folders as labels):
  each image -> histogram of the words assigned to its top-k patches (scored by the
  memory's own distances), tf-idf weighted, L2-normalised
  knn1 / knn5   leave-one-out nearest-neighbour defect-type accuracy (cosine)
  chance        sum_c p_c^2 (accuracy of guessing types at their frequencies)
  sig_cover     for each type, its best "signature" word (largest gap between the share
                of its images that use the word and the share of other types' images
                that do); mean share of the type's images that use it
  sig_false     mean share of other types' images that use those signature words
  feat_knn1     same LOO 1-NN on the continuous top-k mean feature (reference, no words)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from . import ad_data, cache
from .anomaly import _LazyAD, _norm, cached_vocabulary, greedy_coreset, grid_side, nearest_word, nn_search
from .discovery import topk_feature_descriptor
from .sweep import RUNTIME_KEYS, _append, _env, _run_id, load_results
from .vocab import build_vocabulary, l2n

MEMORIES = ("codebook", "coreset")


# --------------------------------------------------------------------------- word quality

def position_nmi(words: np.ndarray, side: int, seed: int = 0) -> tuple[float, float]:
    """(NMI(word, grid position), NMI with words shuffled) over patches stored image by image."""
    pos = np.tile(np.arange(side * side), len(words) // (side * side))
    nmi = normalized_mutual_info_score(pos, words)
    rng = np.random.default_rng(seed)
    return float(nmi), float(normalized_mutual_info_score(pos, rng.permutation(words)))


def cohesion(x: np.ndarray, words: np.ndarray, k: int) -> float:
    """Size-weighted mean of ||mean member||^2 per word (unit-norm rows); singletons excluded."""
    x = l2n(np.asarray(x, np.float32))
    sums = np.zeros((k, x.shape[1]), np.float64)
    np.add.at(sums, words, x)
    n = np.bincount(words, minlength=k).astype(np.float64)
    ok = n >= 2
    # unbiased mean pairwise cosine within a word: (||sum||^2 - n) / (n (n - 1))
    pair = ((sums[ok] ** 2).sum(1) - n[ok]) / (n[ok] * (n[ok] - 1))
    return float(np.average(pair, weights=n[ok]))


# --------------------------------------------------------------------------- defect language

def word_histograms(words: np.ndarray, scores: np.ndarray, n_images: int, k: int, topk: int = 10) -> np.ndarray:
    """(n_images, K) counts of the words assigned to each image's top-k scoring patches."""
    w = words.reshape(n_images, -1)
    s = scores.reshape(n_images, -1)
    topk = min(topk, s.shape[1])
    top = np.argpartition(-s, topk - 1, axis=1)[:, :topk]
    h = np.zeros((n_images, k), np.float32)
    rows = np.repeat(np.arange(n_images), topk)
    np.add.at(h, (rows, w[np.arange(n_images)[:, None], top].ravel()), 1.0)
    return h


def tfidf(h: np.ndarray) -> np.ndarray:
    df = (h > 0).sum(0)
    idf = np.log((1 + len(h)) / (1 + df)) + 1.0
    return l2n(h * idf[None, :])


def loo_knn(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Leave-one-out k-NN accuracy with cosine similarity (rows L2-normalised)."""
    x = l2n(np.asarray(x, np.float32))
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    correct = 0
    for i in range(len(y)):
        lab, cnt = np.unique(y[nn[i]], return_counts=True)
        best = lab[cnt == cnt.max()]
        # ties broken toward the nearest neighbour's label
        pred = y[nn[i, 0]] if y[nn[i, 0]] in best else best[0]
        correct += pred == y[i]
    return float(correct / len(y))


def signature_words(h: np.ndarray, y: np.ndarray) -> dict:
    """Best single word per type: max(share of type images using it - share of other images using it)."""
    use = h > 0
    cover, false = [], []
    per_type = {}
    for t in np.unique(y):
        a, b = use[y == t].mean(0), use[y != t].mean(0)
        w = int(np.argmax(a - b))
        cover.append(a[w])
        false.append(b[w])
        per_type[str(t)] = {"word": w, "cover": float(a[w]), "false": float(b[w])}
    return {"sig_cover": float(np.mean(cover)), "sig_false": float(np.mean(false)), "per_type": per_type}


def chance_accuracy(y: np.ndarray) -> float:
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float((p ** 2).sum())


# --------------------------------------------------------------------------- memories

def memory_words(memory: str, tr_desc: np.ndarray, te_desc: np.ndarray, k: int, seed: int, vocab=None):
    """Word id and distance for train and test patches, for the codebook or the same-size coreset."""
    if memory == "codebook":
        tr_d, tr_w = nearest_word(tr_desc, vocab)
        te_d, te_w = nearest_word(te_desc, vocab)
        return tr_w, te_w, te_d
    if memory == "coreset":
        bank = np.asarray(tr_desc[np.sort(greedy_coreset(tr_desc, k, seed=seed))], np.float32)
        _, tr_w = nn_search(tr_desc, bank)
        te_d, te_w = nn_search(te_desc, bank)
        return tr_w, te_w, te_d
    raise KeyError(memory)


def evaluate(tr_desc, te_desc, n_train: int, n_test: int, side: int, types: np.ndarray, anom: np.ndarray,
             memory: str, k: int, seed: int, topk: int = 10, vocab=None, max_train: int = 300_000) -> dict:
    tr_w, te_w, te_d = memory_words(memory, tr_desc, te_desc, k, seed, vocab)
    # word quality on a whole-image subsample of training patches (keeps grid positions aligned)
    per = side * side
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n_train, min(n_train, max(1, max_train // per)), replace=False))
    sel = (keep[:, None] * per + np.arange(per)[None, :]).ravel()
    nmi, nmi_shuf = position_nmi(tr_w[sel], side, seed)
    coh = cohesion(np.asarray(tr_desc[sel], np.float32), tr_w[sel], k)

    h = word_histograms(te_w, te_d, n_test, k, topk)[anom]
    y = types[anom]
    x = tfidf(h)
    sig = signature_words(h, y)
    feat = topk_feature_descriptor(te_desc, n_test, te_d, topk)[anom]
    return {"position_nmi": nmi, "position_nmi_shuffled": nmi_shuf, "position_nmi_gain": nmi - nmi_shuf,
            "cohesion": coh, "words_used_train": int(len(np.unique(tr_w[sel]))),
            "knn1": loo_knn(x, y, 1), "knn5": loo_knn(x, y, 5), "chance": chance_accuracy(y),
            "feat_knn1": loo_knn(feat, y, 1), "sig_cover": sig["sig_cover"], "sig_false": sig["sig_false"],
            "n_images": int(anom.sum()), "n_types": int(len(np.unique(y))), "signatures": sig["per_type"]}


# --------------------------------------------------------------------------- runner

def run(cfg: dict, progress: bool = True) -> pd.DataFrame:
    data_cfg = dict(cfg["data"])
    cats = data_cfg.get("categories", "all")
    cats = ad_data.categories(data_cfg.get("name", "mvtec")) if cats == "all" else list(cats)
    cache_dir, out_dir = Path(cfg["cache_dir"]), Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{cfg['name']}.jsonl"
    prev = load_results(jsonl)
    done = set(prev["run_id"]) if "run_id" in prev else set()

    ex = cfg["extractor"]
    ex_name, ex_params = ex["name"], ex.get("params", {})
    cache_params = {kk: v for kk, v in ex_params.items() if kk not in RUNTIME_KEYS}
    vocab_cfg = dict(cfg.get("vocab", {}))
    seeds = vocab_cfg.pop("seeds", None) or [0]
    vocab_cfg.pop("seed", None)
    ks = cfg.get("k", [64, 256, 1024])
    memories = cfg.get("memories", list(MEMORIES))
    topk = int(cfg.get("topk", 10))
    normalize = cfg.get("normalize", True)
    drop = set(cfg.get("drop_types", ["combined"]))
    mask_size = data_cfg.get("mask_size", 256)
    env = _env()

    for cat in cats:
        tag = f"{data_cfg.get('name', 'mvtec')}_{cat}_s{data_cfg.get('size', 448)}_m{mask_size}"
        want = [_run_id(cfg["name"], cat, m, k, s) for m in memories for k in ks for s in seeds]
        if all(w in done for w in want):
            continue
        ds = _LazyAD(data_cfg, cat, cache_dir, tag)
        extractor = None

        def extract(images):
            nonlocal extractor
            from .extract import get_extractor
            extractor = extractor or get_extractor(ex_name, **ex_params)
            return extractor(images, progress=progress)

        tr, _ = cache.get_or_compute(cache_dir, f"{tag}_train", ex_name, cache_params,
                                     lambda: extract(ds.split.train_images))
        te, _ = cache.get_or_compute(cache_dir, f"{tag}_test", ex_name, cache_params,
                                     lambda: extract(ds.split.test_images))
        meta = ds.get_meta()
        side = grid_side(te)
        tr_desc, te_desc = _norm(tr.desc, normalize), _norm(te.desc, normalize)
        types = np.array(meta.types)
        anom = (np.asarray(meta.labels) == 1) & ~np.isin(types, list(drop))
        if len(np.unique(types[anom])) < 2:
            continue
        for k in ks:
            for seed in seeds:
                vocab = None
                for m in memories:
                    rid = _run_id(cfg["name"], cat, m, k, seed)
                    if rid in done:
                        continue
                    if m == "codebook" and vocab is None:
                        vkey = {"tag": tag, "extractor": ex_name, "params": cache_params, "normalize": normalize,
                                "k": k, "seed": seed, "vocab": vocab_cfg}
                        vocab, _ = cached_vocabulary(cache_dir, vkey,
                                                     lambda: build_vocabulary(tr_desc, k, seed=seed, **vocab_cfg))
                    r = evaluate(tr_desc, te_desc, tr.n_images, te.n_images, side, types, anom, m, k, seed,
                                 topk, vocab)
                    row = {"run_id": rid, "config": cfg["name"], "category": cat, "extractor": ex_name,
                           "memory": m, "k": k, "vocab_seed": seed, "topk": topk, **r, **env}
                    _append(jsonl, row)
                    if progress:
                        print(f"[{cat:>10}] {m:<8} K={k:<5} s={seed}  posNMI+{r['position_nmi_gain']:.3f} "
                              f"coh={r['cohesion']:.3f}  1-NN {r['knn1']:.2f} (chance {r['chance']:.2f}, "
                              f"feat {r['feat_knn1']:.2f})  sig {r['sig_cover']:.2f}/{r['sig_false']:.2f}")
        del tr, te, tr_desc, te_desc

    df = load_results(jsonl)
    df.drop(columns=["signatures"], errors="ignore").to_csv(out_dir / f"{cfg['name']}.csv", index=False)
    return df


SUMMARY_COLS = ["position_nmi_gain", "cohesion", "knn1", "knn5", "chance", "feat_knn1", "sig_cover", "sig_false"]


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Memory x K: mean over seeds, then categories."""
    cols = [c for c in SUMMARY_COLS if c in df]
    t = df.groupby(["memory", "k", "category"])[cols].mean().groupby(level=[0, 1]).mean()
    return t.sort_index(level=[1, 0])


# --------------------------------------------------------------------------- figure

def _crop(img: np.ndarray, p: int, side: int, ctx: int = 3) -> np.ndarray:
    """Patch p of a side x side grid with ctx patches of context on each side of a square crop."""
    cell = img.shape[0] / side
    r, c = divmod(int(p), side)
    lo_r, hi_r = int(max(0, (r - ctx // 2) * cell)), int(min(img.shape[0], (r + ctx // 2 + 1) * cell))
    lo_c, hi_c = int(max(0, (c - ctx // 2) * cell)), int(min(img.shape[1], (c + ctx // 2 + 1) * cell))
    return img[lo_r:hi_r, lo_c:hi_c]


def defect_vocabulary_figure(train_images, test_images, tr_w: np.ndarray, te_w: np.ndarray, te_d: np.ndarray,
                             types: np.ndarray, anom: np.ndarray, side: int, k: int, topk: int = 10,
                             n_word_examples: int = 5, title: str = ""):
    """One row per defect type: its worst patch in two test images, then normal training patches of the
    type's signature word (what the vocabulary would call it in a report)."""
    import matplotlib.pyplot as plt

    n_test = len(test_images)
    h = word_histograms(te_w, te_d, n_test, k, topk)
    sig = signature_words(h[anom], types[anom])["per_type"]
    per = side * side
    tr_w2 = tr_w.reshape(-1, per)
    te_d2 = te_d.reshape(-1, per)
    rows = sorted(sig)
    ncol = 2 + n_word_examples
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.6 * ncol, 1.7 * len(rows)), squeeze=False)
    for i, t in enumerate(rows):
        w = sig[t]["word"]
        imgs = np.where(anom & (types == t))[0][:2]
        for j in range(2):
            ax = axes[i, j]
            if j < len(imgs):
                ii = imgs[j]
                ax.imshow(_crop(test_images[ii], te_d2[ii].argmax(), side))
            ax.set_xticks([]), ax.set_yticks([])
        axes[i, 0].set_ylabel(f"{t}\nword {w}\n{sig[t]['cover']:.0%} / {sig[t]['false']:.0%}", fontsize=7)
        img_idx, pos = np.where(tr_w2 == w)
        pick = np.linspace(0, len(img_idx) - 1, min(n_word_examples, len(img_idx))).astype(int) if len(img_idx) else []
        for j in range(n_word_examples):
            ax = axes[i, 2 + j]
            if j < len(pick):
                ax.imshow(_crop(train_images[img_idx[pick[j]]], pos[pick[j]], side))
            ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_title("defect", fontsize=8)
    axes[0, 1].set_title("defect", fontsize=8)
    axes[0, 2].set_title("signature word: normal examples", fontsize=8, loc="left")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig
