"""Unsupervised defect-type discovery (P0 anomaly follow-up, feeds P3).

Question: after the codebook flags suspicious patches, can we group anomalous
images by *kind* of defect without any labels? Ground truth is MVTec AD's
defect-type folder names (e.g. bottle: broken_large, broken_small, contamination).

Per category, fit on normal training images only:
  1. build (or load from cache) the K-word vocabulary;
  2. score every test patch by its distance to the nearest word;
  3. describe each test image by what its most suspicious patches look like;
  4. cluster the images and compare clusters with defect types (NMI, ARI, accuracy).

Image descriptors (compared):
  topk_resid  mean residual (patch - nearest word) of the top-k patches, L2-normalised:
              "what the normal vocabulary cannot explain" (VLAD-style)
  topk_feat   mean raw feature of the top-k patches, L2-normalised
  cls         global CLS embedding (baseline, ignores where the defect is)
  mask_feat   mean feature of patches inside the ground-truth mask (upper bound;
              uses labels, reported only as a reference)

Settings:
  oracle      cluster the truly anomalous images, k = number of defect types
              (isolates the discovery question from detection errors)
  detected    cluster the N highest-scoring test images (N = number of anomalous
              images), k = number of types + 1; normal images that slip in are
              labelled "good" and count as their own class

Chance level: NMI of the same clustering against shuffled labels (mean of 20).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import ad_data, cache
from .anomaly import _LazyAD, _norm, cached_vocabulary, grid_side, nearest_word, topk_image_scores
from .eval.clustering import cluster_accuracy
from .sweep import RUNTIME_KEYS, _append, _env, _run_id, load_results
from .vocab import build_vocabulary, l2n

DESCRIPTORS = ("topk_resid", "topk_feat", "cls", "mask_feat")


def image_descriptors(desc: np.ndarray, n_images: int, dist: np.ndarray, resid: np.ndarray,
                      k: int, masks: np.ndarray | None, side: int, global_: np.ndarray | None) -> dict:
    """Per-image descriptors from per-patch features, distances and residuals."""
    d = desc.reshape(n_images, side * side, -1)
    r = resid.reshape(n_images, side * side, -1)
    s = dist.reshape(n_images, side * side)
    k = min(k, side * side)
    top = np.argpartition(-s, k - 1, axis=1)[:, :k]
    rows = np.arange(n_images)[:, None]
    out = {"topk_feat": l2n(d[rows, top].mean(1)), "topk_resid": l2n(r[rows, top].mean(1))}
    if global_ is not None:
        out["cls"] = l2n(np.asarray(global_, np.float32))
    if masks is not None:
        mf = np.empty((n_images, d.shape[2]), np.float32)
        for i in range(n_images):
            g = cv2.resize(masks[i].astype(np.float32), (side, side), interpolation=cv2.INTER_AREA).reshape(-1)
            sel = g >= 0.25
            mf[i] = d[i, sel].mean(0) if sel.any() else d[i, s[i].argmax()]
        out["mask_feat"] = l2n(mf)
    return out


def cluster_scores(x: np.ndarray, y: list, n_clusters: int, algo: str, seeds=(0, 1, 2, 3, 4),
                   n_perm: int = 20) -> dict:
    """NMI / ARI / accuracy of clustering x into n_clusters vs labels y, with a shuffled-label chance NMI."""
    y = np.asarray(y)
    if n_clusters < 2 or len(y) <= n_clusters:
        return {}
    preds = []
    if algo == "kmeans":
        preds = [KMeans(n_clusters=n_clusters, n_init=10, random_state=s).fit_predict(x) for s in seeds]
    elif algo == "ward":
        preds = [AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(x)]
    else:
        raise KeyError(algo)
    rng = np.random.default_rng(0)
    nmi, ari, acc, chance = [], [], [], []
    for p in preds:
        nmi.append(normalized_mutual_info_score(y, p))
        ari.append(adjusted_rand_score(y, p))
        acc.append(cluster_accuracy(y, p))
        chance.append(np.mean([normalized_mutual_info_score(rng.permutation(y), p) for _ in range(n_perm)]))
    return {"nmi": float(np.mean(nmi)), "nmi_std": float(np.std(nmi)), "ari": float(np.mean(ari)),
            "acc": float(np.mean(acc)), "nmi_chance": float(np.mean(chance)),
            "nmi_above_chance": float(np.mean(nmi) - np.mean(chance))}


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
    cache_params = {k: v for k, v in ex_params.items() if k not in RUNTIME_KEYS}
    vocab_cfg = dict(cfg.get("vocab", {}))
    seeds = vocab_cfg.pop("seeds", None) or [0]
    vocab_cfg.pop("seed", None)
    ks = cfg.get("k", [1024])
    topk = int(cfg.get("topk", 10))
    normalize = cfg.get("normalize", True)
    drop = set(cfg.get("drop_types", []))
    algos = cfg.get("algorithms", ["kmeans", "ward"])
    settings = cfg.get("settings", ["oracle", "detected"])
    mask_size = data_cfg.get("mask_size", 256)
    env = _env()

    for cat in cats:
        tag = f"{data_cfg.get('name', 'mvtec')}_{cat}_s{data_cfg.get('size', 448)}_m{mask_size}"
        ds = _LazyAD(data_cfg, cat, cache_dir, tag)
        want = [_run_id(cfg["name"], cat, k, s, st, dname, a) for k in ks for s in seeds for st in settings
                for dname in DESCRIPTORS for a in algos]
        if all(w in done for w in want):
            continue

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
        labels = np.asarray(meta.labels)

        for k in ks:
            for seed in seeds:
                vkey = {"tag": tag, "extractor": ex_name, "params": cache_params, "normalize": normalize,
                        "k": k, "seed": seed, "vocab": vocab_cfg}
                vocab, _ = cached_vocabulary(cache_dir, vkey,
                                             lambda: build_vocabulary(tr_desc, k, seed=seed, **vocab_cfg))
                dist, word = nearest_word(te_desc, vocab)
                resid = vocab.transform(te_desc) - vocab.centers[word]
                img_score = topk_image_scores(dist, te.n_images, topk)
                desc_all = image_descriptors(te_desc, te.n_images, dist, resid, topk, meta.masks, side, te.global_)

                anom = (labels == 1) & ~np.isin(types, list(drop))
                n_types = len(set(types[anom]))
                n_flag = int(anom.sum())
                flagged = np.argsort(-img_score)[:n_flag]
                sel_by = {"oracle": (np.where(anom)[0], n_types),
                          "detected": (flagged, n_types + 1)}
                for st in settings:
                    idx, n_cl = sel_by[st]
                    idx = idx[~np.isin(types[idx], list(drop))]
                    y = types[idx]
                    for dname, x in desc_all.items():
                        for a in algos:
                            rid = _run_id(cfg["name"], cat, k, seed, st, dname, a)
                            if rid in done:
                                continue
                            sc = cluster_scores(x[idx], list(y), n_cl, a)
                            if not sc:
                                continue
                            row = {"run_id": rid, "config": cfg["name"], "category": cat, "extractor": ex_name,
                                   "k": k, "vocab_seed": seed, "setting": st, "descriptor": dname, "algorithm": a,
                                   "n_images": int(len(idx)), "n_types": n_types, "n_clusters": n_cl,
                                   "n_good_in_set": int((y == "good").sum()), "topk": topk, **sc, **env}
                            _append(jsonl, row)
                            if progress:
                                print(f"[{cat:>10}] K={k} s={seed} {st:<8} {dname:<10} {a:<6} "
                                      f"NMI={sc['nmi']:.3f} (chance {sc['nmi_chance']:.3f})  ARI={sc['ari']:.3f}")
                del vocab, dist, word, resid, desc_all
        del tr, te, tr_desc, te_desc

    df = load_results(jsonl)
    df.to_csv(out_dir / f"{cfg['name']}.csv", index=False)
    return df


def summary(df: pd.DataFrame, metric: str = "nmi_above_chance") -> pd.DataFrame:
    """Setting x descriptor x algorithm rows; mean over categories and seeds."""
    t = df.groupby(["setting", "algorithm", "descriptor"])[["nmi", "nmi_chance", "nmi_above_chance", "ari", "acc"]].mean()
    return t.sort_values(["setting", "algorithm", metric], ascending=[True, True, False])


def cluster_category(cfg: dict, category: str, descriptor: str = "topk_resid", k: int = 1024, seed: int = 0,
                     algo: str = "kmeans"):
    """Cluster one category's truly anomalous images; returns (image indices, cluster ids, defect types)."""
    data_cfg = dict(cfg["data"])
    mask_size = data_cfg.get("mask_size", 256)
    tag = f"{data_cfg.get('name', 'mvtec')}_{category}_s{data_cfg.get('size', 448)}_m{mask_size}"
    cache_dir = Path(cfg["cache_dir"])
    ds = _LazyAD(data_cfg, category, cache_dir, tag)
    ex = cfg["extractor"]
    cache_params = {kk: v for kk, v in ex.get("params", {}).items() if kk not in RUNTIME_KEYS}
    tr, _ = cache.get_or_compute(cache_dir, f"{tag}_train", ex["name"], cache_params, lambda: None)
    te, _ = cache.get_or_compute(cache_dir, f"{tag}_test", ex["name"], cache_params, lambda: None)
    meta = ds.get_meta()
    normalize = cfg.get("normalize", True)
    vocab_cfg = dict(cfg.get("vocab", {}))
    vocab_cfg.pop("seeds", None)
    vocab_cfg.pop("seed", None)
    tr_desc, te_desc = _norm(tr.desc, normalize), _norm(te.desc, normalize)
    vkey = {"tag": tag, "extractor": ex["name"], "params": cache_params, "normalize": normalize,
            "k": k, "seed": seed, "vocab": vocab_cfg}
    vocab, _ = cached_vocabulary(cache_dir, vkey, lambda: build_vocabulary(tr_desc, k, seed=seed, **vocab_cfg))
    dist, word = nearest_word(te_desc, vocab)
    resid = vocab.transform(te_desc) - vocab.centers[word]
    side = grid_side(te)
    x = image_descriptors(te_desc, te.n_images, dist, resid, int(cfg.get("topk", 10)), meta.masks, side,
                          te.global_)[descriptor]
    types = np.array(meta.types)
    drop = set(cfg.get("drop_types", []))
    idx = np.where((np.asarray(meta.labels) == 1) & ~np.isin(types, list(drop)))[0]
    n = len(set(types[idx]))
    if algo == "kmeans":
        pred = KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(x[idx])
    else:
        pred = AgglomerativeClustering(n_clusters=n, linkage="ward").fit_predict(x[idx])
    return idx, pred, types[idx]
