"""Config-driven sweep: extractor x vocabulary size K x assignment method.

Each run is appended to `<name>.jsonl` as soon as it finishes, so a Colab
disconnect loses at most the run in progress, and rows with different
columns (e.g. the global baseline vs. codebook runs) never collide. A clean
`<name>.csv` is rewritten from the JSONL at the end of every sweep.
Re-running the same config skips runs already recorded.

Images are loaded only when some extractor's features are not cached yet;
labels are cached next to the features, so a fresh Colab runtime with a warm
Drive cache skips the dataset download entirely.
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import cache, data
from .encode import encode
from .eval import evaluate_clustering
from .extract import get_extractor
from .timing import budget
from .vocab import build_vocabulary


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _run_id(*parts) -> str:
    return "|".join(str(p) for p in parts)


def _append(jsonl: Path, row: dict) -> None:
    with jsonl.open("a") as f:
        f.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def load_results(jsonl: Path) -> pd.DataFrame:
    """Read results; tolerates a truncated last line from a killed runtime."""
    rows = []
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


class _LazyDataset:
    """Loads images on first use; labels come from cache when available."""

    def __init__(self, name: str, kwargs: dict, cache_dir: Path, tag: str):
        self.name, self.kwargs = name, kwargs
        self.labels_path = cache_dir / f"{tag}__labels.npy"
        self._images = None
        self._labels = np.load(self.labels_path) if self.labels_path.exists() else None

    def _load(self):
        self._images, labels = data.load(self.name, **self.kwargs)
        if self._labels is not None and not np.array_equal(self._labels, labels):
            raise RuntimeError(f"Cached labels at {self.labels_path} do not match the dataset; delete them.")
        self._labels = labels
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.labels_path, labels)

    @property
    def images(self):
        if self._images is None:
            self._load()
        return self._images

    @property
    def labels(self):
        if self._labels is None:
            self._load()
        return self._labels


# Parameters that change speed but not the features; excluded from the cache key.
RUNTIME_KEYS = {"n_jobs", "batch_size", "device", "progress"}


def expand_grid(params: dict) -> list[dict]:
    """{'knn': [3, 5], 'sigma_scale': 1.0} -> [{'knn': 3, 'sigma_scale': 1.0}, {'knn': 5, ...}]"""
    keys = sorted(params)
    vals = [v if isinstance(v, list) else [v] for v in (params[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)] or [{}]


def variant_label(params: dict) -> str:
    return ",".join(f"{k}={params[k]}" for k in sorted(params))


def _env() -> dict:
    info = {"n_cpu": os.cpu_count()}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run(cfg: dict, progress: bool = True) -> pd.DataFrame:
    """Run the grid: extractor x K x vocab seed x assignment x encode-param variant."""
    ds_cfg = dict(cfg["dataset"])
    ds_name = ds_cfg.pop("name")
    cache_dir = Path(cfg["cache_dir"])
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{cfg['name']}.jsonl"
    prev = load_results(jsonl)
    done = set(prev["run_id"]) if "run_id" in prev else set()

    ds_tag = f"{ds_name}_" + "_".join(f"{k}{v}" for k, v in sorted(ds_cfg.items()) if k != "root")
    ds = _LazyDataset(ds_name, ds_cfg, cache_dir, ds_tag)
    eval_cfg = cfg.get("eval", {}).get("clustering", {})
    vocab_cfg = dict(cfg["vocab"])
    ks = vocab_cfg.pop("k")
    seeds = vocab_cfg.pop("seeds", None) or [vocab_cfg.get("seed", 0)]
    vocab_cfg.pop("seed", None)
    methods = cfg["assignments"]
    vlad_max_k = cfg.get("vlad_max_k", 256)
    env = _env()

    for ex in cfg["extractors"]:
        ex_name, ex_params = ex["name"], ex.get("params", {})
        cache_params = {k: v for k, v in ex_params.items() if k not in RUNTIME_KEYS}
        vcfg = {**vocab_cfg, **ex.get("vocab", {})}
        variants = {m: expand_grid({**cfg.get("encode", {}).get(m, {}), **ex.get("encode", {}).get(m, {})})
                    for m in methods}

        feats, info = cache.get_or_compute(
            cache_dir, ds_tag, ex_name, cache_params,
            lambda: get_extractor(ex_name, **ex_params)(ds.images, progress=progress))
        labels = ds.labels
        base = {"config": cfg["name"], "dataset": ds_tag, "extractor": ex_name,
                "n_images": feats.n_images, "desc_dim": feats.dim,
                "desc_per_image": float(feats.counts().mean()),
                "extract_seconds": info.get("extract_seconds"), **env}

        if cfg.get("global_baseline", True) and feats.global_ is not None:
            rid = _run_id(cfg["name"], ex_name, "global", 0)
            if rid not in done:
                row = {"run_id": rid, **base, "assignment": "global", "variant": "", "k": 0,
                       "vocab_seed": -1, "enc_dim": feats.global_.shape[1]}
                with budget(row, "eval"):
                    row.update(evaluate_clustering(feats.global_, labels, **eval_cfg))
                _append(jsonl, row)
                if progress:
                    print(f"[{ex_name:>12}] global                      NMI={row['nmi']:.3f}")

        for k in ks:
            for seed in seeds:
                todo = [(m, p) for m in methods if not (m == "vlad" and k > vlad_max_k)
                        for p in variants[m]
                        if _run_id(cfg["name"], ex_name, m, k, variant_label(p), seed) not in done]
                if not todo:
                    continue
                vrow: dict = {}
                with budget(vrow, "vocab"):
                    vocab = build_vocabulary(feats.desc, k, seed=seed, **vcfg)
                for m, p in todo:
                    var = variant_label(p)
                    row = {"run_id": _run_id(cfg["name"], ex_name, m, k, var, seed), **base,
                           "assignment": m, "variant": var, "encode_params": json.dumps(p, sort_keys=True),
                           "k": k, "vocab_seed": seed, **vrow, "empty_words": vocab.info["empty_words"],
                           "vocab_params": json.dumps(vcfg, sort_keys=True)}
                    with budget(row, "encode"):
                        x = encode(m, feats, vocab, **p)
                    row["enc_dim"] = x.shape[1]
                    with budget(row, "eval"):
                        row.update(evaluate_clustering(x, labels, **eval_cfg))
                    _append(jsonl, row)
                    if progress:
                        print(f"[{ex_name:>12}] {m:<5} K={k:<5} s={seed} {var:<22} "
                              f"NMI={row['nmi']:.3f}  ACC={row['acc']:.3f}")

    df = load_results(jsonl)
    df.to_csv(out_dir / f"{cfg['name']}.csv", index=False)  # clean, rectangular copy
    return df


def select_best_variant(df: pd.DataFrame, method: str = "soft", metric: str = "nmi") -> dict:
    """Per extractor, the encode params with the best mean `metric` across K and seeds.

    Use on a TUNING split only; the selection is then fixed for the reported split.
    """
    d = df[df["assignment"] == method]
    best = {}
    for ex, g in d.groupby("extractor"):
        means = g.groupby("encode_params")[metric].mean()
        best[ex] = json.loads(means.idxmax())
    return best


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run a BoVW sweep from a YAML config.")
    p.add_argument("config")
    p.add_argument("--cache-dir")
    p.add_argument("--results-dir")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.cache_dir:
        cfg["cache_dir"] = a.cache_dir
    if a.results_dir:
        cfg["results_dir"] = a.results_dir
    df = run(cfg)
    cols = ["extractor", "assignment", "k", "nmi", "acc", "encode_seconds"]
    print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    main()
