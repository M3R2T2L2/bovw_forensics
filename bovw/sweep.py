"""Config-driven sweep: extractor x vocabulary size K x assignment method.

One CSV row per run, written as soon as the run finishes, so a Colab
disconnect loses at most the run in progress. Re-running the same config
skips rows already in the CSV.
"""

import json
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


def _append(csv: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    df.to_csv(csv, mode="a", header=not csv.exists(), index=False)


def run(cfg: dict, progress: bool = True) -> pd.DataFrame:
    ds_cfg = dict(cfg["dataset"])
    ds_name = ds_cfg.pop("name")
    cache_dir = Path(cfg["cache_dir"])
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{cfg['name']}.csv"
    done = set(pd.read_csv(csv)["run_id"]) if csv.exists() else set()

    images, labels = data.load(ds_name, **ds_cfg)
    ds_tag = f"{ds_name}_" + "_".join(f"{k}{v}" for k, v in sorted(ds_cfg.items()) if k != "root")
    eval_cfg = cfg.get("eval", {}).get("clustering", {})
    ks = cfg["vocab"]["k"]
    methods = cfg["assignments"]
    vlad_max_k = cfg.get("vlad_max_k", 256)

    for ex in cfg["extractors"]:
        ex_name, ex_params = ex["name"], ex.get("params", {})
        vcfg = {**{k: v for k, v in cfg["vocab"].items() if k != "k"}, **ex.get("vocab", {})}

        feats, info = cache.get_or_compute(
            cache_dir, ds_tag, ex_name, ex_params,
            lambda: get_extractor(ex_name, **ex_params)(images, progress=progress))
        base = {"config": cfg["name"], "dataset": ds_tag, "extractor": ex_name,
                "n_images": feats.n_images, "desc_dim": feats.dim,
                "desc_per_image": float(feats.counts().mean()),
                "extract_seconds": info.get("extract_seconds")}

        if cfg.get("global_baseline", True) and feats.global_ is not None:
            rid = _run_id(cfg["name"], ex_name, "global", 0)
            if rid not in done:
                row = {"run_id": rid, **base, "assignment": "global", "k": 0, "enc_dim": feats.global_.shape[1]}
                with budget(row, "eval"):
                    row.update(evaluate_clustering(feats.global_, labels, **eval_cfg))
                _append(csv, row)
                if progress:
                    print(f"[{ex_name:>12}] global          NMI={row['nmi']:.3f}")

        for k in ks:
            todo = [m for m in methods if not (m == "vlad" and k > vlad_max_k)
                    and _run_id(cfg["name"], ex_name, m, k) not in done]
            if not todo:
                continue
            vrow: dict = {}
            with budget(vrow, "vocab"):
                vocab = build_vocabulary(feats.desc, k, **vcfg)
            for m in todo:
                row = {"run_id": _run_id(cfg["name"], ex_name, m, k), **base, "assignment": m, "k": k,
                       **vrow, "empty_words": vocab.info["empty_words"],
                       "vocab_params": json.dumps(vcfg, sort_keys=True)}
                with budget(row, "encode"):
                    x = encode(m, feats, vocab, **cfg.get("encode", {}).get(m, {}))
                row["enc_dim"] = x.shape[1]
                with budget(row, "eval"):
                    row.update(evaluate_clustering(x, labels, **eval_cfg))
                _append(csv, row)
                if progress:
                    print(f"[{ex_name:>12}] {m:<5} K={k:<5}  NMI={row['nmi']:.3f}  ACC={row['acc']:.3f}")

    return pd.read_csv(csv)


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
