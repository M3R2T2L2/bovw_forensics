"""On-disk feature cache.

Extraction is the expensive step, so features are computed once per
(dataset, extractor, params) and reused by every vocabulary / encoding run.
Descriptors are stored as float16 .npy (memory-mappable), offsets as int64.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .extract import LocalFeatures


def cache_key(dataset: str, extractor: str, params: dict) -> str:
    blob = json.dumps({"dataset": dataset, "extractor": extractor, "params": params}, sort_keys=True)
    return f"{dataset}__{extractor}__{hashlib.sha1(blob.encode()).hexdigest()[:10]}"


def save(feats: LocalFeatures, path: Path, meta: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "desc.npy", feats.desc.astype(np.float16, copy=False))
    np.save(path / "offsets.npy", feats.offsets)
    if feats.global_ is not None:
        np.save(path / "global.npy", feats.global_.astype(np.float16))
    (path / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))


def load(path: Path, mmap: bool = True) -> LocalFeatures:
    mode = "r" if mmap else None
    desc = np.load(path / "desc.npy", mmap_mode=mode)
    offsets = np.load(path / "offsets.npy")
    g = path / "global.npy"
    global_ = np.load(g).astype(np.float32) if g.exists() else None
    return LocalFeatures(desc=desc, offsets=offsets, global_=global_)


def get_or_compute(root: Path, dataset: str, extractor: str, params: dict,
                   compute: Callable[[], LocalFeatures]) -> tuple[LocalFeatures, dict]:
    """Return cached features, computing and saving them on a miss.

    The returned info dict records whether it was a hit and the extraction time,
    which feeds the compute-budget columns of the results table.
    """
    path = Path(root) / cache_key(dataset, extractor, params)
    if (path / "meta.json").exists():
        meta = json.loads((path / "meta.json").read_text())
        # Load into RAM: encoders re-read descriptors on every run, and Drive-backed
        # memory maps are slow. 8,000 STL-10 images of dense SIFT is ~1.5 GB (float16).
        return load(path, mmap=False), {"cache_hit": True, **meta}
    t0 = time.perf_counter()
    feats = compute()
    secs = time.perf_counter() - t0
    meta = {"dataset": dataset, "extractor": extractor, "params": params,
            "extract_seconds": secs, "n_images": feats.n_images,
            "n_descriptors": int(feats.desc.shape[0]), "dim": feats.dim}
    save(feats, path, meta)
    return feats, {"cache_hit": False, **meta}
