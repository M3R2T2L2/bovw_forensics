"""Anomaly-detection datasets: MVTec AD and a synthetic stand-in for tests.

Both return an `ADSplit`:
  train_images  normal images only (list of HxWx3 uint8)
  test_images   normal + anomalous
  test_labels   0 = good, 1 = anomalous
  test_masks    (N, mask_size, mask_size) uint8 ground truth (all zero for good)
  test_types    defect type per test image ("good" for normal)

MVTec AD is CC BY-NC-SA 4.0 (non-commercial). Accept the licence at
https://www.mvtec.com/company/research/datasets/mvtec-ad before downloading.
Expected layout (the official archive, extracted):
  <root>/<category>/train/good/*.png
  <root>/<category>/test/<defect>/*.png
  <root>/<category>/ground_truth/<defect>/*_mask.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MVTEC_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


VISA_CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
                   "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]

# VisA (Zou et al., ECCV 2022), CC BY 4.0. Official archive and 1-class split
# (split_csv/1cls.csv from github.com/amazon-science/spot-diff, Apache-2.0, vendored).
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SPLIT_CSV = Path(__file__).resolve().parent / "resources" / "visa_1cls.csv"


# 3CAD (Yang et al., AAAI 2025): 3C-product parts from real production lines, MVTec-style layout.
# Official Google Drive archive (English defect names) from github.com/EnquanYang2022/3CAD.
# No licence is stated; treat as research use and confirm with the authors before publishing.
THREECAD_CATEGORIES = ["Aluminum_Camera_Cover", "Aluminum_Ipad", "Aluminum_Middle_Frame", "Aluminum_New_Ipad",
                       "Aluminum_New_Middle_Frame", "Aluminum_Pc", "Copper_Stator", "Iron_Stator"]
THREECAD_GDRIVE_ID = "1BIX0H8TZp0wmrAnXPw8_aCAIX1j1Fzwz"
# (train images, test images) per category, from the 3CAD README, for a load sanity check.
THREECAD_COUNTS = {"Aluminum_Camera_Cover": (784, 1446), "Aluminum_Ipad": (2096, 2047),
                   "Aluminum_Middle_Frame": (1548, 1479), "Aluminum_New_Middle_Frame": (1072, 1406),
                   "Aluminum_New_Ipad": (2233, 4936), "Aluminum_Pc": (1698, 3161),
                   "Copper_Stator": (409, 959), "Iron_Stator": (653, 1112)}


def count_mvtec_style(root: str, category: str) -> tuple[int, int]:
    base = Path(root) / category
    n_test = sum(len(_images_in(d)) for d in (base / "test").iterdir() if d.is_dir())
    return len(_images_in(base / "train" / "good")), n_test
_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def categories(dataset: str) -> list:
    return {"mvtec": MVTEC_CATEGORIES, "visa": VISA_CATEGORIES, "3cad": THREECAD_CATEGORIES}[dataset]


def find_root(path: str, marker: str) -> str:
    """The directory that directly contains `marker` (a category folder), searching two levels down."""
    p = Path(path)
    cands = [p] + sorted(c for c in p.iterdir() if c.is_dir()) if p.is_dir() else []
    cands += [g for c in cands[1:] for g in sorted(c.iterdir()) if g.is_dir()]
    for c in cands:
        if (c / marker).is_dir():
            return str(c)
    raise FileNotFoundError(f"No folder containing '{marker}' under {path}")


def _images_in(d: Path) -> list:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXT) if d.is_dir() else []


def _find_mask(gt_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}_mask", stem):
        for ext in (".png", ".bmp", ".jpg", ".tif", ".PNG"):
            q = gt_dir / f"{name}{ext}"
            if q.exists():
                return q
    return None


def load_mvtec_style(root: str, category: str, size: int = 448, mask_size: int = 256) -> ADSplit:
    """MVTec-layout loader tolerant of other image formats and mask names (used for 3CAD).

    <root>/<category>/train/good/*, <root>/<category>/test/<type>/*,
    <root>/<category>/ground_truth/<type>/<stem>[_mask].<ext>; any non-zero mask pixel = anomaly.
    Images are decoded in parallel (half-resolution JPEG decode when large enough).
    """
    from concurrent.futures import ThreadPoolExecutor

    base = Path(root) / category
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found")
    train_paths = _images_in(base / "train" / "good")
    tests = [(d.name, p) for d in sorted(x for x in (base / "test").iterdir() if x.is_dir()) for p in _images_in(d)]
    if not train_paths or not tests:
        raise FileNotFoundError(f"{base}: expected train/good and test/<type> image folders")
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as ex:
        train = list(ex.map(lambda q: _read_rgb(q, size, fast=True), train_paths))
        imgs = list(ex.map(lambda t: _read_rgb(t[1], size, fast=True), tests))
    labels, masks, types, missing = [], [], [], []
    for t, p in tests:
        types.append(t)
        if t == "good":
            labels.append(0)
            masks.append(np.zeros((mask_size, mask_size), np.uint8))
            continue
        labels.append(1)
        mp = _find_mask(base / "ground_truth" / t, p.stem)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp else None
        if m is None:
            missing.append(f"{t}/{p.name}")
            m = np.zeros((mask_size, mask_size), np.uint8)
        masks.append((cv2.resize(m, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))
    if missing:
        raise FileNotFoundError(f"{len(missing)} anomalous images have no mask in {base / 'ground_truth'}, "
                                f"e.g. {missing[:3]}")
    return ADSplit(train, imgs, np.array(labels), np.stack(masks), types)


@dataclass
class ADSplit:
    train_images: list
    test_images: list
    test_labels: np.ndarray
    test_masks: np.ndarray
    test_types: list


def _read_rgb(path: Path, size: int, fast: bool = False) -> np.ndarray:
    """Read, convert to RGB and resize to size x size.

    fast=True decodes JPEGs at half resolution (much quicker for VisA's ~1.5k px
    images) and falls back to a full decode if that would be smaller than `size`.
    """
    img = None
    if fast:
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
        if img is not None and min(img.shape[:2]) < size:
            img = None
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_mvtec(root: str, category: str, size: int = 448, mask_size: int = 256) -> ADSplit:
    base = Path(root) / category
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; see the module docstring for the expected layout.")
    train = [_read_rgb(p, size) for p in sorted((base / "train" / "good").glob("*.png"))]
    imgs, labels, masks, types = [], [], [], []
    for d in sorted(p for p in (base / "test").iterdir() if p.is_dir()):
        for p in sorted(d.glob("*.png")):
            imgs.append(_read_rgb(p, size))
            types.append(d.name)
            if d.name == "good":
                labels.append(0)
                masks.append(np.zeros((mask_size, mask_size), np.uint8))
            else:
                labels.append(1)
                m = cv2.imread(str(base / "ground_truth" / d.name / f"{p.stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise FileNotFoundError(f"mask for {p}")
                masks.append((cv2.resize(m, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST) > 127)
                             .astype(np.uint8))
    return ADSplit(train, imgs, np.array(labels), np.stack(masks), types)


def find_visa_root(path: str) -> str:
    """The directory that holds candle/Data/..., wherever the archive put it."""
    p = Path(path)
    for cand in [p, *sorted(p.iterdir())] if p.is_dir() else []:
        if (cand / "candle" / "Data" / "Images").is_dir():
            return str(cand)
    raise FileNotFoundError(f"No VisA layout (candle/Data/Images) under {path}")


def load_visa(root: str, category: str, size: int = 448, mask_size: int = 256,
              split_csv: str | Path = VISA_SPLIT_CSV) -> ADSplit:
    """VisA with the official 1-class split: train = normal only; test = normal + anomalous.

    Masks are binarised (any non-zero pixel = anomaly), as in the official prepare_data.py.
    All anomalies share the type "anomaly" (the 1-class split has no defect-type labels).
    """
    import csv

    base = Path(root)
    if not (base / category).is_dir():
        raise FileNotFoundError(f"{base / category} not found")
    train, imgs, labels, masks, types = [], [], [], [], []
    with open(split_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["object"] == category]
    if not rows:
        raise KeyError(f"'{category}' not in {split_csv}")
    from concurrent.futures import ThreadPoolExecutor

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as ex:  # cv2 decoding releases the GIL
        decoded = list(ex.map(lambda r: _read_rgb(base / r["image"], size, fast=True), rows))
    for r, img in zip(rows, decoded):
        if r["split"] == "train":
            train.append(img)
            continue
        imgs.append(img)
        if r["label"] == "normal":
            labels.append(0)
            types.append("good")
            masks.append(np.zeros((mask_size, mask_size), np.uint8))
        else:
            labels.append(1)
            types.append("anomaly")
            m = cv2.imread(str(base / r["mask"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"mask {r['mask']}")
            masks.append((cv2.resize(m, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST) > 0)
                         .astype(np.uint8))
    return ADSplit(train, imgs, np.array(labels), np.stack(masks), types)


def load_synthetic_ad(n_train: int = 40, n_good: int = 20, n_bad: int = 20, size: int = 128,
                      mask_size: int = 64, seed: int = 0) -> ADSplit:
    """Stripe texture = normal; a square of noise or rotated stripes = defect."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    def texture(theta, phase):
        return np.sin(0.35 * (np.cos(theta) * xx + np.sin(theta) * yy) + phase)

    def normal():
        return texture(0.3 + rng.normal(0, 0.03), rng.uniform(0, 2 * np.pi))

    def to_img(t):
        g = (t * 90 + 128 + rng.normal(0, 8, t.shape)).clip(0, 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)

    train = [to_img(normal()) for _ in range(n_train)]
    imgs, labels, masks, types = [], [], [], []
    for _ in range(n_good):
        imgs.append(to_img(normal()))
        labels.append(0)
        masks.append(np.zeros((mask_size, mask_size), np.uint8))
        types.append("good")
    for i in range(n_bad):
        t = normal()
        s = int(rng.integers(size // 5, size // 3))
        y0, x0 = rng.integers(0, size - s, 2)
        kind = "noise" if i % 2 else "rotated"
        patch = rng.normal(0, 1, (s, s)) if kind == "noise" else texture(1.6, 0.0)[y0:y0 + s, x0:x0 + s]
        t[y0:y0 + s, x0:x0 + s] = patch
        full = np.zeros((size, size), np.uint8)
        full[y0:y0 + s, x0:x0 + s] = 1
        imgs.append(to_img(t))
        labels.append(1)
        masks.append(cv2.resize(full, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST))
        types.append(kind)
    return ADSplit(train, imgs, np.array(labels), np.stack(masks), types)


# Official archive; URL as used by anomalib (main branch, 2026), which tracks MVTec's share links.
MVTEC_URL = ("https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/"
             "download/420938113-1629960298/mvtec_anomaly_detection.tar.xz")
MVTEC_SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"


def sha256sum(path, chunk: int = 1 << 24) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_archive(path: str, url: str = MVTEC_URL, sha256: str | None = MVTEC_SHA256,
                   stub_bytes: int = 100_000_000) -> str:
    """Download `url` to `path` only if `path` is missing or corrupt; verify the checksum.

    Downloads go to `<path>.part` and are renamed only after the checksum
    matches, so a failed or partial download never masquerades as the archive.
    """
    import os
    import shutil
    import subprocess
    import urllib.request

    if os.path.isfile(path):
        if sha256 is None or sha256sum(path) == sha256:
            return path
        if os.path.getsize(path) >= stub_bytes:
            # Never delete a real-sized file (e.g. a manual upload): let the user decide.
            raise RuntimeError(f"{path} does not match the expected SHA-256. If you uploaded it yourself and "
                               "trust it, call ensure_archive(path, sha256=None); otherwise delete it and retry.")
        os.remove(path)  # small stub left by an earlier failed download
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    if os.path.exists(part):
        os.remove(part)
    try:
        if shutil.which("wget") and not url.startswith("file:"):
            subprocess.run(["wget", "-q", "--show-progress", "-O", part, url], check=True)
        else:
            urllib.request.urlretrieve(url, part)
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        raise RuntimeError(
            f"Download failed ({e}). The link may have moved again: download "
            "mvtec_anomaly_detection.tar.xz from https://www.mvtec.com/company/research/datasets/mvtec-ad "
            f"and upload it to {path}.") from e
    got = sha256sum(part) if sha256 else None
    if sha256 and got != sha256:
        os.remove(part)
        raise RuntimeError(f"Checksum mismatch (got {got[:12]}..., expected {sha256[:12]}...). "
                           f"Download the archive manually and upload it to {path}.")
    os.replace(part, path)
    return path
