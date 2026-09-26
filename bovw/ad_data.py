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

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MVTEC_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


@dataclass
class ADSplit:
    train_images: list
    test_images: list
    test_labels: np.ndarray
    test_masks: np.ndarray
    test_types: list


def _read_rgb(path: Path, size: int) -> np.ndarray:
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


def ensure_archive(path: str, url: str = MVTEC_URL, sha256: str = MVTEC_SHA256) -> str:
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
        os.remove(path)  # stale or partial file from an earlier failed attempt
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
