"""Handcrafted descriptors: SIFT (sparse or dense grid) and ORB.

Small images (e.g. STL-10 at 96 px) give very few detected keypoints, so
images are upscaled to `resize` first and dense SIFT is offered as the
standard BoVW choice for small images.

ORB descriptors are binary (256 bits). They are unpacked to 0/1 floats so the
same Euclidean k-means can be used; this approximates Hamming k-majority
clustering and is noted as a limitation in P0.

Extraction runs on a thread pool (OpenCV releases the GIL), with one OpenCV
object per thread. `n_jobs` defaults to all CPUs; it changes speed only, so
it is excluded from the feature-cache key.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm.auto import tqdm

from . import LocalFeatures


def _to_gray(img: np.ndarray, resize: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    if resize and gray.shape[0] != resize:
        gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_LINEAR)
    return gray


class _Parallel:
    """Maps `self._one` over images on a thread pool, preserving order."""

    dim: int
    name: str

    def __init__(self, n_jobs: int | None = None):
        self.n_jobs = max(1, n_jobs or os.cpu_count() or 1)
        self._local = threading.local()

    def _cv(self):
        if not hasattr(self._local, "obj"):
            self._local.obj = self._make()
        return self._local.obj

    def __call__(self, images, progress: bool = True) -> LocalFeatures:
        cv2.setNumThreads(1)  # avoid oversubscription: parallelism is across images
        try:
            with ThreadPoolExecutor(self.n_jobs) as pool:
                it = pool.map(self._one, images)
                out = list(tqdm(it, total=len(images), desc=f"{self.name} x{self.n_jobs}", disable=not progress))
        finally:
            cv2.setNumThreads(-1)
        # float16 halves peak RAM (8,000 STL-10 images of dense SIFT: ~1.5 GB instead of ~3 GB)
        return LocalFeatures.from_list(out, dim=self.dim, dtype=np.float16)


class SIFTExtractor(_Parallel):
    dim, name = 128, "sift"

    def __init__(self, dense: bool = True, resize: int = 224, step: int = 8, size: int = 16,
                 max_kp: int = 500, root_sift: bool = True, n_jobs: int | None = None):
        super().__init__(n_jobs)
        self.dense, self.resize, self.step, self.size = dense, resize, step, size
        self.max_kp, self.root_sift = max_kp, root_sift
        self.name = "dense_sift" if dense else "sift"

    def _make(self):
        return cv2.SIFT_create(nfeatures=self.max_kp)

    def _grid(self, h: int, w: int):
        half = self.size // 2
        return [cv2.KeyPoint(float(x), float(y), float(self.size))
                for y in range(half, h - half + 1, self.step)
                for x in range(half, w - half + 1, self.step)]

    def _one(self, img: np.ndarray) -> np.ndarray:
        gray = _to_gray(img, self.resize)
        sift = self._cv()
        if self.dense:
            _, d = sift.compute(gray, self._grid(*gray.shape))
        else:
            _, d = sift.detectAndCompute(gray, None)
        if d is None:
            return np.zeros((0, 128), np.float16)
        d = d.astype(np.float32)
        if self.root_sift:  # Arandjelovic & Zisserman 2012
            d /= d.sum(axis=1, keepdims=True) + 1e-7
            d = np.sqrt(d)
        return d.astype(np.float16)


class ORBExtractor(_Parallel):
    dim, name = 256, "orb"

    def __init__(self, resize: int = 224, max_kp: int = 500, fast_threshold: int = 5, n_jobs: int | None = None):
        super().__init__(n_jobs)
        self.resize, self.max_kp, self.fast_threshold = resize, max_kp, fast_threshold

    def _make(self):
        return cv2.ORB_create(nfeatures=self.max_kp, fastThreshold=self.fast_threshold, edgeThreshold=15)

    def _one(self, img: np.ndarray) -> np.ndarray:
        gray = _to_gray(img, self.resize)
        _, d = self._cv().detectAndCompute(gray, None)
        if d is None:
            return np.zeros((0, 256), np.float16)
        return np.unpackbits(d, axis=1).astype(np.float16)
