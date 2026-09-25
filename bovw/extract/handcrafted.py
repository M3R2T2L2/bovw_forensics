"""Handcrafted descriptors: SIFT (sparse or dense grid) and ORB.

Small images (e.g. STL-10 at 96 px) give very few detected keypoints, so
images are upscaled to `resize` first and dense SIFT is offered as the
standard BoVW choice for small images.

ORB descriptors are binary (256 bits). They are unpacked to 0/1 floats so the
same Euclidean k-means can be used; this approximates Hamming k-majority
clustering and is noted as a limitation in P0.
"""

import cv2
import numpy as np
from tqdm.auto import tqdm

from . import LocalFeatures


def _to_gray(img: np.ndarray, resize: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    if resize and gray.shape[0] != resize:
        gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_LINEAR)
    return gray


class SIFTExtractor:
    def __init__(self, dense: bool = True, resize: int = 224, step: int = 8, size: int = 16,
                 max_kp: int = 500, root_sift: bool = True):
        self.dense, self.resize, self.step, self.size = dense, resize, step, size
        self.max_kp, self.root_sift = max_kp, root_sift
        self.sift = cv2.SIFT_create(nfeatures=max_kp)

    def _grid(self, h: int, w: int):
        half = self.size // 2
        return [cv2.KeyPoint(float(x), float(y), float(self.size))
                for y in range(half, h - half + 1, self.step)
                for x in range(half, w - half + 1, self.step)]

    def _one(self, img: np.ndarray) -> np.ndarray:
        gray = _to_gray(img, self.resize)
        if self.dense:
            _, d = self.sift.compute(gray, self._grid(*gray.shape))
        else:
            _, d = self.sift.detectAndCompute(gray, None)
        if d is None:
            return np.zeros((0, 128), np.float32)
        d = d.astype(np.float32)
        if self.root_sift:  # Arandjelovic & Zisserman 2012
            d /= d.sum(axis=1, keepdims=True) + 1e-7
            d = np.sqrt(d)
        return d

    def __call__(self, images, progress: bool = True) -> LocalFeatures:
        it = tqdm(images, desc="sift", disable=not progress)
        return LocalFeatures.from_list([self._one(im) for im in it], dim=128)


class ORBExtractor:
    def __init__(self, resize: int = 224, max_kp: int = 500, fast_threshold: int = 5):
        self.resize = resize
        self.orb = cv2.ORB_create(nfeatures=max_kp, fastThreshold=fast_threshold, edgeThreshold=15)

    def _one(self, img: np.ndarray) -> np.ndarray:
        gray = _to_gray(img, self.resize)
        _, d = self.orb.detectAndCompute(gray, None)
        if d is None:
            return np.zeros((0, 256), np.float32)
        return np.unpackbits(d, axis=1).astype(np.float32)

    def __call__(self, images, progress: bool = True) -> LocalFeatures:
        it = tqdm(images, desc="orb", disable=not progress)
        return LocalFeatures.from_list([self._one(im) for im in it], dim=256)
