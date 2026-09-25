"""Local-descriptor extractors.

Every extractor maps a list of HxWx3 uint8 RGB images to a `LocalFeatures`
object: a ragged set of per-image descriptors stored flat, plus an optional
global (per-image) embedding.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LocalFeatures:
    """Ragged per-image descriptors stored as one flat array.

    desc:     (N_total, D) float32 or float16, all descriptors of all images stacked
    offsets:  (n_images + 1,) int64, image i owns desc[offsets[i]:offsets[i+1]]
    global_:  optional (n_images, G) float32 per-image embedding (e.g. CLS token)
    """

    desc: np.ndarray
    offsets: np.ndarray
    global_: Optional[np.ndarray] = None

    @property
    def n_images(self) -> int:
        return len(self.offsets) - 1

    @property
    def dim(self) -> int:
        return self.desc.shape[1]

    def image(self, i: int) -> np.ndarray:
        return self.desc[self.offsets[i] : self.offsets[i + 1]]

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    @staticmethod
    def from_list(per_image: list, global_: Optional[np.ndarray] = None, dim: Optional[int] = None,
                  dtype=np.float32) -> "LocalFeatures":
        """Stack per-image descriptor arrays into one preallocated array (single copy)."""
        if dim is None:
            dim = next((d.shape[1] for d in per_image if d.size), 0)
        counts = [int(np.prod(d.shape) // dim) if dim else 0 for d in per_image]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        desc = np.empty((int(offsets[-1]), dim), dtype=dtype)
        for i, d in enumerate(per_image):
            desc[offsets[i]:offsets[i + 1]] = d.reshape(-1, dim)
        return LocalFeatures(desc=desc, offsets=offsets, global_=global_)


def get_extractor(name: str, **kwargs):
    """Factory: 'sift', 'dense_sift', 'orb', or a foundation model key such as 'dinov2_s'."""
    from . import handcrafted

    if name == "sift":
        return handcrafted.SIFTExtractor(dense=False, **kwargs)
    if name == "dense_sift":
        return handcrafted.SIFTExtractor(dense=True, **kwargs)
    if name == "orb":
        return handcrafted.ORBExtractor(**kwargs)

    from . import foundation  # lazy: needs torch

    return foundation.FoundationExtractor(name, **kwargs)
