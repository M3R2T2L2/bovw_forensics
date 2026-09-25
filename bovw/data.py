"""Dataset loaders returning (images: list[HxWx3 uint8], labels: np.ndarray).

Start with STL-10 (96 px, 10 classes); add retrieval and anomaly sets as P0
grows. `synthetic` is a torch-free toy set for tests and smoke runs.
"""

import numpy as np


def stratified_subset(labels: np.ndarray, n_per_class: int | None, seed: int = 0) -> np.ndarray:
    if not n_per_class:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(labels == c), size=min(n_per_class, (labels == c).sum()), replace=False)
           for c in np.unique(labels)]
    return np.sort(np.concatenate(idx))


def load_stl10(root: str = "./data", split: str = "test", n_per_class: int | None = None, seed: int = 0):
    from torchvision.datasets import STL10

    ds = STL10(root=root, split=split, download=True)
    images = ds.data.transpose(0, 2, 3, 1)  # (N, 3, 96, 96) -> (N, 96, 96, 3)
    labels = np.asarray(ds.labels)
    idx = stratified_subset(labels, n_per_class, seed)
    return [images[i] for i in idx], labels[idx]


def load_synthetic(n_per_class: int = 30, n_classes: int = 4, size: int = 96, seed: int = 0):
    """Textured toy images: each class is a different oriented stripe frequency + noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    images, labels = [], []
    for c in range(n_classes):
        theta, freq = np.pi * c / n_classes, 0.15 + 0.1 * c
        for _ in range(n_per_class):
            phase = rng.uniform(0, 2 * np.pi)
            base = np.sin(freq * (np.cos(theta) * xx + np.sin(theta) * yy) + phase)
            img = (base * 90 + 128 + rng.normal(0, 20, base.shape)).clip(0, 255).astype(np.uint8)
            images.append(np.repeat(img[..., None], 3, axis=2))
            labels.append(c)
    return images, np.array(labels)


LOADERS = {"stl10": load_stl10, "synthetic": load_synthetic}


def load(name: str, **kwargs):
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)
