# bovw-forensics

Shared pipeline for the **BoVW Forensics Research Program**: Bag of Visual Words over handcrafted and foundation-model descriptors, built for sweeps that are reproducible and aware of compute cost.

First consumer: **P0, the systematic revisit paper.** It compares SIFT/ORB, foundation-model codebooks, and global embeddings across clustering, retrieval, and anomaly detection, sweeping vocabulary size and assignment method.

## Pipeline

```
images → extract → cache → vocab → encode → eval → results CSV → plots
```

| Module | Contents |
|---|---|
| `bovw/extract/` | `dense_sift`, `sift` (RootSIFT), `orb`; `dinov2_s/b` (+ register variants) patch tokens + CLS |
| `bovw/cache.py` | Features computed once per (dataset, extractor, params) and stored as float16 `.npy` |
| `bovw/vocab.py` | Optional PCA/whitening → (spherical) MiniBatch k-means on a descriptor sample |
| `bovw/encode.py` | `hard`, `soft` (kernel codebook), `vlad` (intra-norm); power + L2 normalisation |
| `bovw/eval/` | Clustering: NMI, ARI, Hungarian accuracy over several seeds |
| `bovw/timing.py` | Wall time, peak RSS, and peak GPU memory for every stage |
| `bovw/sweep.py` | YAML-driven grid; writes each row on completion and skips finished runs on rerun |
| `bovw/plots.py` | Metric vs. K (small multiples), cost vs. metric, summary table |

## Quick start

**Colab (recommended):** open `notebooks/00_p0_minimal_stl10.ipynb` and follow the cells.

**Local, no GPU (smoke test):**

```bash
pip install -e ".[dev]"
pytest -q
python -m bovw.sweep configs/smoke_synthetic.yaml
```

**Local with foundation models:** `pip install -e ".[foundation,dev]"`, then
`python -m bovw.sweep configs/p0_minimal_stl10.yaml --cache-dir ./cache --results-dir ./results`.

## Adding things

- **Backbone:** add an entry to `BACKBONES` in `bovw/extract/foundation.py` (HF id + number of prefix tokens).
- **Dataset:** add a loader to `LOADERS` in `bovw/data.py` returning `(list of HxWx3 uint8, labels)`.
- **Assignment method:** add a function to `ENCODERS` in `bovw/encode.py`.

## Known limitations (v0.1)

- ORB binary descriptors are unpacked to 0/1 floats and clustered with Euclidean k-means, which approximates Hamming k-majority.
- VLAD dimension is K × d; `vlad_max_k` (default 256) caps it.
- Only the clustering task is wired up so far. Retrieval and anomaly detection come next.

## License

MIT
