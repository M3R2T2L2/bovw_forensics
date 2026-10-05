# bovw-forensics

Shared pipeline for the **BoVW Forensics Research Program**: Bag of Visual Words over handcrafted and foundation-model descriptors, built for sweeps that are reproducible and aware of compute cost.

First consumer: **P0, the systematic revisit paper.** It compares SIFT/ORB, foundation-model codebooks, and global embeddings across clustering, retrieval, and anomaly detection, sweeping vocabulary size and assignment method.

## Pipeline

```
images → extract → cache → vocab → encode → eval → results CSV → plots
```

| Module | Contents |
|---|---|
| `bovw/extract/` | `dense_sift`, `sift` (RootSIFT), `orb` (thread-parallel); `dinov2_s/b` (+ register variants) patch tokens + CLS |
| `bovw/cache.py` | Features computed once per (dataset, extractor, params) and stored as float16 `.npy` |
| `bovw/vocab.py` | Optional PCA/whitening → (spherical) MiniBatch k-means on a descriptor sample |
| `bovw/encode.py` | `hard`, `soft` (kernel codebook; `knn`, `sigma_scale`), `vlad` (intra-norm); power + L2 normalisation |
| `bovw/eval/` | Clustering: NMI, ARI, Hungarian accuracy over several seeds |
| `bovw/anomaly.py` | Unsupervised anomaly detection: PatchCore-style patch kNN, CLS kNN, codebook word distance (raw and radius-normalised), histogram kNN; image/pixel AUROC, per-defect-type AUROC; GPU kNN with NumPy fallback |
| `bovw/ad_data.py` | MVTec AD, VisA and 3CAD loaders (images, labels, masks) and a synthetic defect set for tests; VisA's official 1-class split is vendored in `bovw/resources/` (from amazon-science/spot-diff, Apache-2.0) |
| `bovw/timing.py` | Wall time, peak RSS, and peak GPU memory for every stage |
| `bovw/sweep.py` | YAML grid over extractor × K × vocab seed × assignment × encode-param variants (lists in `encode:` expand); per-extractor overrides; appends to `<name>.jsonl`, rewrites `<name>.csv`, skips finished runs; `select_best_variant` for tuning splits |
| `bovw/plots.py` | `aggregate` over vocab seeds; metric vs. K, soft-assignment sensitivity, cost vs. metric, summary table |

## Quick start

**Colab (recommended):**

- `notebooks/00_p0_minimal_stl10.ipynb`: 2,000-image smoke run (~15 min)
- `notebooks/01_p0_stl10_full.ipynb`: tune soft assignment on the train split, then the full 8,000-image test run with 3 vocabulary seeds
- `notebooks/02_p0_anomaly_mvtec.ipynb`: anomaly detection on MVTec AD (CC BY-NC-SA 4.0: accept the licence first)
- `notebooks/03_p0_anomaly_followup.ipynb`: MVTec follow-up: AUPRO, top-k image scores, matched-memory coreset, pill false alarms
- `notebooks/04_p0_anomaly_visa.ipynb`: preregistered replication on VisA (CC BY 4.0); see `docs/preregistration_visa.md`
- `notebooks/06_p0_anomaly_3cad.ipynb`: preregistered replication on 3CAD, a dataset neither DINOv2 nor PatchCore was developed on; see `docs/preregistration_3cad.md`
- `notebooks/07_p0_hybrid.ipynb`: hybrid memory (visual words + residual-coreset patches) on MVTec AD and VisA, exploratory
- `notebooks/08_p0_hybrid_3cad.ipynb`: preregistered test of the 75/25 hybrid on 3CAD; see `docs/preregistration_hybrid_3cad.md`
- `notebooks/05_p0_defect_discovery.ipynb`: unsupervised defect-type discovery on MVTec AD (clusters anomalous images by kind of defect)

Open in Colab: `https://colab.research.google.com/github/M3R2T2L2/bovw_forensics/blob/main/notebooks/<notebook>.ipynb`

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
- Clustering and anomaly detection are wired up; retrieval (Revisited Oxford/Paris) is next.
- Anomaly scoring uses the full normal patch bank for `patch_knn` (exact, no coreset): an upper bound for PatchCore-style methods.

## License

MIT
