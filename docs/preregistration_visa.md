# Preregistration: VisA replication of the MVTec AD codebook result

Committed before any VisA run; the commit timestamp is the registration date.
Config: `configs/p0_anomaly_visa.yaml`. Checks: `bovw.anomaly.prereg_checks`.

## Background (MVTec AD, DINOv2-S, 15 categories, 3 seeds)

At matched memory (M stored vectors = K words), a K-word codebook localized
defects better than PatchCore's greedy coreset (AUPRO 0.933 vs 0.914 at 1,024)
and, with a top-10 patch image score, tied it on image AUROC (0.986 each).
The top-10 rule was chosen after seeing MVTec results, so it is fixed here.

## Fixed in advance

- Dataset: VisA, official 1-class split (`split_csv/1cls.csv`), all 12 categories.
- Features: DINOv2-S, 448 px, L2-normalised patch tokens (32 x 32 grid).
- Methods: full-bank patch kNN; codebook distance (spherical k-means);
  greedy coreset kNN. M = K in {64, 256, 1024}; vocabulary / coreset seeds {0, 1, 2}.
- **Primary image score: mean of the 10 highest patch scores (top-10).**
  Max, top-3 and top-30 are reported as secondary.
- Pixel maps: bilinear upsampling to 256 px, Gaussian sigma 4.
- Metrics: image AUROC, pixel AUROC, AUPRO up to 30% FPR.
- Aggregation: mean over categories, then over seeds.

## Predictions

| ID | Prediction | Pass if |
|----|-----------|---------|
| P1 | Codebook localizes better than the coreset at every memory size | codebook AUPRO > coreset AUPRO at M = 64, 256 and 1024 |
| P2 | Codebook detects images as well as the coreset | codebook top-10 image AUROC >= coreset - 0.005 at M = 1024 |
| P3 | Codebook pixel AUROC is at least the coreset's | codebook pixel AUROC >= coreset at M = 1024 |
| P4 | Codebook stays close to the full bank | within 0.01 of patch kNN on top-10 image AUROC, pixel AUROC and AUPRO at K = 1024 |

Every prediction is reported whether it passes or fails. No settings change
after the run; any further analysis is labelled exploratory.
