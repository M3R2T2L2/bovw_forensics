# Preregistration: 3CAD replication of the codebook-vs-coreset result

Committed before any 3CAD run; the commit timestamp is the registration date.
Config: `configs/p0_anomaly_3cad.yaml`. Checks: `bovw.anomaly.prereg_checks`.

## Why 3CAD

3CAD (Yang et al., AAAI 2025) was released after DINOv2's pretraining data was
collected, comes from a company's production lines, and PatchCore has not been
reported on it (the authors' benchmark covers 11 other methods). It is the
first dataset in this study that neither the backbone nor the coreset method
was developed or tuned on.

## Fixed in advance (identical to the VisA preregistration unless noted)

- Dataset: 3CAD, official train/test folders (English defect names), all 8 categories.
- Features: DINOv2-S, 448 px, L2-normalised patch tokens (32 x 32 grid).
- Methods: full-bank patch kNN; codebook distance (spherical k-means);
  greedy coreset kNN. M = K in {64, 256, 1024}; seeds {0, 1, 2}.
- **Primary image score: mean of the 10 highest patch scores (top-10).**
  Max, top-3 and top-30 are secondary.
- Pixel maps: bilinear upsampling to 256 px, Gaussian sigma 4.
- Metrics: image AUROC, pixel AUROC, AUPRO up to 30% FPR.
- Aggregation: mean over categories, then over seeds.
- **Implementation difference (memory only):** 3CAD's test sets are up to
  ~5,000 images, so features are normalised in place in float16, maps are
  stored in float16, and when a category has more than 10^8 test pixels the
  pixel metrics are computed from 65,536-bin histograms (tested to agree with
  the exact metrics within 0.005). Features are cached on the Colab disk, not Drive.

## Predictions (identical to VisA, kept for a strict replication)

| ID | Prediction | Pass if |
|----|-----------|---------|
| P1 | Codebook localizes better than the coreset at every memory size | codebook AUPRO > coreset AUPRO at M = 64, 256 and 1024 |
| P2 | Codebook detects images as well as the coreset | codebook top-10 image AUROC >= coreset - 0.005 at M = 1024 |
| P3 | Codebook pixel AUROC is at least the coreset's | codebook pixel AUROC >= coreset at M = 1024 |
| P4 | Codebook stays close to the full bank | within 0.01 of patch kNN on top-10 image AUROC, pixel AUROC and AUPRO at K = 1024 |

P2 and P4 failed on VisA. They are kept unchanged: 3CAD's memory banks are
larger still (up to ~2.3M patches), so P4 is expected to be hard to pass.
Every prediction is reported whether it passes or fails; no settings change
after the run, and any further analysis is labelled exploratory.

## Licence

No licence is stated in the 3CAD repository. Results are for research; confirm
terms with the authors before publication.
