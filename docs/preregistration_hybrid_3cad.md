# Preregistration: the 75/25 hybrid memory on 3CAD

Committed before any hybrid run on 3CAD; the commit timestamp is the registration date.
Config: `configs/p0_hybrid_3cad.yaml`. Checks: `bovw.anomaly.prereg_checks_hybrid`.

## Background (exploratory, MVTec AD and VisA, DINOv2-S, 3 seeds)

A hybrid memory of M vectors (75% k-means visual words + 25% real patches chosen by
greedy k-center started from the words, i.e. the patches the words cover worst)
gave the best max-score image AUROC of any compressed method at M = 1,024 on both
datasets (MVTec 0.980 vs. codebook 0.972 and coreset 0.979; VisA 0.931 vs. 0.918
and 0.929), with AUPRO 0.004 (MVTec) and 0.012 (VisA) below the codebook. At
M = 256 the AUPRO gap was 0.006 and 0.017. The design and the 75/25 split were
chosen after seeing those results; 3CAD is the confirmatory test.

## Fixed in advance

- Dataset: 3CAD, official train/test folders, all 8 categories (same loader, splits,
  image size 448, mask size 256 and low-memory settings as `p0_anomaly_3cad`).
- Features: DINOv2-S, 448 px, L2-normalised patch tokens, re-extracted for this run.
- Methods, all in the same run on the same features and seeds {0, 1, 2}:
  codebook (K = M words), coreset (M patches, greedy k-center), hybrid 75/25
  (round(0.75 M) words + the rest residual-coreset patches). M in {256, 1024}.
- **Primary image score: maximum patch score** (the hybrid's motivation is to remove
  single-patch false alarms without a top-k workaround). Top-3/10/30 are secondary.
- Metrics: image AUROC, pixel AUROC, AUPRO up to 30% FPR; pixel metrics as in
  `p0_anomaly_3cad` (histogram-binned above 10^8 pixels).
- Aggregation: mean over categories, then over seeds.

## Predictions

| ID | Prediction | Pass if |
|----|-----------|---------|
| H1 | The hybrid detects at least as well as the coreset | hybrid max-score image AUROC >= coreset at M = 256 and M = 1024 |
| H2 | The hybrid detects better than the codebook | hybrid max-score image AUROC > codebook at M = 1024 |
| H3 | The hybrid keeps the codebook's localization | hybrid AUPRO >= codebook AUPRO - 0.02 at M = 256 and M = 1024 |
| H4 | The hybrid localizes better than the coreset | hybrid AUPRO > coreset AUPRO at M = 256 and M = 1024 |

H2 is the hardest: on 3CAD the codebook already out-detects the coreset (top-10
image AUROC 0.801 vs. 0.770 at M = 1024 in `p0_anomaly_3cad`), so there may be
little left for the real patches to fix. Every prediction is reported whether it
passes or fails; no settings change after the run; anything else is exploratory.

## Licence

No licence is stated for 3CAD; research use, confirm with the authors before publication.

## Amendment log

(none)
