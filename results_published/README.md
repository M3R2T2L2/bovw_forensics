# Published per-run results

Raw per-run outputs (one row per category x extractor x method x memory size x seed),
copied from the Drive `results/` folder so the tables in the write-up can be checked.
Columns are documented in `bovw/anomaly.py` and `bovw/sweep.py`.

| File | Run | Notebook |
|---|---|---|
| p0_stl10_full.csv | STL-10 clustering, 8,000 test images | 01 |
| p0_tune_soft_stl10train.csv | soft-assignment tuning on the STL-10 train split | 01 |
| p0_anomaly_mvtec.csv | MVTec AD: codebook, coreset, random sample, full bank, CLS, SIFT | 02 |
| p0_anomaly_mvtec_v2.csv | MVTec AD follow-up: AUPRO, top-k scores, pill maps (DINOv2-S) | 03 |
| p0_anomaly_visa.csv | VisA, preregistered | 04 |
| p0_anomaly_3cad.csv | 3CAD, preregistered | 06 |
| p0_hybrid_mvtec.csv, p0_hybrid_visa.csv | hybrid memory, exploratory | 07 |
| p0_hybrid_3cad.csv | hybrid on 3CAD, preregistered (with rerun codebook/coreset) | 08 |
| p0_defect_discovery_mvtec.csv | defect-type discovery, exploratory | 05 |
| stats_paired_category_bootstrap.csv | paired per-category differences, bootstrap CIs over categories (`bovw/stats.py`) | — |

Per-image scores and saved maps are on Drive (`results/scores`, `results/maps`), not here.
