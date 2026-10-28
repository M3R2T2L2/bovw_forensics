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
| p0_review_controls_summary.csv | review controls, MVTec + VisA: medoids, trimmed coreset, GPU k-means to M = 16,384, sigma/FPR sensitivity (means over categories and seeds) | 09 |
| p0_review_image_bootstrap.csv | two-level (category, then image) bootstrap of image-AUROC differences (`bovw/review.py`) | 09 |
| p0_review_timing.csv | same-hardware timings, T4, K = 1,024 | 09 |
| p0_review_discovery_summary.csv | defect discovery with top-10 patches chosen by codebook, full bank and coreset; true vs. estimated k | 09 |
| p0_review_stl10_summary.csv | STL-10 stability rerun (n_init 20, 10 clustering seeds) with mean-pool and CLS+mean-pool baselines | 09 |
| p0_review_controls_paired.csv | paired per-category bootstrap CIs for the review controls (medoids, trimming, large M, sigma/FPR), MVTec + VisA + pooled | 10 |
| p0_review_image_bootstrap_3cad.csv | two-level bootstrap of image-AUROC differences on 3CAD | 10 |
| p0_lexicon_mvtec_summary.csv, p0_lexicon_mvtec_paired.csv | defect-vocabulary tests (`bovw/lexicon.py`): word purity and word-defect consistency, codebook vs. same-size coreset | 10 |
| p0_dinov2b_mvtec.csv, p0_dinov2b_mvtec_paired.csv | DINOv2-B on MVTec: per-run results and codebook-vs-coreset paired CIs | 11 |
| p0_3cad_train_inspection.csv | 3CAD training folders: counts, sizes, duplicates, train/test overlap | 12 |
| p0_3cad_hires_vs_448.csv | 3CAD at 560 vs. 448 px, 7 categories (Aluminum_New_Ipad excluded: RAM) | 12 |
| p0_dinov2breg_mvtec_paired.csv | DINOv2-B with registers on MVTec: codebook-vs-coreset paired CIs (register control) | 11 |
| p0_dinov2b_visa_paired.csv, p0_dinov2b_3cad_paired.csv | DINOv2-B on VisA (12) and 3CAD (7; Aluminum_New_Ipad skipped): paired CIs | 11 |
| p0_3cad_pad_vs_stretch.csv, p0_3cad_pad_per_category_aupro1024.csv | 3CAD with images padded to square and duplicate/leaked training images removed, vs. the original stretched run | 13 |
| stats_family_bootstrap.csv | codebook-vs-coreset intervals resampling families of related categories (pcb1-4, macaroni1-2, paired 3CAD aluminium parts) instead of categories | — |
| p0_random_mvtec.csv, p0_random_means.csv, p0_random_paired_aupro.csv | uniform random sample of real patches as the memory (AUPRO), vs. codebook and coreset; per-run file for MVTec, VisA/3CAD per-run files on Drive | 14 |
| p0_token_norms_mvtec.csv | DINOv2-B with/without registers: token-norm percentile of each normal image's top-scoring patch, K = 64 | 14 |
| p0_3cad_near_duplicates.csv | dHash (64-bit) near-duplicate counts within train and train-vs-test, 3CAD | 14 |

Per-run files behind the review summaries: `p0_review_mvtec.csv`, `p0_review_visa.csv` (notebook 09 controls), `p0_review_discovery_mvtec.csv`, `p0_lexicon_mvtec.csv`, `p0_dinov2breg_mvtec.csv`, `p0_dinov2b_visa.csv`, `p0_dinov2b_3cad.csv`, `p0_3cad_pad.csv`. Only the STL-10 rerun's per-run file (`p0_review_stl10.csv`, 12 rows) is still on Drive; its summary is here.

Per-image scores and saved maps are on Drive (`results/scores`, `results/maps`), not here.
