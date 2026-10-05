"""Generates the Colab notebooks in notebooks/ (keeps them diff-friendly in git)."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

def setup_cells():
    return [
    md("## 1 · Setup"),
    code("""!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: DINOv2 extraction will be slow"

from google.colab import drive
drive.mount('/content/drive')"""),
    code("""# Get the code. Option A: GitHub (default). Option B: set REPO_URL = "" to install
# from bovw-forensics.zip uploaded to MyDrive/bovw-forensics/.
REPO_URL = "https://github.com/M3R2T2L2/bovw_forensics.git"
ZIP_PATH = "/content/drive/MyDrive/bovw-forensics/bovw-forensics.zip"

import os, shutil, subprocess, zipfile
shutil.rmtree("/content/bovw_forensics", ignore_errors=True)
if REPO_URL:
    subprocess.run(["git", "clone", "-q", REPO_URL, "/content/bovw_forensics"], check=True)
else:
    zipfile.ZipFile(ZIP_PATH).extractall("/content/")
    os.rename("/content/bovw-forensics", "/content/bovw_forensics")

%cd /content/bovw_forensics
!pip install -q -e ".[dev]"   # torch, torchvision, transformers come preinstalled on Colab"""),
    code("""# 30-second check that the torch-free core works in this runtime
!python -m pytest -q"""),
    code("""# STL-10 on Drive: restore it here (skips the ~45 min download), or save it after the first download.
import os, shutil
DRIVE_STL = "/content/drive/MyDrive/bovw-forensics/data/stl10_binary"
LOCAL_STL = "/content/data/stl10_binary"
if os.path.isdir(DRIVE_STL) and not os.path.isdir(LOCAL_STL):
    shutil.copytree(DRIVE_STL, LOCAL_STL); print("Restored STL-10 from Drive")
elif os.path.isdir(LOCAL_STL) and not os.path.isdir(DRIVE_STL):
    shutil.copytree(LOCAL_STL, DRIVE_STL); print("Saved STL-10 to Drive for future runtimes")
else:
    print("STL-10 will download on first use; re-run this cell afterwards to save it to Drive"
          if not os.path.isdir(LOCAL_STL) else "STL-10 already local and on Drive")"""),
    ]


cells_00 = [
    md("""# P0 · Minimal loop: BoVW vs. global embeddings on STL-10

Runs the first slice of the systematic revisit paper (P0):

- **Descriptors:** DINOv2-S patch tokens and dense RootSIFT
- **Vocabulary size K:** 64, 256, 1024
- **Assignment:** hard, soft, VLAD (VLAD only up to K = 256)
- **Baseline:** DINOv2 CLS embedding
- **Task:** unsupervised clustering (NMI, ARI, Hungarian accuracy; 3 seeds)

**Runtime:** about 15–25 min on a T4 GPU (*Runtime → Change runtime type → T4 GPU*).
Features are cached on Google Drive, so reruns skip extraction, and finished runs are skipped too."""),

    *setup_cells(),

    md("## 2 · Configure\nEdit the YAML or override here. Set `n_per_class=None` for all 8,000 test images once the small run looks right."),
    code("""from bovw.sweep import load_config, run

cfg = load_config("configs/p0_minimal_stl10.yaml")
cfg["dataset"]["n_per_class"] = 200        # 2,000 images
# cfg["vocab"]["k"] = [64, 256]            # uncomment for a faster first pass
cfg"""),

    md("## 3 · Run the sweep\nOne row per (extractor, K, assignment) is appended to the results CSV on Drive as it finishes."),
    code("""df = run(cfg)
df.tail()"""),

    md("## 4 · Results"),
    code("""from bovw import plots
plots.summary_table(df, "nmi")"""),
    code("""fig = plots.metric_vs_k(df, "nmi")
fig.savefig(f"{cfg['results_dir']}/{cfg['name']}_nmi_vs_k.png", dpi=200, bbox_inches="tight")"""),
    code("""fig = plots.cost_vs_metric(df, "nmi")
fig.savefig(f"{cfg['results_dir']}/{cfg['name']}_cost_vs_nmi.png", dpi=200, bbox_inches="tight")"""),
    code("""# Compute budget, including one-off extraction time per descriptor
cost_cols = ["extractor", "assignment", "k", "enc_dim", "desc_per_image", "extract_seconds",
             "vocab_seconds", "encode_seconds", "eval_seconds", "encode_peak_rss_mb"]
df[[c for c in cost_cols if c in df]].round(2)"""),

    md("""## 5 · What to check before widening the grid

1. **Global vs. BoVW:** does any DINOv2 codebook beat the CLS baseline? At what K?
2. **SIFT gap:** how far below DINOv2 does dense SIFT sit, and does VLAD narrow it?
3. **Empty words:** a large `empty_words` count at K = 1024 means the vocabulary is oversized for 2,000 images.
4. **Cost:** where does the NMI-vs-time curve flatten?

**Next:** add ORB, soft-assignment `knn` sweep, then retrieval (Revisited Oxford) and anomaly detection (MVTec AD)."""),
]

cells_01 = [
    md("""# P0 · STL-10 full run: tune on train, report on test

1. **Tune** soft assignment (`knn` × `sigma_scale`) on 2,000 **train** images and pick the best setting per descriptor.
2. **Report** on all 8,000 **test** images with 3 vocabulary seeds × 3 clustering seeds, using the tuned setting.

Tuning never touches the test split, so the reported numbers are clean.

**Runtime (T4, 2 vCPU):** step 1 ≈ 25–35 min, step 2 ≈ 1–2 h. Every finished run is saved on Drive:
if Colab disconnects, run *Setup* again, then the cell that was running. It picks up where it stopped."""),
    *setup_cells(),

    md("## 2 · Tune soft assignment on the train split"),
    code("""import json
from bovw.sweep import load_config, run, select_best_variant
from bovw import plots

cfg_t = load_config("configs/p0_tune_soft_stl10train.yaml")
df_t = run(cfg_t)"""),
    code("""fig = plots.soft_sensitivity(df_t, "nmi")
fig.savefig(f"{cfg_t['results_dir']}/{cfg_t['name']}_soft_sensitivity.png", dpi=200, bbox_inches="tight")"""),
    code("""best = select_best_variant(df_t, "soft", "nmi")
json.dump(best, open(f"{cfg_t['results_dir']}/{cfg_t['name']}_best_soft.json", "w"), indent=2)
best"""),

    md("## 3 · Full run on the test split (8,000 images)"),
    code("""cfg = load_config("configs/p0_stl10_full.yaml")
for ex in cfg["extractors"]:
    ex.setdefault("encode", {})["soft"] = best[ex["name"]]     # tuned on train, fixed here
[(ex["name"], ex["encode"]["soft"]) for ex in cfg["extractors"]]"""),
    code("""df = run(cfg)"""),

    md("## 4 · Results"),
    code("""plots.summary_table(df, "nmi")"""),
    code("""plots.summary_table(df, "acc")"""),
    code("""fig = plots.metric_vs_k(df, "nmi")
fig.savefig(f"{cfg['results_dir']}/{cfg['name']}_nmi_vs_k.png", dpi=200, bbox_inches="tight")"""),
    code("""fig = plots.cost_vs_metric(df, "nmi")
fig.savefig(f"{cfg['results_dir']}/{cfg['name']}_cost_vs_nmi.png", dpi=200, bbox_inches="tight")"""),
    code("""# Stability: spread of clustering accuracy across all seeds (lower = more stable)
agg = plots.aggregate(df)
agg.pivot_table(index=["extractor", "assignment"], columns="k", values="acc_std").round(3)"""),
    code("""# Compute budget per run (seconds); extraction is one-off per descriptor
cost_cols = ["extractor", "assignment", "k", "vocab_seed", "enc_dim", "extract_seconds",
             "vocab_seconds", "encode_seconds", "eval_seconds", "n_cpu", "gpu"]
df[[c for c in cost_cols if c in df]].round(2)"""),

    md("""## 5 · What to check

1. **Does the codebook win hold at 8,000 images and across vocabulary seeds?** Compare hard K=256 and VLAD K=256 with `global`.
2. **Does tuned soft assignment close the gap to hard at K=64?** If not, the soft deficit is real, not a tuning artifact.
3. **Stability:** is the CLS baseline's accuracy spread still much larger than the codebooks'?
4. **Cost:** extraction time per descriptor, now with SIFT parallelised.

**Next:** retrieval (Revisited Oxford/Paris) and anomaly detection (MVTec AD)."""),
]


cells_02 = [
    md("""# P0 · Anomaly detection on MVTec AD

Unsupervised: every model sees only **normal** training images. Scored per category, then averaged.

| Method | What it scores |
|---|---|
| `patch_knn` | PatchCore-style baseline: distance to the nearest of *all* normal patches |
| `global_knn` | CLS-embedding baseline (image level only) |
| `codebook_dist` | distance to the nearest of K visual words |
| `codebook_norm` | that distance divided by the word's own radius |
| `hist_knn` | bag-of-words histogram distance (image level only) |
| `coreset_knn` | matched memory: PatchCore greedy coreset of M = K normal patches |
| `random_knn` | matched memory: M = K normal patches sampled at random |

**Licence:** MVTec AD is CC BY-NC-SA 4.0 (non-commercial). Accept it at
https://www.mvtec.com/company/research/datasets/mvtec-ad before downloading.

**Runtime (T4, 2 vCPU):** about 1–1.5 h for all 15 categories from scratch (much less when features are cached), plus a one-time ~5 GB download.
Finished runs are saved on Drive; after a disconnect, run *Setup* and *Data* again, then the run cell."""),
    *setup_cells(),

    md("## 2 · Data\nThe archive is kept on Drive so it downloads only once; each new runtime extracts it locally (~3–5 min)."),
    code("""import os, subprocess
from bovw import ad_data
DRIVE_TAR = "/content/drive/MyDrive/bovw-forensics/data/mvtec_anomaly_detection.tar.xz"
LOCAL_ROOT = "/content/mvtec"

# Set to True only after accepting the licence on the MVTec page above.
I_ACCEPT_MVTEC_LICENCE = False

if not os.path.isfile(DRIVE_TAR) or os.path.getsize(DRIVE_TAR) < 1e9:   # missing, or a stub from a failed download
    if not I_ACCEPT_MVTEC_LICENCE:
        raise SystemExit("Accept the MVTec AD licence, then set I_ACCEPT_MVTEC_LICENCE = True "
                         "(or upload mvtec_anomaly_detection.tar.xz to MyDrive/bovw-forensics/data/).")
# Downloads only if missing or corrupt; verifies the SHA-256 (hashing ~5 GB takes about a minute).
ad_data.ensure_archive(DRIVE_TAR)

if not os.path.isdir(f"{LOCAL_ROOT}/bottle"):
    os.makedirs(LOCAL_ROOT, exist_ok=True)
    subprocess.run(["tar", "-xf", DRIVE_TAR, "-C", LOCAL_ROOT], check=True)
print(sorted(os.listdir(LOCAL_ROOT)))"""),

    md("## 3 · Run\nStart with three categories to check everything end to end, then set `categories = \"all\"`."),
    code("""from bovw.sweep import load_config
from bovw import anomaly, plots

cfg = load_config("configs/p0_anomaly_mvtec.yaml")
cfg["data"]["root"] = LOCAL_ROOT
cfg["data"]["categories"] = "all"   # or e.g. ["bottle", "carpet", "screw"] for a quick pass
df = anomaly.run(cfg)"""),

    md("## 4 · Results"),
    code("""anomaly.summary(df, "image_auroc").round(3)"""),
    code("""anomaly.summary(df, "pixel_auroc").round(3)"""),
    code("""fig = plots.anomaly_vs_k(df)
fig.savefig(f"{cfg['results_dir']}/{cfg['name']}_auroc_vs_k.png", dpi=200, bbox_inches="tight")"""),
    code("""# Compute budget per method (seconds, mean over categories)
cols = [c for c in ["vocab_seconds", "score_seconds", "score_peak_gpu_mb", "score_peak_rss_mb"] if c in df]
df.groupby(["extractor", "method", "k"])[cols].mean().round(1)"""),

    md("""## 5 · What to check

1. **Codebook vs. matched-memory baselines:** at the same number of stored vectors (M = K), does `codebook_dist` beat `coreset_knn` and `random_knn`?
1. **Codebook vs. full patch bank:** how close does `codebook_dist` get to `patch_knn`, and at what K? The bank has ~300k patches per category; K is at most 1,024.
2. **Radius normalisation:** does `codebook_norm` beat plain word distance, especially on pixel AUROC?
3. **Image-level methods:** do `hist_knn` / `global_knn` hold up without localisation?
4. **Per-defect-type AUROC** is stored in `image_auroc_by_type`, useful later for P3 (not all anomalies are equal)."""),
]


def _data_cells():
    """The Data section of notebook 02 (licence flag, Drive archive, local extract), copied."""
    import copy
    i = next(j for j, c in enumerate(cells_02) if c["cell_type"] == "markdown" and c["source"].startswith("## 2 · Data"))
    return [copy.deepcopy(cells_02[i]), copy.deepcopy(cells_02[i + 1])]


cells_03 = [
    md("""# P0 · Anomaly follow-up: words localize, coresets detect — why?

At matched memory (M stored vectors = K words), the codebook wins pixel AUROC in 15/15 MVTec categories,
while PatchCore's greedy coreset wins image AUROC. This notebook tests the explanation:

| Check | Prediction if the explanation is right |
|---|---|
| **AUPRO** (every defect region counts equally) | codebook still beats the coreset on localization |
| **Normal-pixel spread** | codebook background is quieter (lower relative IQR, higher pixel d′) |
| **Top-k image score** (mean of the k highest patches) | closes most of the codebook's image-AUROC gap |
| **Pill false alarms** | the codebook's worst normal images peak on speckles or print |
| **Hybrid** (coreset image score + codebook map) | best of both |

DINOv2-S only; 3 seeds. **Runtime (T4):** about 2 h, mostly k-means on CPU. Vocabularies are cached on
Drive, so a rerun after a disconnect resumes quickly. Uses the MVTec archive already on Drive."""),
    *setup_cells(),
    *_data_cells(),

    md("## 3 · Run"),
    code("""from bovw.sweep import load_config
from bovw import anomaly, plots

cfg = load_config("configs/p0_anomaly_mvtec_v2.yaml")
cfg["data"]["root"] = LOCAL_ROOT
df = anomaly.run(cfg)"""),

    md("""## 4 · Results

`img` = max-patch image AUROC; `img_topK` = mean of the K highest patches; `pix` = pixel AUROC;
`aupro` = region overlap up to 30% FPR; `pixel_dprime` = defect-vs-normal separation in normal-pixel SDs.
Means over 15 categories and 3 seeds."""),
    code("""t = anomaly.followup_table(df)
t.to_csv(f"{cfg['results_dir']}/{cfg['name']}_followup.csv")
t.round(3)"""),
    code("""# Per-category AUPRO at M = K = 1024 (codebook vs coreset)
d = df[df.k.fillna(0).astype(int).isin([0, 1024])]
d.pivot_table(index="category", columns="method", values="aupro").round(3)"""),

    md("## 5 · Pill: normal-pixel spread and false alarms"),
    code("""from bovw import ad_data
pill = ad_data.load_mvtec(LOCAL_ROOT, "pill", cfg["data"]["size"], cfg["data"]["mask_size"])
spread = anomaly.normal_pixel_spread(cfg["results_dir"], cfg["name"], "pill", pill.test_masks)
spread.to_csv(f"{cfg['results_dir']}/{cfg['name']}_pill_spread.csv", index=False)
spread.round(3)"""),
    code("""fig = plots.false_alarms(cfg["results_dir"], cfg["name"], "pill", pill.test_images, pill.test_labels)
fig.savefig(f"{cfg['results_dir']}/{cfg['name']}_pill_false_alarms.png", dpi=110, bbox_inches="tight")"""),
]


cells_04 = [
    md("""# P0 · Anomaly replication on VisA (preregistered)

Tests whether the MVTec AD result holds on a second dataset, with every setting fixed in advance:
see [`docs/preregistration_visa.md`](https://github.com/M3R2T2L2/bovw_forensics/blob/main/docs/preregistration_visa.md).
**Primary image score: mean of the top-10 patch scores.**

| ID | Prediction |
|---|---|
| P1 | codebook AUPRO > coreset AUPRO at M = 64, 256, 1024 |
| P2 | codebook top-10 image AUROC >= coreset - 0.005 at M = 1024 |
| P3 | codebook pixel AUROC >= coreset at M = 1024 |
| P4 | codebook within 0.01 of the full bank at K = 1024 (image top-10, pixel, AUPRO) |

**Data:** VisA (Zou et al., ECCV 2022), CC BY 4.0, official 1-class split; 12 categories, 10,821 images.
**Runtime (T4):** about 2 h, plus a one-time ~1.8 GB download to Drive. Rerun after a disconnect to resume."""),
    *setup_cells(),

    md("## 2 · Data\nThe archive is kept on Drive so it downloads only once; each new runtime extracts it locally."),
    code("""import os, subprocess
from bovw import ad_data
DRIVE_TAR = "/content/drive/MyDrive/bovw-forensics/data/VisA_20220922.tar"
LOCAL = "/content/visa"

if not os.path.isfile(DRIVE_TAR) or os.path.getsize(DRIVE_TAR) < 1e9:
    ad_data.ensure_archive(DRIVE_TAR, url=ad_data.VISA_URL, sha256=None)
if not os.path.isdir(LOCAL):
    os.makedirs(LOCAL)
    subprocess.run(["tar", "-xf", DRIVE_TAR, "-C", LOCAL], check=True)
VISA_ROOT = ad_data.find_visa_root(LOCAL)
print(VISA_ROOT, sorted(c for c in os.listdir(VISA_ROOT) if c in ad_data.VISA_CATEGORIES))
print("archive SHA-256 (record in the paper):", ad_data.sha256sum(DRIVE_TAR))"""),

    md("## 3 · Run"),
    code("""from bovw.sweep import load_config
from bovw import anomaly

cfg = load_config("configs/p0_anomaly_visa.yaml")
cfg["data"]["root"] = VISA_ROOT
df = anomaly.run(cfg)"""),

    md("## 4 · Preregistered checks"),
    code("""import pandas as pd
pd.set_option("display.max_colwidth", 120)
checks = anomaly.prereg_checks(df)
checks.to_csv(f"{cfg['results_dir']}/{cfg['name']}_prereg.csv", index=False)
checks"""),

    md("## 5 · Full table (secondary metrics are exploratory)"),
    code("""t = anomaly.followup_table(df)
t.to_csv(f"{cfg['results_dir']}/{cfg['name']}_followup.csv")
t.round(3)"""),
    code("""d = df[df.k.fillna(0).astype(int).isin([0, 1024])]
pd.concat({m: d.pivot_table(index="category", columns="method", values=m)
           for m in ["image_auroc_top10", "pixel_auroc", "aupro"]}, axis=1).round(3)"""),
]


cells_05 = [
    md("""# P0 → P3 · Unsupervised defect-type discovery (MVTec AD)

After the codebook flags suspicious patches, can we sort anomalous images by **kind** of defect with no labels?
Each image is described by its 10 most suspicious patches; images are clustered and compared with MVTec's
defect-type folders (NMI, ARI, accuracy; chance = shuffled labels).

| Descriptor | What it encodes |
|---|---|
| `topk_resid` | what the normal vocabulary *cannot* explain: patch minus its nearest word |
| `topk_feat` | what the suspicious patches look like |
| `cls` | whole-image embedding (baseline) |
| `mask_feat` | features inside the true defect mask (upper bound, uses labels) |

**Settings:** `oracle` clusters the truly anomalous images; `detected` clusters the images the detector flags,
so false alarms join as a "good" class. Reuses cached features and vocabularies from notebook 03:
about 10–15 min on a T4. Exploratory, not preregistered."""),
    *setup_cells(),
    *_data_cells(),

    md("## 3 · Run"),
    code("""from bovw.sweep import load_config
from bovw import discovery

cfg = load_config("configs/p0_defect_discovery_mvtec.yaml")
cfg["data"]["root"] = LOCAL_ROOT
df = discovery.run(cfg)"""),

    md("## 4 · Results\n`nmi_above_chance` is the headline: NMI minus the shuffled-label NMI of the same clustering."),
    code("""s = discovery.summary(df)
s.to_csv(f"{cfg['results_dir']}/{cfg['name']}_summary.csv")
s.round(3)"""),
    code("""# Per category, oracle setting, k-means
o = df[(df.setting == "oracle") & (df.algorithm == "kmeans")]
t = o.pivot_table(index=["category"], columns="descriptor", values="nmi_above_chance").round(3)
t.insert(0, "n_types", o.groupby("category").n_types.first())
t.to_csv(f"{cfg['results_dir']}/{cfg['name']}_per_category.csv")
t"""),

    md("## 5 · Look at the clusters\nRows = discovered clusters; each tile is titled with its true defect type."),
    code("""import matplotlib.pyplot as plt
from bovw import ad_data

CATEGORY = "bottle"   # try metal_nut, hazelnut, pill, cable
idx, pred, types = discovery.cluster_category(cfg, CATEGORY, "topk_resid")
split = ad_data.load_mvtec(LOCAL_ROOT, CATEGORY, 224, 64)
n_cl, per = pred.max() + 1, 6
fig, axes = plt.subplots(n_cl, per, figsize=(1.9 * per, 2.1 * n_cl), squeeze=False)
for c in range(n_cl):
    members = idx[pred == c][:per]
    for j in range(per):
        ax = axes[c][j]; ax.set_axis_off()
        if j < len(members):
            ax.imshow(split.test_images[members[j]])
            ax.set_title(types[pred == c][j], fontsize=7)
    axes[c][0].text(-0.1, 0.5, f"cluster {c}\\n(n={int((pred == c).sum())})", transform=axes[c][0].transAxes,
                    ha="right", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{cfg['results_dir']}/{cfg['name']}_{CATEGORY}_clusters.png", dpi=110, bbox_inches="tight")"""),
]


cells_06 = [
    md("""# P0 · Anomaly replication on 3CAD (preregistered)

A dataset neither DINOv2 nor PatchCore was developed on: 3CAD (Yang et al., AAAI 2025), 27,039 images of
3C-product parts from real production lines, 8 categories, 47 defect types. Settings and predictions are
fixed in [`docs/preregistration_3cad.md`](https://github.com/M3R2T2L2/bovw_forensics/blob/main/docs/preregistration_3cad.md),
identical to VisA. **Primary image score: mean of the top-10 patch scores.**

| ID | Prediction |
|---|---|
| P1 | codebook AUPRO > coreset AUPRO at M = 64, 256, 1024 |
| P2 | codebook top-10 image AUROC >= coreset - 0.005 at M = 1024 |
| P3 | codebook pixel AUROC >= coreset at M = 1024 |
| P4 | codebook within 0.01 of the full bank at K = 1024 (image top-10, pixel, AUPRO) |

**Licence:** none stated by the authors; research use, confirm before publishing.
**Runtime (T4):** about 3–4 h. Features are cached on the Colab disk (too large for Drive); results go to Drive.
After a disconnect, run everything again: finished categories are skipped. A High-RAM runtime is safer if you have one."""),
    *setup_cells(),

    md("## 2 · Data\nDownloads the official archive from Google Drive with `gdown` (or uses a copy you placed in `MyDrive/bovw-forensics/data/`)."),
    code("""import os, glob, shutil, subprocess, torch
if not torch.cuda.is_available():   # check before the long download
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
from bovw import ad_data
LOCAL = "/content/3cad"
DRIVE_COPY = glob.glob("/content/drive/MyDrive/bovw-forensics/data/3CAD*")
ARCHIVE = "/content/3cad_archive"

if not os.path.isdir(LOCAL):
    if DRIVE_COPY:
        shutil.copy(DRIVE_COPY[0], ARCHIVE)
    else:
        import gdown
        out = gdown.download(id=ad_data.THREECAD_GDRIVE_ID, output=ARCHIVE, quiet=False)
        if out is None:
            raise SystemExit("gdown could not fetch the archive (Drive quota?). Open "
                             "https://drive.google.com/file/d/" + ad_data.THREECAD_GDRIVE_ID +
                             " , download it, upload to MyDrive/bovw-forensics/data/ as 3CAD.<ext>, and rerun.")
    os.makedirs(LOCAL)
    magic = open(ARCHIVE, "rb").read(4)
    fmt = "zip" if magic[:2] == b"PK" else ("gztar" if magic[:2] == b"\\x1f\\x8b" else "tar")
    shutil.unpack_archive(ARCHIVE, LOCAL, fmt)
    print("archive SHA-256 (record in the paper):", ad_data.sha256sum(ARCHIVE))
    os.remove(ARCHIVE)

ROOT = ad_data.find_root(LOCAL, "Copper_Stator")
for c in ad_data.THREECAD_CATEGORIES:
    got, want = ad_data.count_mvtec_style(ROOT, c), ad_data.THREECAD_COUNTS[c]
    print(f"{c:<28} train/test {got}  expected {want}  {'OK' if got == want else 'MISMATCH'}")"""),

    md("## 3 · Run"),
    code("""from bovw.sweep import load_config
from bovw import anomaly

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all. "
                     "(On CPU, feature extraction takes hours per category.)")
print("GPU:", torch.cuda.get_device_name(0))

cfg = load_config("configs/p0_anomaly_3cad.yaml")
cfg["data"]["root"] = ROOT
df = anomaly.run(cfg)"""),

    md("## 4 · Preregistered checks"),
    code("""import pandas as pd
pd.set_option("display.max_colwidth", 120)
checks = anomaly.prereg_checks(df)
checks.to_csv(f"{cfg['results_dir']}/{cfg['name']}_prereg.csv", index=False)
checks"""),

    md("## 5 · Full table (secondary metrics are exploratory)"),
    code("""t = anomaly.followup_table(df)
t.to_csv(f"{cfg['results_dir']}/{cfg['name']}_followup.csv")
t.round(3)"""),
    code("""d = df[df.k.fillna(0).astype(int).isin([0, 1024])]
pd.concat({m: d.pivot_table(index="category", columns="method", values=m)
           for m in ["image_auroc_top10", "pixel_auroc", "aupro"]}, axis=1).round(3)"""),
]


cells_07 = [
    md("""# P0 · Hybrid memory: visual words + real patches (exploratory)

One memory bank of M vectors: **half visual words** (k-means averages, which cover the common middle of
normal appearance) and **half real patches**, picked greedily as the normal patches the words cover *worst*
(rare-but-normal details). A test patch is scored by its distance to the nearest item of either kind.

Compared at the same M (64, 256, 1,024) with the codebook and coreset results already on Drive.
Reuses cached features: **no dataset download**. Runtime (T4): about 1.5–2 h per word fraction for MVTec AD + VisA, mostly k-means; finished runs are skipped.
Exploratory (designed after seeing these datasets); a confirmatory test should be preregistered."""),
    *setup_cells(),

    md("## 2 · Run (MVTec AD, then VisA)"),
    code("""import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
from bovw.sweep import load_config
from bovw import anomaly

results = {}
for ds in ["mvtec", "visa"]:
    cfg = load_config(f"configs/p0_hybrid_{ds}.yaml")
    results[ds] = anomaly.run(cfg)"""),

    md("## 3 · Compare with codebook and coreset\nMeans over categories and 3 seeds; `hybrid_w50` = 50% words, 50% real patches; `hybrid_w75` = 75% words, 25% real patches."),
    code("""import pandas as pd
RES = "/content/drive/MyDrive/bovw-forensics/results"
base = {"mvtec": "p0_anomaly_mvtec_v2.csv", "visa": "p0_anomaly_visa.csv"}
tables = {ds: anomaly.hybrid_compare(pd.read_csv(f"{RES}/{base[ds]}"), results[ds]) for ds in results}
out = pd.concat(tables, names=["dataset"])
out.to_csv(f"{RES}/p0_hybrid_compare.csv")
out.round(3)"""),
]


def write(cells, name):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"  # deterministic ids: regenerating does not churn git diffs
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
    out = Path(__file__).resolve().parents[1] / "notebooks" / name
    nbf.write(nb, out)
    print(f"wrote {out}")


write(cells_00, "00_p0_minimal_stl10.ipynb")
write(cells_01, "01_p0_stl10_full.ipynb")
write(cells_02, "02_p0_anomaly_mvtec.ipynb")
write(cells_03, "03_p0_anomaly_followup.ipynb")
write(cells_04, "04_p0_anomaly_visa.ipynb")
write(cells_05, "05_p0_defect_discovery.ipynb")
write(cells_06, "06_p0_anomaly_3cad.ipynb")
write(cells_07, "07_p0_hybrid.ipynb")
