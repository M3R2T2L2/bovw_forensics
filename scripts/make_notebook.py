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


def _cells_06_data():
    """The 3CAD data section of notebook 06 (GPU check, download, count check), copied."""
    import copy
    i = next(j for j, c in enumerate(cells_06) if c["cell_type"] == "markdown" and c["source"].startswith("## 2 · Data"))
    return [copy.deepcopy(cells_06[i]), copy.deepcopy(cells_06[i + 1])]


cells_08 = [
    md("""# P0 · Hybrid memory on 3CAD (preregistered)

Confirmatory test of the 75/25 hybrid (75% visual words + 25% real patches the words cover worst),
designed on MVTec AD and VisA. Predictions are fixed in
[`docs/preregistration_hybrid_3cad.md`](https://github.com/M3R2T2L2/bovw_forensics/blob/main/docs/preregistration_hybrid_3cad.md).
**Primary image score: the maximum patch score.**

| ID | Prediction |
|---|---|
| H1 | hybrid max-score image AUROC >= coreset at M = 256 and 1,024 |
| H2 | hybrid max-score image AUROC > codebook at M = 1,024 |
| H3 | hybrid AUPRO >= codebook - 0.02 at M = 256 and 1,024 |
| H4 | hybrid AUPRO > coreset at M = 256 and 1,024 |

Codebook and coreset are rerun here on the same features. **Runtime (T4):** about 4–5 h (feature extraction
about 3 h). Needs a GPU runtime; after a disconnect, Run all again: finished runs are skipped."""),
    *setup_cells(),
    *_cells_06_data(),

    md("## 3 · Run"),
    code("""from bovw.sweep import load_config
from bovw import anomaly

cfg = load_config("configs/p0_hybrid_3cad.yaml")
cfg["data"]["root"] = ROOT
df = anomaly.run(cfg)"""),

    md("## 4 · Preregistered checks"),
    code("""import pandas as pd
pd.set_option("display.max_colwidth", 120)
checks = anomaly.prereg_checks_hybrid(df)
checks.to_csv(f"{cfg['results_dir']}/{cfg['name']}_prereg.csv", index=False)
checks"""),

    md("## 5 · Full table (secondary metrics are exploratory)"),
    code("""t = anomaly.followup_table(df)
t.to_csv(f"{cfg['results_dir']}/{cfg['name']}_followup.csv")
t.round(3)"""),
]


cells_09 = [
    md("""# P0 · Referee-report controls (exploratory)

Controls requested in the review, run on the **cached** MVTec AD, VisA and STL-10 features (no dataset download):

| Control | Question |
|---|---|
| k-means medoids | is the codebook's edge from **averaging** or from **where** its vectors sit? |
| k-center after trimming 1% / 5% outliers | is the coreset hurt by isolated patches? |
| M = 4,096 and 16,384 | do the methods converge at larger memory? (GPU k-means, calibrated at 1,024) |
| σ ∈ {0, 2, 4, 8}, AUPRO at 5% / 10% / 30% FPR | is the localization gap an artefact of smoothing or of the FPR limit? |
| image-level bootstrap | CIs that include test-image sampling, not only categories |
| same-hardware timing | what do codebook and coreset really cost on one GPU? |
| defect discovery, three patch selectors, estimated k | does the codebook matter for discovery, without being told the number of types? |
| STL-10: 10 seeds, n_init = 20, mean-pooled and CLS+mean baselines | is the codebook's stability real? |

**Runtime (T4):** about 2–3 h. Every finished run is saved; Run all again after a disconnect to resume."""),
    *setup_cells(),

    md("## 2 · Anomaly controls (MVTec AD, then VisA)"),
    code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA starts
import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
import pandas as pd
from bovw.sweep import load_config
from bovw import anomaly, discovery, review
from bovw.sweep import run as sweep_run

RES = "/content/drive/MyDrive/bovw-forensics/results"
CACHE = "/content/drive/MyDrive/bovw-forensics/cache"
ctrl = {}
for ds in ["mvtec", "visa"]:
    ctrl[ds] = anomaly.run(load_config(f"configs/p0_review_{ds}.yaml"))"""),
    code("""tabs = pd.concat({ds: review.control_table(df) for ds, df in ctrl.items()}, names=["dataset"])
tabs.to_csv(f"{RES}/p0_review_controls_summary.csv")
tabs.round(3)"""),

    md("## 3 · Image-level bootstrap (MVTec AD, VisA)\nTwo-level bootstrap: categories, then test images within each. Max-patch image AUROC (the per-image scores saved by every run)."),
    code("""from bovw import ad_data
specs = []
for ds, cfgname, hyb in [("mvtec", "p0_anomaly_mvtec_v2", "p0_hybrid_mvtec"), ("visa", "p0_anomaly_visa", "p0_hybrid_visa")]:
    cats = ad_data.categories(ds)
    for k in [64, 256, 1024]:
        specs.append({"label": f"{ds}: codebook - coreset", "cfg_name": cfgname, "dataset": ds, "categories": cats,
                      "a": ("codebook_dist", k), "b": ("coreset_knn", k)})
    for k in [256, 1024]:
        specs.append({"label": f"{ds}: hybrid75 - codebook", "cfg_name": hyb, "cfg_name_b": cfgname, "dataset": ds,
                      "categories": cats, "a": ("hybrid_w75", k), "b": ("codebook_dist", k)})
img = review.image_level_table(specs, results_dir=RES, cache_dir=CACHE, n_boot=2000)
img.to_csv(f"{RES}/p0_review_image_bootstrap.csv", index=False)
img.round(4)"""),

    md("## 4 · Same-hardware timing (one GPU, three categories)"),
    code("""from bovw import cache as fcache
from bovw.anomaly import _norm
rows = []
for ds, cat in [("mvtec", "bottle"), ("visa", "pcb1"), ("visa", "macaroni1")]:
    tag = f"{ds}_{cat}_s448_m256"
    tr, _ = fcache.get_or_compute(CACHE, f"{tag}_train", "dinov2_s", {"image_size": 448}, lambda: None)
    te, _ = fcache.get_or_compute(CACHE, f"{tag}_test", "dinov2_s", {"image_size": 448}, lambda: None)
    rows.append({"dataset": ds, "category": cat, **review.benchmark(_norm(tr.desc, True), _norm(te.desc, True), k=1024)})
timing = pd.DataFrame(rows)
timing.to_csv(f"{RES}/p0_review_timing.csv", index=False)
timing.round(2)"""),

    md("## 5 · Defect-type discovery with three patch selectors (MVTec AD)"),
    code("""disc = discovery.run(load_config("configs/p0_review_discovery_mvtec.yaml"))
ds_ = disc.groupby(["setting", "descriptor"])[["nmi", "nmi_chance", "nmi_above_chance", "ari", "acc", "n_clusters"]].mean()
ds_.to_csv(f"{RES}/p0_review_discovery_summary.csv")
ds_.round(3)"""),

    md("## 6 · STL-10 clustering stability"),
    code("""stl = sweep_run(load_config("configs/p0_review_stl10.yaml"))
st = review.stl10_stability(stl)
st.to_csv(f"{RES}/p0_review_stl10_summary.csv", index=False)
st.round(3)"""),
]


_DATA_ANY = """import os, glob, shutil, subprocess
from bovw import ad_data

def get_mvtec():
    tar, local = "/content/drive/MyDrive/bovw-forensics/data/mvtec_anomaly_detection.tar.xz", "/content/mvtec"
    if not os.path.isfile(tar):
        raise SystemExit("Put mvtec_anomaly_detection.tar.xz in MyDrive/bovw-forensics/data/ (see notebook 02).")
    if not os.path.isdir(f"{local}/bottle"):
        os.makedirs(local, exist_ok=True)
        subprocess.run(["tar", "-xf", tar, "-C", local], check=True)
    return local

def get_visa():
    tar, local = "/content/drive/MyDrive/bovw-forensics/data/VisA_20220922.tar", "/content/visa"
    if not os.path.isfile(tar) or os.path.getsize(tar) < 1e9:
        ad_data.ensure_archive(tar, url=ad_data.VISA_URL, sha256=None)
    if not os.path.isdir(local):
        os.makedirs(local)
        subprocess.run(["tar", "-xf", tar, "-C", local], check=True)
    return ad_data.find_visa_root(local)

def get_3cad():
    local, archive = "/content/3cad", "/content/3cad_archive"
    if not os.path.isdir(local):
        copy = glob.glob("/content/drive/MyDrive/bovw-forensics/data/3CAD*")
        if not copy:
            raise SystemExit("Put the 3CAD archive in MyDrive/bovw-forensics/data/ (see notebook 06).")
        shutil.copy(copy[0], archive)
        os.makedirs(local)
        magic = open(archive, "rb").read(4)
        fmt = "zip" if magic[:2] == b"PK" else ("gztar" if magic[:2] == b"\\x1f\\x8b" else "tar")
        shutil.unpack_archive(archive, local, fmt)
        os.remove(archive)
    return ad_data.find_root(local, "Copper_Stator")
"""

cells_10 = [
    md("""# P0 · Review follow-ups: paired intervals, defect vocabulary, 3CAD bootstrap (exploratory)

| Section | Question |
|---|---|
| Paired intervals for the controls | are the medoid, trimming, large-M and σ/FPR differences larger than category-to-category noise? |
| Defect vocabulary | do words mean consistent things (position, appearance), and do defects of one type trigger the same words? Codebook vs. the same-size coreset used as a vocabulary |
| Example figures | for each defect type: the defect, and normal examples of its signature word |
| 3CAD image-level bootstrap | the two-level bootstrap from notebook 09, on 3CAD |

Reuses cached MVTec features and vocabularies. The figures need the MVTec images and the 3CAD bootstrap needs
the 3CAD archive (both already on Drive). **Runtime (T4):** about 1–1.5 h. Finished runs are saved;
Run all again after a disconnect to resume."""),
    *setup_cells(),
    *_data_cells(),

    md("## 3 · Paired category-level intervals for the controls (MVTec AD, VisA)"),
    code("""import pandas as pd
from bovw import stats

RES = "/content/drive/MyDrive/bovw-forensics/results"
CACHE = "/content/drive/MyDrive/bovw-forensics/cache"
ctrl = {ds: pd.read_csv(f"{RES}/p0_review_{ds}.csv") for ds in ["mvtec", "visa"]}
main = ["image_auroc", "image_auroc_top10", "pixel_auroc", "aupro"]
extra = ["aupro_f5", "aupro_f10", "pixel_auroc_s0", "aupro_s0", "pixel_auroc_s8", "aupro_s8"]
pairs = [("medoid_knn", "codebook_dist", (64, 256, 1024), main + extra),
         ("medoid_knn", "coreset_knn", (64, 256, 1024), main + extra),
         ("coreset_trim1", "coreset_knn", (64, 256, 1024, 4096, 16384), main),
         ("coreset_trim5", "coreset_knn", (64, 256, 1024, 4096, 16384), main),
         ("codebook_gpu", "coreset_knn", (1024, 4096, 16384), main),
         ("codebook_gpu", "codebook_dist", (1024,), main),
         ("codebook_dist", "coreset_knn", (64, 1024), extra)]
paired = pd.concat([stats.compare(ctrl, a, b, ks=ks, metrics=m) for a, b, ks, m in pairs], ignore_index=True)
paired.to_csv(f"{RES}/p0_review_controls_paired.csv", index=False)
show = paired[paired.dataset == "pooled"].drop(columns=["p_sign", "p_wilcoxon"])
show.round(3).to_string(max_rows=200)"""),

    md("## 4 · Is the vocabulary a usable defect language? (MVTec AD)"),
    code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
from bovw.sweep import load_config
from bovw import lexicon

lcfg = load_config("configs/p0_lexicon_mvtec.yaml")
lcfg["data"]["root"] = LOCAL_ROOT
lex = lexicon.run(lcfg)
lsum = lexicon.summary(lex)
lsum.to_csv(f"{RES}/p0_lexicon_mvtec_summary.csv")
lsum.round(3)"""),
    code("""# Paired codebook - coreset differences over categories (seeds averaged), with bootstrap CIs
lx = lex.rename(columns={"memory": "method"})
lpaired = stats.compare({"mvtec": lx}, "codebook", "coreset", ks=(64, 256, 1024),
                        metrics=["position_nmi_gain", "cohesion", "knn1", "knn5", "sig_cover", "sig_false"])
lpaired.to_csv(f"{RES}/p0_lexicon_mvtec_paired.csv", index=False)
lpaired.drop(columns=["p_sign", "p_wilcoxon"]).round(3)"""),

    md("## 5 · Example figures: each defect type and its signature word (K = 256, seed 0)"),
    code("""import matplotlib.pyplot as plt
import numpy as np
from bovw import ad_data, cache as fcache
from bovw.anomaly import _norm, cached_vocabulary, grid_side
from bovw.vocab import build_vocabulary

K, SEED = 256, 0
vcfg = {"max_descriptors": 200000, "spherical": True}
for cat in ["cable", "screw", "metal_nut", "hazelnut"]:
    tag = f"mvtec_{cat}_s448_m256"
    tr, _ = fcache.get_or_compute(CACHE, f"{tag}_train", "dinov2_s", {"image_size": 448}, lambda: None)
    te, _ = fcache.get_or_compute(CACHE, f"{tag}_test", "dinov2_s", {"image_size": 448}, lambda: None)
    trd, ted = _norm(tr.desc, True), _norm(te.desc, True)
    vkey = {"tag": tag, "extractor": "dinov2_s", "params": {"image_size": 448}, "normalize": True,
            "k": K, "seed": SEED, "vocab": vcfg}
    vocab, _ = cached_vocabulary(CACHE, vkey, lambda: build_vocabulary(trd, K, seed=SEED, **vcfg))
    tr_w, te_w, te_d = lexicon.memory_words("codebook", trd, ted, K, SEED, vocab)
    split = ad_data.load_mvtec(LOCAL_ROOT, cat, 448, 256)
    types = np.array(split.test_types)
    anom = (np.asarray(split.test_labels) == 1) & (types != "combined")
    fig = lexicon.defect_vocabulary_figure(split.train_images, split.test_images, tr_w, te_w, te_d, types, anom,
                                           grid_side(te), K, title=f"MVTec {cat}: defect type -> signature word (K = {K})")
    fig.savefig(f"{RES}/p0_lexicon_{cat}_K{K}.png", dpi=130, bbox_inches="tight")
    plt.show()"""),

    md("""## 6 · Image-level bootstrap on 3CAD
Needs only the test labels. They are rebuilt once from the 3CAD archive in `MyDrive/bovw-forensics/data/`
and saved to Drive, so later runs need no archive. If the archive is not on Drive, this section is skipped
(download it in a browser from the link in notebook 06 and upload it there)."""),
    code(_DATA_ANY + """
from pathlib import Path
from bovw import review
from bovw.anomaly import _LazyAD

META = Path(f"{CACHE}/3cad_meta")            # labels only (small); 3CAD features were never on Drive
cats = ad_data.categories("3cad")
need = [c for c in cats if not (META / f"3cad_{c}_s448_m256__meta.npz").exists()]
ok = True
if need:
    try:
        ROOT3 = get_3cad()
        for cat in need:
            _LazyAD({"name": "3cad", "root": ROOT3, "size": 448, "mask_size": 256}, cat, META,
                    f"3cad_{cat}_s448_m256").get_meta()
    except SystemExit as e:
        ok = False
        print("Skipping the 3CAD bootstrap:", e)
if ok:
    specs = [{"label": "3cad: codebook - coreset", "cfg_name": "p0_anomaly_3cad", "dataset": "3cad", "categories": cats,
              "a": ("codebook_dist", k), "b": ("coreset_knn", k)} for k in [64, 256, 1024]]
    specs += [{"label": "3cad: hybrid75 - codebook", "cfg_name": "p0_hybrid_3cad", "dataset": "3cad", "categories": cats,
               "a": ("hybrid_w75", k), "b": ("codebook_dist", k)} for k in [256, 1024]]
    img3 = review.image_level_table(specs, results_dir=RES, cache_dir=META, n_boot=2000)
    img3.to_csv(f"{RES}/p0_review_image_bootstrap_3cad.csv", index=False)
    display(img3.round(4))"""),

    md("## 7 · Record the environment (pinned package versions for the archive)"),
    code("""import platform, subprocess, torch
lock = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
head = (f"# Colab environment, {pd.Timestamp.now():%Y-%m-%d}; Python {platform.python_version()}; "
        f"torch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}\\n")
open(f"{RES}/requirements-colab.lock", "w").write(head + lock)
print(head, len(lock.splitlines()), "packages written to requirements-colab.lock on Drive")"""),
]


cells_11 = [
    md("""# P0 · Second backbone: DINOv2-B (exploratory)

The referee asked whether the codebook-vs-coreset result depends on DINOv2-S. This reruns the main comparison
with **DINOv2-B** (768-d patch tokens), same protocol: 448 px, M = K ∈ {64, 256, 1,024}, 3 seeds, codebook vs.
greedy coreset (plus the full bank on MVTec AD and VisA).

**One dataset per Colab session:** set `DATASET` below to `"mvtec"`, then `"visa"`, then `"3cad"`. For the register control, set `BACKBONE = "dinov2breg"` with `DATASET = "mvtec"`.
**Runtime (T4):** MVTec about 2–3 h, VisA about 4–5 h, 3CAD about 5–6 h. Finished runs are saved to Drive;
after a disconnect, Run all again."""),
    *setup_cells(),

    md("## 2 · Data"),
    code("""DATASET = "mvtec"     # "mvtec", then "visa", then "3cad" (one per session)
BACKBONE = "dinov2b"  # "dinov2breg" = DINOv2-B with registers (control, MVTec only)

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
""" + _DATA_ANY + """
ROOT = {"mvtec": get_mvtec, "visa": get_visa, "3cad": get_3cad}[DATASET]()
print(DATASET, ROOT)"""),

    md("## 3 · Run"),
    code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from bovw.sweep import load_config
from bovw import anomaly, stats

cfg = load_config(f"configs/p0_{BACKBONE}_{DATASET}.yaml")
cfg["data"]["root"] = ROOT
# 3CAD's Aluminum_New_Ipad (4,936 test images) exceeds Colab RAM with 768-d features; set [] on a high-RAM runtime
SKIP_CATEGORIES = ["Aluminum_New_Ipad"] if DATASET == "3cad" else []
if SKIP_CATEGORIES:
    cfg["data"]["categories"] = [c for c in ad_data.categories(DATASET) if c not in SKIP_CATEGORIES]
df = anomaly.run(cfg)"""),

    md("## 4 · Codebook vs. coreset with DINOv2-B (paired over categories)"),
    code("""RES = cfg["results_dir"]
pb = stats.compare({DATASET: df}, "codebook_dist", "coreset_knn", ks=(64, 256, 1024))
pb.to_csv(f"{RES}/p0_{BACKBONE}_{DATASET}_paired.csv", index=False)
print(anomaly.followup_table(df).round(3).to_string())
pb.drop(columns=["p_sign", "p_wilcoxon"]).round(3)"""),
]

cells_12 = [
    md("""# P0 · 3CAD: training folders and higher resolution (exploratory)

1. **Training folders.** Counts, image sizes, exact duplicates (within train, and train images that also appear
   in test), any non-`good` training folders, and a thumbnail sheet per category to eyeball defects in
   the "defect-free" training data.
2. **560 px instead of 448 px** (40 × 40 patch grid). Tests whether the low 3CAD scores are partly a resolution
   floor. DINOv2-S, codebook vs. coreset, M = K ∈ {64, 256, 1,024}, 3 seeds; the full bank is skipped.

**Runtime (T4):** section 1 about 10 min; section 2 about 6–8 h (feature extraction). Finished categories
are saved to Drive; an interrupted category re-extracts after Run all."""),
    *setup_cells(),

    md("## 2 · Data"),
    code("""import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
""" + _DATA_ANY + """
ROOT = get_3cad()
print(ROOT)"""),

    md("## 3 · Training-folder inspection"),
    code("""import pandas as pd
import matplotlib.pyplot as plt
from bovw import review

RES = "/content/drive/MyDrive/bovw-forensics/results"
rows = []
for cat in ad_data.THREECAD_CATEGORIES:
    info, fig = review.inspect_train_folder(ROOT, cat, n_thumbs=48)
    rows.append(info)
    fig.savefig(f"{RES}/p0_3cad_train_{cat}.png", dpi=110, bbox_inches="tight")
    plt.show()
insp = pd.DataFrame(rows)
insp.to_csv(f"{RES}/p0_3cad_train_inspection.csv", index=False)
insp"""),

    md("""## 4 · 560 px run
**Aluminum_New_Ipad** (4,936 test images, about 7.9 million patches at 560 px) needs more RAM than a standard
Colab runtime has: the session crashes and restarts after each run. It is skipped by default; set
`SKIP_CATEGORIES = []` on a high-RAM runtime to include it."""),
    code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from bovw.sweep import load_config
from bovw import anomaly, stats

SKIP_CATEGORIES = ["Aluminum_New_Ipad"]
cfg = load_config("configs/p0_3cad_hires.yaml")
cfg["data"]["root"] = ROOT
cfg["data"]["categories"] = [c for c in ad_data.THREECAD_CATEGORIES if c not in SKIP_CATEGORIES]
hi = anomaly.run(cfg)"""),

    md("""## 5 · 560 px vs. 448 px, and codebook vs. coreset at 560 px
Reads the saved results from Drive, so it works in a fresh session (run Setup, then this cell).
Both resolutions are compared on the same categories and seeds only."""),
    code("""import pandas as pd
from pathlib import Path
from bovw import stats
from bovw.sweep import load_results

RES = "/content/drive/MyDrive/bovw-forensics/results"
hi = load_results(Path(RES) / "p0_3cad_hires.jsonl")
lo = pd.read_csv(f"{RES}/p0_anomaly_3cad.csv")
lo = lo[lo.method.isin(["codebook_dist", "coreset_knn"])]
key = ["category", "method", "k", "vocab_seed"]
for d in (hi, lo):
    d["k"] = d["k"].astype(int)
n_runs = hi.drop_duplicates(key).groupby("category").size()
complete = n_runs[n_runs >= 18].index          # 2 methods x 3 memory sizes x 3 seeds
print("left out (incomplete at 560 px):", sorted(set(n_runs.index) - set(complete)))
common = hi[hi.category.isin(complete)][key].merge(lo[key]).drop_duplicates()
hi_c, lo_c = hi.merge(common), lo.merge(common)
print("categories compared:", sorted(common.category.unique()))
cols = ["image_auroc", "image_auroc_top10", "pixel_auroc", "aupro"]
cmp_ = pd.concat({"448": lo_c, "560": hi_c}, names=["px"]).reset_index(0)
cmp_ = cmp_.groupby(["px", "method", "k", "category"])[cols].mean().groupby(level=[0, 1, 2]).mean().unstack(0).round(3)
cmp_.to_csv(f"{RES}/p0_3cad_hires_vs_448.csv")
ph = stats.compare({"3cad_560": hi_c}, "codebook_dist", "coreset_knn", ks=(64, 256, 1024))
ph.to_csv(f"{RES}/p0_3cad_hires_paired.csv", index=False)
res_gain = stats.compare({"3cad": pd.concat([hi_c.assign(method=hi_c.method + "_560"),
                                              lo_c.assign(method=lo_c.method + "_448")])},
                         "codebook_dist_560", "codebook_dist_448", ks=(64, 256, 1024))
res_gain = pd.concat([res_gain, stats.compare({"3cad": pd.concat([hi_c.assign(method=hi_c.method + "_560"),
                                                                   lo_c.assign(method=lo_c.method + "_448")])},
                                              "coreset_knn_560", "coreset_knn_448", ks=(64, 256, 1024))])
res_gain.to_csv(f"{RES}/p0_3cad_hires_gain.csv", index=False)
display(cmp_)
display(ph.drop(columns=["p_sign", "p_wilcoxon"]).round(3))
res_gain.drop(columns=["p_sign", "p_wilcoxon"]).round(3)"""),
]


cells_13 = [
    md("""# P0 · 3CAD with padded (not stretched) images and a cleaned training set (exploratory)

Notebook 12 found that resolution does not explain 3CAD's low scores, that most categories mix image shapes
(stretched to a square at 448 px), and that Aluminum_Camera_Cover has duplicate and leaked training images.
This run pads every image and mask to a square before resizing and drops duplicate and leaked training
images, then compares with the original 448 px run on the same categories and seeds.

**Runtime (T4):** about 4 h, mostly feature extraction. Finished runs are saved to Drive; after a disconnect,
Run all again (an interrupted category re-extracts). Needs the 3CAD archive in `MyDrive/bovw-forensics/data/`."""),
    *setup_cells(),

    md("## 2 · Data"),
    code("""import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
""" + _DATA_ANY + """
ROOT = get_3cad()
print(ROOT)"""),

    md("## 3 · Run"),
    code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from bovw.sweep import load_config
from bovw import anomaly

cfg = load_config("configs/p0_3cad_pad.yaml")
cfg["data"]["root"] = ROOT
pad = anomaly.run(cfg)"""),

    md("""## 4 · Padded vs. stretched, and codebook vs. coreset when padded
Reads the saved results from Drive, so it works in a fresh session (run Setup, then this cell)."""),
    code("""import pandas as pd
from pathlib import Path
from bovw import stats
from bovw.sweep import load_results

RES = "/content/drive/MyDrive/bovw-forensics/results"
new = load_results(Path(RES) / "p0_3cad_pad.jsonl")
old = pd.read_csv(f"{RES}/p0_anomaly_3cad.csv")
old = old[old.method.isin(["codebook_dist", "coreset_knn"])]
key = ["category", "method", "k", "vocab_seed"]
for d in (new, old):
    d["k"] = d["k"].astype(int)
n_runs = new.drop_duplicates(key).groupby("category").size()
complete = n_runs[n_runs >= 18].index
print("compared:", sorted(complete), " incomplete:", sorted(set(n_runs.index) - set(complete)))
common = new[new.category.isin(complete)][key].merge(old[key]).drop_duplicates()
new_c, old_c = new.merge(common), old.merge(common)
cols = ["image_auroc", "image_auroc_top10", "pixel_auroc", "aupro"]
cmp_ = pd.concat({"stretched": old_c, "padded": new_c}, names=["resize"]).reset_index(0)
cmp_ = cmp_.groupby(["resize", "method", "k", "category"])[cols].mean().groupby(level=[0, 1, 2]).mean().unstack(0).round(3)
cmp_.to_csv(f"{RES}/p0_3cad_pad_vs_stretch.csv")
both = pd.concat([new_c.assign(method=new_c.method + "_pad"), old_c.assign(method=old_c.method + "_stretch")])
gain = pd.concat([stats.compare({"3cad": both}, f"{m}_pad", f"{m}_stretch", ks=(64, 256, 1024))
                  for m in ["codebook_dist", "coreset_knn"]])
gain.to_csv(f"{RES}/p0_3cad_pad_gain.csv", index=False)
pp = stats.compare({"3cad_pad": new_c}, "codebook_dist", "coreset_knn", ks=(64, 256, 1024))
pp.to_csv(f"{RES}/p0_3cad_pad_paired.csv", index=False)
per_cat = pd.concat({"stretched": old_c, "padded": new_c}, names=["resize"]).reset_index(0)
per_cat = per_cat[per_cat.k == 1024].pivot_table(index="category", columns=["method", "resize"], values="aupro").round(3)
per_cat.to_csv(f"{RES}/p0_3cad_pad_per_category_aupro1024.csv")
display(cmp_)
display(per_cat)
display(gain.drop(columns=["p_sign", "p_wilcoxon"]).round(3))
pp.drop(columns=["p_sign", "p_wilcoxon"]).round(3)"""),
]


_RANDOM_CMP = """import pandas as pd
from pathlib import Path
from bovw import stats
from bovw.sweep import load_results

RES = "/content/drive/MyDrive/bovw-forensics/results"
BASE = {"mvtec": "p0_anomaly_mvtec_v2", "visa": "p0_anomaly_visa", "3cad": "p0_anomaly_3cad"}

def random_comparison(ds):
    base = pd.read_csv(f"{RES}/{BASE[ds]}.csv")
    base = base[base.method.isin(["codebook_dist", "coreset_knn"]) & (base.extractor == "dinov2_s")]
    rnd = load_results(Path(RES) / f"p0_random_{ds}.jsonl")
    d = pd.concat([base, rnd], ignore_index=True)
    d["k"] = d["k"].astype(int)
    m = ["aupro", "pixel_auroc", "image_auroc", "image_auroc_top10"]
    out = pd.concat([stats.compare({ds: d}, a, b, ks=(64, 256, 1024), metrics=m)
                     for a, b in [("codebook_dist", "random_knn"), ("random_knn", "coreset_knn")]])
    means = d.groupby(["method", "k", "category"])[m].mean().groupby(level=[0, 1]).mean().round(3)
    return out, means
"""

cells_14 = [
    md("""# P0 · Second referee report: random-sample baseline, token norms, near-duplicates (exploratory)

| Section | Question |
|---|---|
| Random sample of real patches, with AUPRO (MVTec AD, VisA, 3CAD) | how much of the codebook-vs-coreset gap is "k-means is good" and how much is "greedy k-center is a poor choice at small M"? |
| Token norms (DINOv2-B, with and without registers, MVTec AD) | do the highest-scoring patches of normal images have unusually large token norms, as the artifact-token explanation of the K = 64 collapse predicts? |
| Near-duplicates (3CAD) | does a perceptual hash find training images that nearly match test images, beyond exact copies? |

MVTec AD and VisA reuse cached features (no download). 3CAD needs its archive in `MyDrive/bovw-forensics/data/`
and re-extracts features. **Runtime (T4):** about 1 h, plus 3–4 h for the 3CAD random run (last section).
Finished runs are saved; Run all again after a disconnect to resume."""),
    *setup_cells(),

    md("## 2 · Random-sample baseline on MVTec AD and VisA"),
    code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then Run all.")
from bovw.sweep import load_config
from bovw import anomaly
for ds in ["mvtec", "visa"]:
    anomaly.run(load_config(f"configs/p0_random_{ds}.yaml"))"""),
    code(_RANDOM_CMP + """
tabs = []
for ds in ["mvtec", "visa"]:
    out, means = random_comparison(ds)
    out.to_csv(f"{RES}/p0_random_{ds}_paired.csv", index=False)
    means.to_csv(f"{RES}/p0_random_{ds}_means.csv")
    display(means)
    tabs.append(out)
pd.concat(tabs).drop(columns=["p_sign", "p_wilcoxon"]).round(3)"""),

    md("## 3 · Token norms of the highest-scoring normal patches (DINOv2-B, MVTec AD, K = 64, seed 0)"),
    code("""from bovw import ad_data, review
CACHE = "/content/drive/MyDrive/bovw-forensics/cache"
rows = []
for ex in ["dinov2_b", "dinov2_b_reg"]:
    for cat in ad_data.MVTEC_CATEGORIES:
        try:
            rows.append({"category": cat, **review.token_norm_check(CACHE, f"mvtec_{cat}_s448_m256", ex, k=64)})
        except Exception as e:
            print(ex, cat, "skipped:", e)
tn = pd.DataFrame(rows)
tn.to_csv(f"{RES}/p0_token_norms_mvtec.csv", index=False)
cols = ["codebook_top_patch_norm_pct_median", "codebook_top_patch_in_top1pct_norm",
        "coreset_top_patch_norm_pct_median", "coreset_top_patch_in_top1pct_norm",
        "spearman_codebook_score_vs_norm", "spearman_coreset_score_vs_norm"]
tn.groupby("extractor")[cols].mean().round(3)"""),

    md("## 4 · 3CAD: perceptual near-duplicates, then the random-sample baseline"),
    code("""import torch
""" + _DATA_ANY + """
ROOT3 = get_3cad()
nd = pd.DataFrame([review.near_duplicates(ROOT3, c, thresholds=(0, 4, 8)) for c in ad_data.THREECAD_CATEGORIES])
nd.to_csv(f"{RES}/p0_3cad_near_duplicates.csv", index=False)
nd"""),
    code("""cfg = load_config("configs/p0_random_3cad.yaml")
cfg["data"]["root"] = ROOT3
anomaly.run(cfg)
out, means = random_comparison("3cad")
out.to_csv(f"{RES}/p0_random_3cad_paired.csv", index=False)
means.to_csv(f"{RES}/p0_random_3cad_means.csv")
display(means)
out.drop(columns=["p_sign", "p_wilcoxon"]).round(3)"""),
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
write(cells_08, "08_p0_hybrid_3cad.ipynb")
write(cells_09, "09_p0_review_controls.ipynb")
write(cells_10, "10_p0_review_followups.ipynb")
write(cells_11, "11_p0_dinov2b.ipynb")
write(cells_12, "12_p0_3cad_hires.ipynb")
write(cells_13, "13_p0_3cad_pad.ipynb")
write(cells_14, "14_p0_review2.ipynb")
