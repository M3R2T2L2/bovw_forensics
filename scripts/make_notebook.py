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

**Licence:** MVTec AD is CC BY-NC-SA 4.0 (non-commercial). Accept it at
https://www.mvtec.com/company/research/datasets/mvtec-ad before downloading.

**Runtime (T4, 2 vCPU):** about 1–1.5 h for all 15 categories, plus a one-time ~5 GB download.
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

1. **Codebook vs. full patch bank:** how close does `codebook_dist` get to `patch_knn`, and at what K? The bank has ~300k patches per category; K is at most 1,024.
2. **Radius normalisation:** does `codebook_norm` beat plain word distance, especially on pixel AUROC?
3. **Image-level methods:** do `hist_knn` / `global_knn` hold up without localisation?
4. **Per-defect-type AUROC** is stored in `image_auroc_by_type`, useful later for P3 (not all anomalies are equal)."""),
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
