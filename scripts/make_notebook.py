"""Generates notebooks/00_p0_minimal_stl10.ipynb (keeps the notebook diff-friendly in git)."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# P0 · Minimal loop: BoVW vs. global embeddings on STL-10

Runs the first slice of the systematic revisit paper (P0):

- **Descriptors:** DINOv2-S patch tokens and dense RootSIFT
- **Vocabulary size K:** 64, 256, 1024
- **Assignment:** hard, soft, VLAD (VLAD only up to K = 256)
- **Baseline:** DINOv2 CLS embedding
- **Task:** unsupervised clustering (NMI, ARI, Hungarian accuracy; 3 seeds)

**Runtime:** about 15–25 min on a T4 GPU (*Runtime → Change runtime type → T4 GPU*).
Features are cached on Google Drive, so reruns skip extraction, and finished runs are skipped too."""),

    md("## 1 · Setup"),
    code("""!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: DINOv2 extraction will be slow"

from google.colab import drive
drive.mount('/content/drive')"""),
    code("""# Get the code. Option A: GitHub (set REPO_URL once the repo is pushed).
# Option B: upload bovw-forensics.zip to MyDrive/bovw-forensics/ and leave REPO_URL empty.
REPO_URL = "https://github.com/M3R2T2L2/bovw_forensics.git"   # set to "" to install from the Drive zip instead
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

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "00_p0_minimal_stl10.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
