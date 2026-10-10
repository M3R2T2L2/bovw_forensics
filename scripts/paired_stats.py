"""Rebuild results_published/stats_paired_category_bootstrap.csv from the published per-run CSVs."""
from pathlib import Path

import pandas as pd

from bovw import stats

R = Path(__file__).resolve().parents[1] / "results_published"
ds = {n: pd.read_csv(R / f) for n, f in [("MVTec AD", "p0_anomaly_mvtec_v2.csv"), ("VisA", "p0_anomaly_visa.csv"),
                                         ("3CAD", "p0_anomaly_3cad.csv")]}
hy = {"MVTec AD": pd.concat([ds["MVTec AD"], pd.read_csv(R / "p0_hybrid_mvtec.csv")]),
      "VisA": pd.concat([ds["VisA"], pd.read_csv(R / "p0_hybrid_visa.csv")]),
      "3CAD": pd.read_csv(R / "p0_hybrid_3cad.csv")}  # 3CAD hybrid run reran codebook/coreset on its own features
out = pd.concat([stats.compare(ds, "codebook_dist", "coreset_knn"),
                 stats.compare(hy, "hybrid_w75", "codebook_dist", ks=(256, 1024)),
                 stats.compare(hy, "hybrid_w75", "coreset_knn", ks=(256, 1024))], ignore_index=True)
out.to_csv(R / "stats_paired_category_bootstrap.csv", index=False)
print(f"wrote {len(out)} rows")
