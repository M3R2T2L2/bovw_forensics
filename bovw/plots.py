"""Standard P0 figures from a sweep results DataFrame.

Colors follow a fixed categorical order keyed by assignment method, so a
method keeps its color across every figure and paper.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {"hard": "#2a78d6", "soft": "#eb6834", "vlad": "#1baf7a"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e8e7e3"
EXTRACTOR_MARKERS = ["o", "s", "^", "D", "v", "P"]
KNN_RAMP = ["#86b6ef", "#2a78d6", "#104281", "#0d366b"]  # ordered: light -> dark as knn grows
METRICS = ("nmi", "ari", "acc")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse vocabulary seeds: mean per (extractor, assignment, variant, k).

    `<metric>_std` becomes the total spread: within-run (clustering seeds) plus
    between vocabulary seeds, sqrt(mean(within var) + var(seed means)).
    """
    d = df.copy()
    d["variant"] = d.get("variant", pd.Series("", index=d.index)).fillna("")
    keys = ["extractor", "assignment", "variant", "k"]
    out = []
    for key, g in d.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row["n_vocab_seeds"] = len(g)
        for m in METRICS:
            if m not in g:
                continue
            within = (g[f"{m}_std"] ** 2).mean() if f"{m}_std" in g else 0.0
            row[m] = g[m].mean()
            row[f"{m}_std"] = float(np.sqrt(within + g[m].var(ddof=0)))
        for c in ("vocab_seconds", "encode_seconds", "eval_seconds", "extract_seconds", "enc_dim"):
            if c in g:
                row[c] = g[c].mean()
        out.append(row)
    return pd.DataFrame(out)


def _one_variant(d: pd.DataFrame, method: str, variants: dict | None) -> pd.DataFrame:
    """Rows of one encode-param variant for `method` (the chosen one if several were run)."""
    s = d[d["assignment"] == method]
    if s["variant"].nunique() <= 1:
        return s
    want = (variants or {}).get(method)
    if want is None:
        raise ValueError(f"'{method}' has several variants {sorted(s['variant'].unique())}; "
                         f"pass variants={{'{method}': '<one of them>'}} or use soft_sensitivity().")
    return s[s["variant"] == want]


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _direct_labels(ax, ends, min_gap_frac: float = 0.07):
    """Label line ends, merging labels that land on the same value and spreading near ones."""
    if not ends:
        return
    ends.sort(key=lambda e: e[0])
    merged = []
    for y, lab, x in ends:
        if merged and abs(y - merged[-1][0]) < 1e-3:
            merged[-1][1] += f" / {lab}"
        else:
            merged.append([y, lab, x])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    ys = [m[0] for m in merged]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    pts_per_unit = ax.bbox.height * 72 / ax.figure.dpi / (hi - lo)
    for (y0, lab, x), y in zip(merged, ys):
        ax.annotate(lab, (x, y0), xytext=(8, (y - y0) * pts_per_unit), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2, annotation_clip=False)


def metric_vs_k(df: pd.DataFrame, metric: str = "nmi", title: str | None = None, variants: dict | None = None):
    """Small multiples: one panel per extractor; line per assignment; dashed global baseline.

    Vocabulary seeds are averaged; error bars show the total spread.
    """
    df = aggregate(df)
    exts = list(dict.fromkeys(df["extractor"]))
    # Independent y per panel: descriptor families can differ by an order of magnitude.
    fig, axes = plt.subplots(1, len(exts), figsize=(4.4 * len(exts), 3.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, ex in zip(axes[0], exts):
        _style(ax)
        d = df[df["extractor"] == ex]
        ends = []  # (y, label) at the right edge, for direct labels
        for m, color in METHOD_COLORS.items():
            s = _one_variant(d, m, variants).sort_values("k")
            if s.empty:
                continue
            err = s.get(f"{metric}_std")
            ax.errorbar(s["k"], s[metric], yerr=err, color=color, lw=2, marker="o", ms=5,
                        capsize=0, elinewidth=1, label=m)
            ends.append([float(s[metric].iloc[-1]), m, float(s["k"].iloc[-1])])
        g = d[d["assignment"] == "global"]
        if not g.empty:
            ax.axhline(float(g[metric].iloc[0]), color=MUTED, lw=1.25, ls="--", label="global embedding")
        # Direct labels only for lines reaching the right edge (VLAD stops early; legend covers it).
        kmax = max(e[2] for e in ends) if ends else 0
        _direct_labels(ax, [e for e in ends if e[2] == kmax])
        ax.set_xscale("log", base=2)
        ks = sorted(d.loc[d["k"] > 0, "k"].unique())
        ax.set_xticks(ks, [str(int(k)) for k in ks])
        ax.set_title(ex, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Vocabulary size K", color=INK_2, fontsize=9)
    axes[0][0].set_ylabel(metric.upper(), color=INK_2, fontsize=9)
    seen = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc="upper right", frameon=False,
                   fontsize=9, ncol=len(seen))
    fig.suptitle(title or f"Clustering {metric.upper()} by vocabulary size", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def cost_vs_metric(df: pd.DataFrame, metric: str = "nmi"):
    """Compute budget (vocab + encode seconds, log scale) vs. metric; color = method, marker = extractor."""
    d = aggregate(df)
    d["cost_s"] = d[["vocab_seconds", "encode_seconds"]].fillna(0).sum(1).clip(lower=1e-3)
    exts = list(dict.fromkeys(d["extractor"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    _style(ax)
    for i, ex in enumerate(exts):
        for m, color in {**METHOD_COLORS, "global": MUTED}.items():
            s = d[(d["extractor"] == ex) & (d["assignment"] == m)]
            if not s.empty:
                ax.scatter(s["cost_s"], s[metric], s=60, color=color, edgecolor="#fcfcfb", lw=2,
                           marker=EXTRACTOR_MARKERS[i % len(EXTRACTOR_MARKERS)], zorder=3)
    from matplotlib.ticker import FuncFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Vocabulary + encoding time (s, log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel(metric.upper(), color=INK_2, fontsize=9)
    ax.set_title(f"Cost vs. {metric.upper()} (extraction time excluded)", loc="left", color=INK, fontsize=12)
    method_h = [plt.Line2D([], [], ls="", marker="o", color=c, label=m)
                for m, c in {**METHOD_COLORS, "global": MUTED}.items() if m in set(d["assignment"])]
    ext_h = [plt.Line2D([], [], ls="", marker=EXTRACTOR_MARKERS[i % 6], markerfacecolor="none",
                        markeredgecolor=INK_2, label=ex) for i, ex in enumerate(exts)]
    leg1 = ax.legend(handles=method_h, title="Assignment", frameon=False, fontsize=9, title_fontsize=9,
                     loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.add_artist(leg1)
    ax.legend(handles=ext_h, title="Extractor", frameon=False, fontsize=9, title_fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def summary_table(df: pd.DataFrame, metric: str = "nmi") -> pd.DataFrame:
    """Extractor x assignment (x variant) rows, K columns, 'mean ± std' cells over all seeds."""
    d = aggregate(df)
    d["cell"] = d.apply(lambda r: f"{r[metric]:.3f} ± {r.get(f'{metric}_std', np.nan):.3f}", axis=1)
    d["k"] = d["k"].where(d["assignment"] != "global", 0).astype(int).astype(str).replace({"0": "—"})
    idx = ["extractor", "assignment"] + (["variant"] if d["variant"].nunique() > 1 else [])
    return d.pivot_table(index=idx, columns="k", values="cell", aggfunc="first").fillna("")


def soft_sensitivity(df: pd.DataFrame, metric: str = "nmi"):
    """Rows = extractor, cols = K. x = sigma_scale (log2), line per knn; dashed = hard at same K."""
    import json

    d = aggregate(df)
    soft = d[d["assignment"] == "soft"].copy()
    if soft.empty:
        raise ValueError("No soft-assignment rows.")
    params = soft["variant"].map(lambda v: dict(kv.split("=") for kv in v.split(",")) if v else {})
    soft["knn"] = params.map(lambda p: int(p.get("knn", 5)))
    soft["sigma_scale"] = params.map(lambda p: float(p.get("sigma_scale", 1.0)))
    exts = list(dict.fromkeys(soft["extractor"]))
    ks = sorted(soft["k"].unique())
    fig, axes = plt.subplots(len(exts), len(ks), figsize=(3.4 * len(ks), 2.9 * len(exts)),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    knns = sorted(soft["knn"].unique())
    colors = {kn: KNN_RAMP[min(i, len(KNN_RAMP) - 1)] for i, kn in enumerate(knns)} if len(knns) > 1 \
        else {knns[0]: KNN_RAMP[1]}
    for r, ex in enumerate(exts):
        for c, k in enumerate(ks):
            ax = axes[r][c]
            _style(ax)
            g = soft[(soft["extractor"] == ex) & (soft["k"] == k)]
            for kn in knns:
                s = g[g["knn"] == kn].sort_values("sigma_scale")
                if not s.empty:
                    ax.errorbar(s["sigma_scale"], s[metric], yerr=s[f"{metric}_std"], color=colors[kn],
                                lw=2, marker="o", ms=4, elinewidth=1, label=f"knn={kn}")
            h = d[(d["extractor"] == ex) & (d["assignment"] == "hard") & (d["k"] == k)]
            if not h.empty:
                ax.axhline(float(h[metric].iloc[0]), color=MUTED, lw=1.25, ls="--")
                ax.text(0.98, float(h[metric].iloc[0]), "hard", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=8.5, color=INK_2)
            ax.set_xscale("log", base=2)
            xs = sorted(g["sigma_scale"].unique())
            ax.set_xticks(xs, [f"{x:g}" for x in xs])
            ax.set_title(f"{ex} · K={int(k)}", loc="left", color=INK, fontsize=10)
            if c == 0:
                ax.set_ylabel(metric.upper(), color=INK_2, fontsize=9)
            if r == len(exts) - 1:
                ax.set_xlabel("sigma scale (× median NN distance)", color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=len(labels))
    fig.suptitle(f"Soft assignment sensitivity ({metric.upper()})", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


AD_COLORS = {"codebook_dist": "#2a78d6", "codebook_norm": "#eb6834", "hist_knn": "#1baf7a",
             "coreset_knn": "#8a5cc2", "random_knn": "#a0a39b"}
AD_LABELS = {"codebook_dist": "word distance", "codebook_norm": "word distance / radius",
             "hist_knn": "histogram kNN", "patch_knn": "patch kNN (PatchCore-style)", "global_knn": "CLS kNN",
             "coreset_knn": "coreset kNN (M = K)", "random_knn": "random-sample kNN (M = K)"}


def anomaly_vs_k(df: pd.DataFrame, metrics: tuple = ("image_auroc", "pixel_auroc")):
    """Mean AUROC over categories vs vocabulary size; rows = extractor, cols = metric.

    Codebook methods are lines; baselines that do not depend on K are horizontal
    reference lines (dashed: patch kNN, dotted: CLS kNN). Five series share one
    legend; direct labels are omitted because the method names are long.
    """
    d = df.copy()
    mean = d.groupby(["extractor", "method", "k"])[list(metrics)].mean().reset_index()  # over categories + seeds
    exts = list(dict.fromkeys(d["extractor"]))
    fig, axes = plt.subplots(len(exts), len(metrics), figsize=(4.6 * len(metrics), 3.4 * len(exts)), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for r, ex in enumerate(exts):
        for c, met in enumerate(metrics):
            ax = axes[r][c]
            _style(ax)
            m = mean[mean["extractor"] == ex]
            for meth, color in AD_COLORS.items():
                s = m[m["method"] == meth].sort_values("k").dropna(subset=[met])
                if s.empty:
                    continue
                ax.plot(s["k"], s[met], color=color, lw=2, marker="o", ms=5, label=AD_LABELS[meth])
            for meth, ls in (("patch_knn", "--"), ("global_knn", ":")):
                s = m[m["method"] == meth].dropna(subset=[met])
                if not s.empty:
                    ax.axhline(float(s[met].iloc[0]), color=MUTED, lw=1.5, ls=ls, label=AD_LABELS[meth])
            ks = sorted(m.loc[m["k"] > 0, "k"].unique())
            if ks:
                ax.set_xscale("log", base=2)
                ax.set_xticks(ks, [str(int(k)) for k in ks])
            ax.set_title(f"{ex} · {met.replace('_', ' ')}", loc="left", color=INK, fontsize=10.5)
            if r == len(exts) - 1:
                ax.set_xlabel("Vocabulary size K", color=INK_2, fontsize=9)
            if c == 0:
                ax.set_ylabel("Mean AUROC over categories", color=INK_2, fontsize=9)
    seen = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", ncol=min(len(seen), 4),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def false_alarms(results_dir, cfg_name: str, category: str, images: list, labels: np.ndarray,
                 methods=(("codebook_dist", 1024), ("coreset_knn", 1024)), extractor: str = "dinov2_s",
                 seed: int = 0, n: int = 4):
    """The normal test images each method ranks as most anomalous, with both methods' maps.

    Columns: image, then one map per method. Each map is divided by that method's
    99th-percentile normal-pixel score, so 1.0 means "as high as the noisiest 1%
    of normal pixels" in both columns. A cross marks the max-scoring location.
    """
    from pathlib import Path

    from .sweep import _run_id

    def rid(meth, k):
        return _run_id(cfg_name, category, extractor, meth, k, 0 if meth == "patch_knn" else seed).replace("|", "__")

    labels = np.asarray(labels)
    good = np.where(labels == 0)[0]
    maps, scores = {}, {}
    for meth, k in methods:
        maps[meth] = np.load(Path(results_dir) / "maps" / f"{rid(meth, k)}.npy").astype(np.float32)
        scores[meth] = np.load(Path(results_dir) / "scores" / f"{rid(meth, k)}.npy")
    picks = []
    for meth, _ in methods:  # worst false alarms of each method, deduplicated
        r = scores[meth][good]
        for i in good[np.argsort(-r)]:
            if i not in picks:
                picks.append(int(i))
                break
        if len(picks) >= n:
            break
    first = methods[0][0]
    for i in good[np.argsort(-scores[first][good])]:
        if len(picks) >= n:
            break
        if int(i) not in picks:
            picks.append(int(i))
    scale = {}
    for meth in maps:
        neg = maps[meth][labels == 0]
        scale[meth] = float(np.quantile(neg, 0.99))
    ncol = 1 + len(methods)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.6 * ncol, 2.6 * len(picks)), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for r, i in enumerate(picks):
        ax = axes[r][0]
        ax.imshow(images[i])
        ax.set_axis_off()
        if r == 0:
            ax.set_title("normal test image", fontsize=9, color=INK, loc="left")
        for c, (meth, k) in enumerate(methods, start=1):
            ax = axes[r][c]
            mp = maps[meth][i] / scale[meth]
            ax.imshow(images[i], alpha=1.0)
            ax.imshow(mp, cmap="magma", vmin=0.5, vmax=1.5, alpha=0.6,
                      extent=(0, images[i].shape[1], images[i].shape[0], 0))
            y, x = np.unravel_index(mp.argmax(), mp.shape)
            sy, sx = images[i].shape[0] / mp.shape[0], images[i].shape[1] / mp.shape[1]
            ax.plot(x * sx, y * sy, "x", color="#ffffff", ms=9, mew=2)
            rank = int((scores[meth] >= scores[meth][i]).sum())
            ax.set_title(f"{AD_LABELS.get(meth, meth)} · rank {rank}/{len(labels)}", fontsize=8.5, color=INK, loc="left")
            ax.set_axis_off()
    fig.suptitle(f"{category}: normal images ranked most anomalous (maps / normal p99)", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return fig
