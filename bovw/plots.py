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


def metric_vs_k(df: pd.DataFrame, metric: str = "nmi", title: str | None = None):
    """Small multiples: one panel per extractor; line per assignment; dashed global baseline."""
    exts = list(dict.fromkeys(df["extractor"]))
    fig, axes = plt.subplots(1, len(exts), figsize=(4.2 * len(exts), 3.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, ex in zip(axes[0], exts):
        _style(ax)
        d = df[df["extractor"] == ex]
        ends = []  # (y, label) at the right edge, for direct labels
        for m, color in METHOD_COLORS.items():
            s = d[d["assignment"] == m].sort_values("k")
            if s.empty:
                continue
            err = s.get(f"{metric}_std")
            ax.errorbar(s["k"], s[metric], yerr=err, color=color, lw=2, marker="o", ms=5,
                        capsize=0, elinewidth=1, label=m)
            ends.append([float(s[metric].iloc[-1]), m, float(s["k"].iloc[-1])])
        _direct_labels(ax, ends)
        g = d[d["assignment"] == "global"]
        if not g.empty:
            y = float(g[metric].iloc[0])
            ax.axhline(y, color=MUTED, lw=1.25, ls="--")
            ax.text(0.02, y, " global embedding", transform=ax.get_yaxis_transform(),
                    va="bottom", fontsize=9, color=INK_2)
        ax.set_xscale("log", base=2)
        ks = sorted(d.loc[d["k"] > 0, "k"].unique())
        ax.set_xticks(ks, [str(int(k)) for k in ks])
        ax.set_title(ex, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Vocabulary size K", color=INK_2, fontsize=9)
    axes[0][0].set_ylabel(metric.upper(), color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=len(labels))
    fig.suptitle(title or f"Clustering {metric.upper()} by vocabulary size", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def cost_vs_metric(df: pd.DataFrame, metric: str = "nmi"):
    """Compute budget (vocab + encode seconds, log scale) vs. metric; color = method, marker = extractor."""
    d = df.copy()
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
    """Extractor x assignment rows, K columns, 'mean ± std' cells."""
    d = df.copy()
    d["cell"] = d.apply(lambda r: f"{r[metric]:.3f} ± {r.get(f'{metric}_std', np.nan):.3f}", axis=1)
    d["k"] = d["k"].where(d["assignment"] != "global", 0).astype(int).astype(str).replace({"0": "—"})
    return d.pivot_table(index=["extractor", "assignment"], columns="k", values="cell", aggfunc="first").fillna("")
