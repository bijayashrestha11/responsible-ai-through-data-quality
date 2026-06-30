"""Generate publication figures for the paper from the saved grid results.

Reads results/tables/<dataset>/grid.csv (no re-running the benchmark) and writes polished
figures to results/figures/paper/ as both PDF (for LaTeX) and PNG (for preview/GitHub):

  fig0_gate_schematic   the validation gate's place in the pipeline (conceptual)
  fig1_scatter_contrast Adult vs COMPAS: D_max vs Delta-EO (the headline + the honest twist)
  fig2_detector_roc     early-warning ROC on Adult
  fig3_mechanism_r      predictive strength by mechanism, both datasets
  fig4_aggregation      aggregation comparison (max/mean/weighted/MI) on Adult

Usage (from repo root):  python experiment/make_figures.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

OUT = os.path.join(_REPO, "results", "figures", "paper")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "legend.frameon": False, "axes.titlesize": 10,
})
MECH = ["mcar", "mar", "mnar"]
COL = {"mcar": "#4C72B0", "mar": "#DD8452", "mnar": "#C44E52"}


def load(ds):
    return pd.read_csv(os.path.join(_REPO, "results", "tables", ds, "grid.csv"))


def pear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _scatter(ax, df, title):
    for m in MECH:
        s = df[df.mechanism == m]
        ax.scatter(s.statistic_max, s.eo_gap_delta, s=14, alpha=0.55,
                   color=COL[m], edgecolor="none", label=m.upper())
    x, y = df.statistic_max.to_numpy(), df.eo_gap_delta.to_numpy()
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, "k--", lw=1.3)
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_title(f"{title}  ($r = {pear(x, y):+.2f}$)")
    ax.set_xlabel(r"$D_{\max}$  (pre-training)")
    ax.set_ylabel(r"$\Delta$ equalized-odds gap")


def fig_scatter_contrast(da, dc):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    _scatter(axes[0], da, "Adult: signal holds")
    _scatter(axes[1], dc, "COMPAS: reverses")
    axes[0].legend(title="mechanism", fontsize=7)
    fig.tight_layout()
    save(fig, "fig1_scatter_contrast")


def fig_detector_roc(da):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for tau, c in [(0.02, "#4C72B0"), (0.05, "#C44E52")]:
        y = (da.eo_gap_delta > tau).astype(int).to_numpy()
        sc = da.statistic_max.to_numpy()
        if y.min() != y.max():
            fpr, tpr, _ = roc_curve(y, sc)
            ax.plot(fpr, tpr, color=c, lw=1.8,
                    label=rf"$\Delta$EO $> {tau}$  (AUROC {roc_auc_score(y, sc):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Early-warning detector (Adult)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig2_detector_roc")


def fig_mechanism_r(da, dc):
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ra = [pear(da[da.mechanism == m].statistic_max, da[da.mechanism == m].eo_gap_delta) for m in MECH]
    rc = [pear(dc[dc.mechanism == m].statistic_max, dc[dc.mechanism == m].eo_gap_delta) for m in MECH]
    x = np.arange(len(MECH))
    w = 0.38
    ax.bar(x - w / 2, ra, w, label="Adult", color="#55A868")
    ax.bar(x + w / 2, rc, w, label="COMPAS", color="#C44E52")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in MECH])
    ax.set_ylabel(r"Pearson $r$  ($D_{\max}$, $\Delta$EO)")
    ax.set_title("Predictive strength by mechanism")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_mechanism_r")


def fig_aggregation(da):
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    cols = [("statistic_max", "max"), ("statistic_mean", "mean"),
            ("statistic_weighted", "weighted"), ("statistic_mi", "MI-wtd")]
    vals = [pear(da[c], da.eo_gap_delta) for c, _ in cols]
    colors = ["#4C72B0", "#8172B3", "#937860", "#55A868"]
    bars = ax.bar([lbl for _, lbl in cols], vals, color=colors)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel(r"Pearson $r$ vs $\Delta$EO")
    ax.set_title("Aggregation comparison (Adult)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_aggregation")


def fig_gate_schematic():
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    def box(cx, cy, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.2, edgecolor="#333333", facecolor=fc))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2, color="#333333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.3, color=color))

    box(1.1, 1.5, 1.8, 0.9, "Ingest /\nTransform", "#ECECEC")
    box(3.9, 1.5, 2.1, 1.1, r"Missingness-Disparity" "\n" r"Gate:  $D_{\max} \leq \tau$ ?", "#FCE5CD")
    box(7.4, 2.2, 2.2, 0.9, "Impute → Train\n→ Model", "#E2EFDA")
    box(7.4, 0.7, 2.2, 0.9, "Halt / Alert", "#F4CCCC")
    arrow(2.0, 1.5, 2.85, 1.5)
    arrow(4.95, 1.8, 6.3, 2.2, "#38761D")
    arrow(4.95, 1.2, 6.3, 0.7, "#990000")
    ax.text(5.5, 2.25, "pass", fontsize=8, color="#38761D")
    ax.text(5.5, 0.72, "fail", fontsize=8, color="#990000")
    save(fig, "fig0_gate_schematic")


def main():
    da, dc = load("adult"), load("compas")
    fig_gate_schematic()
    fig_scatter_contrast(da, dc)
    fig_detector_roc(da)
    fig_mechanism_r(da, dc)
    fig_aggregation(da)
    print(f"Wrote 5 figures (PDF+PNG) to {os.path.relpath(OUT, _REPO)}/")


if __name__ == "__main__":
    main()
