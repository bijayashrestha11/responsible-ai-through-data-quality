"""Analysis of the grid results (experiment-plan.md, "Analysis" / "Outputs").

Produces the three things the paper needs:
  1. scatter + regression : pre-training statistic vs post-training fairness target
  2. detector             : threshold on the statistic -> flag target exceedance (AUROC)
  3. regime table         : where the signal holds vs. fails, by mechanism

Plus the max/mean/weighted aggregation comparison and the EO-vs-DP contrast.

PRIMARY target is the BASELINE-CORRECTED EO gap (``eo_gap_delta``): the EO gap of a cell minus
the EO gap of its matching no-disparity baseline (same mechanism/base_rate/seed, group_gap=0).
This isolates the contribution of group-correlated missingness from Adult's inherent model
unfairness and from missingness that is present but not group-correlated. Absolute ``eo_gap``
is kept as a secondary target.

Uses numpy for correlations (no scipy dependency); roc_auc_score from scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float:
    xr = pd.Series(np.asarray(x)).rank().to_numpy()
    yr = pd.Series(np.asarray(y)).rank().to_numpy()
    return _pearson(xr, yr)


def add_baseline_delta(df) -> pd.DataFrame:
    """Add eo_gap_delta / dp_gap_delta = gap minus the matching group_gap==0 baseline cell
    (same mechanism, base_rate, seed)."""
    keys = ["mechanism", "base_rate", "seed"]
    base = df[df["group_gap"] == 0].groupby(keys)[["eo_gap", "dp_gap"]].mean()
    base = base.rename(columns={"eo_gap": "eo_base", "dp_gap": "dp_base"})
    out = df.merge(base, on=keys, how="left")
    out["eo_gap_delta"] = out["eo_gap"] - out["eo_base"]
    out["dp_gap_delta"] = out["dp_gap"] - out["dp_base"]
    return out


def scatter_statistic_vs_target(df, outpath, stat="statistic_max", target="eo_gap_delta") -> float:
    x = df[stat].to_numpy(float)
    yv = df[target].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for mech, sub in df.groupby("mechanism"):
        ax.scatter(sub[stat], sub[target], label=mech, alpha=0.55, s=20)
    if np.std(x) > 0:
        b1, b0 = np.polyfit(x, yv, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, "k--", lw=1.2)
    r = _pearson(x, yv)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title(f"D_max vs {target}  (Pearson r = {r:.2f})")
    ax.set_xlabel("missingness-disparity statistic  (D_max, pre-training)")
    ax.set_ylabel(f"{target}  (post-training)")
    ax.legend(title="mechanism")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return r


def detector_auroc(df, thresholds, stat="statistic_max", target="eo_gap_delta") -> pd.DataFrame:
    score = df[stat].to_numpy(float)
    rows = []
    for tau in thresholds:
        label = (df[target] > tau).astype(int).to_numpy()
        auroc = float("nan") if label.min() == label.max() else roc_auc_score(label, score)
        rows.append({"target": target, "threshold": tau, "positives": int(label.sum()),
                     "n": int(len(label)), "auroc": auroc})
    return pd.DataFrame(rows)


def aggregation_comparison(df, target="eo_gap_delta") -> pd.DataFrame:
    rows = []
    for stat in ["statistic_max", "statistic_mean", "statistic_weighted"]:
        rows.append({"statistic": stat, "target": target,
                     "pearson_r": _pearson(df[stat], df[target]),
                     "spearman_r": _spearman(df[stat], df[target])})
    return pd.DataFrame(rows)


def eo_vs_dp_contrast(df, stat="statistic_max") -> pd.DataFrame:
    rows = []
    for target in ["eo_gap_delta", "dp_gap_delta", "eo_gap", "dp_gap"]:
        if target in df.columns:
            rows.append({"target": target, "pearson_r": _pearson(df[stat], df[target])})
    return pd.DataFrame(rows)


def regime_table(df, stat="statistic_max", target="eo_gap_delta") -> pd.DataFrame:
    rows = []
    for mech, sub in df.groupby("mechanism"):
        rows.append({"mechanism": mech, "n": int(len(sub)),
                     "mean_statistic": float(sub[stat].mean()),
                     f"mean_{target}": float(sub[target].mean()),
                     "pearson_r": _pearson(sub[stat], sub[target]),
                     "spearman_r": _spearman(sub[stat], sub[target])})
    return pd.DataFrame(rows)
