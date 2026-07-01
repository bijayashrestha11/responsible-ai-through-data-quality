"""Test the detectability <- sample-size axis WITHIN Adult (Task 5 / issue #22).

The "detectability depends on sample size" claim currently rests on one point (German, n=1000),
where size is confounded with feature count, base rate, and domain. Here we hold EVERYTHING else
fixed and vary only Adult's row count: re-run the full 450-cell grid on stratified subsamples of
Adult at several n, and watch pooled r(D_max, Delta-EO) and the noise indicator mean |Delta EO|.

Falsifiable by design: if r degrades toward 0 as n shrinks (and Adult-at-~1000 collapses toward
German's null while mean |Delta EO| stays non-trivial), the sample-size explanation is supported
with all else fixed. If r stays strong at small n, the German null is NOT primarily sample size.

Writes results/tables/adult/subsample.csv.  Usage:  python experiment/subsample_adult.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from experiment.analyze import _pearson, add_baseline_delta  # noqa: E402
from experiment.benchmark.load_adult import load_adult  # noqa: E402
from experiment.sweep import run_grid  # noqa: E402

TABLES = os.path.join(_REPO, "results", "tables", "adult")
SIZES = [1000, 2000, 5000, 10000, 20000, None]  # None = full


def subsample(X, y, g, n, seed=0):
    """Stratified subsample by (group, y) to n rows, BEFORE any train/test split."""
    if n is None or n >= len(X):
        return X.reset_index(drop=True), y, g
    strat = g.astype(int) * 2 + y.astype(int)
    idx, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=strat, random_state=seed)
    return X.iloc[idx].reset_index(drop=True), y[idx], g[idx]


def auroc(df, tau=0.02):
    lab = (df.eo_gap_delta > tau).astype(int).to_numpy()
    if lab.min() == lab.max():
        return float("nan")
    return float(roc_auc_score(lab, df.statistic_max.to_numpy()))


def main():
    X, y, g, cols, driver = load_adult()
    g = np.asarray(g)
    rows = []
    for n in SIZES:
        Xs, ys, gs = subsample(X, y, g, n)
        df = add_baseline_delta(run_grid(Xs, ys, gs, cols, driver, progress=False))
        rows.append({
            "n": len(Xs),
            "pooled_r": round(_pearson(df.statistic_max, df.eo_gap_delta), 4),
            "r_mcar": round(_pearson(*[df[df.mechanism == "mcar"][c] for c in
                                      ("statistic_max", "eo_gap_delta")]), 4),
            "r_mar": round(_pearson(*[df[df.mechanism == "mar"][c] for c in
                                     ("statistic_max", "eo_gap_delta")]), 4),
            "r_mnar": round(_pearson(*[df[df.mechanism == "mnar"][c] for c in
                                      ("statistic_max", "eo_gap_delta")]), 4),
            "auroc_0.02": round(auroc(df), 4),
            "mean_abs_delta_eo": round(float(df.eo_gap_delta.abs().mean()), 4),
        })
        print(f"  n={len(Xs):6d}  pooled_r={rows[-1]['pooled_r']:+.3f}  "
              f"mean|dEO|={rows[-1]['mean_abs_delta_eo']:.4f}")
    out = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    out.to_csv(os.path.join(TABLES, "subsample.csv"), index=False)
    print("\n" + out.to_string(index=False))

    # verdict
    small = out.iloc[0]   # n=1000
    full = out.iloc[-1]   # full
    print(f"\nVERDICT: pooled r goes {full.pooled_r:+.3f} (n={int(full.n)}) -> "
          f"{small.pooled_r:+.3f} (n={int(small.n)}); mean|dEO| stays "
          f"{full.mean_abs_delta_eo:.3f} -> {small.mean_abs_delta_eo:.3f}.")
    if abs(small.pooled_r) < 0.15 and small.mean_abs_delta_eo > 0.005:
        print("  -> Adult-at-small-n collapses toward null while the effect magnitude persists:")
        print("     the sample-size / noise-floor explanation of the German null is SUPPORTED.")
    elif abs(small.pooled_r) >= 0.25:
        print("  -> r stays strong at small n: the German null is NOT primarily sample size;")
        print("     the detectability axis must be weakened in the paper.")
    else:
        print("  -> partial/ambiguous: report the r-vs-n curve honestly without overclaiming.")


if __name__ == "__main__":
    main()
