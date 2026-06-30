"""One command -> regenerates every figure and table (README reproducibility contract).

Per dataset: load -> sweep the full grid -> baseline-correct -> save results/tables/<dataset>/
grid.csv -> produce the scatter figure, detector AUROC, aggregation comparison, EO-vs-DP
contrast, and the where-it-holds/fails regime table (under results/figures|tables/<dataset>/).

PRIMARY fairness target = eo_gap_delta (group-correlation effect, baseline-corrected).
Absolute eo_gap is reported as the secondary target.

Usage (from repo root):
  python experiment/run_all.py                  # Adult (default)
  python experiment/run_all.py --dataset compas # COMPAS confirmatory arm
  python experiment/run_all.py --dataset all    # both
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from experiment.analyze import (  # noqa: E402
    add_baseline_delta, aggregation_comparison, detector_auroc, eo_vs_dp_contrast,
    regime_table, scatter_statistic_vs_target,
)
from experiment.benchmark.load_adult import load_adult  # noqa: E402
from experiment.benchmark.load_compas import load_compas  # noqa: E402
from experiment.sweep import run_grid  # noqa: E402

LOADERS = {"adult": load_adult, "compas": load_compas}

PRIMARY_TARGET = "eo_gap_delta"
SECONDARY_TARGET = "eo_gap"


def run_dataset(dataset):
    figures = os.path.join(_REPO, "results", "figures", dataset)
    tables = os.path.join(_REPO, "results", "tables", dataset)
    os.makedirs(figures, exist_ok=True)
    os.makedirs(tables, exist_ok=True)

    X, y, group, cont_cols, mar_driver = LOADERS[dataset]()
    print(f"[{dataset}] loaded: {X.shape[0]} rows x {X.shape[1]} features | "
          f"inject cols = {cont_cols} | privileged-group base rate = {group.mean():.2f}")

    print(f"[{dataset}] sweeping grid...")
    df = run_grid(X, y, group, cont_cols, mar_driver)
    df = add_baseline_delta(df)
    df.to_csv(os.path.join(tables, "grid.csv"), index=False)
    print(f"[{dataset}] -> {len(df)} cells written")

    r_primary = scatter_statistic_vs_target(
        df, os.path.join(figures, "statistic_vs_eo_delta.png"), target=PRIMARY_TARGET)
    r_secondary = scatter_statistic_vs_target(
        df, os.path.join(figures, "statistic_vs_eo_abs.png"), target=SECONDARY_TARGET)

    detector = detector_auroc(df, thresholds=[0.02, 0.05, 0.10], target=PRIMARY_TARGET)
    regimes = regime_table(df, target=PRIMARY_TARGET)
    aggcmp = aggregation_comparison(df, target=PRIMARY_TARGET)
    eodp = eo_vs_dp_contrast(df)
    detector.to_csv(os.path.join(tables, "detector_auroc.csv"), index=False)
    regimes.to_csv(os.path.join(tables, "regime_table.csv"), index=False)
    aggcmp.to_csv(os.path.join(tables, "aggregation_comparison.csv"), index=False)
    eodp.to_csv(os.path.join(tables, "eo_vs_dp_contrast.csv"), index=False)

    print(f"[{dataset}] r(D_max, {PRIMARY_TARGET}) = {r_primary:.3f}  "
          f"| secondary r(D_max, {SECONDARY_TARGET}) = {r_secondary:.3f}  (n={len(df)})")
    print(f"[{dataset}] detector AUROC (predict {PRIMARY_TARGET} exceedance):")
    print(detector.to_string(index=False))
    print(f"[{dataset}] regime (where it holds vs. fails):")
    print(regimes.to_string(index=False))
    print(f"[{dataset}] aggregation comparison:")
    print(aggcmp.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "compas", "all"], default="adult")
    args = ap.parse_args()
    datasets = ["adult", "compas"] if args.dataset == "all" else [args.dataset]
    for d in datasets:
        print(f"\n========== {d.upper()} ==========")
        run_dataset(d)
    print("\nDone. Figures -> results/figures/<dataset>/  |  tables -> results/tables/<dataset>/")


if __name__ == "__main__":
    main()
