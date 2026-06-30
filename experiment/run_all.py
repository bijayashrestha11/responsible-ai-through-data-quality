"""One command -> regenerates every figure and table (README reproducibility contract).

Pipeline: load Adult -> sweep the full grid -> baseline-correct -> save results/tables/grid.csv
-> produce the scatter figure, detector AUROC, aggregation comparison, EO-vs-DP contrast, and
the where-it-holds/fails regime table.

PRIMARY fairness target = eo_gap_delta (group-correlation effect, baseline-corrected).
Absolute eo_gap is reported as the secondary target.

Usage (from repo root):  python experiment/run_all.py
COMPAS is the confirmatory second dataset and is intentionally not wired here yet.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from experiment.analyze import (  # noqa: E402
    add_baseline_delta, aggregation_comparison, detector_auroc, eo_vs_dp_contrast,
    regime_table, scatter_statistic_vs_target,
)
from experiment.benchmark.load_adult import load_adult  # noqa: E402
from experiment.sweep import run_grid  # noqa: E402

FIGURES = os.path.join(_REPO, "results", "figures")
TABLES = os.path.join(_REPO, "results", "tables")

PRIMARY_TARGET = "eo_gap_delta"
SECONDARY_TARGET = "eo_gap"


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)

    X, y, group, cont_cols, mar_driver = load_adult()
    print(f"Adult loaded: {X.shape[0]} rows x {X.shape[1]} features | "
          f"inject cols = {cont_cols} | protected = sex")

    print("Sweeping grid...")
    df = run_grid(X, y, group, cont_cols, mar_driver)
    df = add_baseline_delta(df)
    grid_path = os.path.join(TABLES, "grid.csv")
    df.to_csv(grid_path, index=False)
    print(f"  -> {len(df)} cells written to {os.path.relpath(grid_path, _REPO)}")

    # 1. scatter + regression, for primary (delta) and secondary (absolute) targets
    r_primary = scatter_statistic_vs_target(
        df, os.path.join(FIGURES, "statistic_vs_eo_delta.png"), target=PRIMARY_TARGET)
    r_secondary = scatter_statistic_vs_target(
        df, os.path.join(FIGURES, "statistic_vs_eo_abs.png"), target=SECONDARY_TARGET)

    # 2. detector (on the primary, baseline-corrected target)
    detector = detector_auroc(df, thresholds=[0.02, 0.05, 0.10], target=PRIMARY_TARGET)
    detector.to_csv(os.path.join(TABLES, "detector_auroc.csv"), index=False)

    # 3. regime + comparisons
    regimes = regime_table(df, target=PRIMARY_TARGET)
    aggcmp = aggregation_comparison(df, target=PRIMARY_TARGET)
    eodp = eo_vs_dp_contrast(df)
    regimes.to_csv(os.path.join(TABLES, "regime_table.csv"), index=False)
    aggcmp.to_csv(os.path.join(TABLES, "aggregation_comparison.csv"), index=False)
    eodp.to_csv(os.path.join(TABLES, "eo_vs_dp_contrast.csv"), index=False)

    print(f"\nHeadline r(D_max, {PRIMARY_TARGET}) = {r_primary:.3f}  "
          f"| secondary r(D_max, {SECONDARY_TARGET}) = {r_secondary:.3f}  (n={len(df)})")
    print(f"\nDetector — AUROC for predicting {PRIMARY_TARGET} exceedance:")
    print(detector.to_string(index=False))
    print("\nWhere it holds vs. fails, by mechanism:")
    print(regimes.to_string(index=False))
    print("\nAggregation comparison (which statistic predicts best):")
    print(aggcmp.to_string(index=False))
    print("\nEO vs DP contrast (expect EO predicted more cleanly):")
    print(eodp.to_string(index=False))
    print(f"\nFigures -> {os.path.relpath(FIGURES, _REPO)}/  |  "
          f"tables -> {os.path.relpath(TABLES, _REPO)}/")


if __name__ == "__main__":
    main()
