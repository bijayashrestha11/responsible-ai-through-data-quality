"""Place all three datasets against the Task 1 boundary condition (Task 2 / issue #13).

Task 1 found (via the encoding-flip test) that the SIGN of r(D_max, Delta-EO) is governed by which
group carries the excess missingness: positive when it falls on the minority / lower-base-rate
group. This script computes the structural scalars + the observed pooled r + the effect size for
each dataset, so the third dataset (German) can confirm, break, or refine that rule.

Writes results/tables/boundary_check.csv.  Usage:  python experiment/boundary_check.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment.analyze import _pearson  # noqa: E402
from experiment.benchmark.load_adult import load_adult  # noqa: E402
from experiment.benchmark.load_compas import load_compas  # noqa: E402
from experiment.benchmark.load_german import load_german  # noqa: E402

LOADERS = {"adult": load_adult, "compas": load_compas, "german": load_german}


def main():
    rows = []
    for name, loader in LOADERS.items():
        _, y, g, _, _ = loader()
        g, y = np.asarray(g), np.asarray(y)
        n0, n1 = int((g == 0).sum()), int((g == 1).sum())
        br0, br1 = float(y[g == 0].mean()), float(y[g == 1].mean())
        d = pd.read_csv(os.path.join(_REPO, "results", "tables", name, "grid.csv"))
        r = _pearson(d.statistic_max, d.eo_gap_delta)
        mean_abs = float(d.eo_gap_delta.abs().mean())
        unpriv_minority = n0 < n1
        predicted = "+ (minority rule)" if unpriv_minority else "- (majority rule)"
        if abs(r) < 0.05:
            outcome = "NULL (no usable effect)"
        elif (r > 0) == unpriv_minority:
            outcome = "matches minority rule"
        else:
            outcome = "breaks minority rule"
        rows.append({
            "dataset": name,
            "unpriv_frac": round(n0 / (n0 + n1), 3),
            "unpriv_is_minority": unpriv_minority,
            "baserate_diff_unpriv_minus_priv": round(br0 - br1, 4),
            "corr_group_target": round(float(np.corrcoef(g, y)[0, 1]), 4),
            "mean_abs_delta_eo": round(mean_abs, 4),
            "pooled_r": round(r, 4),
            "predicted_sign": predicted,
            "outcome": outcome,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_REPO, "results", "tables", "boundary_check.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
