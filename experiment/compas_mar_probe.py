"""Explain (or flag) the weak COMPAS MAR signal (Task 6 / issue #21).

COMPAS regime: MCAR r=-0.52, MNAR r=-0.53 (strong) but MAR r=-0.07 (flat). Adult's MAR is strong
(r=0.44). MAR ties injected missingness to a rank of the observed driver; if that driver is highly
CONCENTRATED (many equal values), the within-tie order is arbitrary and MAR degrades toward random
(MCAR-like) for the tied mass, flattening the induced disparity pattern.

Probe: quantify the concentration/ties of each dataset's MAR driver and compare, alongside the
per-mechanism r and mean|Delta EO| from the regime tables. Honest verdict either way.

Usage:  python experiment/compas_mar_probe.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment.benchmark.load_adult import load_adult  # noqa: E402
from experiment.benchmark.load_compas import load_compas  # noqa: E402
from experiment.benchmark.load_german import load_german  # noqa: E402


def concentration(x):
    x = np.asarray(x, float)
    n = len(x)
    vals, counts = np.unique(x, return_counts=True)
    p = counts / n
    return {
        "n": n,
        "frac_zero": round(float((x == 0).mean()), 3),
        "modal_frac": round(float(counts.max() / n), 3),
        "tie_prob_herfindahl": round(float((p ** 2).sum()), 3),   # P(two random rows share a value)
        "frac_nonunique": round(float(counts[counts > 1].sum() / n), 3),
    }


def main():
    specs = [("adult", load_adult, "hours-per-week"),
             ("compas", load_compas, "priors_count"),
             ("german", load_german, "credit_amount")]
    rows = []
    for name, loader, driver in specs:
        X, _, _, _, mar_driver = loader()
        col = mar_driver  # the actual driver the harness uses
        c = concentration(X[col].to_numpy())
        # attach the per-mechanism MAR r from the regime table
        reg = pd.read_csv(os.path.join(_REPO, "results", "tables", name, "regime_table.csv"))
        mar = reg[reg.mechanism == "mar"].iloc[0]
        rows.append({"dataset": name, "mar_driver": col, **c,
                     "r_mar": round(float(mar.pearson_r), 3),
                     "mean_delta_eo_mar": round(float(mar.iloc[3]), 4)})  # signed mean effect
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    a = df[df.dataset == "adult"].iloc[0]
    co = df[df.dataset == "compas"].iloc[0]
    print("\n--- verdict ---")
    print(f"COMPAS MAR driver ({co.mar_driver}): tie-prob {co.tie_prob_herfindahl}, "
          f"frac_zero {co.frac_zero}, modal {co.modal_frac}  |  r_mar {co.r_mar}")
    print(f"Adult  MAR driver ({a.mar_driver}): tie-prob {a.tie_prob_herfindahl}, "
          f"frac_zero {a.frac_zero}, modal {a.modal_frac}  |  r_mar {a.r_mar}")
    if co.tie_prob_herfindahl > 1.5 * a.tie_prob_herfindahl:
        print("=> COMPAS's MAR driver is markedly MORE concentrated/tied than Adult's, consistent")
        print("   with MAR degrading toward random (MCAR-like) and flattening the signal. EXPLAINED.")
    else:
        print("=> COMPAS's MAR driver is NOT markedly more concentrated than Adult's; the tie")
        print("   hypothesis does not cleanly explain the weak MAR signal. FLAG as unexplained.")


if __name__ == "__main__":
    main()
