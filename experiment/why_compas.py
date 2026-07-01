"""Why does the D_max -> Delta-EO correlation reverse on COMPAS? (Task 1 / issue #12)

Goal: upgrade "dataset-dependent" to a boundary condition by finding a structural property whose
sign predicts the sign of the pooled correlation.

Probes:
  1. Structural scalars for each (dataset, encoding): is the unprivileged group (label 0, the one
     the injection concentrates missingness in) a minority or majority? signed base-rate
     difference (unpriv - priv); group size ratio; corr(protected, target).
  2. Leading hypothesis (encoding flip): re-run the grid with the protected-group encoding flipped
     (so the OTHER group carries the excess missingness). Adult is included as a control. If the
     sign flips in both, the direction is governed by which group -- relative to advantage/size --
     carries the excess missingness.

Writes results/tables/compas/why_reversal.csv. All grids reuse the existing harness unchanged.

Usage (from repo root):  python experiment/why_compas.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment.analyze import _pearson, add_baseline_delta  # noqa: E402
from experiment.benchmark.load_adult import load_adult  # noqa: E402
from experiment.benchmark.load_compas import load_compas  # noqa: E402
from experiment.sweep import run_grid  # noqa: E402

TABLES = os.path.join(_REPO, "results", "tables", "compas")


def structural(y, group, config):
    g, y = np.asarray(group), np.asarray(y)
    n0, n1 = int((g == 0).sum()), int((g == 1).sum())
    br0, br1 = float(y[g == 0].mean()), float(y[g == 1].mean())
    return {
        "config": config,
        "unpriv_frac": round(n0 / (n0 + n1), 3),
        "unpriv_is_minority": bool(n0 < n1),
        "size_ratio_unpriv_over_priv": round(n0 / n1, 3),
        "base_rate_priv": round(br1, 4),
        "base_rate_unpriv": round(br0, 4),
        "baserate_diff_unpriv_minus_priv": round(br0 - br1, 4),
        "corr_group_target": round(float(np.corrcoef(g, y)[0, 1]), 4),
    }


def r_from_cached(ds):
    d = pd.read_csv(os.path.join(_REPO, "results", "tables", ds, "grid.csv"))
    return _pearson(d.statistic_max, d.eo_gap_delta)


def r_from_run(X, y, group, cols, driver):
    d = add_baseline_delta(run_grid(X, y, group, cols, driver, progress=False))
    return _pearson(d.statistic_max, d.eo_gap_delta)


def main():
    Xa, ya, ga, ca, dra = load_adult()
    Xc, yc, gc, cc, drc = load_compas()
    ga, gc = np.asarray(ga), np.asarray(gc)

    ra, rc = r_from_cached("adult"), r_from_cached("compas")
    print("Running flipped-encoding grids (Adult control + COMPAS)... ~a few minutes")
    ra_flip = r_from_run(Xa, ya, 1 - ga, ca, dra)
    rc_flip = r_from_run(Xc, yc, 1 - gc, cc, drc)

    rows = []
    for y, g, cfg, r in [
        (ya, ga, "adult  orig  (unpriv=Female)", ra),
        (ya, 1 - ga, "adult  flip  (unpriv=Male)", ra_flip),
        (yc, gc, "compas orig  (unpriv=African-American)", rc),
        (yc, 1 - gc, "compas flip  (unpriv=Caucasian)", rc_flip),
    ]:
        s = structural(y, g, cfg)
        s["pooled_r"] = round(float(r), 4)
        rows.append(s)
    df = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, "why_reversal.csv"), index=False)

    print("\n" + df.to_string(index=False))

    print("\n--- encoding-flip test ---")
    print(f"  COMPAS: orig r={rc:+.3f}  ->  flipped r={rc_flip:+.3f}   "
          f"({'SIGN FLIPS' if np.sign(rc) != np.sign(rc_flip) else 'same sign'})")
    print(f"  Adult : orig r={ra:+.3f}  ->  flipped r={ra_flip:+.3f}   "
          f"({'SIGN FLIPS' if np.sign(ra) != np.sign(ra_flip) else 'same sign'})")

    # Which structural scalar's sign separates the four configs by sign(pooled_r)?
    print("\n--- candidate boundary conditions (does the scalar separate +r from -r across all 4?) ---")
    sign_r = np.sign(df.pooled_r.to_numpy())
    checks = {
        "unpriv_is_minority (True->+r)": np.where(df.unpriv_is_minority, 1, -1),
        "sign(baserate_diff unpriv-priv)": np.sign(df.baserate_diff_unpriv_minus_priv.to_numpy()),
        "sign(corr_group_target)": np.sign(df.corr_group_target.to_numpy()),
    }
    for name, pred in checks.items():
        aligned = np.all(pred == sign_r) or np.all(pred == -sign_r)
        print(f"  {'SEPARATES' if aligned else 'no      '}  {name}: {pred.tolist()}  vs sign(r) {sign_r.astype(int).tolist()}")
    print(f"\nWrote {os.path.relpath(os.path.join(TABLES, 'why_reversal.csv'), _REPO)}")


if __name__ == "__main__":
    main()
