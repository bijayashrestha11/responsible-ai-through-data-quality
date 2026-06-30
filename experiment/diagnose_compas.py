"""Diagnose the COMPAS reversal (issue #6).

The COMPAS arm found r(D_max, eo_gap_delta) = -0.31 (vs +0.40 on Adult) — the predictive
relationship reverses. Before treating that as a clean counterexample, test whether it is a
structural artifact of COMPAS's heavily zero-inflated injection features (priors_count, juv_*)
interacting with MNAR masking + median imputation, or a real, robust effect.

Three probes:
  1. Zero-inflation profile   — how zero-inflated are COMPAS injection features vs Adult's?
  2. Noise floor              — is the |Δ(EO gap)| effect above seed-to-seed noise?
  3. Design variants          — does the sign of r recover when we change the injection set or
                                the imputation (the two things the artifact hypothesis blames)?

Usage (from repo root):  python experiment/diagnose_compas.py
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


def zero_inflation(X, cols):
    return {c: float((X[c] == 0).mean()) for c in cols}


def noise_floor(df):
    """Mean within-(mechanism,base_rate,group_gap) seed std of eo_gap — the noise in the EO
    estimate — compared to the mean magnitude of the disparity effect |eo_gap_delta|."""
    seed_std = df.groupby(["mechanism", "base_rate", "group_gap"])["eo_gap"].std().mean()
    effect = df["eo_gap_delta"].abs().mean()
    return float(seed_std), float(effect)


def variant_r(X, y, group, inject_columns, mar_driver, impute_strategy, label):
    df = run_grid(X, y, group, inject_columns, mar_driver, progress=False,
                  impute_strategy=impute_strategy)
    df = add_baseline_delta(df)
    r = _pearson(df["statistic_max"], df["eo_gap_delta"])
    print(f"  [{label}] r(D_max, eo_gap_delta) = {r:+.3f}  (n={len(df)})")
    return {"variant": label, "inject": ",".join(inject_columns),
            "impute": impute_strategy, "pearson_r": r}


def main():
    os.makedirs(TABLES, exist_ok=True)
    Xc, yc, gc, cont_c, driver_c = load_compas()
    Xa, ya, ga, cont_a, _ = load_adult()

    # --- Probe 1: zero-inflation profile -------------------------------------------------
    zi_c = zero_inflation(Xc, cont_c)
    zi_a = zero_inflation(Xa, cont_a)
    print("Probe 1 — fraction of zeros in injection features:")
    print("  COMPAS:", {k: round(v, 2) for k, v in zi_c.items()})
    print("  Adult :", {k: round(v, 2) for k, v in zi_a.items()})
    least_zi = sorted(cont_c, key=lambda c: zi_c[c])  # least zero-inflated first

    # --- Probe 2: noise floor (uses the baseline full-design COMPAS run) ------------------
    print("\nProbe 2 — noise floor (full design, median impute):")
    base = run_grid(Xc, yc, gc, cont_c, driver_c, progress=False)
    base = add_baseline_delta(base)
    seed_std, effect = noise_floor(base)
    print(f"  mean seed-to-seed std of eo_gap = {seed_std:.4f}")
    print(f"  mean |eo_gap_delta| (the effect) = {effect:.4f}")
    print(f"  effect-to-noise ratio = {effect / seed_std:.2f}  "
          f"({'BELOW noise — reversal likely noise-driven' if effect < seed_std else 'above noise'})")

    # --- Probe 3: design variants (does the sign recover?) -------------------------------
    print("\nProbe 3 — does the sign of r recover under different injection/imputation?")
    rows = [{"variant": "baseline (full, median)", "inject": ",".join(cont_c),
             "impute": "median", "pearson_r": _pearson(base["statistic_max"], base["eo_gap_delta"])}]
    print(f"  [baseline (full, median)] r = {rows[0]['pearson_r']:+.3f}")
    rows.append(variant_r(Xc, yc, gc, [least_zi[0]], driver_c, "median",
                          f"least-zero-inflated only ({least_zi[0]})"))
    rows.append(variant_r(Xc, yc, gc, least_zi[:2], driver_c, "median",
                          f"two least zero-inflated ({','.join(least_zi[:2])})"))
    rows.append(variant_r(Xc, yc, gc, cont_c, driver_c, "mean", "full, mean impute"))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "diagnosis.csv"), index=False)

    # --- Verdict -------------------------------------------------------------------------
    variant_rs = [r["pearson_r"] for r in rows[1:]]
    recovered = any(r > 0 for r in variant_rs)
    all_negative = all(r["pearson_r"] < 0 for r in rows)
    print("\nVERDICT:")
    print("  Effect size: the per-cell effect (mean |Δ EO gap| = %.3f) is %s the seed-to-seed"
          % (effect, "BELOW" if effect < seed_std else "above"))
    print("  spread (%.3f). Caveat: that spread also includes disparity-PATTERN variation (the"
          % seed_std)
    print("  per-cell weights are seed-derived), so it OVERSTATES pure noise — a rough upper bound.")
    if recovered:
        print("  Artifact check: the sign RECOVERS (r > 0) under a cleaner injection set / different")
        print("  imputation -> consistent with a zero-inflation + median-imputation ARTIFACT.")
    else:
        print("  Artifact check: the negative sign is ROBUST — it does NOT recover under an")
        print("  all-clean-feature injection (age) or mean imputation. So it is NOT a simple")
        print("  zero-inflation/median artifact.")
    print("\n  BEST READ: " + (
        "COMPAS shows a weak but robust negative/null relationship — the Adult-positive signal\n"
        "  does NOT generalize. Report predictive validity as demonstrated on Adult and\n"
        "  dataset/regime-dependent, with COMPAS the domain where it does not hold. The effect\n"
        "  is small (partly within noise), so frame it as 'no usable signal on COMPAS', not a\n"
        "  strong opposite effect." if all_negative else
        "mixed across variants — see the per-variant table before concluding."))
    print(f"\nWrote {os.path.relpath(os.path.join(TABLES, 'diagnosis.csv'), _REPO)}")


if __name__ == "__main__":
    main()
