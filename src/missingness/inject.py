"""Missingness injection: MCAR / MAR / MNAR with a controllable group-correlation knob.

Start from a (near-)complete dataset and inject NaNs so the mechanism and the
group-correlation are *known* — the control real data cannot give (experiment-plan.md,
"Independent variable"). Three knobs:

  - mechanism : 'mcar' | 'mar' | 'mnar'
  - group_gap : difference in overall missingness rate between protected groups (0..~0.30),
                concentrated in the unprivileged group (label 0)
  - base_rate : overall missingness rate (e.g. 0.05, 0.15, 0.30)

Group labels are binary: 1 = privileged, 0 = unprivileged.
"""
from __future__ import annotations

import numpy as np


def _per_group_rates(base_rate: float, group_gap: float) -> dict:
    """Privileged (1) gets base - gap/2; unprivileged (0) gets base + gap/2 — i.e. missingness
    concentrates in the unprivileged group. Both clamped to [0, 1]."""
    lo = float(np.clip(base_rate - group_gap / 2.0, 0.0, 1.0))
    hi = float(np.clip(base_rate + group_gap / 2.0, 0.0, 1.0))
    return {1: lo, 0: hi}


def _rank01(x) -> np.ndarray:
    """Map values to [0, 1] by rank (lowest -> 0, highest -> 1)."""
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.zeros(len(x))
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1)


def inject_missingness(df, columns, group, *, mechanism, base_rate, group_gap,
                       seed, mar_driver=None, column_weights=None):
    """Return a copy of ``df`` with NaNs injected into ``columns``.

    mechanism:
      'mcar' — within a group, rows are masked uniformly at random.
      'mar'  — missingness probability rises with an OBSERVED driver column (``mar_driver``):
               depends on other observed data, not on the value being masked.
      'mnar' — missingness probability rises with the masked value itself.

    column_weights: optional per-column scale in [0, 1], aligned to ``columns``. Each column's
      effective base_rate and group_gap are multiplied by its weight, so missingness can be
      HETEROGENEOUS across features (different disparities per feature). Default None = uniform
      (every column gets the full base_rate / group_gap). Heterogeneity is what lets the max /
      mean / weighted aggregations of the disparity statistic diverge and be compared.

    The per-group target rate is hit exactly (deterministic given ``seed``): for MAR/MNAR the
    highest-propensity rows in the group are masked; for MCAR a random subset is.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    group = np.asarray(group)

    for j, col in enumerate(columns):
        w = 1.0 if column_weights is None else float(column_weights[j])
        rates = _per_group_rates(base_rate * w, group_gap * w)
        vals = out[col].to_numpy()
        if mechanism == "mcar":
            propensity = None
        elif mechanism == "mar":
            driver = out[mar_driver].to_numpy() if mar_driver is not None else vals
            propensity = _rank01(driver)
        elif mechanism == "mnar":
            propensity = _rank01(vals)
        else:
            raise ValueError(f"unknown mechanism {mechanism!r}")

        mask = np.zeros(len(out), dtype=bool)
        for g, rate in rates.items():
            idx = np.where(group == g)[0]
            if rate <= 0 or len(idx) == 0:
                continue
            k = int(round(rate * len(idx)))
            if k <= 0:
                continue
            k = min(k, len(idx))
            if mechanism == "mcar":
                chosen = rng.choice(idx, size=k, replace=False)
            else:
                chosen = idx[np.argsort(-propensity[idx])[:k]]
            mask[chosen] = True

        out.loc[out.index[mask], col] = np.nan
    return out
