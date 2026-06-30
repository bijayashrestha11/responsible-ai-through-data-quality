"""The missingness-disparity statistic — the artifact the paper proposes.

LOCKED (experiment-plan.md, "The metric under test"): the primary statistic is the
**max across features** of the absolute difference in per-feature missingness rate between
the two protected groups, computed PRE-TRAINING on the injected data (the upstream check):

    D_max = max_f | miss_rate(group=1, f) - miss_rate(group=0, f) |

It is cheap, model-free, and computable at a pipeline validation gate.

'mean' and 'weighted' are SECONDARY aggregations, kept so the benchmark can report which
aggregation best predicts the post-training equalized-odds gap (an empirical result, not an
assumption). 'weighted' uses a MODEL-FREE weight (per-feature overall missingness prevalence)
so the statistic stays genuinely pre-model — preserving the ingestion-time differentiation
against model-dependent risk scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY = "max"  # locked primary aggregation


def per_feature_disparity(df, group, columns=None) -> pd.Series:
    """Absolute missingness-rate gap between groups, per feature."""
    group = np.asarray(group)
    columns = list(df.columns) if columns is None else list(columns)
    g1, g0 = group == 1, group == 0
    gaps = {}
    for col in columns:
        miss = df[col].isna().to_numpy()
        r1 = miss[g1].mean() if g1.any() else 0.0
        r0 = miss[g0].mean() if g0.any() else 0.0
        gaps[col] = abs(r1 - r0)
    return pd.Series(gaps, dtype=float)


def missingness_disparity(df, group, columns=None, aggregation=PRIMARY) -> float:
    """Aggregate the per-feature disparities into the scalar statistic."""
    gaps = per_feature_disparity(df, group, columns)
    if len(gaps) == 0:
        return 0.0
    if aggregation == "max":
        return float(gaps.max())
    if aggregation == "mean":
        return float(gaps.mean())
    if aggregation == "weighted":
        cols = gaps.index
        prevalence = pd.Series({c: float(df[c].isna().mean()) for c in cols})
        total = prevalence.sum()
        if total <= 0:
            return float(gaps.mean())
        return float((gaps * (prevalence / total)).sum())
    raise ValueError(f"unknown aggregation {aggregation!r}")
