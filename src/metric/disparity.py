"""The missingness-disparity statistic — the artifact the paper proposes.

LOCKED (experiment-plan.md, "The metric under test"): the primary statistic is the
**max across features** of the absolute difference in per-feature missingness rate between
the two protected groups, computed PRE-TRAINING on the injected data (the upstream check):

    D_max = max_f | miss_rate(group=1, f) - miss_rate(group=0, f) |

It is cheap, model-free, and computable at a pipeline validation gate.

'mean', 'weighted', and 'mi_weighted' are SECONDARY aggregations, kept so the benchmark can
report which aggregation best predicts the post-training equalized-odds gap (an empirical
result, not an assumption). All three stay MODEL-FREE so the statistic remains genuinely
pre-model — preserving the ingestion-time differentiation against model-dependent risk scores:
  - 'weighted'    weights each feature's disparity by its overall missingness prevalence.
  - 'mi_weighted' weights by the feature's mutual information with the target (an importance
    proxy computed without fitting a model), to test whether up-weighting disparities on
    target-relevant features beats the importance-blind max/mean. See compute_mi_weights().
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


def compute_mi_weights(df, y, columns=None, seed=0) -> pd.Series:
    """Per-feature mutual information with the target — a MODEL-FREE, PRE-training importance proxy.

    Computed on the OBSERVED (non-missing) rows of each feature in the injected data, so it needs
    no imputation and no model fit — preserving the ingestion-time differentiation. Features with
    too few observed rows or a single-class observed target get weight 0.
    """
    from sklearn.feature_selection import mutual_info_classif  # lazy: only mi_weighted needs sklearn

    y = np.asarray(y)
    columns = list(df.columns) if columns is None else list(columns)
    weights = {}
    for col in columns:
        obs = df[col].notna().to_numpy()
        if obs.sum() < 10 or len(np.unique(y[obs])) < 2:
            weights[col] = 0.0
            continue
        xcol = df.loc[obs, [col]].to_numpy(dtype=float)
        mi = mutual_info_classif(xcol, y[obs], discrete_features=False, random_state=seed)
        weights[col] = float(max(mi[0], 0.0))
    return pd.Series(weights, dtype=float)


def missingness_disparity(df, group, columns=None, aggregation=PRIMARY, weights=None) -> float:
    """Aggregate the per-feature disparities into the scalar statistic.

    'mi_weighted' requires precomputed per-feature `weights` (see compute_mi_weights); the other
    aggregations ignore `weights`.
    """
    gaps = per_feature_disparity(df, group, columns)
    if len(gaps) == 0:
        return 0.0
    if aggregation == "max":
        return float(gaps.max())
    if aggregation == "mean":
        return float(gaps.mean())
    if aggregation == "weighted":
        w = pd.Series({c: float(df[c].isna().mean()) for c in gaps.index})
    elif aggregation == "mi_weighted":
        if weights is None:
            raise ValueError("aggregation='mi_weighted' requires precomputed `weights` "
                             "(per-feature MI); see compute_mi_weights().")
        w = pd.Series(weights).reindex(gaps.index).fillna(0.0)
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    total = w.sum()
    if total <= 0:
        return float(gaps.mean())
    return float((gaps * (w / total)).sum())
