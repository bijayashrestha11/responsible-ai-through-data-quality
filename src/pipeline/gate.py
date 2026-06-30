"""The missingness-disparity validation gate — the data-engineering contribution.

This is the artifact placed at a pre-training data-validation stage: it computes the group-wise
missingness-disparity statistic on incoming data and FAILS the pipeline when it exceeds a
threshold. It needs only the feature columns and a binary protected-group label — NO target
labels and NO model — so it runs at a true ingestion / transformation gate, before training.

The gate is framework-agnostic by design. `src/pipeline/integrations.py` shows how this same core
plugs into Great Expectations, Deequ (PyDeequ), and TensorFlow Data Validation — frameworks that
natively measure completeness but carry no group-wise / fairness-linked notion. The gate adds
exactly that missing leg, which is what answers the "just Deequ with a fairness label" critique:
it is a *demographic* disparity check with empirically-characterized predictive validity (see the
benchmark), not a bare null-rate constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.metric.disparity import PRIMARY, per_feature_disparity, missingness_disparity


@dataclass
class GateResult:
    """Outcome of one gate evaluation. `passed=False` means the pipeline should halt/alert."""
    statistic: float
    threshold: float
    aggregation: str
    passed: bool
    per_feature: dict = field(default_factory=dict)  # column -> group missingness-rate gap
    group_rates: dict = field(default_factory=dict)  # column -> {group: null-rate}
    message: str = ""

    def __bool__(self):  # truthy == passed, so `if gate(...):` reads naturally
        return self.passed


def evaluate_gate(df, group, columns, *, threshold, aggregation=PRIMARY) -> GateResult:
    """Compute the missingness-disparity statistic and check it against `threshold`.

    df         : incoming data (a pandas DataFrame at the validation stage)
    group      : binary protected-group label aligned to df rows (1 = privileged, 0 = other)
    columns    : feature columns to check for group-correlated missingness
    threshold  : fail if statistic > threshold (tune from the benchmark detector results)
    aggregation: 'max' (D_max, default/locked), 'mean', or 'weighted'
    """
    group = np.asarray(group)
    gaps = per_feature_disparity(df, group, columns)
    statistic = missingness_disparity(df, group, columns=columns, aggregation=aggregation)
    passed = statistic <= threshold

    g1, g0 = group == 1, group == 0
    group_rates = {
        c: {1: float(df[c].isna().to_numpy()[g1].mean()) if g1.any() else 0.0,
            0: float(df[c].isna().to_numpy()[g0].mean()) if g0.any() else 0.0}
        for c in columns
    }
    worst = gaps.idxmax() if len(gaps) else None
    msg = (f"PASS: {aggregation} missingness disparity {statistic:.3f} <= {threshold:.3f}"
           if passed else
           f"FAIL: {aggregation} missingness disparity {statistic:.3f} > {threshold:.3f}"
           f" (worst feature: {worst}, gap {float(gaps.max()):.3f})")
    return GateResult(statistic=statistic, threshold=threshold, aggregation=aggregation,
                      passed=passed, per_feature={c: float(v) for c, v in gaps.items()},
                      group_rates=group_rates, message=msg)


class MissingnessDisparityGate:
    """Reusable, configured gate. Construct once with a policy, then call on each batch.

    Example:
        gate = MissingnessDisparityGate(columns=cont_cols, threshold=0.10)
        result = gate(df, group)
        if not result:
            raise PipelineHalt(result.message)
    """

    def __init__(self, columns, *, threshold, aggregation=PRIMARY):
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.columns = list(columns)
        self.threshold = float(threshold)
        self.aggregation = aggregation

    def __call__(self, df, group) -> GateResult:
        return evaluate_gate(df, group, self.columns, threshold=self.threshold,
                             aggregation=self.aggregation)
