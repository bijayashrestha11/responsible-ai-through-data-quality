"""Plugging the missingness-disparity gate into real data-validation frameworks.

These frameworks natively measure per-column completeness (null rate) but have NO group-wise /
fairness-linked notion. The pattern in every case is the same: let the framework measure the
per-group, per-column null rate (its native strength), then compose those measurements into the
group-wise disparity statistic and enforce the threshold (our contribution). That composition is
exactly the "fairness leg" the frameworks lack.

- Great Expectations: `great_expectations_gate` — RUNNABLE. Uses GE's built-in
  `expect_column_proportion_of_non_null_values_to_be_between` on each protected-group subset to
  obtain observed non-null proportions from GE's own engine, then composes the gate. Verified in
  an isolated environment (see experiment/demo_gate.py and DEVLOG); GE pins older numpy/pandas, so
  it is an OPTIONAL dependency installed separately (requirements-gate.txt).
- Deequ (PyDeequ): `deequ_check_spec` — interface code. PyDeequ's `Completeness` analyzer gives the
  per-group null rate; compose identically. Requires Spark/JVM (not run here).
- TFDV: `tfdv_group_null_rates` — interface code. TFDV's per-group statistics give null fractions;
  compose identically. Requires TensorFlow (not run here).

All framework imports are lazy so this module imports cleanly without them installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.gate import GateResult, evaluate_gate
from src.metric.disparity import PRIMARY, missingness_disparity, per_feature_disparity


# ---------------------------------------------------------------------------------------------
# Great Expectations  (runnable; optional dependency)
# ---------------------------------------------------------------------------------------------
def ge_non_null_proportions(df, columns):
    """Use Great Expectations' own validation engine to measure the non-null proportion of each
    column, returning {column: observed_proportion}. Raises ImportError if GE is not installed.

    This intentionally routes the measurement THROUGH GE (not pandas) so the gate's inputs come
    from the framework's validation result — demonstrating a genuine in-framework integration.
    """
    import great_expectations as gx  # lazy

    context = gx.get_context()
    datasource = context.sources.add_or_update_pandas(name="md_gate")
    asset = datasource.add_dataframe_asset(name="batch")
    batch_request = asset.build_batch_request(dataframe=df)
    validator = context.get_validator(batch_request=batch_request)

    # Use GE's built-in null check; read the observed counts from its result (we want the
    # measurement, not GE's pass/fail of "all non-null").
    proportions = {}
    for col in columns:
        res = validator.expect_column_values_to_not_be_null(column=col)
        r = res.result
        n = r.get("element_count", 0) or 0
        nulls = r.get("unexpected_count", 0) or 0
        proportions[col] = float(1.0 - nulls / n) if n else 1.0
    return proportions


def great_expectations_gate(df, group, columns, *, threshold, aggregation=PRIMARY) -> GateResult:
    """Run the gate with per-group null rates measured by Great Expectations' engine.

    GE measures the per-group, per-column non-null proportion; we turn that into the group-wise
    disparity statistic and enforce the threshold. Same GateResult contract as evaluate_gate().
    """
    group = np.asarray(group)
    g1_df, g0_df = df[group == 1], df[group == 0]
    p1 = ge_non_null_proportions(g1_df, columns)
    p0 = ge_non_null_proportions(g0_df, columns)

    gaps = pd.Series({c: abs((1.0 - p1[c]) - (1.0 - p0[c])) for c in columns}, dtype=float)
    if aggregation == "max":
        statistic = float(gaps.max())
    elif aggregation == "mean":
        statistic = float(gaps.mean())
    else:
        raise ValueError(f"GE adapter supports max/mean; got {aggregation!r}")

    passed = statistic <= threshold
    msg = (f"[GE] {'PASS' if passed else 'FAIL'}: {aggregation} missingness disparity "
           f"{statistic:.3f} {'<=' if passed else '>'} {threshold:.3f}")
    return GateResult(statistic=statistic, threshold=float(threshold), aggregation=aggregation,
                      passed=passed, per_feature={c: float(v) for c, v in gaps.items()},
                      group_rates={c: {1: 1.0 - p1[c], 0: 1.0 - p0[c]} for c in columns},
                      message=msg)


# ---------------------------------------------------------------------------------------------
# Deequ / PyDeequ  (interface; requires Spark/JVM — not executed here)
# ---------------------------------------------------------------------------------------------
def deequ_group_completeness(spark_df, columns, group_col):
    """Measure per-group, per-column Completeness with PyDeequ, returning
    {group_value: {column: completeness}}. Requires a running Spark session + PyDeequ.

    Sketch (PyDeequ AnalysisRunner):
        from pydeequ.analyzers import AnalysisRunner, AnalyzerContext, Completeness
        for g in spark_df.select(group_col).distinct().collect():
            sub = spark_df.filter(spark_df[group_col] == g[0])
            runner = AnalysisRunner(spark).onData(sub)
            for c in columns:
                runner = runner.addAnalyzer(Completeness(c))
            result = AnalyzerContext.successMetricsAsDataFrame(spark, runner.run())
            ...  # collect completeness per column
    Then feed the resulting null-rate gaps into evaluate_gate-style composition.
    """
    raise NotImplementedError("Requires Spark/JVM + PyDeequ; see docstring for the integration.")


# ---------------------------------------------------------------------------------------------
# TensorFlow Data Validation  (interface; requires TensorFlow — not executed here)
# ---------------------------------------------------------------------------------------------
def tfdv_group_null_rates(df, columns, group):
    """Measure per-group null rates via TFDV statistics, returning {group: {column: null_rate}}.

    Sketch:
        import tensorflow_data_validation as tfdv
        for g in (0, 1):
            stats = tfdv.generate_statistics_from_dataframe(df[group == g])
            ...  # read num_missing / num_non_missing per feature from the FeatureNameStatistics
    Then compose the per-group null rates into the disparity statistic + threshold, identically.
    """
    raise NotImplementedError("Requires TensorFlow + TFDV; see docstring for the integration.")
