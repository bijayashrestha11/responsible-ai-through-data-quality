"""One grid cell, end-to-end (experiment-plan.md, "Pipeline: per grid cell").

    1. inject missingness        (known mechanism, group-correlation, rate)
    2. compute the disparity statistic on the injected TRAIN data  -- PRE-TRAINING upstream check
    3. impute (fixed: median)    -- held constant so imputation is not a confound
    4. train a fixed model       (logistic regression)
    5. fairness on held-out test -- EO gap (primary), DP gap (secondary)
    6. return (statistic, eo_gap, ...) + cell params + seed

This is the unit run_all.py will sweep over the full grid. Deterministic given ``seed``.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.fairness.metrics import demographic_parity_gap, equalized_odds_gap
from src.metric.disparity import PRIMARY, compute_mi_weights, missingness_disparity
from src.missingness.inject import inject_missingness


def run_cell(X, y, group, *, inject_columns, mechanism, base_rate, group_gap, seed,
             mar_driver=None, aggregation=PRIMARY):
    y = np.asarray(y)
    group = np.asarray(group)

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    ytr, yte = y[tr], y[te]
    gtr, gte = group[tr], group[te]

    # Per-CELL randomized disparity pattern: draw per-feature weights from the cell seed, so the
    # disparity is sometimes concentrated in one feature (high max, low mean) and sometimes spread
    # (max ~ mean). Varying the pattern ACROSS cells (not just across features within a cell) is
    # what decouples max/mean/weighted and makes "which aggregation predicts best" a real test.
    # Deterministic given seed; group_gap=0 cells are unaffected (gaps are 0 regardless).
    wrng = np.random.default_rng(seed + 50_000)
    col_weights = wrng.uniform(0.1, 1.0, size=len(inject_columns))

    # 1. inject the (known) data-quality defect into both splits — it exists in production data
    Xtr_m = inject_missingness(Xtr, inject_columns, gtr, mechanism=mechanism,
                               base_rate=base_rate, group_gap=group_gap, seed=seed,
                               mar_driver=mar_driver, column_weights=col_weights)
    Xte_m = inject_missingness(Xte, inject_columns, gte, mechanism=mechanism,
                               base_rate=base_rate, group_gap=group_gap, seed=seed + 10_000,
                               mar_driver=mar_driver, column_weights=col_weights)

    # 2. UPSTREAM CHECK: the statistic on injected train data, before any training. Compute all
    # aggregations (cheap) so the benchmark can report which predicts best; `statistic` remains the
    # locked primary (max) for back-compat. mi_weighted needs model-free MI weights (computed on
    # the injected train data + labels, pre-imputation/pre-model).
    mi_weights = compute_mi_weights(Xtr_m, ytr, columns=inject_columns, seed=seed)
    stats = {}
    for agg in ("max", "mean", "weighted", "mi_weighted"):
        kw = {"weights": mi_weights} if agg == "mi_weighted" else {}
        stats[agg] = missingness_disparity(Xtr_m, gtr, columns=inject_columns,
                                           aggregation=agg, **kw)
    statistic = stats[aggregation]

    # 3-4. fixed median impute + scale + logistic regression
    model = make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler(),
                          LogisticRegression(max_iter=1000))
    model.fit(Xtr_m, ytr)
    yhat = model.predict(Xte_m)

    # 5. fairness on held-out test
    eo = equalized_odds_gap(yte, yhat, gte)
    dp = demographic_parity_gap(yhat, gte)

    return {
        "mechanism": mechanism, "base_rate": base_rate, "group_gap": group_gap,
        "seed": seed, "aggregation": aggregation,
        "statistic": statistic,
        "statistic_max": stats["max"], "statistic_mean": stats["mean"],
        "statistic_weighted": stats["weighted"], "statistic_mi": stats["mi_weighted"],
        "eo_gap": eo["eo_gap"], "tpr_gap": eo["tpr_gap"], "fpr_gap": eo["fpr_gap"],
        "dp_gap": dp,
    }
