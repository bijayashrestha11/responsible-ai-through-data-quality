"""Adult (Census Income) loader — the primary benchmark dataset (public, reproducible).

Returns:
  X         : numeric feature frame (categoricals one-hot encoded)
  y         : binary target (income > 50K)
  group     : binary protected attribute (sex: 1 = Male, 0 = Female)
  cont_cols : continuous columns to inject missingness into
  mar_driver: an observed column used as the MAR driver

We start from the COMPLETE rows (drop native missing) so that all missingness in the
experiment is injected and therefore of known mechanism. Cached as CSV under results/cache/
so reruns are fast and offline (no pyarrow dependency).
"""
from __future__ import annotations

import os

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE = os.path.join(_REPO, "results", "cache", "adult.csv")

CONTINUOUS = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
MAR_DRIVER = "hours-per-week"


def load_adult(use_cache=True):
    if use_cache and os.path.exists(_CACHE):
        df = pd.read_csv(_CACHE)
    else:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("adult", version=2, as_frame=True).frame.copy()
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        try:
            df.to_csv(_CACHE, index=False)
        except OSError:
            pass

    df = df.dropna().reset_index(drop=True)

    target_col = "class" if "class" in df.columns else df.columns[-1]
    y = df[target_col].astype(str).str.contains(">50K").astype(int).to_numpy()
    group = (df["sex"].astype(str).str.strip() == "Male").astype(int).to_numpy()

    features = df.drop(columns=[target_col])
    cat_cols = list(features.select_dtypes(exclude="number").columns)
    X = pd.get_dummies(features, columns=cat_cols, drop_first=True)
    X = X.astype(float)

    cont_cols = [c for c in CONTINUOUS if c in X.columns]
    driver = MAR_DRIVER if MAR_DRIVER in X.columns else cont_cols[0]
    return X, y, group, cont_cols, driver
