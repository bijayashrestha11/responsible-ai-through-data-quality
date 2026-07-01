"""German Credit (Statlog) loader — the third benchmark dataset (Task 2 / issue #13).

Small (1000 rows), clean, standard in fairness work. Returns numeric X, binary y (good credit),
binary protected group (age: 1 = older/privileged, 0 = young/unprivileged, threshold 25 per the
Kamiran & Calders convention), continuous injection columns, and a MAR driver. We start from the
complete rows (the dataset has no missing values) so all missingness is injected. Cached as CSV.

Source: OpenML `credit-g` (dataset 31).
"""
from __future__ import annotations

import os

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE = os.path.join(_REPO, "results", "cache", "german.csv")

AGE_THRESHOLD = 25
CONTINUOUS = ["duration", "credit_amount", "installment_commitment", "residence_since",
              "existing_credits"]
MAR_DRIVER = "credit_amount"


def load_german(use_cache=True):
    if use_cache and os.path.exists(_CACHE):
        df = pd.read_csv(_CACHE)
    else:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("credit-g", version=1, as_frame=True).frame.copy()
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        try:
            df.to_csv(_CACHE, index=False)
        except OSError:
            pass

    df = df.dropna().reset_index(drop=True)

    target_col = "class" if "class" in df.columns else df.columns[-1]
    y = (df[target_col].astype(str).str.strip() == "good").astype(int).to_numpy()
    group = (df["age"].astype(float) > AGE_THRESHOLD).astype(int).to_numpy()  # 1 = older (privileged)

    features = df.drop(columns=[target_col])
    cat_cols = list(features.select_dtypes(exclude="number").columns)
    X = pd.get_dummies(features, columns=cat_cols, drop_first=True).astype(float)

    cont_cols = [c for c in CONTINUOUS if c in X.columns]
    driver = MAR_DRIVER if MAR_DRIVER in X.columns else cont_cols[0]
    return X, y, group, cont_cols, driver
