"""COMPAS (ProPublica two-year recidivism) loader — the confirmatory benchmark dataset.

Public and standard in fairness work; smaller and noisier than Adult, so it serves as a
robustness check that the signal isn't an Adult artifact. Returns numeric X, binary y
(two_year_recid), binary protected group (race: 1 = Caucasian, 0 = African-American),
continuous injection columns, and a MAR driver. We start from the COMPLETE rows so all
missingness in the experiment is injected (known mechanism). Cached as CSV under results/cache/.

Source: ProPublica compas-analysis (compas-scores-two-years.csv), with ProPublica's standard
quality filter applied.
"""
from __future__ import annotations

import os

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE = os.path.join(_REPO, "results", "cache", "compas.csv")
_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
        "master/compas-scores-two-years.csv")

FEATURES = ["sex", "age", "race", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "priors_count", "c_charge_degree"]
CONTINUOUS = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
MAR_DRIVER = "priors_count"


def load_compas(use_cache=True):
    if use_cache and os.path.exists(_CACHE):
        df = pd.read_csv(_CACHE)
    else:
        df = pd.read_csv(_URL)
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        try:
            df.to_csv(_CACHE, index=False)
        except OSError:
            pass

    # ProPublica's standard quality filter (drops mis-joined / ambiguous records)
    df = df[(df["days_b_screening_arrest"] <= 30) &
            (df["days_b_screening_arrest"] >= -30) &
            (df["is_recid"] != -1) &
            (df["c_charge_degree"] != "O") &
            (df["score_text"] != "N/A")].copy()

    # restrict to the two largest race groups -> binary protected attribute
    df = df[df["race"].isin(["Caucasian", "African-American"])].copy()

    df = df[FEATURES + ["two_year_recid"]].dropna().reset_index(drop=True)

    y = df["two_year_recid"].astype(int).to_numpy()
    group = (df["race"] == "Caucasian").astype(int).to_numpy()

    features = df.drop(columns=["two_year_recid"])
    cat_cols = list(features.select_dtypes(exclude="number").columns)
    X = pd.get_dummies(features, columns=cat_cols, drop_first=True).astype(float)

    cont_cols = [c for c in CONTINUOUS if c in X.columns]
    driver = MAR_DRIVER if MAR_DRIVER in X.columns else cont_cols[0]
    return X, y, group, cont_cols, driver
