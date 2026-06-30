"""Fairness metrics. Primary: equalized-odds gap (TPR/FPR gaps). Secondary: demographic parity.

    equalized_odds_gap = max( |TPR_1 - TPR_0|, |FPR_1 - FPR_0| )   over the binary protected group

Design choice: report equalized-odds degradation *specifically*, not "fairness" in
the abstract. Chouldechova's impossibility theorem means that with differing group base rates,
equalized odds / predictive parity / calibration cannot jointly hold — naming the chosen axis
pre-empts the incompatibility critique.
"""
from __future__ import annotations

import numpy as np


def _tpr_fpr(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos, neg = y_true == 1, y_true == 0
    tpr = y_pred[pos].mean() if pos.any() else 0.0
    fpr = y_pred[neg].mean() if neg.any() else 0.0
    return float(tpr), float(fpr)


def equalized_odds_gap(y_true, y_pred, group) -> dict:
    group = np.asarray(group)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    t1, f1 = _tpr_fpr(y_true[group == 1], y_pred[group == 1])
    t0, f0 = _tpr_fpr(y_true[group == 0], y_pred[group == 0])
    tpr_gap, fpr_gap = abs(t1 - t0), abs(f1 - f0)
    return {"eo_gap": max(tpr_gap, fpr_gap), "tpr_gap": tpr_gap, "fpr_gap": fpr_gap}


def demographic_parity_gap(y_pred, group) -> float:
    group = np.asarray(group)
    y_pred = np.asarray(y_pred).astype(int)
    p1 = y_pred[group == 1].mean() if (group == 1).any() else 0.0
    p0 = y_pred[group == 0].mean() if (group == 0).any() else 0.0
    return float(abs(p1 - p0))
