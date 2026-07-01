"""Derive the gate's D_max threshold from the Adult benchmark (Task 3 / issue #14).

The gate currently asserts a round threshold (0.10). We instead *derive* a recommended value from
the detector data: treat "harmful" as a baseline-corrected EO-gap increase above EO_TARGET, use
D_max as the score, and report the threshold at two operating points:
  - max-F1 (balanced), and
  - highest recall subject to precision >= a target (default 0.8; precision-first for a gate that
    should not cry wolf).

Calibrated on ADULT. Because the sign of the D_max->EO relationship is group-structure-dependent
(Task 1) and vanishes at small n (Task 2, German), this threshold is regime-dependent: it is a
starting recommendation for Adult-like settings, not a universal constant.

Writes results/tables/adult/threshold.csv.  Usage:  python experiment/derive_threshold.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

TABLES = os.path.join(_REPO, "results", "tables", "adult")
EO_TARGETS = [0.02, 0.05]     # "harm" = baseline-corrected EO-gap increase above this
TARGET_PRECISION = 0.8


def operating_points(score, label):
    prec, rec, thr = precision_recall_curve(label, score)
    # precision_recall_curve returns prec/rec of length n+1, thr of length n (aligned to prec[:-1])
    prec, rec = prec[:-1], rec[:-1]
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    best_f1 = int(np.argmax(f1))
    pts = [{"operating_point": "max_F1", "threshold": float(thr[best_f1]),
            "precision": float(prec[best_f1]), "recall": float(rec[best_f1]),
            "f1": float(f1[best_f1])}]
    ok = np.where(prec >= TARGET_PRECISION)[0]
    if len(ok):
        # among thresholds meeting the precision target, take the one with the highest recall
        j = ok[int(np.argmax(rec[ok]))]
        pts.append({"operating_point": f"precision>={TARGET_PRECISION}",
                    "threshold": float(thr[j]), "precision": float(prec[j]),
                    "recall": float(rec[j]), "f1": float(f1[j])})
    else:
        pts.append({"operating_point": f"precision>={TARGET_PRECISION}",
                    "threshold": float("nan"), "precision": float("nan"),
                    "recall": float("nan"), "f1": float("nan")})
    return pts


def main():
    d = pd.read_csv(os.path.join(TABLES, "grid.csv"))
    score = d.statistic_max.to_numpy()
    rows = []
    for eo in EO_TARGETS:
        label = (d.eo_gap_delta > eo).astype(int).to_numpy()
        for pt in operating_points(score, label):
            pt = {"eo_harm_target": eo, "positives": int(label.sum()), "n": int(len(label)), **pt}
            rows.append(pt)
    df = pd.DataFrame(rows).round(4)
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, "threshold.csv"), index=False)
    print(df.to_string(index=False))

    # headline recommendation: max-F1 at the primary harm target (0.02)
    rec_row = df[(df.eo_harm_target == EO_TARGETS[0]) & (df.operating_point == "max_F1")].iloc[0]
    print(f"\nRecommended D_max threshold (Adult, max-F1 @ EO-harm>{EO_TARGETS[0]}): "
          f"{rec_row.threshold:.3f}  (precision {rec_row.precision:.2f}, recall {rec_row.recall:.2f})")


if __name__ == "__main__":
    main()
