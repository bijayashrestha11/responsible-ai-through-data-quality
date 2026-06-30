# Benchmark Experiment Plan

The controlled validation that earns the paper's central claim. This operationalizes the
benchmark arm — the contribution, the locked fairness metric, and the empirical design. The
claims-data slice is separate and minimal (one figure/table) — not covered here.

---

## What this experiment must prove

That a **group-wise missingness-disparity statistic, measured on the data before training,
predicts the post-training equalized-odds gap.** If that predictive link holds (and we can say
where it fails), the upstream check is justified. If it doesn't, the contribution narrows and
we report that honestly — the experiment is falsifiable by design.

This is also what adjudicates the Bhatti (null) vs. Min/Asif/Vaidya (positive) tension: we
characterize the *regime* in which the signal is predictive rather than asserting it always is.

---

## Datasets

- **Primary: Adult (Census Income).** Protected attributes: sex, race. Large, well-understood,
  standard in fairness work. Binary target (income >50K).
- **Secondary: COMPAS.** Protected attribute: race. Smaller, noisier, different domain —
  serves as a robustness check that the signal isn't an Adult artifact.
- Use both so a reviewer can't dismiss results as dataset-specific. If time is tight, Adult is
  the non-negotiable one; COMPAS is the confirmatory second.

---

## Independent variable: injected missingness (this is the core design)

Start from the (near-)complete dataset and **inject** missingness so the mechanism and
group-correlation are *known* — that control is the whole point and what real data can't give.

Vary three knobs:
1. **Mechanism:** MCAR / MAR / MNAR.
2. **Group correlation:** how strongly missingness concentrates in the protected group —
   sweep from 0 (uniform) to strong (e.g. missingness rate gap of 0.0, 0.05, 0.10, 0.20, 0.30
   between groups).
3. **Overall rate:** total missingness (e.g. 5%, 15%, 30%) — note Min/Asif/Vaidya found harm
   "even for little missingness," so include a low-rate cell to test that directly.

Grid over these; multiple seeds per cell (fixed seed list for reproducibility). Each cell
produces one (missingness-disparity statistic, equalized-odds gap) observation.

---

## The metric under test (the artifact)

Define a concrete **missingness-disparity statistic** computed pre-training, e.g. the
max/absolute difference in per-feature missingness rate between protected groups, aggregated
across features (decide: max across features vs. mean vs. weighted). Keep it simple and
defensible — it must be cheap to compute in a pipeline. This statistic is the thing the paper
proposes; pin its definition down first and justify it.

---

## Pipeline: per grid cell

1. Inject missingness per the cell's (mechanism, group-correlation, rate).
2. **Compute the missingness-disparity statistic on the injected data** — pre-training. This
   is the upstream check.
3. Impute (a fixed, standard strategy — e.g. simple/median — so imputation isn't a confound;
   note that imputation choice is itself a known fairness lever, cite Feng et al., and hold it
   constant).
4. Train a fixed model (logistic regression primary; one tree-based model for robustness).
5. Compute fairness metrics on a held-out test set: **equalized-odds gap (primary)**,
   demographic-parity gap (secondary).
6. Record the (statistic, EO gap) pair plus all cell parameters and seed.

---

## Analysis: does the statistic predict the gap?

- **Correlation / regression:** across all cells, how well does the pre-training statistic
  predict the post-training EO gap? Report strength and confidence.
- **Threshold / early-warning framing:** can a threshold on the statistic flag "EO gap will
  exceed X" with useful precision/recall? Report it as a detector (this is the "early-warning"
  claim made concrete — e.g. AUROC of the statistic for predicting a gap-exceedance label).
- **Where it fails:** identify regimes (mechanism, rate, correlation) where the statistic does
  NOT predict the gap. This honesty is a feature — it's the adjudication of the literature
  conflict and pre-empts the "you overclaimed" critique.
- **EO vs. DP contrast:** show (as expected) the statistic predicts the EO gap more cleanly
  than the DP gap; name Chouldechova's impossibility theorem where base rates differ.

---

## Outputs the paper needs from this

- A scatter/regression figure: pre-training statistic vs. post-training EO gap.
- A detector-performance result (threshold → gap-exceedance).
- A table of where the signal holds vs. fails, by regime.
- The pipeline-gate framing: this statistic, computed at the validation stage, is the check.

---

## Reproducibility (non-negotiable)

- Fixed seed list; every cell reproducible.
- One command regenerates all figures/tables from cached intermediate outputs.
- Public datasets only in the benchmark arm — no access friction, fully rerunnable by a
  reviewer or a professor evaluating the repo.

---

## Deliberately OUT of scope (deadline guards)

- Imputation-method comparison (hold imputation fixed; cite it as a known separate lever).
- Mitigation/repair of the unfairness (we *detect*, we don't fix — that's future work).
- Class imbalance, drift, label noise (named as related future work in the paper).
- The claims-data arm (separate, minimal, not part of this controlled study).
