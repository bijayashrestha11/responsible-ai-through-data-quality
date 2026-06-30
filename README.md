# Responsible AI Through Data Quality

Group-correlated missingness as an automatable **upstream** data-quality check that predicts
downstream fairness degradation — *before a model is trained* — with a defended pipeline
placement.

Most fairness work intervenes at the model. This project intervenes upstream, in the data
pipeline: a cheap, model-free **missingness-disparity statistic** computed at ingestion that
forecasts equalized-odds degradation, plus an empirical characterization of where that signal
holds and where it fails.

> **Status:** benchmark built and reproducible; results are preliminary (work in progress).
> The benchmark design and rationale live in [`experiment-plan.md`](experiment-plan.md).

---

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiment/run_all.py      # ~80s; regenerates all tables + figures
```

The first run fetches **Adult** (Census Income) via `scikit-learn`'s OpenML loader and caches
it under `results/cache/` (gitignored); later runs are offline. Seeds are fixed, so a clone
reproduces every figure and table.

## The check (the proposed artifact)

`D_max` — the **maximum across features** of the absolute difference in per-feature
missingness rate between protected groups, computed pre-training:

```
D_max = max_f | miss_rate(group=1, f) − miss_rate(group=0, f) |
```

Model-free and cheap to compute at a pipeline validation gate. `mean` and `weighted`
aggregations are implemented as secondary variants for comparison; the benchmark finds they
predict comparably, so the simplest (`max`) is the locked primary.

## What the benchmark does

For each grid cell — `mechanism × group-correlation × overall-rate × seed` — it injects
missingness of *known* mechanism (MCAR / MAR / MNAR) and group-correlation into the complete
data, computes `D_max` pre-training, then imputes (fixed median), trains a fixed model
(logistic regression), and measures the **equalized-odds gap** (primary) and demographic-parity
gap (secondary) on a held-out test set. The primary target is the *baseline-corrected* EO gap
(Δ above the matching no-disparity cell), which isolates the group-correlated-missingness effect
from the dataset's inherent unfairness.

## Results so far (Adult, 450 cells)

- `D_max` predicts the baseline-corrected EO gap with Pearson **r ≈ 0.40** pooled, rising to
  **r ≈ 0.52 under MNAR** — the regime that is both most harmful and most predictable.
- Early-warning detector: **AUROC ≈ 0.74–0.78** for flagging EO-gap exceedance.
- The statistic predicts EO degradation more cleanly than demographic parity, as expected.
- Ceiling and how to lift it: because `D_max` is model-free it is blind to feature importance,
  which caps its predictive power. A model-free **MI-weighted** variant (weight each feature's
  disparity by its mutual information with the target) is the best-predicting aggregation
  (**r ≈ 0.42** vs 0.40 for `D_max`) — a modest, consistent lift that confirms the harm depends
  on feature importance. `D_max` remains the recommended default: it needs no labels, so it can
  run at a true pre-label ingestion gate, whereas MI-weighting requires labels at the gate.

Figures: `results/figures/`. Tables: `results/tables/`.

---

## Repository layout

```
experiment-plan.md         # benchmark design + rationale
requirements.txt
README.md                  # this file

src/
  missingness/inject.py    # MCAR/MAR/MNAR injection with a group-correlation knob
  metric/disparity.py      # the missingness-disparity statistic (D_max; +mean/weighted)
  fairness/metrics.py      # equalized-odds + demographic-parity gaps
  pipeline/run_cell.py     # one grid cell, end to end

experiment/
  benchmark/load_adult.py  # Adult loader (public, cached)
  configs/grid.py          # grid + seed definitions
  sweep.py                 # run the full grid
  analyze.py               # scatter, detector AUROC, regime + aggregation comparison
  run_all.py               # one command → regenerates all figures/tables
  claims/                  # (reserved) single real-world demonstration slice

notes/
  sources.md               # citations + verification status
  lit-review-notes.md      # organized related-work notes

results/
  figures/  tables/        # committed outputs
  cache/                   # cached intermediates (gitignored)
paper/                     # (planned) outline, related-work, main.tex
```

## Data handling

- **Benchmark:** Adult (and, planned, COMPAS) — public datasets, fully reproducible, no
  special access.
- **Claims/EHR:** raw data is **never** committed — only derived, aggregate, non-identifiable
  outputs. No PHI or identifiers in code, commits, or prose.

## Reproducibility

Fixed seeds; `python experiment/run_all.py` regenerates every figure and table. A reviewer can
clone and rerun the benchmark arm with no special access.

## Roadmap

Tracked as GitHub issues:

1. ~~Importance-aware (MI-weighted) statistic~~ — **done**: implemented and validated (best
   aggregation, r ≈ 0.42; see Results). `D_max` stays the labels-free default.
2. **COMPAS confirmatory arm** — robustness that the signal isn't an Adult artifact.
3. **Literature** — verify citations and draft the related-work section.

Planned beyond that: implement the check as a validation-stage gate inside an existing
data-quality framework (Deequ / TFDV / Great Expectations).
