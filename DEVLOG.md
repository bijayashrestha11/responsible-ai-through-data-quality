# Development Log

A running record of *what* was built and *how* / *why* the design evolved — so any future
session (or collaborator) can resume without re-deriving the reasoning. Newest entries first.
For the public-facing overview see [`README.md`](README.md); for the benchmark design see
[`experiment-plan.md`](experiment-plan.md).

> Note: this project was built with AI-assisted pair-programming. All design decisions,
> interpretations, and the argument are the author's; the assistant did the legwork (scaffolding,
> harness, figures, drafting from direction). Citations are verified by reading the source.

---

## Project in one line

Group-correlated missingness as a cheap, **model-free, pre-training** data-quality check
(`D_max`) that predicts downstream **equalized-odds** degradation — with an empirical map of
where the signal holds and fails. The contribution is the upstream/pipeline lens, not a new
fairness metric.

---

## Current state (2026-06-30)

- **Benchmark: built and reproducible, two datasets.** `python experiment/run_all.py
  [--dataset adult|compas|all]` regenerates every table and figure from public data, fixed
  seeds. Outputs under `results/{tables,figures}/<dataset>/`.
- **Statistic locked:** `D_max` (max across features of the per-feature group missingness-rate
  gap). `mean`, `weighted`, and `mi_weighted` exist as secondary aggregations for comparison.
- **Adult (primary, 450 cells):** `D_max` predicts the baseline-corrected EO gap at Pearson
  r ≈ 0.40 pooled, ≈ 0.52 under MNAR; detector AUROC ≈ 0.74–0.78; EO predicted more cleanly
  than DP.
- **COMPAS (confirmatory, 450 cells): does NOT replicate — the relationship REVERSES**
  (r ≈ −0.31 pooled, MNAR −0.53). The signal is dataset-dependent. Possibly partly a
  zero-inflated-feature + median-impute artifact — flagged for diagnosis (see detailed log).
  This is the project's most important "where it fails" result so far.
- **Published:** private GitHub repo `bijayashrestha11/responsible-ai-through-data-quality`.
  `research-brief.md` and `CLAUDE.md` are kept local-only (gitignored). Plan: flip public at
  preprint time.
- **Issues:** #1 (MI-weighted) DONE (merged, PR #4). #2 (COMPAS) DONE (branch feat/compas-arm).
  #3 literature verification open. Candidate new issue: diagnose the COMPAS reversal.

---

## Module map (what each file does)

```
src/missingness/inject.py   inject_missingness(): MCAR/MAR/MNAR with a group-correlation knob
                            (group_gap) + per-feature column_weights for heterogeneous patterns.
                            MAR depends on an observed driver; MNAR on the masked value itself.
src/metric/disparity.py     per_feature_disparity() + missingness_disparity(aggregation=...).
                            Aggregations: max (locked), mean, weighted (by missingness
                            prevalence), mi_weighted (by MI with target — PR #4).
src/fairness/metrics.py     equalized_odds_gap() = max(|ΔTPR|, |ΔFPR|); demographic_parity_gap().
src/pipeline/run_cell.py    one grid cell end-to-end: inject → statistic (pre-training) →
                            median-impute → logreg → EO/DP on held-out test. Deterministic by seed.
experiment/benchmark/       load_adult.py: Adult via OpenML; sex as protected group.
                            load_compas.py: ProPublica COMPAS via URL + standard filter; race
                            (Caucasian vs African-American) as protected group. Both dropna to
                            complete data, one-hot cats, return (X, y, group, cont_cols, mar_driver);
                            cached CSV (gitignored).
experiment/configs/grid.py  SEEDS, MECHANISMS, GROUP_GAPS, BASE_RATES, DEMO_CELL.
experiment/sweep.py         run_grid(): product of the grid through run_cell.
experiment/analyze.py       add_baseline_delta(), scatter, detector AUROC, regime table,
                            aggregation comparison, EO-vs-DP contrast.
experiment/run_all.py       --dataset {adult,compas,all}; per-dataset outputs under
                            results/{tables,figures}/<dataset>/.
```

---

## Locked design decisions (and why)

1. **Statistic = `D_max`** (max-across-features). Simplest, most interpretable ("the single
   worst feature"), cheap, and — critically — **needs no labels**, so it runs at a true
   pre-label ingestion gate. The aggregation comparison (below) showed alternatives don't beat
   it enough to justify their cost.
2. **Primary fairness target = baseline-corrected EO gap (`eo_gap_delta`)** = a cell's EO gap
   minus its matching `group_gap=0` cell (same mechanism/base_rate/seed). This isolates the
   *group-correlated-missingness* effect from (a) the dataset's inherent unfairness and (b)
   missingness that is present but not group-correlated. Absolute EO gap kept as secondary.
3. **Equalized odds primary, demographic parity secondary.** The causal story (uneven data
   corruption → uneven error rates) maps onto error-rate fairness. Chouldechova's impossibility
   theorem is named so the EO-specific framing pre-empts the incompatibility critique.
4. **Imputation held fixed (median)** so it isn't a confound; **model fixed (logistic
   regression).** Imputation choice is a known separate fairness lever — out of scope here.
5. **Reproducibility:** fixed seed list; public data only in the benchmark arm; one command
   regenerates everything from cache.

---

## How the results evolved (the important part)

Each iteration changed the design *because the previous run revealed something*. Reading this
in order explains why the headline number moved.

| # | Change | Cells | Headline r | What it taught us |
|---|---|---|---|---|
| 1 | First full grid; **uniform** injection (all features same rate), absolute EO target | 225 | 0.25 | Signal real but modest. max=mean=weighted **identical** (all features injected the same → degenerate). Absolute EO diluted by Adult's baseline unfairness. |
| 2 | **Baseline correction** (`eo_gap_delta`) + heterogeneous within-cell weights | 225 | **0.47** | Baseline correction was the big lever (0.25→0.47); absolute-target r fell to 0.13, confirming dilution. MNAR r=0.78. But aggregations **still identical** — fixed weight pattern makes max/mean/weighted collinear *across* cells. |
| 3 | **Randomized disparity pattern per cell** (weights drawn from seed) + 10 seeds | 450 | 0.40 | Aggregations finally **decouple** (weighted 0.401 > max 0.398 > mean 0.388 — choice barely matters → keep `D_max`). Headline dropped 0.47→0.40 because random patterns sometimes land disparity on *low-importance* features; the model-free statistic is blind to importance. MNAR 0.52, detector AUROC 0.74–0.78. **More honest numbers.** |
| 4 | **MI-weighted aggregation** (importance-aware, still model-free) — *PR #4* | 450 | 0.42 (mi) | mi_weighted is the **best** aggregation (0.418 vs 0.398 for `D_max`), confirming harm depends on feature importance. Modest lift. Costs labels at the gate → `D_max` stays default. |
| 5 | **COMPAS confirmatory arm** (#2) — second dataset, same harness | 450 | **−0.31** | Signal does **not** replicate — it **reverses** on COMPAS (MNAR −0.53; mean Δ EO gap negative). Dataset-dependent. Possibly partly a zero-inflated-feature + median-impute artifact — flagged for diagnosis. The honest "where it fails." |

Key takeaways for the paper:
- The predictive signal is **real but moderate** on Adult (r ≈ 0.4), **strongest under MNAR**
  (the "interesting case"), and the design honestly reports *where it fails*.
- The model-free statistic's ceiling is set by its **blindness to feature importance**;
  MI-weighting partially lifts it but trades away the no-labels property.
- **Generalization is NOT established:** COMPAS does not replicate (the sign flips). The current
  honest position is that the signal is **regime/dataset-dependent**, which is itself the
  adjudication of the Bhatti vs Min/Asif/Vaidya tension — not a universal claim.
- These are falsifiable-by-design results, not cherry-picked — the experiment is built to
  characterize the regime, not to assert the signal always holds.

---

## Detailed feature log (newest first)

*Convention: add a detailed entry here after every feature — what was built, how, why, the
result, threats to validity, and follow-ups.*

### 2026-06-30 — #2 COMPAS confirmatory arm  (branch `feat/compas-arm`)

**What was built**
- `experiment/benchmark/load_compas.py` — loads ProPublica's `compas-scores-two-years.csv`
  (cached to `results/cache/compas.csv`), applies ProPublica's standard quality filter
  (`|days_b_screening_arrest| ≤ 30`, `is_recid ≠ -1`, `c_charge_degree ≠ 'O'`,
  `score_text ≠ 'N/A'`), restricts to the two largest race groups, and returns: `X` (one-hot),
  `y = two_year_recid`, `group = (race == 'Caucasian')` [1 = privileged], continuous inject
  columns `[age, priors_count, juv_fel/misd/other_count]`, MAR driver `priors_count`. **5,278
  complete rows, 8 base features.**
- `experiment/run_all.py` — added a `--dataset {adult,compas,all}` switch and per-dataset output
  dirs `results/{tables,figures}/<dataset>/`. The entire sweep/analysis path is reused unchanged
  (the harness was already dataset-agnostic). Removed the now-orphaned top-level result files.

**How / why.** COMPAS is the *confirmatory* dataset (per experiment-plan): smaller, noisier,
different domain — a robustness check that the Adult signal isn't a dataset artifact. The loader
mirrors `load_adult` exactly so nothing else in the pipeline changes; only the data differs.

**Result — the signal does NOT replicate; it reverses.**
- Adult (re-confirmed, byte-identical to PR #4): `r(D_max, eo_gap_delta) = +0.40`; MNAR `+0.52`.
- COMPAS: `r(D_max, eo_gap_delta) = −0.31` pooled (mean −0.30, weighted −0.31, mi −0.33). By
  mechanism: MNAR −0.53, MCAR −0.52, MAR −0.07. **Mean Δ(EO gap) is negative** for MCAR/MNAR
  (−0.019, −0.021): injecting group-correlated missingness tends to *reduce* the measured EO gap
  vs the matched no-disparity baseline. Detector AUROC ~0.63, but since the underlying
  correlation is negative the raw statistic is **not a usable early-warning score** on COMPAS.

**Interpretation (honest).** This is a genuine negative / "where it fails" result and the most
consequential finding so far: the predictive relationship is **dataset-dependent**, and on COMPAS
the sign flips. It directly echoes the Bhatti (null) vs Min/Asif/Vaidya (positive) tension the
project set out to adjudicate — landing on "regime-dependent," not "universal." It should be
reported, not buried.

**Threat to validity (investigate before trusting the sign/magnitude).** COMPAS is ~9× smaller
than Adult and its injection features are heavily **zero-inflated counts** (`priors_count`,
`juv_*`). So (a) the Δ(EO gap) values are tiny (~0.02) and may be **noise-dominated**, and
(b) MNAR masking ("hide the highest values") removes the few high-count records, which plausibly
*reduces* the model's group-differential reliance and hence the gap. The reversal may therefore
be partly a **structural artifact of zero-inflated features + median imputation**, not a clean
counterexample.

**Follow-up (candidate next issue).** Diagnose the COMPAS reversal: (i) compare Δ(EO gap) against
a noise floor (seed/shuffle variance); (ii) inject into less zero-inflated features and/or vary
imputation; (iii) examine per-cell rather than pooled. Then decide: true counterexample (report
the signal as Adult-regime-specific) vs artifact (fix injection and re-test).

**Repro:** `python experiment/run_all.py --dataset all` (~3 min) →
`results/{tables,figures}/{adult,compas}/`.

---

## Repo / workflow history (this build)

- Consolidated a duplicate set of planning docs to repo root; removed the old layout
  (`drafts/`, `experiments/`, `sources/`) and built the canonical layout (`src/`, `experiment/`,
  `results/`, `notes/`, `paper/`).
- Seeded `notes/sources.md` (verify-every-citation checklist; all rows UNVERIFIED) and
  `notes/lit-review-notes.md` (4-part structure). Dropped an earlier literature scan and its
  DispaRisk/DQD nearest-neighbors per author decision; current competitors are tracked in the
  (local-only) brief.
- Nested everything under `responsible-ai-through-data-quality/`.
- Published private; tidied the README to public-facing; swept all tracked files clean of
  references to the local-only `research-brief.md` / `CLAUDE.md`.

---

## Next steps

1. **Diagnose the COMPAS reversal** (candidate new issue) — noise floor, alternative injection
   features / imputation, per-cell view. Decide: true counterexample vs zero-inflation artifact.
   This now gates how strongly the generalization claim can be made.
2. **#3 Literature** — verify §-referenced citations by reading sources; organize; draft
   `paper/related-work.md`.
3. Beyond: implement the check as a validation-stage gate inside an existing data-quality
   framework (Deequ / TFDV / Great Expectations) — the data-engineering contribution.
