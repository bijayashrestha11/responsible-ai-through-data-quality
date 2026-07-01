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

- **Benchmark: built and reproducible, three datasets.** `python experiment/run_all.py
  [--dataset adult|compas|german|all]` regenerates every table and figure from public data, fixed
  seeds. Outputs under `results/{tables,figures}/<dataset>/`.
- **Generalization map (3 datasets):** Adult r=+0.40 (excess on minority), COMPAS r=−0.31 (excess
  on majority), German r≈0 (null). Two axes: SIGN←group structure (Task 1, encoding-flip),
  DETECTABILITY←sample size (Task 2 + **Task 5: controlled** — Adult subsampled to n=1000 → r≈0.007,
  reproducing German's null with all else fixed; r rises monotonically to 0.40 at full n). See
  `results/tables/boundary_check.csv`, `results/tables/adult/subsample.csv`.
- **Statistic locked:** `D_max` (max across features of the per-feature group missingness-rate
  gap). `mean`, `weighted`, and `mi_weighted` exist as secondary aggregations for comparison.
- **Adult (primary, 450 cells):** `D_max` predicts the baseline-corrected EO gap at Pearson
  r ≈ 0.40 pooled, ≈ 0.52 under MNAR; detector AUROC ≈ 0.74–0.78; EO predicted more cleanly
  than DP.
- **COMPAS (confirmatory, 450 cells): reverses** (r ≈ −0.31). **Diagnosed (#6): not a
  zero-inflation/median artifact.** **Explained (Task 1): a boundary condition** — flipping the
  protected-group encoding flips the sign in BOTH datasets (COMPAS −0.31→+0.42, Adult
  +0.40→−0.36), so the sign is governed by *which group carries the excess missingness* (positive
  when it falls on the smaller/lower-base-rate group). Magnitude |r|≈0.4 preserved throughout.
  Benign for the gate (fires on |D_max|); the third dataset (Task 2) disambiguates the confound.
- **Published:** private GitHub repo `bijayashrestha11/responsible-ai-through-data-quality`.
  `research-brief.md` and `CLAUDE.md` are kept local-only (gitignored). Plan: flip public at
  preprint time.
- **Issues:** #1 (MI-weighted) DONE (merged). #2 (COMPAS) DONE (merged). #6 (diagnose COMPAS
  reversal) DONE (merged). #3 (literature verification + related-work draft) DONE (branch
  feat/literature-verification). All tracked issues complete; remaining work is optional
  (third dataset, Deequ/TFDV gate, paper integration).
- **Literature:** all 16 §6–§8 citations VERIFIED against primary sources (see notes/sources.md);
  lit-review-notes organized; paper/related-work.md drafted.
- **Pipeline placement (third leg) DONE:** `src/pipeline/gate.py` validation-stage gate (no
  target/model needed); runnable Great Expectations integration (`integrations.py`, verified in an
  isolated venv); Deequ/TFDV adapters as interface code. `experiment/demo_gate.py` shows PASS on
  clean / FAIL on disparate batches, Adult + COMPAS. **All three contribution legs now exist:
  validated metric, predictive validity (Adult), pipeline placement.**
- **Paper assembled:** `paper/main.tex` (acmart numeric) + `paper/references.bib` (16 verified).
  Source-only (no local LaTeX); compiles on arXiv/Overleaf. Author to revise into voice + fill
  placeholders.

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
                            impute (impute_strategy, default median) → logreg → EO/DP on held-out
                            test. Deterministic by seed.
src/pipeline/gate.py        MissingnessDisparityGate / evaluate_gate(): the validation-stage gate
                            (no target/model) → GateResult(statistic, threshold, passed, ...).
src/pipeline/integrations.py  framework adapters: great_expectations_gate (runnable) + Deequ/TFDV
                            interface code. Same compose pattern on each framework's null-rate.
experiment/benchmark/       load_adult.py: Adult via OpenML; sex as protected group.
                            load_compas.py: ProPublica COMPAS via URL + standard filter; race
                            (Caucasian vs African-American) as protected group.
                            load_german.py: OpenML credit-g (1000 rows); age (>25) as protected
                            group. All dropna to complete data, one-hot cats, return
                            (X, y, group, cont_cols, mar_driver); cached CSV (gitignored).
experiment/boundary_check.py  places all 3 datasets vs Task 1's sign rule ->
                            results/tables/boundary_check.csv.
experiment/derive_threshold.py  derives the gate D_max threshold (max-F1, precision-target) from
                            the Adult detector data -> results/tables/adult/threshold.csv.
experiment/subsample_adult.py  tests detectability<-sample-size within Adult (full grid at several
                            n) -> results/tables/adult/subsample.csv.
experiment/configs/grid.py  SEEDS, MECHANISMS, GROUP_GAPS, BASE_RATES, DEMO_CELL.
experiment/sweep.py         run_grid(): product of the grid through run_cell.
experiment/analyze.py       add_baseline_delta(), scatter, detector AUROC, regime table,
                            aggregation comparison, EO-vs-DP contrast.
experiment/run_all.py       --dataset {adult,compas,all}; per-dataset outputs under
                            results/{tables,figures}/<dataset>/.
experiment/diagnose_compas.py  probes the COMPAS reversal: zero-inflation profile, noise floor,
                            injection/imputation variants -> results/tables/compas/diagnosis.csv.
experiment/why_compas.py    boundary-condition analysis: structural scalars + encoding-flip test
                            -> results/tables/compas/why_reversal.csv.
experiment/demo_gate.py     demonstrates the gate: clean batch PASS / disparate batch FAIL on
                            Adult + COMPAS; runs the GE-backed path if GE is installed.
experiment/make_figures.py  builds the paper figures from the saved grid CSVs ->
                            paper/figures/ (PDF+PNG; self-contained with the paper).
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

### 2026-07-01 — Task 6: COMPAS MAR anomaly — flagged, not force-explained  (branch `analysis/compas-mar-anomaly`)

**Question.** COMPAS regime: MCAR r=−0.52, MNAR r=−0.53 (strong) but **MAR r=−0.07** (flat). Is the
weak MAR signal explained by COMPAS's zero-inflated MAR driver (`priors_count`, 32% zero)
producing tied propensity ranks that flatten the pattern?

**What was done.** `experiment/compas_mar_probe.py`: quantify the concentration/ties of each
dataset's MAR driver (fraction zero, modal fraction, Herfindahl tie-probability, non-unique
fraction) against the per-mechanism r.

**Result — hypothesis REFUTED.** Adult's MAR driver `hours-per-week` is *more* concentrated
(tie-prob **0.243**, modal **0.472**) than COMPAS's `priors_count` (tie-prob **0.160**, modal
0.316, 32% zero), yet Adult's MAR signal is **strong (r=+0.44)** while COMPAS's is **flat
(−0.07)**. If concentration/ties flattened MAR, Adult would be the *more* flattened, not less. So
driver concentration does **not** explain the weak COMPAS MAR.

**Verdict (honest).** Per the task's falsifiable design, we flag it: the weak COMPAS MAR is an
**unexplained per-mechanism variation**, not attributable to the driver. Ready sentence for
`main.tex` (Task 7 folds it in): *"On COMPAS the MAR signal is notably weaker than MCAR/MNAR; a
probe rules out the obvious cause (its zero-inflated MAR driver), since Adult's MAR driver is in
fact more concentrated yet yields a strong MAR signal—so we report this per-mechanism variation as
unexplained rather than over-interpret it."* Repro: `python experiment/compas_mar_probe.py`.

### 2026-07-01 — Task 5: detectability axis, tested within Adult  (branch `analysis/detectability-subsample`)

**Question.** Is "detectability ← sample size" real, or is German's null confounded with its other
differences (feature count, base rate, domain)?

**What was done.** `experiment/subsample_adult.py`: re-run the full 450-cell grid on stratified
`(group,y)` subsamples of Adult at n ∈ {1000, 2000, 5000, 10000, 20000, full}, subsampling BEFORE
the split, everything else byte-identical. → `results/tables/adult/subsample.csv`. Added
`fig6_subsample` (r vs n) to `make_figures.py`.

**Result — supported, under control.** Pooled r rises monotonically with n while the effect
magnitude persists: n=1000 → **+0.007** (mean|ΔEO|=0.024); 5000 → +0.050; 10000 → +0.140; 20000 →
+0.328; full (45 222) → **+0.398** (mean|ΔEO|=0.010). **Adult-at-1000 (r≈0.007) is indistinguishable
from German's null (r≈−0.02)** — with features, base rate, and domain all held fixed. So sample
size alone reproduces the null. The noise-floor mechanism is confirmed: |ΔEO| stays ~0.01–0.02
(larger at small n) while the correlation collapses.

**Falsifiability.** Designed to be able to refute the axis (r staying strong at small n would have
done so); it did not. The full r-vs-n curve is reported, not just the endpoints.

**Implication.** The detectability axis is now a controlled result, not a one-point assertion; the
German null is reframed as "Adult shrunk to n=1000 behaves the same." Task 7 folds the curve
(`fig6_subsample`) into the paper. Repro: `python experiment/subsample_adult.py` (~5 min).

### 2026-07-01 — Task 4: paper hardening + submission-prep  (branch `feat/paper-hardening`)

**What was done.** Folded Tasks 1–3 into `paper/main.tex`:
- **Abstract + intro** rewritten to give the bounded/negative result **equal billing** — the
  three-dataset map (Adult holds, COMPAS reverses, German null) and the two governing axes (sign ←
  group structure; detectability ← sample size) are now front-and-centre, framed as a contribution.
- **Datasets** → three; **Results** → boundary-condition (encoding-flip) paragraph + German
  detectability paragraph; **Gate** → benchmark-derived threshold ($D_{\max}\approx0.10$,
  high-recall screen) and the regime-dependence caveat.
- **Figures** (`make_figures.py`): the headline scatter is now a **3-panel** Adult/COMPAS/German
  contrast; added **fig5** (encoding-flip bars). 6 figures total, all labels/refs/targets verified.
- **Tooling precision audit:** the paper now states GE is **executed** end-to-end and Deequ/TFDV
  are **interface-compatible but not executed** (Spark/TF) — no sentence implies they were run.
- **MDLA re-verified** (web, 2026-07-01): still preprint v1, no gate/predictor, no published
  version. Sharpened the related-work framing per the finding — MDLA already claims a
  pre-deployment posture, so the differentiator is **automation + a predictive threshold, not
  timing**. Reworded the "closest competitors" paragraph accordingly.
- Added `paper/SUBMISSION_CHECKLIST.md` (author metadata, arXiv endorsement, compile steps,
  license, category).

**Left to the author (per project rules).** The argument, final voice, and the
author/affiliation/email placeholders; the first real LaTeX compile (no toolchain here — draft is
structure-checked only).

### 2026-07-01 — Task 3: derive the gate threshold from the benchmark  (branch `feat/threshold-from-benchmark`)

**What was done.** `experiment/derive_threshold.py`: treats "harm" as a baseline-corrected EO-gap
increase above EO_TARGET, uses `D_max` as the score, and reports the threshold at two operating
points — max-F1 and highest-recall-subject-to-precision≥0.8 — for EO_TARGET ∈ {0.02, 0.05} on
Adult. → `results/tables/adult/threshold.csv`.

**Result — closes the loop.** The F1-optimal `D_max` threshold (harm = EO increase > 0.02) is
**0.103** — which *vindicates the round 0.10* the gate/demo already used (it was not arbitrary).
Operating point: **recall 0.72, precision 0.26**. A precision≥0.8 point is **unreachable** with
`D_max` alone.

**Honest framing.** The gate is a **high-recall, low-precision screen** — a sensitive early
warning that flags data for human review, not a precise oracle (as expected from a moderate
predictor, r≈0.4). That is the appropriate role for a validation gate. The threshold is
**Adult-calibrated and regime-dependent**: Task 1 (sign ← group structure) and Task 2 (detectability
← sample size) mean it does not transfer unchanged to COMPAS (opposite sign) or German (null). The
gate still *detects* the disparity via |D_max| everywhere; what varies is the mapping from D_max to
EO-harm.

**Updated.** `demo_gate.py` documents the 0.10 as the benchmark-derived (rounded F1-optimal) value
with the recall/precision caveat. Repro: `python experiment/derive_threshold.py`.

### 2026-07-01 — Task 2: third dataset (German Credit) — a detectability axis  (branch `feat/third-dataset`)

**What was done.** `experiment/benchmark/load_german.py` (OpenML `credit-g`, 1000 rows; protected =
age, older = privileged, threshold 25; 5 numeric inject columns; y = good credit) wired into
`run_all.py --dataset {…, german, all}`. Ran the full 450-cell grid.
`experiment/boundary_check.py` places all three datasets against Task 1's sign rule →
`results/tables/boundary_check.csv`.

**Prediction (from Task 1).** German is structurally **Adult-like** — the unprivileged group
(young, ≤25) is a **minority (19%)** with a **lower** good-credit base rate (baserate-diff −0.15,
corr +0.13) — so the minority sign-rule predicted **positive r**.

**Result — NULL, not positive.** German pooled **r = −0.02**; per-mechanism all ≈0; detector AUROC
still modest (0.69–0.80). The tell: mean $|\Delta\mathrm{EO}|$ = **0.020**, *larger* than Adult's
0.010 — so the effect is not small, it is **noise-dominated**. At n=1000 (test ~300) the EO-gap
estimate fluctuates more than the injected disparity moves it, dissolving the linear relationship
(consistent with #6's COMPAS noise-floor finding).

**Synthesis — the boundary condition has two axes:**
1. **SIGN ← group structure** (which group carries the excess missingness): established decisively
   by Task 1's encoding-flip test on Adult + COMPAS. German does **not** contradict it (null, not
   opposite-sign).
2. **DETECTABILITY ← sample size**: German (n=1000) is below where a clean linear signal rises
   above EO-estimate noise; Adult (45k) is well above; COMPAS (5.3k) is between (signal present,
   noisier).

Three datasets = 1 clear positive (Adult +0.40), 1 clear negative (COMPAS −0.31), 1 null
(German ≈0) — a richer, more honest generalization map than "2 of 3."

**Honesty.** German's three structural scalars still co-vary (minority + negative baserate-diff +
positive corr, like Adult), so it does **not** disambiguate which scalar drives the sign — that
stays open (needs a dataset where they diverge). What German adds is the orthogonal
detectability/sample-size axis. Repro: `python experiment/run_all.py --dataset german` +
`python experiment/boundary_check.py`.

### 2026-07-01 — Task 1: WHY the COMPAS reversal (a boundary condition)  (branch `analysis/compas-why`)

**Question.** Can the COMPAS sign reversal ($r=-0.31$ vs Adult $+0.40$) be explained as a
*boundary condition* rather than just "dataset-dependent"?

**What was done.** `experiment/why_compas.py`: computed structural scalars for Adult and COMPAS
(is the excess-missingness group a minority/majority; signed base-rate difference unpriv$-$priv;
size ratio; corr(protected, target)), and ran the leading hypothesis — the **encoding-flip
test**: re-run each grid with the protected-group encoding flipped so the OTHER group carries the
injected excess missingness (Adult as control). → `results/tables/compas/why_reversal.csv`.

**Result — decisive.** Flipping the encoding **flips the sign of the correlation in both
datasets**, with magnitude preserved:
- COMPAS: $-0.31 \rightarrow +0.42$
- Adult: $+0.40 \rightarrow -0.36$

So the sign is **not a fixed dataset property** — it is governed by *which group carries the
excess missingness*. Only the sign moves; $|r|\approx0.4$ throughout (the statistic carries
predictive signal in every case).

**Boundary condition.** Across all four configs, three scalars each perfectly separate $+r$ from
$-r$: the excess-missingness group being the **minority** ($\to +r$), the sign of the base-rate
difference, and the sign of corr(protected, target). In Adult vs COMPAS these three move together,
so two datasets cannot isolate a single cause — but the reportable mechanism is: **the statistic
predicts EO degradation with positive sign when the excess missingness falls on the smaller /
lower-base-rate group, and the sign reverses when it falls on the larger / higher-base-rate
group.**

**Implications.** (a) A major upgrade over "does not generalize": the reversal is a controlled,
reproducible consequence of group structure (consistent with #6's not-an-artifact finding).
(b) It is **benign for the gate** — the gate fires on $|D_{\max}|$ regardless of sign, so
detection is unaffected; what is structure-dependent is the *direction* of downstream EO harm,
which matters for interpretation, not for flagging. (c) Task 2's third dataset is exactly what is
needed to disentangle the three confounded scalars.

**Honesty.** Cannot isolate which of the three confounded scalars is causal from two datasets;
stated as such. Repro: `python experiment/why_compas.py` (~3 min).

### 2026-07-01 — Paper assembly (acmart draft)  (branch `feat/paper-assembly`)

**What was built**
- `paper/main.tex` — full draft in ACM `acmart` (sigconf, nonacm, numeric citations). Sections:
  introduction (3-leg contribution), related work, the $D_{\max}$ statistic, the
  predictive-validity benchmark (methods), results (Adult holds / COMPAS reverses), the
  validation gate, discussion + limitations, conclusion, reproducibility, AI-use disclosure,
  references.
- `paper/references.bib` — 16 BibTeX entries, all VERIFIED (DOIs/identifiers); preprints noted.
- **Figures** (`experiment/make_figures.py` → `paper/figures/`, PDF+PNG): a gate
  schematic, the Adult-vs-COMPAS contrast scatter (headline + the honest twist), the Adult
  detector ROC, predictive-strength-by-mechanism bars, and the aggregation comparison. 5 figures
  embedded in `main.tex` (all `\label`/`\ref` matched, all targets present). Generated from the
  saved grid CSVs (no re-run needed).

**How / why.** Assembled "draft from direction" from `research-brief` (argument), `experiment-plan`
(method), `results/tables/<dataset>/` (**exact numbers quoted verbatim**), and the verified
`sources.md`/`related-work.md`. The author owns the argument, interpretation, and final voice
(flagged in a header comment). Numbers used: Adult $r=0.40$ pooled / MNAR $0.52$ / detector AUROC
$0.74$–$0.78$ / EO $0.40$ vs DP $0.30$ / aggregations max $0.398$, mi $0.418$; COMPAS $r=-0.31$ +
diagnosis variants. Impossibility results cited via `caton2024` (avoids an unverified Chouldechova
cite); `feng2023` framed correctly (impute-then-classify is harmful).

**Verified.** All 16 `\cite` keys resolve to bib entries; braces balanced. **No local LaTeX
toolchain** (pdflatex/acmart absent) → the paper is **source-only**; compiles on arXiv/Overleaf
(`pdflatex; bibtex; pdflatex x2`).

**Honesty.** The draft states the scoped claim (Adult holds, COMPAS does not → dataset-dependent),
labels all preprints, and includes the AI-use disclosure per the project compliance note.

**Follow-up.** Author revises into voice; fill author/affiliation/email placeholders; optionally
add figures from `results/figures/<dataset>/`; arrange the arXiv endorsement early; compile + submit.

### 2026-07-01 — Validation-stage gate (pipeline placement)  (branch `feat/validation-gate`)

**What was built** — the third leg of the contribution (pipeline placement):
- `src/pipeline/gate.py` — framework-agnostic `MissingnessDisparityGate` / `evaluate_gate()`.
  Computes the group-wise missingness-disparity statistic on a DataFrame and returns a
  `GateResult` (statistic, threshold, passed, per-feature gaps, per-group null rates, message).
  Needs **only features + the protected-group label — NO target, NO model** — so it runs at a
  true pre-training ingestion gate.
- `src/pipeline/integrations.py` — real adapters. `great_expectations_gate` is **RUNNABLE**: it
  routes per-group null measurement through GE's own validation engine
  (`expect_column_values_to_not_be_null`), then composes the disparity statistic + threshold.
  `deequ_group_completeness` (PyDeequ `Completeness`) and `tfdv_group_null_rates` (TFDV
  statistics) are interface code — same compose pattern, but require Spark/JVM and TensorFlow.
- `experiment/demo_gate.py` — demo: clean batch PASSES, disparate batch FAILS, on Adult + COMPAS;
  runs the GE path if GE is importable.
- Made the `sklearn` import lazy in `disparity.py` (only `mi_weighted` needs it) so the gate runs
  in a minimal environment.

**How / why.** Deequ/TFDV/GE natively measure completeness but carry no group-wise / fairness
notion. The integration pattern is identical for all three: let the framework measure the
per-group, per-column null rate (its native strength), compose into the disparity statistic,
enforce the threshold (our fairness leg). This concretely answers the "just Deequ with a fairness
label" critique — it is a *demographic* disparity check with empirically-characterized predictive
validity, not a bare null-rate constraint.

**Result (runnable).**
- Core gate (main venv): clean batch `D_max = 0.000` → PASS; disparate batch `D_max = 0.300` →
  FAIL (worst feature `age`), on both Adult and COMPAS.
- GE integration (isolated venv): the **same gate, measured through GE's engine**, FAILs the
  disparate batch at 0.300 — verified end-to-end.

**Env note.** GE 0.18.x pins numpy 1.26 / pandas 2.3, conflicting with the benchmark's numpy 2.x /
pandas 3.x, so GE is an **optional** dependency in a separate `.venv-gate` (`requirements-gate.txt`,
gitignored). Deequ (Spark/JVM) and TFDV (TensorFlow) don't install on Python 3.14, so their
adapters are interface code against the documented APIs.

**Threshold.** Demo uses `D_max` threshold 0.10 — the gate's policy knob; tune from the benchmark
detector (AUROC) results.

**Follow-up.** On infra with Spark/TF, run the Deequ/TFDV adapters to show cross-framework parity.

### 2026-07-01 — #3 Literature verification + related-work draft  (branch `feat/literature-verification`)

**What was done**
- **Verified all 16 §6–§8 citations** against primary sources (DOI/arXiv/publisher/proceedings
  pages) using 4 parallel research agents under strict integrity rules (fetch + record the URL +
  a confirming snippet; mark PARTIAL/UNVERIFIED honestly; never guess). Cross-checked their
  evidence before recording.
- Rewrote `notes/sources.md`: every entry now `VERIFIED` with full APA-7 reference,
  DOI/identifier, fetched URL, and a claim-check snippet.
- Organized `notes/lit-review-notes.md` into the 4-part structure (did / did NOT / relation).
- Drafted `paper/related-work.md` (draft-from-direction; author owns the synthesis line).

**Corrections caught during verification (this is why the rule exists)**
- `bhatti2025`: our citation had **no title**. Actual: "The influence of missing data mechanisms
  and simple missing data handling techniques on fairness" (Bhatti, Sandrock & Nienkemper-Swanepoel).
- `grafberger2022`: the article title is "Data distribution debugging in machine learning
  pipelines" — **"mlinspect" is the tool name, not the title**. VLDB Journal 31(5),
  DOI 10.1007/s00778-021-00726-w.
- `feng2023`: **claim reworded** — the paper frames impute-then-classify as *harmful* to fairness,
  not imputation as a positive "lever." Avoid mis-citation.
- `tfdv`: a canonical paper exists (Caveness et al. 2020, SIGMOD) — cite it, not just docs.

**Caveats recorded (not papered over)**
- `caton2024`: metadata confirmed via the ACM DL *listing*, not a rendered full page.
- `mdla2026`: medRxiv (Cloudflare) blocked a direct render; verified via the **resolving DOI**
  (302 → medRxiv) + search-indexed *verbatim* snippets. The "integrate into clinical AI validation
  pipelines" recommendation is consistent but not first-party quoted → that sub-claim is PARTIAL.

**Result.** Zero `UNVERIFIED` rows remain. The verified set confirms the gap: each closest
neighbor misses exactly one axis (model-level: jeong2022/feng2023; post-hoc audit: mdla2026;
different signal: suarezferreira2025; fairness-agnostic tooling: deequ/tfdv/great_expectations/
grafberger2022). The Bhatti-vs-Min tension reconciles as mechanism-*type* vs
presence/correlation of missingness.

**Follow-up.** Optional first-party render of the MDLA recommendation sentence and a caton2024
full-page snippet; normalize citation style to ACM `acmart` numeric at paper integration.

### 2026-06-30 — #6 Diagnose the COMPAS reversal  (branch `fix/diagnose-compas-reversal`)

**Question.** Is COMPAS's reversed correlation (`r(D_max, eo_gap_delta) = −0.31`) a true
counterexample or a structural artifact of zero-inflated count features + median imputation?

**What was built**
- `experiment/diagnose_compas.py` — three probes: (1) zero-inflation profile of the injection
  features (COMPAS vs Adult); (2) noise floor = mean within-(mech,rate,gap) seed-std of `eo_gap`
  vs mean `|eo_gap_delta|`; (3) design variants that re-run the COMPAS grid under cleaner
  injection sets and mean imputation, reporting pooled `r` each time. Writes
  `results/tables/compas/diagnosis.csv`.
- Parametrized imputation: `run_cell(..., impute_strategy="median")` and
  `run_grid(..., impute_strategy=...)` — default unchanged, so this probe could vary it.

**Findings**
- *Probe 1:* COMPAS injection features are heavily zero-inflated (`juv_*` 92–96% zero,
  `priors_count` 32%); only `age` is clean. (Adult also has zero-inflated features but clean
  ones too.)
- *Probe 2:* mean `|Δ EO gap|` = 0.027 vs mean seed-spread of `eo_gap` = 0.062 → effect is
  below the spread. **Caveat:** that spread also includes disparity-*pattern* variation (weights
  are seed-derived), so it overstates pure noise — treat as a rough upper bound.
- *Probe 3 (decisive):* the negative sign does **not** recover — `age`-only injection −0.42,
  two-least-zero-inflated −0.54, mean imputation −0.22. All negative.

**Verdict.** **NOT a zero-inflation/median artifact** — the sign is robust to exactly the knobs
the artifact hypothesis blamed. Honest read: COMPAS shows a **weak but robust negative/null**
relationship; the Adult-positive signal **does not generalize**. Because the effect is small
(partly within noise), frame it as *"no usable signal on COMPAS"*, not a strong opposite effect.

**Implication for the paper.** Predictive validity is demonstrated on Adult and is
**dataset/regime-dependent**; COMPAS is a domain where the check does not hold. One positive +
one null/negative benchmark ⇒ the claim must be **scoped, not universal** — which is itself the
adjudication of the Bhatti (null) vs Min/Asif/Vaidya (positive) tension.

**Follow-up.** Understand *why* the sign is negative on COMPAS (where unprivileged = majority);
and seek a third dataset to map the boundary of where the signal holds.

**Repro:** `python experiment/diagnose_compas.py` (~40s).

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

1. **Frame the scoped claim** — Adult: signal holds (r ≈ 0.4–0.52, MNAR-strongest); COMPAS:
   does not hold (weak/null negative, diagnosed not-an-artifact). The paper claims *demonstrated,
   dataset/regime-dependent* predictive validity — not universal. Consider a third dataset to map
   the boundary.
2. **Paper integration** — assemble related-work + methods + results into `paper/`; normalize
   citations to ACM `acmart` numeric; capture the two outstanding first-party snippets
   (MDLA recommendation sentence, caton2024 full page).
3. On infra with Spark/TensorFlow, run the Deequ/TFDV adapters end-to-end (the GE adapter is
   already runnable) to show cross-framework parity.
