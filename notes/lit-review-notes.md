# Literature Review Notes

Organized as a four-part walk that delivers the reader to the contribution. Every work here is
`VERIFIED` in `sources.md`. For each: *what it did*, *what it did NOT do*, and *how our
contribution relates* (build-on / adjacent / differentiate-from). The author owns the synthesis
line; these are organized notes, not finished prose.

---

## Part 1 — Data quality → fairness (the established link we build ON)

**martinezplumed2021** — *Did:* argues missing values are a neglected dimension of ML fairness
("most recent techniques... have simply ignored missing data"). *Did NOT:* propose an operational
detector or pipeline placement. *Relation:* **build-on** — our motivating premise; we answer the
gap it names.

**caton2024** — *Did:* broad survey of ML fairness (metrics, methods, taxonomy). *Did NOT:* treat
missingness as a data-quality signal or address pipeline placement. *Relation:* **build-on /
context** — situates fairness metrics (we cite for equalized-odds choice + the impossibility
results).

## Part 2 — Missingness-and-fairness specifically (the immediate neighborhood)

**goel2021** — *Did:* causal-graph characterization of missingness mechanisms in algorithmic
fairness; ignoring missingness "nullifies any fairness guarantee." *Did NOT:* give a cheap
pre-training detector or pipeline gate. *Relation:* **build-on** — theoretical grounding for why
group-correlated missingness matters.

**wangsingh2021** — *Did:* empirically shows missing values + selection bias impact fairness "even
when prediction accuracy remains fairly constant." *Did NOT:* forecast harm before training.
*Relation:* **build-on** — evidence the harm is real and accuracy-invisible (motivates a
fairness-specific upstream check).

**jeong2022** — *Did:* in-processing fair decision tree (Fair MIP Forest) with
missing-incorporated-as-attribute, minimizing equalized odds without imputation. *Did NOT:*
detect/forecast; it's a model-level mitigation. *Relation:* **adjacent / contrast** — intervenes
at the model; we intervene upstream and only *detect*.

**feng2023** — *Did:* methods to adapt fairness interventions to missing patterns; shows
impute-then-classify "can significantly worsen group fairness and average accuracy." *Did NOT:*
provide an ingestion-time disparity statistic. *Relation:* **adjacent / build-on** — justifies
holding imputation fixed and treating it as a confound, not the fix. (⚠️ do not cite as
"imputation as a lever" — see sources.md.)

**min2025** — *Did:* peer-reviewed evidence that missingness correlated with sensitive attributes
"can exacerbate disparities, even for little missingness." *Did NOT:* operationalize a detector.
*Relation:* **build-on / strongest motivation** — directly supports the low-rate cell in our grid
and the positive side of the tension.

**bhatti2025** *(preprint)* — *Did:* finds the missing-data *mechanism* "does not significantly
impact fairness" at high missingness (listwise deletion most fair on average). *Did NOT:* test
group-*correlated* missingness as a predictor. *Relation:* **adjudicate** — the null side of the
tension; our benchmark characterizes the regime where the signal holds vs. fails (the
mechanism-type vs presence/correlation distinction reconciles Bhatti with min2025/feng2023).

## Part 3 — Pipeline / data-validation tooling (the engineering side)

**schelter2018deequ** — *Did:* declarative large-scale data-quality verification ("unit tests for
data": completeness/null-rate constraints). *Did NOT:* any group-wise or fairness-linked check.
*Relation:* **build-on / host** — the kind of framework our check plugs into; pre-empts "just
Deequ with a fairness label" by adding the demographic-disparity + predictive-validity legs.

**tfdv** (Caveness et al. 2020) — *Did:* schema/completeness validation in continuous TFX
pipelines. *Did NOT:* demographic missingness disparity or fairness linkage. *Relation:*
**adjacent / host** — generic gate; we add the fairness-relevant signal.

**greatexpectations** *(software)* — *Did:* expectation-based data validation (nulls, ranges,
uniqueness). *Did NOT:* fairness linkage. *Relation:* **adjacent / host** — another candidate
home for the check.

**grafberger2022** — *Did:* `mlinspect` — extracts a pipeline DAG and inspects operators for
distribution/technical-bias issues (incl. imputation skew). *Did NOT:* a *predictive*,
ingestion-time missingness-disparity gate; it's a post-hoc debugger on a written pipeline.
*Relation:* **adjacent / differentiate** — diagnoses, doesn't forecast.

**schelter2020fairprep** — *Did:* FairPrep — promotes data to first-class in fairness-intervention
studies (evaluation/design framework). *Did NOT:* a gate or predictor. *Relation:* **adjacent** —
shares the data-centric-fairness spirit; different artifact.

**biswasrajan2021** — *Did:* studies how local fairness of preprocessing stages composes into
global pipeline fairness. *Did NOT:* a missingness-specific upstream detector. *Relation:*
**adjacent** — compositional view of preprocessing; complementary framing.

## Part 4 — The gap + close competitors we differentiate from

**mdla2026** *(preprint, nearest neighbor)* — *Did:* a retrospective four-step *audit* on ICU data
(MIMIC-IV, eICU) testing whether missingness indicators predict demographics; associative, subtle
(racial AUROC 0.543, Cramér's V < 0.10). *Did NOT:* (a) an automated pipeline gate, (b) a
predictive/early-warning threshold, (c) a generalizable cross-dataset metric. *Relation:*
**differentiate (critical)** — we build the pipeline operationalization + predictive framing it
calls for; different setting. MONITOR for a peer-reviewed v2 (convergence risk).

**suarezferreira2025** *(preprint)* — *Did:* proposes a data-centric *early indicator* of fairness
based on **classification complexity** disparities between groups. *Did NOT:* use missingness;
address pipeline placement or automation. *Relation:* **differentiate** — owns the "early
indicator" *framing* but on a different signal; we own missingness + the gating contribution.

---

### The gap, in one line
No verified work provides an **operational, pre-training, group-wise missingness-disparity check,
placed in a data-validation gate, with calibrated predictive validity** for downstream
equalized-odds degradation. Closest neighbors each miss one axis: model-level (jeong2022,
feng2023), post-hoc audit (mdla2026), different signal (suarezferreira2025), or
fairness-agnostic tooling (deequ/tfdv/great_expectations/grafberger2022).
