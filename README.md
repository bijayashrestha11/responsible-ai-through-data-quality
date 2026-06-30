# Responsible AI Through Data Quality

Group-correlated missingness as an automatable upstream data-quality check that predicts
downstream fairness degradation — with a defended pipeline placement.

> **Status:** scaffolding. See `research-brief.md` (the what), `CLAUDE.md` (the how, incl. the
> verify-every-citation rule), and `experiment-plan.md` (the benchmark design).

---

## Repository layout

```
research-brief.md          # contribution, gap, locked decisions — source of truth
CLAUDE.md                  # working rules for Claude Code: citations, conventions, disclosure
experiment-plan.md         # benchmark experiment design
README.md                  # this file

notes/
  sources.md               # VERIFIED citations only: full ref + identifier + verification note
  lit-review-notes.md      # organized per research-brief §9 (4-part structure)

paper/
  outline.md
  related-work.md          # drafted from verified notes
  main.tex                 # acmart numeric (author's choice; arXiv target)

src/
  missingness/             # missingness injection (MCAR/MAR/MNAR, group-correlated)
  metric/                  # the missingness-disparity statistic (the proposed artifact)
  fairness/                # equalized-odds + demographic-parity computation
  pipeline/                # the check as a validation-stage gate (Deequ/GE/TFDV integration)

experiment/
  benchmark/               # Adult + COMPAS controlled study (the scientific core)
  claims/                  # single real-world demonstration slice (minimal; may be cut)
  configs/                 # grid + seed definitions
  run_all.py               # one command → regenerates all figures/tables

results/
  figures/
  tables/
  cache/                   # cached intermediates so run_all is fast + reproducible
```

## Data handling

- **Benchmark:** Adult, COMPAS — public, committable references/loaders, fully reproducible.
- **Claims:** raw data NEVER committed. Only derived, aggregate, non-identifiable outputs.
  No PHI or identifiers in code, commits, or prose.

## Reproducibility contract

Fixed seeds; `python experiment/run_all.py` regenerates every figure and table from cache.
A reviewer (or a professor evaluating this repo) can clone and rerun the benchmark arm with no
special access.

---

## First-session checklist for Claude Code

The two tracks below are independent — run in parallel.

**Track A — Build (critical path to the empirical claim):**
1. Pin the missingness-disparity statistic definition (`src/metric/`) — see experiment-plan
   "metric under test." Decide max-vs-mean aggregation and justify.
2. Implement missingness injection (`src/missingness/`) for MCAR/MAR/MNAR + group correlation.
3. Implement fairness metrics (`src/fairness/`): equalized-odds gap (primary), demographic
   parity (secondary).
4. Wire the per-cell pipeline + grid/seed configs; get Adult running end to end before COMPAS.
5. Run the grid, produce the scatter + detector results.

**Track B — Literature (verify → organize → draft):**
1. Verify every reference in `research-brief.md` §6–§8 against the real source; write to
   `notes/sources.md`. (MDLA already read in full — still record it.)
2. Organize into `notes/lit-review-notes.md` using the 4-part structure (research-brief §9).
3. Draft `paper/related-work.md` from verified notes — author owns the synthesis.

**Before anything:** fill the disclosure statement in `CLAUDE.md` to match the real workflow,
and arrange the arXiv endorsement (it can take days).
