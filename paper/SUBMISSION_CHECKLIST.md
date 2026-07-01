# Submission checklist (arXiv)

The paper draft is complete but not yet submission-ready. These are the author-owned steps.

## Blockers (must do)
- [ ] **Fill author metadata** in `main.tex`: `[Author Name]`, `[Affiliation — independent
  researcher]`, `[email]`, and `\shortauthors`.
- [ ] **Compile it.** There is no LaTeX toolchain in the dev environment, so the draft has been
  structure-checked (braces, labels/refs, figure paths, cite keys) but **never rendered**. First
  build on Overleaf or a local TeX dist:
  ```
  pdflatex main ; bibtex main ; pdflatex main ; pdflatex main
  ```
  Expect minor float placement nudges on first build (`[t]` → `[ht]`/`[h!]` as needed).
- [ ] **Read and own the argument/voice.** This is a draft assembled from direction; the author
  owns the framing, interpretation, and final wording.
- [ ] **arXiv endorsement.** First-time cs submitters need an endorsement — **arrange early, it can
  take days.** Decide primary category (cs.LG / cs.CY / stat.ML) and secondary.
- [ ] **arXiv license** selection.

## How to submit (self-contained)
Upload the `paper/` folder: `main.tex`, `references.bib`, and `figures/`. `\graphicspath{{figures/}}`
resolves relatively — no `../` paths, so it works on arXiv/Overleaf as-is.

## Regenerating figures (if results change)
```
python experiment/make_figures.py      # writes paper/figures/*.{pdf,png}
```
Regenerate the underlying results first if needed: `python experiment/run_all.py --dataset all`.

## Verified content (safe to lean on)
- All 16 references verified against primary sources (`notes/sources.md`); preprints labeled.
- Great Expectations integration **executed** end to end; Deequ/TFDV are **interface-compatible,
  not executed** (the paper says so — do not imply otherwise).
- MDLA (`mdla2026`) re-checked (2026-07-01): still preprint v1, no gate/predictor built — the
  differentiation framing (automation + predictive threshold, *not* timing) is accurate.
- AI-use disclosure is included in the endmatter; keep it honest to the actual workflow.

## Notes
- `research-brief.md` and `CLAUDE.md` are local-only (gitignored) — not part of the submission.
- Open scientific thread (optional strengthening): a dataset in which the three co-varying
  structural properties diverge would isolate which drives the sign of the signal.
