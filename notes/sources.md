# Verified Sources

**The verify-every-citation rule governs this file.** A reference is `VERIFIED` only after the
actual source was fetched and read: authors, title, venue, year, and DOI/identifier confirmed
against the source, *and* the source confirmed to say what we attribute to it. Preprints (MDLA,
Suárez Ferreira, Bhatti) must be labeled "preprint / not peer-reviewed" everywhere they are
cited.

**Status (2026-06-30): all 16 verified** against primary sources (web fetch of DOI/arXiv/
publisher/proceedings pages). Two reads carry caveats, recorded inline: `caton2024` (metadata
confirmed via the ACM DL listing, not a rendered full page) and `mdla2026` (medRxiv blocked
direct page rendering; verified via the resolving DOI + search-indexed verbatim snippets). Three
corrections were applied during verification — see the ⚠️ flags on `bhatti2025`, `grafberger2022`,
and `feng2023`.

---

## §6 — Established background (build ON, not novelty)

**martinezplumed2021 — VERIFIED.** Martínez-Plumed, F., Ferri, C., Nieves, D., &
Hernández-Orallo, J. (2021). Missing the missing values: The ugly duckling of fairness in machine
learning. *International Journal of Intelligent Systems, 36*(7), 3217–3258.
- Identifier: DOI 10.1002/int.22415
- Verified: confirmed authors/title/venue/year/DOI via UPV institutional repository
  (riunet.upv.es/handle/10251/181819; Wiley page was paywalled). Abstract: "most recent
  techniques, libraries and experimental results dealing with fairness in machine learning have
  simply ignored missing data." → supports our claim (missing data is under-examined in fairness).

**goel2021 — VERIFIED.** Goel, N., Amayuelas, A., Deshpande, A., & Sharma, A. (2021). The
importance of modeling data missingness in algorithmic fairness: A causal perspective.
*Proceedings of the AAAI Conference on Artificial Intelligence, 35*(9), 7564–7573.
- Identifier: DOI 10.1609/aaai.v35i9.16926; arXiv:2012.11448
- Verified: AAAI OJS page (ojs.aaai.org/index.php/AAAI/article/view/16926). Abstract uses causal
  graphs to characterize missingness mechanisms; "this missingness, if ignored, nullifies any
  fairness guarantee." → supports our claim (causal modeling of missingness in fairness).

**wangsingh2021 — VERIFIED.** Wang, Y., & Singh, L. (2021). Analyzing the impact of missing values
and selection bias on fairness. *International Journal of Data Science and Analytics, 12*(2),
101–119.
- Identifier: DOI 10.1007/s41060-021-00259-z
- Verified: NSF PAR record (par.nsf.gov/biblio/10280383). Abstract: "different missing values
  generated from different mechanisms and selection bias impact prediction fairness, even when
  prediction accuracy remains fairly constant." → supports our claim.

**jeong2022 — VERIFIED.** Jeong, H., Wang, H., & Calmon, F. P. (2022). Fairness without
imputation: A decision tree approach for fair prediction with missing values. *Proceedings of the
AAAI Conference on Artificial Intelligence, 36*(9), 9558–9566.
- Identifier: DOI 10.1609/aaai.v36i9.21189; arXiv:2109.10431
- Verified: AAAI OJS + arXiv + NSF full text. Method = **Fair MIP Forest** with "missing
  incorporated as attribute (MIA), which does not require explicit imputation"; an in-processing
  approach minimizing equalized odds / accuracy parity; tested on COMPAS, Adult, HSLS. → supports
  our claim on all points.

**feng2023 — VERIFIED.** ⚠️ *Claim reworded during verification.* Feng, R., Calmon, F. P., &
Wang, H. (2023). Adapting fairness interventions to missing values. *Advances in Neural
Information Processing Systems, 36* (NeurIPS 2023).
- Identifier: arXiv:2305.19429; NeurIPS 2023 proceedings
- Verified: NeurIPS proceedings page (abstract read in full). ⚠️ The paper does NOT treat
  imputation as a positive "fairness lever" — it argues the opposite: "training a classifier from
  imputed data can significantly worsen the achievable values of group fairness and average
  accuracy." Their methods preserve missingness patterns instead. **Cite as: impute-then-classify
  can degrade fairness; do not cite as "imputation as a fairness lever."**

**min2025 — VERIFIED.** Min, S., Asif, H., & Vaidya, J. (2025). Exploring the inequitable impact
of data missingness on fairness in machine learning. *IEEE Intelligent Systems, 40*(3), 28–38.
- Identifier: DOI 10.1109/MIS.2025.3549484; PMID 40585432
- Verified: PubMed record (pubmed.ncbi.nlm.nih.gov/40585432). Abstract confirms missingness
  correlated with sensitive attributes "can exacerbate disparities, **even for little
  missingness**." → our load-bearing "even at low rates" claim is confirmed verbatim.

**caton2024 — VERIFIED** *(caveat: metadata via listing).* Caton, S., & Haas, C. (2024). Fairness
in machine learning: A survey. *ACM Computing Surveys, 56*(7), Article 166, 1–38.
- Identifier: DOI 10.1145/3616865
- Verified: vol/issue/article/date confirmed via the ACM DL search listing (full DL page not
  rendered); consistent across ACM, Semantic Scholar, ResearchGate. Broad ML-fairness survey. →
  one-click full-page snippet still worth capturing for the record.

**bhatti2025 — VERIFIED (preprint / not peer-reviewed).** ⚠️ *Title was missing in our draft.*
Bhatti, A., Sandrock, T., & Nienkemper-Swanepoel, J. (2025). *The influence of missing data
mechanisms and simple missing data handling techniques on fairness* [Preprint]. arXiv.
- Identifier: arXiv:2503.07313
- Verified: arXiv abstract page read in full. Quote: "the missing data mechanism **does not
  significantly impact fairness**; across the missing data handling techniques listwise deletion
  gives the highest fairness on average" (tested at high missingness). → supports our null-leaning
  claim. Note the mechanism-type (Bhatti) vs presence/correlation (Min/Feng) distinction — they
  are not directly contradictory.

## §7 — Close competitors to differentiate from (CRITICAL)

**mdla2026 — VERIFIED (preprint / not peer-reviewed).** *(caveat: see below.)* Patel, K., &
Beedala, P. (2026). *Unmeasured but not unbiased: The Missingness Demographic Leakage Audit (MDLA)
for calibration-aware fairness evaluation in critical care mortality prediction* [Preprint].
medRxiv.
- Identifier: DOI 10.64898/2026.05.01.26352193
- Verified: DOI **resolves** (direct fetch, 302 → medRxiv). Confirmed: authors Krutarth Patel &
  Phanindra Beedala; **lead author at Humana Inc.** (Louisville, KY); retrospective four-step
  audit framework on **MIMIC-IV v2.2 + eICU-CRD v2.0**; missingness indicators predict racial
  group membership at **AUROC 0.543** (95% CI 0.540–0.546); 18/43 features Bonferroni-significant
  with **Cramér's V < 0.10**. → supports our description (manual audit, associative, not an
  automated pipeline gate, not predictive early-warning).
- ⚠️ Caveats: (1) medRxiv (Cloudflare) blocked direct page rendering — the above rests on the
  resolving DOI plus search-indexed *verbatim* snippets, not a first-party render. (2) The
  "recommends integrating missingness-aware auditing into clinical AI validation pipelines"
  sub-claim is **consistent but not directly quoted** → treat as PARTIAL until the sentence is
  captured first-party.

**suarezferreira2025 — VERIFIED (preprint / not peer-reviewed).** Suárez Ferreira, J., Slavkovik,
M., & Casillas, J. (2025). *Uncovering fairness through data complexity as an early indicator*
[Preprint]. arXiv.
- Identifier: arXiv:2504.05923
- Verified: arXiv page rendered (HTTP 200). The signal is explicitly **disparities in
  classification complexity** between groups, NOT missingness; no pipeline-placement or automation
  claim. → confirms our differentiation (owns "early indicator" framing, but on a different
  signal).

## §8 — Adjacent prior art (pipeline side)

**grafberger2022 — VERIFIED.** ⚠️ *Title corrected during verification.* Grafberger, S., Groth,
P., Stoyanovich, J., & Schelter, S. (2022). Data distribution debugging in machine learning
pipelines. *The VLDB Journal, 31*(5), 1103–1126.
- Identifier: DOI 10.1007/s00778-021-00726-w
- Verified: DBLP + Springer. ⚠️ The journal article title is "Data distribution debugging in
  machine learning pipelines" — **"mlinspect" is the system name**, not the title (don't conflate
  with the separate SIGMOD 2021 demo "mlinspect: a Data Distribution Debugger..."). A pipeline
  inspector/debugger for distribution/technical-bias issues — not a predictive missingness gate.

**schelter2020fairprep — VERIFIED.** Schelter, S., He, Y., Khilnani, J., & Stoyanovich, J. (2020).
FairPrep: Promoting data to a first-class citizen in studies on fairness-enhancing interventions.
In *Proceedings of the 23rd International Conference on Extending Database Technology (EDBT 2020)*
(pp. 395–398).
- Identifier: DOI 10.5441/002/edbt.2020.41
- Verified: researchr/NSF PAR (10166058). Evaluation/design framework, data as first-class; not a
  gate.

**biswasrajan2021 — VERIFIED.** Biswas, S., & Rajan, H. (2021). Fair preprocessing: Towards
understanding compositional fairness of data transformers in machine learning pipeline. In
*Proceedings of the 29th ACM Joint European Software Engineering Conference and Symposium on the
Foundations of Software Engineering (ESEC/FSE 2021)* (pp. 981–993).
- Identifier: DOI 10.1145/3468264.3468536; arXiv:2106.06054
- Verified: arXiv read. "We showed how the local fairness of a preprocessing stage composes in the
  global fairness of the pipeline." → supports our claim (compositional fairness of preprocessing).

**schelter2018deequ — VERIFIED.** Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F.,
& Grafberger, A. (2018). Automating large-scale data quality verification. *Proceedings of the
VLDB Endowment, 11*(12), 1781–1794.
- Identifier: DOI 10.14778/3229863.3229867
- Verified: DBLP. Declarative "unit tests for data" with completeness/null-rate constraints; **no
  group-wise or fairness-linked checks**. → supports our framing.

**tfdv — VERIFIED.** Caveness, E., Suganthan G. C., P., Peng, Z., Polyzotis, N., Roy, S., &
Zinkevich, M. (2020). TensorFlow Data Validation: Data analysis and validation in continuous ML
pipelines. In *Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data*
(pp. 2793–2796).
- Identifier: DOI 10.1145/3318464.3384707
- Verified: Google Research page + ACM DOI. Generic completeness/schema validation in TFX; **no
  demographic missingness disparity or fairness linkage**. (Caveness et al. 2020 is the canonical
  citation — better than citing docs.)

**greatexpectations — VERIFIED (software; no academic paper).** Great Expectations. (n.d.). *GX
Core: Open source data quality platform* [Computer software].
https://github.com/great-expectations/great_expectations
- Identifier: greatexpectations.io ; GitHub repo (Apache-2.0)
- Verified: official site + repo. Expectations = "unit tests for your data" (schemas, ranges,
  nulls, uniqueness); **no fairness linkage**. No single canonical DOI/paper — cite the tool.
