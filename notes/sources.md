# Verified Sources

**The verify-every-citation rule (CLAUDE.md) governs this file.** A row may move to
`VERIFIED` only after the actual source has been fetched and read: authors, title, venue,
year, and DOI/identifier confirmed against the source, *and* the source confirmed to say
what we attribute to it. Until then it stays `UNVERIFIED` and must not appear in drafted
prose. Preprints (MDLA, Suárez Ferreira, Bhatti) must be labeled "preprint / not
peer-reviewed" everywhere they are cited.

Per-row format: `Status | Cite key | Full reference | Identifier | Verification note`.

---

## §6 — Established background (build ON, not novelty)

| Status | Key | Reference | Identifier | Verified note |
|---|---|---|---|---|
| UNVERIFIED | martinezplumed2021 | Martínez-Plumed et al. (2021). Missing the missing values: The ugly duckling of fairness in ML. *Int. J. Intelligent Systems* 36(7). | — | — |
| UNVERIFIED | goel2021 | Goel et al. (2021). The Importance of Modeling Data Missingness in Algorithmic Fairness: A Causal Perspective. *AAAI* 35(9). | — | — |
| UNVERIFIED | wangsingh2021 | Wang & Singh (2021). Analyzing the impact of missing values and selection bias on fairness. *Int. J. Data Science and Analytics* 12(2). | — | — |
| UNVERIFIED | jeong2022 | Jeong et al. (2022). Fairness without Imputation: A Decision Tree Approach. *AAAI* 36(9). | arXiv:2109.10431 | — |
| UNVERIFIED | feng2023 | Feng, Calmon & Wang (2023). Adapting Fairness Interventions to Missing Values. *NeurIPS* 36. | — | — |
| UNVERIFIED | min2025 | Min, Asif & Vaidya (2025). Exploring the Inequitable Impact of Data Missingness on Fairness in ML. *IEEE Intelligent Systems* 40(3). | — | — |
| UNVERIFIED | caton2024 | Caton & Haas (2024). Fairness in Machine Learning: A Survey. *ACM Computing Surveys* 56(7). | — | — |
| UNVERIFIED | bhatti2025 | Bhatti et al. (2025). [preprint — contested null result on missingness mechanism]. | arXiv:2503.07313 | preprint / not peer-reviewed |

## §7 — Close competitors to differentiate from (CRITICAL)

| Status | Key | Reference | Identifier | Verified note |
|---|---|---|---|---|
| UNVERIFIED | mdla2026 | Patel & Beedala (2026). Unmeasured but Not Unbiased: The Missingness Demographic Leakage Audit. *medRxiv*. | DOI 10.64898/2026.05.01.26352193 | preprint / not peer-reviewed; nearest neighbor, read in full |
| UNVERIFIED | suarezferreira2025 | Suárez Ferreira et al. (2025). Uncovering Fairness through Data Complexity as an Early Indicator. | arXiv:2504.05923 | preprint / not peer-reviewed |

## §8 — Adjacent prior art (pipeline side)

| Status | Key | Reference | Identifier | Verified note |
|---|---|---|---|---|
| UNVERIFIED | grafberger2022 | Grafberger, Groth, Stoyanovich & Schelter (2022). mlinspect. *VLDB Journal* 31. | — | — |
| UNVERIFIED | schelter2020fairprep | Schelter et al. (2020). FairPrep. *EDBT*. | — | — |
| UNVERIFIED | biswasrajan2021 | Biswas & Rajan (2021). Fair Preprocessing. *ESEC/FSE*. | — | — |
| UNVERIFIED | schelter2018deequ | Schelter et al. (2018). Deequ. *PVLDB* 11(12). | — | — |
| UNVERIFIED | tfdv | TensorFlow Data Validation. | — | generic completeness gate; no demographic disparity |
| UNVERIFIED | greatexpectations | Great Expectations. | — | generic completeness gate; no fairness linkage |
