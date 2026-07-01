# Related Work (draft)

> **Draft from direction.** Code drafted this from the `VERIFIED` entries in `notes/sources.md`
> and the organized notes in `notes/lit-review-notes.md`; the **author owns the synthesis line and
> final wording**. Every citation here is verified against its primary source. Preprints (MDLA,
> Suárez Ferreira, Bhatti) are labeled as such. Citation style will be normalized to ACM `acmart`
> numeric at integration; inline author–year is used here for readability.

We organize prior work as a walk from the established premise to the specific gap this paper fills:
(1) data quality as a fairness concern, (2) missingness and fairness specifically, (3) data-
validation tooling, and (4) the two closest competitors we differentiate from.

## Data quality as a fairness concern

A recurring theme in responsible ML is that fairness harms often originate in the data rather than
the model. This data-first stance has a lineage: Sambasivan et al. (2021) show that upstream data
problems compound into downstream "data cascades," and a strand of data documentation — datasheets
(Gebru et al., 2021), data cards (Pushkarna et al., 2022) — records a dataset's provenance,
composition, and quality in prose. Our contribution sits at the quantitative, automated end of that
spectrum: instead of prose documentation, a machine-checkable disparity statistic enforced at a
validation gate. Martínez-Plumed et al. (2021) make the point sharply for missing data, observing
that much fairness work has "simply ignored missing data." For the downstream side, we adopt
equalized odds (Hardt et al., 2016) as our primary metric and name the impossibility result
(Chouldechova, 2017) that constrains it when group base rates differ; a survey (Caton & Haas, 2024)
gives the broader landscape. This paper takes the data-quality premise literally and asks what can
be measured *upstream*, before a model exists.

## Missingness and fairness

The link between missing data and fairness degradation is, by now, established. Goel et al. (2021)
use causal graphs to show that ignoring missingness "nullifies any fairness guarantee," and
Wang & Singh (2021) demonstrate empirically that missing values and selection bias shift fairness
"even when prediction accuracy remains fairly constant" — i.e., the harm is invisible to standard
accuracy monitoring. Min, Asif & Vaidya (2025) strengthen this with peer-reviewed evidence that
missingness correlated with sensitive attributes "can exacerbate disparities, even for little
missingness," which motivates our inclusion of low-rate regimes.

Responses to this problem have been predominantly *model-level* or *post-hoc*. Jeong, Wang &
Calmon (2022) propose an in-processing fair decision tree (Fair MIP Forest) that incorporates
missingness as an attribute and optimizes equalized odds without imputation; Feng, Calmon & Wang
(2023) adapt fairness interventions to missing patterns and, notably, show that naive
impute-then-classify "can significantly worsen group fairness and average accuracy" — a result we
rely on to justify holding imputation fixed as a confound rather than treating it as the remedy.

A genuine tension exists in this literature. Bhatti et al. (2025, preprint) report that the missing
data *mechanism* "does not significantly impact fairness" at high missingness, whereas
Min/Asif/Vaidya and Wang & Singh find clear effects. We read these as compatible rather than
contradictory: Bhatti varies the mechanism *type* under heavy missingness, while the positive
results concern the *presence and group-correlation* of missingness. Our benchmark is designed to
adjudicate exactly this — to characterize the regime in which a group-correlated missingness signal
predicts downstream fairness degradation, and where it does not.

## Data-validation tooling

On the engineering side, mature data-validation frameworks already guard production pipelines.
Deequ (Schelter et al., 2018) offers declarative "unit tests for data," including completeness and
null-rate constraints; TensorFlow Data Validation (Caveness et al., 2020) and Great Expectations
provide schema- and expectation-based gates. None of these carries any group-wise or
fairness-linked notion — they are demographic-agnostic by design. Pipeline-aware fairness tooling
exists but is diagnostic rather than predictive: mlinspect (Grafberger et al., 2022) inspects a
pipeline's operators post hoc for distribution and technical-bias issues (including imputation
skew); FairPrep (Schelter et al., 2020) promotes data to a first-class citizen in
fairness-intervention studies; and Biswas & Rajan (2021) analyze how the fairness of individual
preprocessing stages composes into pipeline-level fairness. Our contribution is to add a single,
fairness-relevant, *predictive* check to precisely this class of validation gate — which is what
makes it a data-engineering artifact rather than another model-level method.

## Closest competitors

Two recent preprints sit nearest our contribution. The Missingness Demographic Leakage Audit
(Patel & Beedala, 2026, medRxiv preprint) is a retrospective, four-step *audit* on ICU datasets
(MIMIC-IV, eICU) testing whether missingness indicators leak demographic information; its findings
are associative and deliberately subtle (racial AUROC ≈ 0.543; Cramér's V < 0.10). It is, by its
own framing, a manual audit rather than an automated pipeline gate, and it measures leakage after
the fact rather than forecasting fairness harm with a calibrated threshold. We build the pipeline
operationalization and predictive framing it points toward, in a non-ICU setting. Suárez Ferreira
et al. (2025, arXiv preprint) propose a data-centric *early indicator* of fairness — but the signal
is disparity in *classification complexity*, not missingness, and the work addresses neither
pipeline placement nor automation. It thus owns the "early indicator" framing on a different
signal.

## The gap

No verified prior work provides an operational, pre-training, group-wise missingness-disparity
statistic, situated in a data-validation gate, with calibrated predictive validity for downstream
equalized-odds degradation. The closest neighbors each miss one axis: they intervene at the model
(Jeong et al., 2022; Feng et al., 2023), audit post hoc (Patel & Beedala, 2026), use a different
signal (Suárez Ferreira et al., 2025), or provide fairness-agnostic tooling (Schelter et al., 2018;
Caveness et al., 2020; Grafberger et al., 2022). This paper occupies that gap — and reports
honestly, via a controlled benchmark, where the signal holds and where it does not.

---

## References (verified — see `notes/sources.md` for evidence)

- Bhatti, A., Sandrock, T., & Nienkemper-Swanepoel, J. (2025). *The influence of missing data
  mechanisms and simple missing data handling techniques on fairness* [Preprint]. arXiv:2503.07313.
- Caton, S., & Haas, C. (2024). Fairness in machine learning: A survey. *ACM Computing Surveys,
  56*(7), Article 166. https://doi.org/10.1145/3616865
- Caveness, E., Suganthan G. C., P., Peng, Z., Polyzotis, N., Roy, S., & Zinkevich, M. (2020).
  TensorFlow Data Validation: Data analysis and validation in continuous ML pipelines. *SIGMOD
  2020*, 2793–2796. https://doi.org/10.1145/3318464.3384707
- Biswas, S., & Rajan, H. (2021). Fair preprocessing: Towards understanding compositional fairness
  of data transformers in machine learning pipeline. *ESEC/FSE 2021*, 981–993.
  https://doi.org/10.1145/3468264.3468536
- Feng, R., Calmon, F. P., & Wang, H. (2023). Adapting fairness interventions to missing values.
  *NeurIPS 36*. arXiv:2305.19429.
- Goel, N., Amayuelas, A., Deshpande, A., & Sharma, A. (2021). The importance of modeling data
  missingness in algorithmic fairness: A causal perspective. *AAAI, 35*(9), 7564–7573.
  https://doi.org/10.1609/aaai.v35i9.16926
- Grafberger, S., Groth, P., Stoyanovich, J., & Schelter, S. (2022). Data distribution debugging in
  machine learning pipelines. *The VLDB Journal, 31*(5), 1103–1126.
  https://doi.org/10.1007/s00778-021-00726-w
- Jeong, H., Wang, H., & Calmon, F. P. (2022). Fairness without imputation: A decision tree approach
  for fair prediction with missing values. *AAAI, 36*(9), 9558–9566.
  https://doi.org/10.1609/aaai.v36i9.21189
- Martínez-Plumed, F., Ferri, C., Nieves, D., & Hernández-Orallo, J. (2021). Missing the missing
  values: The ugly duckling of fairness in machine learning. *International Journal of Intelligent
  Systems, 36*(7), 3217–3258. https://doi.org/10.1002/int.22415
- Min, S., Asif, H., & Vaidya, J. (2025). Exploring the inequitable impact of data missingness on
  fairness in machine learning. *IEEE Intelligent Systems, 40*(3), 28–38.
  https://doi.org/10.1109/MIS.2025.3549484
- Patel, K., & Beedala, P. (2026). *Unmeasured but not unbiased: The Missingness Demographic Leakage
  Audit (MDLA)...* [Preprint]. medRxiv. https://doi.org/10.64898/2026.05.01.26352193
- Schelter, S., He, Y., Khilnani, J., & Stoyanovich, J. (2020). FairPrep: Promoting data to a
  first-class citizen in studies on fairness-enhancing interventions. *EDBT 2020*, 395–398.
  https://doi.org/10.5441/002/edbt.2020.41
- Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F., & Grafberger, A. (2018).
  Automating large-scale data quality verification. *PVLDB, 11*(12), 1781–1794.
  https://doi.org/10.14778/3229863.3229867
- Suárez Ferreira, J., Slavkovik, M., & Casillas, J. (2025). *Uncovering fairness through data
  complexity as an early indicator* [Preprint]. arXiv:2504.05923.
- Wang, Y., & Singh, L. (2021). Analyzing the impact of missing values and selection bias on
  fairness. *International Journal of Data Science and Analytics, 12*(2), 101–119.
  https://doi.org/10.1007/s41060-021-00259-z
- Great Expectations [Computer software]. https://github.com/great-expectations/great_expectations
- Sambasivan, N., Kapania, S., Highfill, H., Akrong, D., Paritosh, P., & Aroyo, L. M. (2021).
  "Everyone wants to do the model work, not the data work": Data cascades in high-stakes AI.
  *CHI 2021*, Article 39. https://doi.org/10.1145/3411764.3445518
- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., &
  Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92.
  https://doi.org/10.1145/3458723
- Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data cards: Purposeful and transparent
  dataset documentation for responsible AI. *ACM FAccT 2022*, 1776–1826.
  https://doi.org/10.1145/3531146.3533231
- Rubin, D. B. (1976). Inference and missing data. *Biometrika, 63*(3), 581–592.
  https://doi.org/10.1093/biomet/63.3.581
- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
  https://doi.org/10.1002/9781119482260
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
  *Advances in NeurIPS 29*. arXiv:1610.02413.
- Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism
  prediction instruments. *Big Data, 5*(2), 153–163. https://doi.org/10.1089/big.2016.0047
