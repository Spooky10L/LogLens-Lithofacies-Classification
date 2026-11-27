# LogLens v1.0 - results summary

**Data:** FORCE 2020 well log and lithofacies dataset - *public benchmark* (118 wells, Norwegian Sea; Zenodo DOI 10.5281/zenodo.4351156, CC BY 4.0 as stated on the record).
**Scope of these numbers:** LogLens' own 80/20 well-wise split of the 118 labelled wells. They are **not** the official FORCE 2020 competition test scoring and must not be compared with leaderboard values. All figures below come from `models/v1/metrics.json`, `models/v1/per_class_metrics_audited.csv` and `models/v1/results_audit.json`; the audit script `python -m src.audit_results` recomputes them from the saved predictions (maximum difference from the saved metrics: 1e-16).

## Setup actually used
| Item | Value |
|---|---|
| Features | GR, RHOB, NPHI, RDEP, DTC, CALI (+ missing-value indicators) |
| Not used as inputs | well name, DEPTH_MD, x/y/z location, confidence curve |
| Rows | 2,337,639 total; 1,431,383 labelled (rows without a label dropped) |
| Invalid values set to NaN | RHOB <= 0: 21,910; DTC <= 0: 9,039 (plus all -999.25) |
| Split | `GroupShuffleSplit(0.2, seed 42)`: 94 training wells, 24 validation wells; **overlap = none** (verified) |
| Training rows | 1,144,513 available; 300,002 used (sampled per well x class, at least one row per group) |
| Validation rows | 286,870 (all labelled samples of the 24 held-out wells) |
| Model | `RandomForestClassifier(n_estimators=150, max_depth=24, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42)` behind a median `SimpleImputer(add_indicator=True)` |
| Versions | Python 3.13.7, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5 |

## Headline metrics (24 held-out wells)
| Model | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| Dummy: always most frequent (Shale) | 0.6662 | 0.0909 | 0.0727 | 0.5327 |
| Dummy: stratified random | 0.4345 | 0.0910 | 0.0901 | 0.4508 |
| **Random Forest** | **0.6786** | **0.4346** | **0.4220** | **0.6924** |

Macro averages cover the **11 classes present in the validation wells** (Basement is absent; see below).

**How to read this:** accuracy is only 0.012 above a model that always says "Shale", because Shale is ~66 % of the validation samples. The real gain is in balanced accuracy (0.43 vs 0.09) and macro-F1 (0.42 vs 0.07): the forest does recognise several other lithologies, but about half of the classes are still found poorly.

## Per-class results (Random Forest)
| Class | Precision | Recall | F1 | Validation samples |
|---|---|---|---|---|
| Shale | 0.883 | 0.772 | 0.824 | 191,101 |
| Anhydrite | 0.900 | 0.906 | 0.903 | 299 |
| Sandstone | 0.550 | 0.719 | 0.623 | 33,802 |
| Coal | 0.504 | 0.566 | 0.533 | 1,216 |
| Chalk | 0.529 | 0.520 | 0.524 | 4,570 |
| Limestone | 0.408 | 0.422 | 0.415 | 14,639 |
| Sandstone/Shale | 0.248 | 0.408 | 0.308 | 26,654 |
| Marl | 0.218 | 0.264 | 0.239 | 6,724 |
| Tuff | 0.259 | 0.169 | 0.204 | 3,222 |
| Halite | 1.000 | 0.035 | 0.068 | 4,295 |
| Dolomite | 0.000 | 0.000 | 0.000 | 348 |
| Basement | n/a | n/a | n/a | **0 - not scorable** |

### Which lithofacies perform poorly, and what the confusion matrix shows
- **Dolomite (F1 0.00, 348 samples):** the forest predicted it 62 times and never correctly. Dolomite occurs in 47 of 118 wells but is only 0.17 % of labelled samples.
- **Halite (recall 0.035):** 4,127 of its 4,295 validation samples (96 %) were predicted as Limestone. All Halite validation samples come from **one well** (16_11-1 ST3), and Halite exists in only 4 wells overall (3 in training), so this number reflects one well and is not a stable estimate. High precision (1.00) comes from only 151 predictions.
- **Tuff (F1 0.20) and Marl (F1 0.24):** spread across Shale, Sandstone/Shale, Sandstone and Limestone.
- **Sandstone/Shale (F1 0.31):** the largest confusion is with Shale. 13.9 % of true Shale is predicted Sandstone/Shale, and 37.2 % of true Sandstone/Shale is predicted Shale. This is a gradational class between two others, and the boundary is interpretive.
- **Limestone / Chalk:** of 4,570 true Chalk samples, 1,181 (26 %) are predicted Limestone and 1,012 (22 %) Sandstone; in the other direction 1,765 true Limestone samples (12 %) are predicted Chalk.
- **Well-to-well spread is large:** per-well accuracy ranges from 0.17 (31_5-4 S) to 0.86 (25_6-3) (`per_well_accuracy_random_forest.csv`).

*Why* these classes fail was not tested. Plausible, untested explanations are overlapping log responses, few training wells, label subjectivity and the lack of vertical context. Treat them as hypotheses.

## The "y_pred contains classes not in y_true" warning
- Cause: the **stratified dummy** baseline draws classes from the training prior, which includes Basement (141 samples, one training well). It predicted Basement 24 times in validation, where Basement has no samples. The Random Forest and the most-frequent dummy never predict an absent class.
- Handling: Basement is reported as **not scorable (n/a)**, never as 0.0 and never removed from the data. Macro averages are computed over the 11 classes present in validation, and that number (`n_classes_scored`) is stored with every metric. `class_coverage.csv` lists Basement and Halite as rare (<= 5 wells).
- Note: the original training output (`per_class_metrics_random_forest.csv`, kept locally and not published) printed 0.0 for Basement. `src/force_v1.py` now writes n/a for such classes, and the published `per_class_metrics_audited.csv` and `PER_CLASS_REPORT.md` apply the same rule to the saved predictions. The model was **not** retrained.

## Feature importance (impurity-based)
RHOB 0.181, RDEP 0.179, DTC 0.168, GR 0.164, CALI 0.129, NPHI 0.114; missing-indicators 0.001-0.015 each. No single log dominates. Impurity importance is biased towards continuous features and splits credit between correlated curves; it is not geological causality.

## Limitations
1. One 80/20 split (seed 42): no repeated grouped cross-validation, so split-to-split uncertainty is unknown.
2. Not the official FORCE competition test/hidden-test scoring; not leaderboard-comparable.
3. Rare classes: Basement (1 well, unscorable), Halite (4 wells), Chalk (15), Anhydrite (10). Their scores rest on 1-4 validation wells.
4. Each sample is classified independently; vertical bed continuity is ignored.
5. Labels are expert interpretations (with a confidence curve), not core-verified truth. The confidence curve is saved for a later weighting experiment but unused.
6. Training used a 300k sample of 1.14M rows; no hyper-parameter tuning, no probability calibration.
7. Median imputation is not a petrophysical estimate; 24-30 % of validation RHOB/NPHI values were missing.
8. Dashboard training-well predictions are in-sample and not a performance estimate.
9. Decision support only. Not validated for operational use; does not replace petrophysical interpretation.
10. LogLens is an independent student research prototype developed for educational and research purposes. It is a machine-learning-based lithofacies classification and decision-support application, not a commercial-grade petrophysical interpretation system.

## Next experiments (not yet done)
Repeated grouped CV; confidence-based sample weights; adding PEF/RMED/DRHO; depth-context (neighbouring-sample) features; penalty-matrix scoring from the competition; evaluation on the official test wells.
