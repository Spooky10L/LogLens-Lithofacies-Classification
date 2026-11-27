# LogLens v1.0 - engineering notes

Dataset: FORCE 2020 well log and lithofacies dataset - **public benchmark** (Zenodo, DOI 10.5281/zenodo.4351156).
Class names come from the competition's own starter notebook (`lithology_keys`).

> **Status: trained and evaluated.** Real numbers, per-class results and limitations are in `RESULTS_SUMMARY.md`; they are not repeated here so they cannot drift out of sync.

## 1. Why these six curves (GR, RHOB, NPHI, RDEP, DTC, CALI)
- They are the six curves present in **all 118** wells (audit result), so every well can be used with the same feature set.
- Each responds to something geologists use to separate rocks: GR (clay/shale content), RHOB and NPHI (density and porosity; their crossover helps separate sand, carbonate, shale), RDEP (deep resistivity: fluid and evaporite/coal signatures), DTC (compressional slowness: compaction, cementation), CALI (hole size: flags washouts that distort RHOB/NPHI).
- CALI is a *quality/context* curve rather than a rock property. It is kept because bad hole conditions change the other readings. Its importance should be read with that in mind.
- Curves available in only some wells (PEF, SP, RMED, ...) are deliberately left out of v1 so that missingness is not driven by which service company logged a well.

## 2. Why missing values are imputed (and flagged)
- Many rows have a gap in one curve (audit: NPHI ~54 %, RHOB ~40 % missing on average per well). Dropping any row with a gap would remove a large share of data and bias it towards wells with complete logs.
- Median imputation (learned from the training wells only, inside the Pipeline) keeps every row usable. The missing-indicator columns tell the forest *that* a value was filled, because "no density log" is itself informative (e.g. cased or washed-out intervals).
- Limitation: a median is not a petrophysical estimate. Imputed rows are less reliable. Indicators only appear for features that had gaps in the training data.
- Physically impossible values (RHOB, RDEP, DTC, CALI <= 0) are set to NaN too; the counts are written to `metrics.json`.

## 3. Why Random Forest
- Handles non-linear, overlapping log responses and mixed scales without feature scaling.
- Robust to outliers and noisy logs; trains quickly on ~300 k rows.
- `class_weight='balanced_subsample'` raises the weight of rare classes in each tree's bootstrap sample.
- Compared against two DummyClassifier baselines so any gain is measured, not assumed.
- Limitation: it classifies each depth sample independently and ignores the vertical sequence (a bed is continuous); this is a known simplification of v1.

## 4. Why validation is well-wise
Neighbouring depth samples (0.152 m apart) are almost copies of each other. A random row split puts near-duplicates in train and test, so the score mostly measures memorisation of that well. Holding out whole wells asks the real question: *how does the model do in a well it has never seen?* v1 uses `GroupShuffleSplit(test_size=0.2, random_state=42)` on the well name; the exact well lists are saved in `models/v1/train_wells.txt` and `val_wells.txt`.

**Scope of the claim:** this is LogLens' own 80/20 split of the 118 labelled wells. It is not the official FORCE 2020 competition test scoring and must not be compared with leaderboard values.

## 5. Why accuracy alone is misleading
About 61 % of labelled samples are Shale. A model that always answers "Shale" would score roughly that in accuracy while finding no sandstone, limestone or coal at all. So v1 reports balanced accuracy (mean recall per class), macro-F1 (every class counts equally), weighted-F1, and per-class precision/recall/F1, always next to the dummy baselines.

Headline macro metrics are computed over the classes present in the validation wells; a class with zero validation samples cannot be scored. Those classes are listed in `class_coverage.csv` and `metrics.json` rather than hidden.

## 6. Rare classes
Rare classes (present in <= 5 of the labelled wells) are flagged in `models/v1/class_coverage.csv`. They are **not** removed. Consequences to expect and report honestly: a rare class may be missing from training or validation after a well-wise split, and its scores are based on very few wells.

## 7. Results (fill from `models/v1/metrics.json` only)
See `RESULTS_SUMMARY.md` (source: `models/v1/metrics.json`, `per_class_metrics_audited.csv`, `results_audit.json`).

## 8. Known limitations of v1
- Synthetic-free but still single-split: one 80/20 draw can be lucky or unlucky; grouped cross-validation is a later step.
- Random sampling of the 300 k training rows is within (well, class) groups; every group keeps at least one row.
- Impurity-based feature importance favours continuous features and can be inflated by correlated curves.
- Labels are expert interpretations (with a confidence curve), not ground truth core descriptions. Confidence is saved in `val_predictions.csv.gz` and `data/processed/force_v1_labelled.csv.gz` for a later sample-weighting experiment but is not used here.
- Predictions are decision-support only and do not replace petrophysical interpretation.
