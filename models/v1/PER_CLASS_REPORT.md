# LogLens v1.0 - per-class results (Random Forest)

Held-out validation wells only (24 wells, 286,870 labelled samples). LogLens' own 80/20 well-wise split, **not** the official FORCE 2020 competition scoring.

Headline: accuracy 0.6786 · balanced accuracy 0.4346 · macro-F1 0.4220 · weighted-F1 0.6924, averaged over the 11 classes that occur in the validation wells.

| Class | Code | Precision | Recall | F1 | Validation samples | Predicted | Train wells | Val wells | Note |
|---|---|---|---|---|---|---|---|---|---|
| Sandstone | 30000 | 0.550 | 0.719 | 0.623 | 33,802 | 44,241 | 94 | 24 |  |
| Sandstone/Shale | 65030 | 0.248 | 0.408 | 0.308 | 26,654 | 43,841 | 93 | 24 |  |
| Shale | 65000 | 0.883 | 0.772 | 0.824 | 191,101 | 167,049 | 94 | 24 |  |
| Marl | 80000 | 0.218 | 0.264 | 0.239 | 6,724 | 8,131 | 71 | 21 |  |
| Dolomite | 74000 | 0.000 | 0.000 | 0.000 | 348 | 62 | 37 | 10 | none of its validation samples identified correctly |
| Limestone | 70000 | 0.408 | 0.422 | 0.415 | 14,639 | 15,140 | 91 | 23 |  |
| Chalk | 70032 | 0.529 | 0.520 | 0.524 | 4,570 | 4,496 | 11 | 4 |  |
| Halite | 88000 | 1.000 | 0.035 | 0.068 | 4,295 | 151 | 3 | 1 | RARE: in only 4 of 118 labelled wells |
| Anhydrite | 86000 | 0.900 | 0.906 | 0.903 | 299 | 301 | 8 | 2 |  |
| Tuff | 99000 | 0.259 | 0.169 | 0.204 | 3,222 | 2,093 | 52 | 15 |  |
| Coal | 90000 | 0.504 | 0.566 | 0.533 | 1,216 | 1,365 | 51 | 12 |  |
| Basement | 93000 | not scorable | not scorable | not scorable | 0 | 0 | 1 | 0 | NOT SCORABLE: no validation samples; RARE: in only 1 of 118 labelled wells |

**Reading this table**
- *not scorable* means the class has no samples in the validation wells, so precision/recall/F1 cannot be measured. It is **not** a score of 0 and the class has not been removed from the data.
- A score of 0.000 (for example Dolomite) is a measured result: the class occurs in validation and none of it was identified correctly.
- Classes present in only a few wells (see the Train/Val wells columns) give unstable scores.

Source: `per_class_metrics_audited.csv`, `class_coverage.csv`, `metrics.json` (all derived from the saved validation predictions; the model was not retrained).
