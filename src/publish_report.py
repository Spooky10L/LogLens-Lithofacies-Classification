"""Build the publication-ready per-class report from the SAVED results (no retraining).

Reads   models/v1/per_class_metrics_audited.csv, class_coverage.csv, metrics.json
Writes  models/v1/PER_CLASS_REPORT.md and models/v1/per_class_report.csv

The original training output (per_class_metrics_random_forest.csv) is left untouched.
That file printed 0.0 for Basement; here a class with no validation samples is shown as
"not scorable", because a score cannot be measured without samples.

Run from LogLens_AI/:  python -m src.publish_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["class_code", "class_name", "precision", "recall", "f1", "support_val", "n_predicted"]
NOT_SCORABLE = "not scorable"


def build_report(audited: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in audited.columns]
    if missing:
        raise ValueError(f"per-class table is missing columns {missing}")
    cov = coverage.set_index("class_name")
    rows = []
    for _, r in audited.iterrows():
        c = cov.loc[r["class_name"]]
        scorable = r["support_val"] > 0
        notes = []
        if not scorable:
            notes.append("NOT SCORABLE: no validation samples")
        elif r["recall"] == 0:
            notes.append("none of its validation samples identified correctly")
        if isinstance(c.get("flags"), str) and c["flags"].startswith("RARE"):
            notes.append(c["flags"].split(";")[0])
        rows.append({
            "class": r["class_name"], "code": int(r["class_code"]),
            "precision": r["precision"] if (scorable or r["n_predicted"] > 0) else np.nan,
            "recall": r["recall"] if scorable else np.nan,
            "f1": r["f1"] if scorable else np.nan,
            "validation_samples": int(r["support_val"]), "predicted_samples": int(r["n_predicted"]),
            "training_wells": int(c["train_wells"]), "validation_wells": int(c["val_wells"]),
            "scorable": bool(scorable), "note": "; ".join(notes),
        })
    return pd.DataFrame(rows)


def _fmt(v) -> str:
    return NOT_SCORABLE if pd.isna(v) else f"{v:.3f}"


def render_markdown(report: pd.DataFrame, metrics: dict) -> str:
    m = metrics["metrics_validation"]["random_forest"]
    lines = [
        "# LogLens v1.0 - per-class results (Random Forest)",
        "",
        "Held-out validation wells only (24 wells, 286,870 labelled samples). LogLens' own 80/20 well-wise "
        "split, **not** the official FORCE 2020 competition scoring.",
        "",
        f"Headline: accuracy {m['accuracy']:.4f} · balanced accuracy {m['balanced_accuracy']:.4f} · "
        f"macro-F1 {m['macro_f1']:.4f} · weighted-F1 {m['weighted_f1']:.4f}, averaged over the "
        f"{m['n_classes_scored']} classes that occur in the validation wells.",
        "",
        "| Class | Code | Precision | Recall | F1 | Validation samples | Predicted | Train wells | Val wells | Note |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in report.iterrows():
        lines.append(f"| {r['class']} | {r['code']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | "
                     f"{r['validation_samples']:,} | {r['predicted_samples']:,} | {r['training_wells']} | "
                     f"{r['validation_wells']} | {r['note']} |")
    lines += [
        "",
        "**Reading this table**",
        "- *not scorable* means the class has no samples in the validation wells, so precision/recall/F1 cannot be "
        "measured. It is **not** a score of 0 and the class has not been removed from the data.",
        "- A score of 0.000 (for example Dolomite) is a measured result: the class occurs in validation and none of "
        "it was identified correctly.",
        "- Classes present in only a few wells (see the Train/Val wells columns) give unstable scores.",
        "",
        "Source: `per_class_metrics_audited.csv`, `class_coverage.csv`, `metrics.json` (all derived from the saved "
        "validation predictions; the model was not retrained).",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("models/v1"))
    d = ap.parse_args().dir
    report = build_report(pd.read_csv(d / "per_class_metrics_audited.csv"), pd.read_csv(d / "class_coverage.csv"))
    metrics = json.loads((d / "metrics.json").read_text())
    report.to_csv(d / "per_class_report.csv", index=False)
    (d / "PER_CLASS_REPORT.md").write_text(render_markdown(report, metrics))
    print((d / "PER_CLASS_REPORT.md").read_text())


if __name__ == "__main__":
    main()
