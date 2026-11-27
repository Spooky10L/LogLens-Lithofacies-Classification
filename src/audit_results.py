"""Audit the SAVED LogLens v1 results (no retraining, pandas/numpy only).

Re-computes every headline metric from models/v1/val_predictions.csv.gz,
checks that training and validation wells do not overlap, explains the
sklearn "y_pred contains classes not in y_true" warning, and writes
models/v1/results_audit.json plus per_class_metrics_audited.csv.

Run from LogLens_AI/:  python -m src.audit_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .force_v1 import LABEL, LITHOLOGY_NAMES

MODELS = {"dummy_most_frequent": "pred_dummy_most_frequent",
          "dummy_stratified": "pred_dummy_stratified",
          "random_forest": "pred_random_forest"}


def per_class_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for code, name in LITHOLOGY_NAMES.items():
        sup, npred = int((y == code).sum()), int((p == code).sum())
        tp = int(((y == code) & (p == code)).sum())
        prec = tp / npred if npred else np.nan
        rec = tp / sup if sup else np.nan
        f1 = (0.0 if tp == 0 else 2 * prec * rec / (prec + rec)) if sup else np.nan
        rows.append({"class_code": code, "class_name": name, "precision": prec, "recall": rec,
                     "f1": f1, "support_val": sup, "n_predicted": npred,
                     "status": "scored" if sup else "NOT SCORABLE: absent from validation wells"})
    return pd.DataFrame(rows)


def headline(y, p, tab: pd.DataFrame) -> dict:
    sc = tab[tab.support_val > 0]
    return {"accuracy": float((y == p).mean()),
            "balanced_accuracy": float(sc.recall.mean()),
            "macro_f1": float(sc.f1.mean()),
            "weighted_f1": float(np.average(sc.f1, weights=sc.support_val)),
            "n_classes_scored": int(len(sc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("models/v1"))
    a = ap.parse_args()
    d = a.dir

    v = pd.read_csv(d / "val_predictions.csv.gz")
    y = v[LABEL].to_numpy()
    saved = json.loads((d / "metrics.json").read_text())["metrics_validation"]
    train_w = {w for w in (d / "train_wells.txt").read_text().splitlines() if w}
    val_w = {w for w in (d / "val_wells.txt").read_text().splitlines() if w}

    audit = {
        "wells": {"n_train": len(train_w), "n_val": len(val_w),
                  "overlap": sorted(train_w & val_w),
                  "validation_csv_wells_equal_val_list": set(v["WELL"]) == val_w},
        "models": {},
    }
    for name, col in MODELS.items():
        p = v[col].to_numpy()
        tab = per_class_table(y, p)
        h = headline(y, p, tab)
        extra = sorted(int(c) for c in set(p) - set(y))
        audit["models"][name] = {
            "recomputed": h,
            "saved_in_metrics_json": saved[name],
            "max_abs_difference": max(abs(h[k] - saved[name][k]) for k in
                                      ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")),
            "predicted_classes_absent_from_y_true": {
                str(c): {"name": LITHOLOGY_NAMES[c], "n_rows_predicted": int((p == c).sum())} for c in extra},
        }
        if name == "random_forest":
            tab.to_csv(d / "per_class_metrics_audited.csv", index=False)
            cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(p, name="pred"))
            off = [(LITHOLOGY_NAMES[int(i)], LITHOLOGY_NAMES[int(j)], int(cm.loc[i, j]),
                    float(cm.loc[i, j] / cm.loc[i].sum()))
                   for i in cm.index for j in cm.columns if i != j and cm.loc[i, j] > 0]
            off.sort(key=lambda t: -t[2])
            audit["top_confusions_random_forest"] = [
                {"true": t, "predicted": q, "n": n, "share_of_true_class": round(s, 3)} for t, q, n, s in off[:10]]
    v["_ok"] = v[LABEL] == v["pred_random_forest"]
    pw = v.groupby("WELL").agg(n_labelled=("_ok", "size"), accuracy=("_ok", "mean")).reset_index()
    pw.to_csv(d / "per_well_accuracy_random_forest.csv", index=False)
    audit["explanation_of_warning"] = (
        "sklearn warns 'y_pred contains classes not in y_true' when a model predicts a class that "
        "has no validation samples. Only dummy_stratified does this (it samples classes from the "
        "training prior, including Basement, which exists in one training well and no validation well). "
        "RandomForest and dummy_most_frequent never predict an absent class. Basement is reported as "
        "NOT SCORABLE, not as a 0.0 score, and is excluded only from the macro averages' denominator, "
        "which is stated alongside every headline number.")
    (d / "results_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ("wells",)}, indent=2))
    for n, m in audit["models"].items():
        print(n, "max |recomputed - saved| =", m["max_abs_difference"],
              "| absent-class predictions:", m["predicted_classes_absent_from_y_true"])
    print(pd.read_csv(d / "per_class_metrics_audited.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
