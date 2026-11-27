"""Train and evaluate a well-log lithofacies classifier with well-wise holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CANDIDATES = [
    "GR", "RHOB", "NPHI", "RDEP", "RMED", "RSHA", "PEF",
    "DTC", "DTS", "SP", "CALI", "BS", "ROP", "MUDWEIGHT",
]


def choose_group_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                       test_size: float = 0.20, random_state: int = 42):
    """Choose a held-out-well split with good class coverage where possible."""
    splitter = GroupShuffleSplit(n_splits=80, test_size=test_size, random_state=random_state)
    all_classes = set(y.unique())
    best = None
    best_score = -1
    for train_idx, test_idx in splitter.split(X, y, groups):
        train_classes = set(y.iloc[train_idx].unique())
        test_classes = set(y.iloc[test_idx].unique())
        shared = len(train_classes & test_classes)
        score = shared * 10000 + len(test_classes) * 100 + len(train_classes)
        if score > best_score:
            best, best_score = (train_idx, test_idx), score
        if train_classes == all_classes and test_classes == all_classes:
            return train_idx, test_idx
    if best is None:
        raise ValueError("Unable to split by well. Check that multiple wells are present.")
    return best


def train(data: pd.DataFrame, model_dir: Path = ROOT / "models") -> dict:
    required = {"WELL", "LITHOLOGY"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    features = [c for c in FEATURE_CANDIDATES if c in data.columns and data[c].notna().any()]
    if len(features) < 3:
        raise ValueError(f"Need at least 3 usable log curves; found {features}")

    work = data.copy()
    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work.loc[~np.isfinite(work[col]), col] = np.nan
    work["LITHOLOGY"] = pd.to_numeric(work["LITHOLOGY"], errors="coerce")
    work = work.dropna(subset=["WELL", "LITHOLOGY"])
    work["LITHOLOGY"] = work["LITHOLOGY"].astype(int)
    # Remove unlabelled/sentinel rows if present. FORCE labels are positive codes.
    work = work[work["LITHOLOGY"] > 0]

    X, y, groups = work[features], work["LITHOLOGY"], work["WELL"].astype(str)
    if groups.nunique() < 4:
        raise ValueError("Use at least four wells to support a meaningful well-wise holdout.")

    train_idx, test_idx = choose_group_split(X, y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    classes = sorted(set(y_train.unique()) | set(y_test.unique()))

    report = classification_report(y_test, predicted, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predicted, labels=classes)
    metrics = {
        "project": "LogLens",
        "model": "RandomForestClassifier",
        "target": "FORCE 2020 interpreted lithology code",
        "features": features,
        "n_rows": int(len(work)),
        "n_wells": int(groups.nunique()),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_wells": sorted(groups.iloc[train_idx].unique().tolist()),
        "test_wells": sorted(groups.iloc[test_idx].unique().tolist()),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predicted)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_test, predicted, average="weighted", zero_division=0)), 4),
        "classes_in_train": sorted(int(v) for v in y_train.unique()),
        "classes_in_test": sorted(int(v) for v in y_test.unique()),
        "validation": "well-wise group holdout; rows from a held-out well are not used for training",
        "data_source": "FORCE 2020 public well-log/lithofacies dataset; cite original Zenodo record",
        "limitations": [
            "Performance depends on curve availability and the held-out wells.",
            "Lithology codes are interpreted labels, not direct measurements.",
            "Generalization to other basins requires external validation.",
        ],
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "classes": classes}, model_dir / "loglens_model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(model_dir / "confusion_matrix.csv")
    pd.DataFrame(report).T.to_csv(model_dir / "classification_report.csv")

    # Permutation importance is evaluated on held-out wells, not training rows.
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model, X_test, y_test, n_repeats=5,
                                  random_state=42, scoring="f1_macro", n_jobs=-1)
    importance = pd.DataFrame({
        "feature": features,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    importance.to_csv(model_dir / "permutation_importance.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LogLens on prepared FORCE 2020 data")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "force2020_prepared.csv")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found. First prepare the FORCE LAS/CSV dataset; see README.md."
        )
    frame = pd.read_csv(args.data)
    result = train(frame, args.model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
