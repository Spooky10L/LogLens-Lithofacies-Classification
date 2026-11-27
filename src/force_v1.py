"""LogLens v1.0 - first reproducible baseline on the real FORCE 2020 LAS data.

Dataset: FORCE 2020 well log and lithofacies dataset (PUBLIC BENCHMARK),
https://doi.org/10.5281/zenodo.4351156 - see README for attribution/licence.

Design (see docs/LOGLENS_V1_ENGINEERING_NOTES.md):
* features: GR, RHOB, NPHI, RDEP, DTC, CALI only (no depth, no location, no confidence)
* rows without a valid lithology label are dropped (derived copy only)
* 80/20 WELL-WISE split (GroupShuffleSplit, seed 42): no well in both partitions
* models: DummyClassifier baselines vs RandomForest(class_weight='balanced_subsample')
* training set capped at ~300k rows by documented stratified sampling per (well, class);
  evaluation uses ALL labelled rows of the held-out wells
* the original LAS files are only ever opened read-only

Run from LogLens_AI/:
    python -m src.force_v1 --data data/force_las --output models/v1
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
NULL_VALUE = -999.25
SEED = 42
FEATURES = ["GR", "RHOB", "NPHI", "RDEP", "DTC", "CALI"]
LABEL = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONFIDENCE = "FORCE_2020_LITHOFACIES_CONFIDENCE"
# Never used as model inputs in v1 (kept as columns for grouping / later experiments).
EXCLUDED_FROM_FEATURES = ["WELL", "DEPTH_MD", "x_loc", "y_loc", "z_loc", CONFIDENCE]
MAX_TRAIN_SAMPLES = 300_000
# A class present in this many wells or fewer (over all 118) is flagged as "rare".
RARE_CLASS_WELL_THRESHOLD = 5

# Official mapping: FORCE 2020 starter notebook, cell "lithology_keys"
# https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
#   (lithology_competition/data/starter_notebook.ipynb)
LITHOLOGY_NAMES = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}

# Physically impossible (<= 0) values for these curves are treated as invalid.
POSITIVE_ONLY = ["RHOB", "RDEP", "DTC", "CALI"]


# ----------------------------------------------------------------------------
# Phase 1 - loading and cleaning (no scikit-learn needed)
# ----------------------------------------------------------------------------
def clean_values(df: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """-999.25, +/-inf and physically impossible values -> NaN (works on a copy)."""
    out = df.copy()
    cols = [c for c in FEATURES + [LABEL, CONFIDENCE] if c in out.columns]
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[cols] = out[cols].replace(NULL_VALUE, np.nan)
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    for c in POSITIVE_ONLY:
        if c in out.columns:
            bad = out[c] <= 0
            if report is not None:
                report[f"{c}_nonpositive_set_to_nan"] = int(bad.sum())
            out.loc[bad, c] = np.nan
    return out


def load_force_las(folder: Path, limit_wells: int | None = None) -> pd.DataFrame:
    """Read LAS files (read-only) into one DataFrame with WELL, DEPTH + curves."""
    from .audit_force_las import parse_las  # same reader used for the audit

    files = sorted(set(folder.rglob("*.las")) | set(folder.rglob("*.LAS")))
    if limit_wells:
        files = files[:limit_wells]
    frames = []
    for f in files:
        _, _, df, null_val = parse_las(f)
        df = df.replace(null_val, np.nan)
        keep = ["DEPT", LABEL, CONFIDENCE] + [c for c in FEATURES if c in df.columns]
        sub = df[[c for c in keep if c in df.columns]].copy()
        for c in FEATURES:  # a missing curve is an all-NaN column (imputer handles it)
            if c not in sub.columns:
                sub[c] = np.nan
        sub.insert(0, "WELL", f.stem)  # well id = filename stem (original name kept)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def build_labelled_table(raw: pd.DataFrame, report: dict) -> pd.DataFrame:
    report["rows_total"] = int(len(raw))
    df = clean_values(raw, report)
    df = df[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(int)
    unknown = sorted(set(df[LABEL]) - set(LITHOLOGY_NAMES))
    if unknown:
        raise ValueError(f"Unmapped lithology codes: {unknown}")
    df["LITHOLOGY_NAME"] = df[LABEL].map(LITHOLOGY_NAMES)
    report["rows_labelled"] = int(len(df))
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Phase 2 - well-wise split and class coverage
# ----------------------------------------------------------------------------
def well_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["WELL"]))
    train_wells = sorted(df.iloc[tr_idx]["WELL"].unique())
    val_wells = sorted(df.iloc[va_idx]["WELL"].unique())
    assert not set(train_wells) & set(val_wells), "well leakage between partitions!"
    return tr_idx, va_idx, train_wells, val_wells


def class_coverage(df: pd.DataFrame, tr_idx, va_idx) -> pd.DataFrame:
    rows = []
    wells_per_class = df.groupby(LABEL)["WELL"].nunique()
    for code, name in LITHOLOGY_NAMES.items():
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]
        n_tr, n_va = int((tr[LABEL] == code).sum()), int((va[LABEL] == code).sum())
        w_all = int(wells_per_class.get(code, 0))
        flags = []
        if w_all <= RARE_CLASS_WELL_THRESHOLD:
            flags.append(f"RARE: in only {w_all} of {df['WELL'].nunique()} labelled wells")
        if n_tr == 0:
            flags.append("ABSENT FROM TRAIN: model cannot learn it")
        if n_va == 0:
            flags.append("ABSENT FROM VALIDATION: cannot be scored")
        rows.append({
            "class_code": code, "class_name": name,
            "train_samples": n_tr, "val_samples": n_va,
            "train_wells": int(tr.loc[tr[LABEL] == code, "WELL"].nunique()),
            "val_wells": int(va.loc[va[LABEL] == code, "WELL"].nunique()),
            "wells_total": w_all, "flags": "; ".join(flags),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Phase 3 - training-set sampling
# ----------------------------------------------------------------------------
def stratified_sample(train: pd.DataFrame, max_samples: int = MAX_TRAIN_SAMPLES,
                      seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Proportional sampling inside every (well, class) group.

    Every group keeps at least 1 row, so no class/well combination disappears.
    Rows are drawn at random within a group (adjacent depth samples are
    correlated, so this is a documented simplification of the full data).
    """
    n = len(train)
    info = {"n_before": n, "max_samples": max_samples, "sampled": False}
    if n <= max_samples:
        info["n_after"] = n
        return train, info
    frac = max_samples / n
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in train.groupby(["WELL", LABEL], sort=True):
        k = min(len(g), max(1, int(round(len(g) * frac))))
        parts.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    out = pd.concat(parts).sort_index()
    info.update(sampled=True, fraction=frac, n_after=int(len(out)))
    return out, info


def make_pipeline(model):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", model),
    ])


# ----------------------------------------------------------------------------
# Phase 4 - evaluation and plots
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, present_labels) -> tuple[dict, pd.DataFrame]:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_recall_fscore_support)

    # Headline macro metrics are over classes PRESENT in the validation labels.
    # A class that is absent from validation cannot be scored and would only add a 0.
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present_labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present_labels,
                                      average="weighted", zero_division=0)),
        "n_classes_scored": len(present_labels),
    }
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LITHOLOGY_NAMES), zero_division=0)
    n_pred = pd.Series(y_pred).value_counts().reindex(list(LITHOLOGY_NAMES)).fillna(0).astype(int).to_numpy()
    per = pd.DataFrame({"class_code": list(LITHOLOGY_NAMES),
                        "class_name": list(LITHOLOGY_NAMES.values()),
                        "precision": p, "recall": r, "f1": f, "support_val": s,
                        "n_predicted": n_pred})
    # A class with no validation samples has NO defined recall/F1 (not "0.0").
    # Report it as NaN so it is never mistaken for a measured score.
    absent = per["support_val"] == 0
    per.loc[absent, ["recall", "f1"]] = np.nan
    per.loc[absent & (per["n_predicted"] == 0), "precision"] = np.nan
    per["status"] = np.where(absent, "NOT SCORABLE: absent from validation wells", "scored")
    return m, per


def make_plots(out: Path, y_true, y_pred, rf_pipe, val: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    codes = list(LITHOLOGY_NAMES)
    names = [LITHOLOGY_NAMES[c] for c in codes]
    cm = confusion_matrix(y_true, y_pred, labels=codes)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out / "confusion_matrix.csv")
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = np.where(cm.sum(1, keepdims=True) > 0, cm / cm.sum(1, keepdims=True), np.nan)

    for arr, title, fname, fmt in [(cm, "Confusion matrix (counts), held-out wells", "confusion_matrix.png", "d"),
                                   (cmn, "Row-normalised confusion matrix (recall per true class)",
                                    "confusion_matrix_normalized.png", ".2f")]:
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True (FORCE interpretation)")
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                v = arr[i, j]
                if np.isfinite(v) and v > 0:
                    ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6,
                            color="white" if v > np.nanmax(arr) / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)

    # Feature importance (impurity-based; biased towards continuous features - see notes)
    imputer, rf = rf_pipe.named_steps["impute"], rf_pipe.named_steps["model"]
    fn = list(imputer.get_feature_names_out(FEATURES))
    imp = pd.Series(rf.feature_importances_, index=fn).sort_values()
    imp.sort_values(ascending=False).to_csv(out / "feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.plot.barh(ax=ax, color="#1f77b4")
    ax.set_xlabel("Mean decrease in impurity (RandomForest)")
    ax.set_title("Feature importance (indicator columns = 'missingindicator_*')")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=150); plt.close(fig)

    # Actual vs predicted: class totals
    tot = pd.DataFrame({"actual": pd.Series(y_true).value_counts(),
                        "predicted": pd.Series(y_pred).value_counts()}).reindex(codes).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(codes))
    ax.bar(x - 0.2, tot["actual"], 0.4, label="actual"); ax.bar(x + 0.2, tot["predicted"], 0.4, label="RF predicted")
    ax.set_yscale("log"); ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("samples (log scale)"); ax.legend()
    ax.set_title("Actual vs predicted class totals, held-out wells")
    fig.tight_layout(); fig.savefig(out / "actual_vs_predicted_class_totals.png", dpi=150); plt.close(fig)

    # Actual vs predicted: depth tracks for up to 4 held-out wells (deterministic: largest label counts)
    wells = val["WELL"].value_counts().index[:4]
    cmap = plt.get_cmap("tab20", len(codes))
    code_idx = {c: i for i, c in enumerate(codes)}
    fig, axes = plt.subplots(1, 2 * len(wells), figsize=(2.2 * len(wells) * 2, 8), sharey=False)
    for k, w in enumerate(wells):
        g = val[val["WELL"] == w].sort_values("DEPT")
        for j, col in enumerate(["y_true", "y_rf"]):
            ax = axes[2 * k + j]
            ax.imshow(g[col].map(code_idx).to_numpy()[:, None], aspect="auto", cmap=cmap,
                      vmin=0, vmax=len(codes) - 1, interpolation="nearest",
                      extent=[0, 1, g["DEPT"].max(), g["DEPT"].min()])
            ax.set_xticks([]); ax.set_title(f"{w}\n{'actual' if j == 0 else 'RF pred'}", fontsize=8)
            if j == 1:
                ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(codes))]
    fig.legend(handles, names, loc="lower center", ncol=6, fontsize=7)
    fig.suptitle("Actual vs predicted lithofacies (labelled intervals only; depth in m)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97]); fig.savefig(out / "actual_vs_predicted_depth_tracks.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/force_las"))
    ap.add_argument("--output", type=Path, default=Path("models/v1"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--max-wells", type=int, default=None, help="smoke-test on the first N files only")
    ap.add_argument("--max-train", type=int, default=MAX_TRAIN_SAMPLES)
    args = ap.parse_args(argv)

    import joblib
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier

    args.output.mkdir(parents=True, exist_ok=True)
    args.processed.mkdir(parents=True, exist_ok=True)

    # Phase 1
    report: dict = {}
    raw = load_force_las(args.data, args.max_wells)
    df = build_labelled_table(raw, report)
    del raw
    df[["WELL", "DEPT", *FEATURES, LABEL, "LITHOLOGY_NAME", CONFIDENCE]].astype(
        {c: "float32" for c in FEATURES + ["DEPT", CONFIDENCE]}
    ).to_csv(args.processed / "force_v1_labelled.csv.gz", index=False)  # derived copy; keeps confidence
    print(f"labelled rows: {len(df):,} from {df['WELL'].nunique()} wells")

    # Phase 2
    tr_idx, va_idx, train_wells, val_wells = well_split(df)
    (args.output / "train_wells.txt").write_text("\n".join(train_wells) + "\n")
    (args.output / "val_wells.txt").write_text("\n".join(val_wells) + "\n")
    cov = class_coverage(df, tr_idx, va_idx)
    cov.to_csv(args.output / "class_coverage.csv", index=False)
    print(cov.to_string(index=False))

    train_full, val = df.iloc[tr_idx], df.iloc[va_idx].copy()
    train, samp = stratified_sample(train_full, args.max_train)
    print(f"train rows {len(train_full):,} -> {len(train):,} used; validation rows {len(val):,} (all used)")
    present = sorted(val[LABEL].unique())

    # Phase 3
    X_tr, y_tr = train[FEATURES], train[LABEL].to_numpy()
    X_va, y_va = val[FEATURES], val[LABEL].to_numpy()
    rf_cfg = dict(n_estimators=150, max_depth=24, min_samples_leaf=3,
                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    models = {
        "dummy_most_frequent": make_pipeline(DummyClassifier(strategy="most_frequent")),
        "dummy_stratified": make_pipeline(DummyClassifier(strategy="stratified", random_state=SEED)),
        "random_forest": make_pipeline(RandomForestClassifier(**rf_cfg)),
    }
    metrics, per_class = {}, {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        val[f"pred_{name}"] = pred
        metrics[name], per_class[name] = evaluate(y_va, pred, present)
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics[name].items()})

    # Phase 4
    val["y_true"], val["y_rf"] = y_va, val["pred_random_forest"]
    val["pred_rf_name"] = val["y_rf"].map(LITHOLOGY_NAMES)
    val[["WELL", "DEPT", *FEATURES, LABEL, "LITHOLOGY_NAME", "pred_dummy_most_frequent",
         "pred_dummy_stratified", "pred_random_forest", "pred_rf_name", CONFIDENCE]].to_csv(
        args.output / "val_predictions.csv.gz", index=False)
    pc = per_class["random_forest"].merge(
        per_class["dummy_most_frequent"][["class_code", "f1"]].rename(columns={"f1": "f1_dummy_most_frequent"}),
        on="class_code")
    pc.to_csv(args.output / "per_class_metrics_random_forest.csv", index=False)
    for n_, d_ in per_class.items():
        d_.to_csv(args.output / f"per_class_metrics_{n_}.csv", index=False)
    make_plots(args.output, y_va, val["y_rf"].to_numpy(), models["random_forest"], val)

    out = {
        "dataset": "FORCE 2020 well log and lithofacies dataset - PUBLIC BENCHMARK (Zenodo 4351156)",
        "claim_scope": "Own 80/20 well-wise validation split of the 118 labelled wells. "
                       "NOT the official FORCE 2020 competition test/hidden-test scoring; "
                       "not comparable to competition leaderboard scores.",
        "seed": SEED, "features": FEATURES, "excluded_from_features": EXCLUDED_FROM_FEATURES,
        "target": LABEL, "class_names": {str(k): v for k, v in LITHOLOGY_NAMES.items()},
        "preprocessing": {"null_value": NULL_VALUE, "positive_only_curves": POSITIVE_ONLY,
                          "imputer": "SimpleImputer(median, add_indicator=True)", **report},
        "split": {"method": "GroupShuffleSplit(test_size=0.2, random_state=42), group=WELL",
                  "n_train_wells": len(train_wells), "n_val_wells": len(val_wells),
                  "train_wells": train_wells, "val_wells": val_wells,
                  "train_rows_full": int(len(train_full)), "val_rows": int(len(val)),
                  "classes_scored_in_validation": [int(c) for c in present]},
        "train_sampling": samp,
        "random_forest_config": rf_cfg,
        "metrics_validation": metrics,
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2))
    joblib.dump(models["random_forest"], args.output / "random_forest_pipeline.joblib")
    print("saved to", args.output)


if __name__ == "__main__":
    main()
