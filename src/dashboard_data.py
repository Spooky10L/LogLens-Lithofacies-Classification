"""Data-loading helpers for the LogLens dashboard (pandas only - no Streamlit/plotly).

Kept separate from app.py so it can be unit-tested without a browser.
Everything here only READS files that the training run already saved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_results import headline, per_class_table
from .force_v1 import CONFIDENCE, FEATURES, LABEL, LITHOLOGY_NAMES

PRED_COL = "pred_random_forest"

# One fixed colour per lithology so actual/predicted strips always match.
LITHOLOGY_COLOURS = {
    "Sandstone": "#f2d16b", "Sandstone/Shale": "#c9a86a", "Shale": "#6f7f6a",
    "Marl": "#8fb8a8", "Dolomite": "#b48ad0", "Limestone": "#6fa8dc",
    "Chalk": "#e8eef2", "Halite": "#f4a6c0", "Anhydrite": "#d46a9c",
    "Tuff": "#e07b39", "Coal": "#222222", "Basement": "#a33a3a",
}


class ResultsNotFound(FileNotFoundError):
    pass


class SchemaError(ValueError):
    """A saved results table does not have the columns the dashboard needs."""


# Columns each saved table must contain (checked at load time, so a schema
# mismatch fails once, with a clear message, instead of as a KeyError mid-render).
REQUIRED_COLUMNS = {
    "val_predictions": ["WELL", "DEPT", LABEL, PRED_COL, *FEATURES],
    "per_class": ["class_code", "class_name", "precision", "recall", "f1", "support_val"],
    "coverage": ["class_name", "train_samples", "val_samples", "train_wells", "val_wells", "wells_total"],
}


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}; found {list(df.columns)}")
    return df


def per_well_table(val: pd.DataFrame, pred_col: str = PRED_COL) -> pd.DataFrame:
    """Accuracy per held-out well, computed directly from the saved predictions.

    Columns are always lower-case: well, n_labelled, accuracy. (The CSV written by
    src.audit_results uses 'WELL'; computing here removes any dependence on that spelling.)
    """
    require_columns(val, ["WELL", LABEL, pred_col], "validation predictions")
    ok = (val[LABEL] == val[pred_col]).astype(float)
    out = (ok.groupby(val["WELL"]).agg(["size", "mean"])
           .rename(columns={"size": "n_labelled", "mean": "accuracy"}).reset_index()
           .rename(columns={"WELL": "well"}))
    return out


def _read_lines(path: Path) -> list[str]:
    return [w for w in path.read_text().splitlines() if w.strip()]


def load_results(model_dir: Path) -> dict:
    """Load every small artefact saved by src.force_v1 / src.audit_results."""
    model_dir = Path(model_dir)
    needed = ["metrics.json", "val_predictions.csv.gz", "train_wells.txt", "val_wells.txt"]
    missing = [n for n in needed if not (model_dir / n).exists()]
    if missing:
        raise ResultsNotFound(f"Missing in {model_dir}: {', '.join(missing)}. "
                              "Run `python -m src.force_v1` first.")
    metrics = json.loads((model_dir / "metrics.json").read_text())
    res = {
        "metrics": metrics,
        "val": require_columns(pd.read_csv(model_dir / "val_predictions.csv.gz"),
                               REQUIRED_COLUMNS["val_predictions"], "val_predictions.csv.gz"),
        "train_wells": _read_lines(model_dir / "train_wells.txt"),
        "val_wells": _read_lines(model_dir / "val_wells.txt"),
    }
    optional = {"confusion": "confusion_matrix.csv", "importance": "feature_importance.csv",
                "coverage": "class_coverage.csv"}
    for key, fname in optional.items():
        p = model_dir / fname
        if p.exists():
            res[key] = pd.read_csv(p, index_col=0) if key in ("confusion", "importance") else pd.read_csv(p)
    if "coverage" in res:
        require_columns(res["coverage"], REQUIRED_COLUMNS["coverage"], "class_coverage.csv")
    if "importance" in res:
        require_columns(res["importance"], ["importance"], "feature_importance.csv")
    res["per_well"] = per_well_table(res["val"])  # always derived from the saved predictions
    audited = model_dir / "per_class_metrics_audited.csv"
    if audited.exists():
        res["per_class"] = pd.read_csv(audited)
    else:  # older runs: recompute from the saved predictions (same maths as the audit)
        v = res["val"]
        res["per_class"] = per_class_table(v[LABEL].to_numpy(), v[PRED_COL].to_numpy())
    require_columns(res["per_class"], REQUIRED_COLUMNS["per_class"], "per-class metrics")
    audit_json = model_dir / "results_audit.json"
    if audit_json.exists():
        res["audit"] = json.loads(audit_json.read_text())
    return res


def load_dataset_stats(audit_dir: Path) -> dict:
    """Optional dataset statistics from the read-only LAS audit (src.audit_force_las)."""
    audit_dir = Path(audit_dir)
    out = {}
    for key, fname in {"manifest": "well_manifest.csv", "classes": "class_distribution.csv",
                       "curves": "curve_inventory.csv"}.items():
        p = audit_dir / fname
        if p.exists():
            out[key] = pd.read_csv(p)
    return out


def load_labelled_all(processed_csv: Path) -> pd.DataFrame | None:
    """Optional: all labelled rows of all 118 wells (derived copy made by src.force_v1)."""
    p = Path(processed_csv)
    return pd.read_csv(p) if p.exists() else None


def well_catalog(res: dict, all_df: pd.DataFrame | None) -> pd.DataFrame:
    """One row per selectable well, flagged held-out vs training."""
    val = set(res["val_wells"])
    wells = sorted(val | (set(res["train_wells"]) if all_df is not None else set()))
    return pd.DataFrame({"well": wells,
                         "split": ["validation (held-out)" if w in val else "training (in-sample)"
                                   for w in wells]})


def well_frame(res: dict, all_df: pd.DataFrame | None, well: str) -> pd.DataFrame:
    """Rows for one well, sorted by depth. Validation wells carry the saved predictions."""
    if well in set(res["val_wells"]):
        df = res["val"][res["val"]["WELL"] == well].copy()
        df["pred_code"] = df[PRED_COL]
    elif all_df is not None and well in set(all_df["WELL"]):
        df = all_df[all_df["WELL"] == well].copy()
        df["pred_code"] = np.nan  # no honest prediction unless the user runs the model
    else:
        raise KeyError(f"Well {well!r} not available")
    df["pred_name"] = df["pred_code"].map(LITHOLOGY_NAMES)
    return df.sort_values("DEPT").reset_index(drop=True)


def metrics_for_subset(df: pd.DataFrame, pred_col: str = "pred_code") -> tuple[dict, pd.DataFrame]:
    """Headline metrics + per-class table for any set of rows with actual labels and predictions."""
    d = df.dropna(subset=[pred_col])
    y, p = d[LABEL].to_numpy().astype(int), d[pred_col].to_numpy().astype(int)
    tab = per_class_table(y, p)
    return headline(y, p, tab), tab


def regularise_depth(df: pd.DataFrame, step: float = 0.152) -> pd.DataFrame:
    """Put a well on a regular depth grid; unlabelled/missing depths become NaN gaps.

    FORCE wells are sampled every 0.152 m (audit result). Without this, a plot
    would stretch the last labelled sample across an unlabelled interval.
    """
    if df.empty:
        return df.copy()
    d0 = float(df["DEPT"].min())
    idx = np.rint((df["DEPT"].to_numpy() - d0) / step).astype(int)
    n = int(idx.max()) + 1
    out = pd.DataFrame(np.nan, index=range(n), columns=[c for c in df.columns
                                                       if pd.api.types.is_numeric_dtype(df[c])])
    num = out.columns
    out.loc[idx, num] = df[num].to_numpy()
    out["DEPT"] = d0 + np.arange(n) * step
    return out


def lithology_index(codes: pd.Series) -> pd.Series:
    order = {c: i for i, c in enumerate(LITHOLOGY_NAMES)}
    return codes.map(order)


def predict_with_model(model_path: Path, df: pd.DataFrame) -> np.ndarray:
    """Optional: apply the saved sklearn pipeline (needs scikit-learn; joblib file is large)."""
    import joblib  # imported lazily so the dashboard still opens without sklearn

    pipe = joblib.load(model_path)
    return pipe.predict(df[FEATURES])
