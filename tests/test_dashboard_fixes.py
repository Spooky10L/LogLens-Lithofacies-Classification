"""Regression tests for the dashboard fixes (KeyError 'well', blank well tracks, confusion matrix)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import dashboard_data as dd
from src import dashboard_plots as dp
from src.audit_results import headline, per_class_table
from src.force_v1 import FEATURES, LABEL, LITHOLOGY_NAMES

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "models" / "v1"
needs_real = pytest.mark.skipif(not (REAL / "val_predictions.csv.gz").exists(),
                                reason="trained results not present")


def _val(n=40, wells=("A", "B"), seed=1):
    rng = np.random.RandomState(seed)
    rows = []
    for w in wells:
        d = pd.DataFrame({c: rng.rand(n) + 1 for c in FEATURES})
        d["WELL"], d["DEPT"] = w, 500 + np.arange(n) * 0.152
        d[LABEL] = rng.choice([30000, 65000, 65030], n)
        d["pred_random_forest"] = np.where(rng.rand(n) < 0.7, d[LABEL], 65000)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ---- Task 1: KeyError 'well' ------------------------------------------------
def test_per_well_table_has_lowercase_well_column_whatever_the_csv_says():
    v = _val()
    pw = dd.per_well_table(v)
    assert list(pw.columns) == ["well", "n_labelled", "accuracy"]  # the app indexes pw["well"]
    assert set(pw["well"]) == {"A", "B"} and (pw["n_labelled"] == 40).all()
    expected = (v[LABEL] == v["pred_random_forest"])[v.WELL == "A"].mean()
    assert pw.set_index("well").loc["A", "accuracy"] == pytest.approx(expected)


def test_root_cause_audit_csv_uses_uppercase_WELL():
    """Documents the original bug: src.audit_results wrote 'WELL'; app.py asked for 'well'."""
    v = _val()
    v["_ok"] = v[LABEL] == v["pred_random_forest"]
    saved = v.groupby("WELL").agg(n_labelled=("_ok", "size"), accuracy=("_ok", "mean")).reset_index()
    assert "WELL" in saved.columns and "well" not in saved.columns
    with pytest.raises(KeyError):
        saved["well"]
    assert "well" in dd.per_well_table(v).columns  # the loader no longer depends on that spelling


def test_schema_errors_are_clear_not_keyerrors():
    with pytest.raises(dd.SchemaError, match="missing columns"):
        dd.require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "demo")


@needs_real
def test_real_results_load_and_per_well_matches_saved_csv():
    res = dd.load_results(REAL)
    pw = res["per_well"]
    assert list(pw.columns) == ["well", "n_labelled", "accuracy"] and len(pw) == 24
    saved = pd.read_csv(REAL / "per_well_accuracy_random_forest.csv")
    merged = pw.merge(saved, left_on="well", right_on="WELL", suffixes=("", "_saved"))
    assert len(merged) == 24
    assert np.allclose(merged["accuracy"], merged["accuracy_saved"])
    assert (merged["n_labelled"] == merged["n_labelled_saved"]).all()


# ---- Task 4: scientific integrity (saved results unchanged) -----------------
@needs_real
def test_headline_metrics_unchanged():
    v = pd.read_csv(REAL / "val_predictions.csv.gz")
    y, p = v[LABEL].to_numpy(), v["pred_random_forest"].to_numpy()
    h = headline(y, p, per_class_table(y, p))
    assert round(h["accuracy"], 4) == 0.6786
    assert round(h["balanced_accuracy"], 4) == 0.4346
    assert round(h["macro_f1"], 4) == 0.4220
    assert round(h["weighted_f1"], 4) == 0.6924
    assert h["n_classes_scored"] == 11


# ---- Task 2: well viewer chart data ----------------------------------------
def _regularised(n=60):
    v = _val(n, wells=("A",))
    v.loc[5:9, "GR"] = np.nan  # a genuine gap must stay a gap, not erase the curve
    res = {"val": v, "val_wells": ["A"], "train_wells": []}
    raw = dd.well_frame(res, None, "A")
    return raw, dd.regularise_depth(raw)


def test_well_figure_curves_are_svg_and_carry_all_the_data():
    raw, g = _regularised()
    fig = dp.well_figure(g)
    curves = {t.name: t for t in fig.data if t.name in ("GR", "RHOB", "NPHI", "RDEP")}
    assert set(curves) == {"GR", "RHOB", "NPHI", "RDEP"}
    for name, t in curves.items():
        assert t.type == "scatter", "WebGL scattergl rendered blank in the live app and in PDF export"
        assert len(t.x) == len(t.y) == len(g)
        assert np.isfinite(np.asarray(t.x, dtype=float)).sum() == raw[name].notna().sum()
    # depth and curve arrays are aligned: every raw sample sits at its own depth
    gr = curves["GR"]
    pos = dict(zip(np.round(np.asarray(gr.y), 2), np.asarray(gr.x, dtype=float)))
    for d, val in zip(raw["DEPT"].round(2), raw["GR"]):
        assert (np.isnan(val) and np.isnan(pos[d])) or pos[d] == pytest.approx(val)


def test_well_figure_lithology_strips_match_the_labels():
    raw, g = _regularised()
    fig = dp.well_figure(g)
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert len(heat) == 3  # actual, predicted, match
    n = raw[LABEL].notna().sum()
    assert np.isfinite(np.asarray(heat[0].z, dtype=float)[:, 0]).sum() == n
    assert np.isfinite(np.asarray(heat[1].z, dtype=float)[:, 0]).sum() == n
    codes = list(LITHOLOGY_NAMES)
    z0 = np.asarray(heat[0].z, dtype=float)[:, 0]
    back = [codes[int(z)] for z in z0[np.isfinite(z0)]]  # class index recovered from the strip
    assert back == list(g[LABEL].dropna().astype(int))
    match = np.asarray(heat[2].z, dtype=float)[:, 0]
    assert np.nansum(match) == (raw[LABEL] == raw["pred_code"]).sum()


def test_all_tracks_share_one_fixed_depth_axis():
    _, g = _regularised()
    fig = dp.well_figure(g)
    ranges = {tuple(fig.layout[k].range) for k in fig.layout if k.startswith("yaxis")}
    assert len(ranges) == 1
    top, bottom = ranges.pop()
    assert top > bottom  # reversed: depth increases downwards
    assert top == pytest.approx(g["DEPT"].max()) and bottom == pytest.approx(g["DEPT"].min())


def test_depth_window_restricts_every_track():
    _, g = _regularised(100)
    lo, hi = g["DEPT"].iloc[20], g["DEPT"].iloc[40]
    fig = dp.well_figure(g, (lo, hi))
    for t in fig.data:
        if t.name in ("GR", "RHOB"):
            assert np.asarray(t.y).min() >= lo - 1e-9 and np.asarray(t.y).max() <= hi + 1e-9


def test_missing_predictions_do_not_break_the_figure():
    raw, _ = _regularised()
    raw = raw.assign(pred_code=np.nan)
    fig = dp.well_figure(dd.regularise_depth(raw))
    assert len([t for t in fig.data if t.type == "heatmap"]) == 3


@needs_real
def test_real_well_15_9_13_has_visible_curves_and_strips():
    res = dd.load_results(REAL)
    raw = dd.well_frame(res, None, "15_9-13")
    assert len(raw) == 18270 and round(raw["DEPT"].min()) == 495 and round(raw["DEPT"].max()) == 3272
    fig = dp.well_figure(dd.regularise_depth(raw))
    for t in fig.data:
        if t.name in ("GR", "RHOB", "RDEP"):
            assert np.isfinite(np.asarray(t.x, dtype=float)).sum() == 18270
        if t.name == "NPHI":
            assert np.isfinite(np.asarray(t.x, dtype=float)).sum() == raw["NPHI"].notna().sum() > 10000
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert np.isfinite(np.asarray(heat[0].z, dtype=float)[:, 0]).sum() == 18270
    assert np.isfinite(np.asarray(heat[1].z, dtype=float)[:, 0]).sum() == 18270


# ---- confusion matrix -------------------------------------------------------
def test_confusion_figure_shows_all_present_classes_and_reports_dropped_ones():
    names = list(LITHOLOGY_NAMES.values())
    cm = pd.DataFrame(0, index=names, columns=names)
    for i, n in enumerate(names[:-1]):  # last class (Basement) has no samples and no predictions
        cm.loc[n, n] = 10 + i
    fig, dropped = dp.confusion_figure(cm, normalise=True)
    assert dropped == ["Basement"]
    assert len(fig.data[0].x) == len(fig.data[0].y) == 11
    fig2, _ = dp.confusion_figure(cm, normalise=False)
    assert np.nansum(np.asarray(fig2.data[0].z, dtype=float)) == cm.to_numpy().sum()


def test_class_that_is_predicted_but_absent_from_truth_is_not_dropped():
    names = list(LITHOLOGY_NAMES.values())
    cm = pd.DataFrame(0, index=names, columns=names)
    cm.loc["Shale", "Shale"] = 5
    cm.loc["Shale", "Basement"] = 1  # predicted Basement even though no true Basement
    _, dropped = dp.confusion_figure(cm)
    assert "Basement" not in dropped


# ---- the whole app ------------------------------------------------------------
@needs_real
def test_app_runs_all_six_tabs_without_exceptions():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6
    assert [t.label for t in at.tabs] == ["Overview", "Well viewer", "Model performance", "Per-class",
                                          "Feature importance", "Limitations"]
