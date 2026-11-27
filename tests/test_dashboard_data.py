"""Tests for the dashboard data layer and the results audit (no Streamlit needed)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import dashboard_data as dd
from src.audit_results import headline, per_class_table
from src.force_v1 import FEATURES, LABEL


def test_absent_class_is_nan_not_zero():
    y = np.array([65000, 65000, 30000, 30000])
    p = np.array([65000, 30000, 30000, 93000])  # 93000 predicted but never true
    tab = per_class_table(y, p)
    b = tab[tab.class_code == 93000].iloc[0]
    assert b.support_val == 0 and np.isnan(b.recall) and np.isnan(b.f1)
    assert b.precision == 0.0  # predicted, never correct: precision IS defined
    assert "NOT SCORABLE" in b.status
    h = headline(y, p, tab)
    assert h["n_classes_scored"] == 2  # only classes present in y_true


def test_headline_matches_hand_calculation():
    y = np.array([65000] * 3 + [30000])
    p = np.array([65000, 65000, 65000, 65000])  # always-Shale on a 3:1 set
    h = headline(y, p, per_class_table(y, p))
    assert h["accuracy"] == pytest.approx(0.75)
    assert h["balanced_accuracy"] == pytest.approx(0.5)


def test_regularise_depth_inserts_gaps():
    df = pd.DataFrame({"DEPT": [0.0, 0.152, 0.152 * 5], "GR": [1.0, 2.0, 3.0]})
    g = dd.regularise_depth(df)
    assert len(g) == 6
    assert g["GR"].isna().sum() == 3  # three missing depths between samples 2 and 3


def _write_results(tmp_path):
    rng = np.random.RandomState(0)
    v = pd.DataFrame({"WELL": ["A"] * 6 + ["B"] * 6, "DEPT": list(np.arange(6) * 0.152) * 2,
                      LABEL: [65000, 65000, 30000, 30000, 65000, 65000] * 2})
    for c in FEATURES:
        v[c] = rng.rand(12)
    v["pred_random_forest"] = v[LABEL]
    v.to_csv(tmp_path / "val_predictions.csv.gz", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"metrics_validation": {}}))
    (tmp_path / "train_wells.txt").write_text("T1\nT2\n")
    (tmp_path / "val_wells.txt").write_text("A\nB\n")


def test_load_results_and_well_frame(tmp_path):
    _write_results(tmp_path)
    res = dd.load_results(tmp_path)
    assert set(res["val_wells"]) == {"A", "B"} and not set(res["val_wells"]) & set(res["train_wells"])
    cat = dd.well_catalog(res, None)
    assert list(cat.well) == ["A", "B"]
    f = dd.well_frame(res, None, "A")
    assert f["pred_name"].notna().all()
    m, _ = dd.metrics_for_subset(f)
    assert m["accuracy"] == 1.0
    with pytest.raises(KeyError):
        dd.well_frame(res, None, "T1")  # training well without the processed file


def test_missing_results_raise_clear_error(tmp_path):
    with pytest.raises(dd.ResultsNotFound):
        dd.load_results(tmp_path)
