"""Tests for LogLens v1 (synthetic mini-tables only; the real dataset is not needed)."""
import numpy as np
import pandas as pd
import pytest

from src import force_v1 as v1


def _toy(n_wells=10, n=60, seed=0):
    rng = np.random.RandomState(seed)
    codes = list(v1.LITHOLOGY_NAMES)[:3]
    rows = []
    for w in range(n_wells):
        d = pd.DataFrame({c: rng.rand(n) + 1 for c in v1.FEATURES})
        d["WELL"] = f"W{w}"
        d["DEPT"] = np.arange(n) * 0.152
        d[v1.LABEL] = rng.choice(codes, n).astype(float)  # LAS data is float
        d[v1.CONFIDENCE] = 1.0
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_mapping_has_the_12_official_classes():
    assert len(v1.LITHOLOGY_NAMES) == 12
    assert v1.LITHOLOGY_NAMES[65000] == "Shale"
    assert v1.LITHOLOGY_NAMES[30000] == "Sandstone"


def test_forbidden_columns_not_in_features():
    for bad in v1.EXCLUDED_FROM_FEATURES:
        assert bad not in v1.FEATURES
    assert v1.FEATURES == ["GR", "RHOB", "NPHI", "RDEP", "DTC", "CALI"]


def test_clean_values_null_inf_and_nonpositive():
    df = _toy(1, 5)
    df.loc[0, "GR"] = -999.25
    df.loc[1, "NPHI"] = np.inf
    df.loc[2, "RDEP"] = 0.0
    rep = {}
    out = v1.clean_values(df, rep)
    assert np.isnan(out.loc[0, "GR"]) and np.isnan(out.loc[1, "NPHI"]) and np.isnan(out.loc[2, "RDEP"])
    assert rep["RDEP_nonpositive_set_to_nan"] == 1
    assert df.loc[0, "GR"] == -999.25  # input untouched


def test_unlabelled_rows_dropped_and_names_mapped():
    df = _toy(2, 10)
    df.loc[:3, v1.LABEL] = -999.25
    rep = {}
    out = v1.build_labelled_table(df, rep)
    assert len(out) == len(df) - 4
    assert out["LITHOLOGY_NAME"].notna().all()


def test_unknown_code_raises():
    df = _toy(1, 5)
    df.loc[0, v1.LABEL] = 12345
    with pytest.raises(ValueError):
        v1.build_labelled_table(df, {})


def test_split_is_well_wise_and_deterministic():
    pytest.importorskip("sklearn")
    df = _toy()
    a = v1.well_split(df)
    b = v1.well_split(df)
    assert a[2] == b[2] and a[3] == b[3]
    assert not set(a[2]) & set(a[3])
    assert set(df.iloc[a[0]]["WELL"]).isdisjoint(set(df.iloc[a[1]]["WELL"]))
    assert len(a[3]) == 2  # 20 % of 10 wells


def test_stratified_sample_caps_size_and_keeps_every_group():
    df = _toy(4, 500)
    out, info = v1.stratified_sample(df, max_samples=300)
    assert info["sampled"] and len(out) <= 300 + df.groupby(["WELL", v1.LABEL]).ngroups
    assert set(map(tuple, df[["WELL", v1.LABEL]].drop_duplicates().values)) == \
        set(map(tuple, out[["WELL", v1.LABEL]].drop_duplicates().values))
    again, _ = v1.stratified_sample(df, max_samples=300)
    assert out.index.equals(again.index)  # reproducible


def test_pipeline_imputes_and_adds_indicators():
    pytest.importorskip("sklearn")
    from sklearn.dummy import DummyClassifier
    df = _toy(2, 20)
    df.loc[0, "GR"] = np.nan
    pipe = v1.make_pipeline(DummyClassifier(strategy="most_frequent"))
    pipe.fit(df[v1.FEATURES], df[v1.LABEL])
    names = list(pipe.named_steps["impute"].get_feature_names_out(v1.FEATURES))
    assert any(n.startswith("missingindicator_") for n in names)
