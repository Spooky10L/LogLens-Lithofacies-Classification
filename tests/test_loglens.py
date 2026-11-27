from src.make_demo_data import make_demo_data
from src.prepare_force import standardize_columns
from src.train import train
import pandas as pd


def test_standardize_force_columns():
    frame = pd.DataFrame({
        "GR": [20.0, 80.0],
        "RHOB": [2.3, 2.5],
        "NPHI": [0.15, 0.3],
        "FORCE_2020_LITHOFACIES_LITHOLOGY": [30000, 65000],
    })
    prepared = standardize_columns(frame, "WELL_A")
    assert "LITHOLOGY" in prepared.columns
    assert prepared["WELL"].iloc[0] == "WELL_A"


def test_training_pipeline_uses_well_groups(tmp_path):
    data = make_demo_data(n_wells=8, rows_per_well=80)
    metrics = train(data, tmp_path)
    assert metrics["n_wells"] == 8
    assert set(metrics["train_wells"]).isdisjoint(set(metrics["test_wells"]))
    assert (tmp_path / "loglens_model.joblib").exists()
