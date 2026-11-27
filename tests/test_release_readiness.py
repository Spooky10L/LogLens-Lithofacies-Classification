"""Release-hygiene tests: publication report, pinned requirements, licence, README links."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import publish_report as pr

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "models" / "v1"
needs_real = pytest.mark.skipif(not (REAL / "per_class_metrics_audited.csv").exists(),
                                reason="trained results not present")


def _toy():
    aud = pd.DataFrame({
        "class_code": [65000, 74000, 93000], "class_name": ["Shale", "Dolomite", "Basement"],
        "precision": [0.9, 0.0, np.nan], "recall": [0.8, 0.0, np.nan], "f1": [0.85, 0.0, np.nan],
        "support_val": [100, 10, 0], "n_predicted": [90, 3, 0]})
    cov = pd.DataFrame({"class_name": ["Shale", "Dolomite", "Basement"], "train_wells": [94, 37, 1],
                        "val_wells": [24, 10, 0],
                        "flags": ["", "", "RARE: in only 1 of 118 labelled wells; ABSENT FROM VALIDATION"]})
    return aud, cov


def test_report_marks_absent_class_not_scorable_but_keeps_real_zero():
    rep = pr.build_report(*_toy())
    b = rep[rep["class"] == "Basement"].iloc[0]
    assert not b["scorable"] and pd.isna(b["f1"]) and pd.isna(b["recall"])
    d = rep[rep["class"] == "Dolomite"].iloc[0]
    assert d["scorable"] and d["f1"] == 0.0  # a genuine measured zero is kept as 0
    assert "none of its validation samples" in d["note"]
    assert "RARE" in b["note"]


def test_markdown_never_prints_a_zero_for_basement():
    md = pr.render_markdown(pr.build_report(*_toy()),
                            {"metrics_validation": {"random_forest": {"accuracy": .6786, "balanced_accuracy": .4346,
                                                                      "macro_f1": .4220, "weighted_f1": .6924,
                                                                      "n_classes_scored": 11}}})
    row = [l for l in md.splitlines() if l.startswith("| Basement")][0]
    assert row.count("not scorable") == 3 and "0.000" not in row
    assert "0.6786" in md and "not** the official FORCE 2020" in md


@needs_real
def test_real_report_matches_saved_metrics_and_original_is_preserved():
    rep = pd.read_csv(REAL / "per_class_report.csv")
    assert len(rep) == 12 and rep["scorable"].sum() == 11
    assert rep.loc[rep["class"] == "Basement", "f1"].isna().all()
    sc = rep[rep.scorable]
    assert round(sc["recall"].mean(), 4) == 0.4346  # balanced accuracy
    assert round(sc["f1"].mean(), 4) == 0.4220  # macro-F1
    # the original training output is preserved, unmodified, locally (it prints Basement as 0.0)
    orig = REAL / "per_class_metrics_random_forest.csv"
    if orig.exists():
        o = pd.read_csv(orig)
        assert float(o.loc[o.class_name == "Basement", "f1"].iloc[0]) == 0.0


def test_requirements_are_exactly_pinned_to_tested_versions():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l and not re.search(r"[<>~]", l) for l in lines)
    pins = dict(l.split("==") for l in lines)
    assert pins["numpy"] == "2.5.5" and pins["pandas"] == "3.1.1" and pins["scikit-learn"] == "1.9.1"
    assert pins["streamlit"] == "1.67.0" and pins["plotly"] == "7.2.0" and pins["matplotlib"] == "3.11.2"
    assert pins["joblib"] == "1.6.0" and pins["pytest"] == "9.2.0"


def test_pins_match_the_installed_environment():
    from importlib.metadata import version
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, pin = line.strip().split("==")
            assert version(name) == pin, f"{name}: installed {version(name)} != pinned {pin}"


def test_licence_is_mit_and_keeps_dataset_licence_separate():
    t = (ROOT / "LICENSE").read_text()
    assert t.startswith("MIT License") and "Permission is hereby granted" in t
    assert "CC BY 4.0" in t and "NOT part of this repository" in t
    readme = (ROOT / "README.md").read_text()
    assert "MIT License" in readme and "CC BY 4.0" in readme
    for author in ("Bormann", "Aursand", "Dilib", "Manral", "Dischington"):
        assert author in readme
    assert "10.5281/zenodo.4351156" in readme


def test_readme_links_resolve_and_do_not_mention_local_only_files():
    readme = (ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(((?!http)[^)#]+)\)", readme):
        assert (ROOT / target).exists(), f"broken README link: {target}"
    for private in ("app_legacy", "INTERVIEW_NOTES", "NEXT_STEPS", "CV_ENTRY"):
        assert private not in readme


def test_gitignore_keeps_private_and_large_files_out():
    gi = (ROOT / ".gitignore").read_text().splitlines()
    for must in ("docs/INTERVIEW_NOTES.md", "docs/NEXT_STEPS.md", "docs/CV_ENTRY.md", "app_legacy.py",
                 ".venv/", "data/force_las/", "data/processed/", "*.joblib", "models/v1_smoke/", "*.las"):
        assert must in gi, f"{must} missing from .gitignore"
