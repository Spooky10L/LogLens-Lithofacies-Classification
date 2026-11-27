"""Prepare FORCE 2020 LAS files or competition CSVs for LogLens.

The raw FORCE dataset must be downloaded separately from its official source.
This script does not redistribute the dataset.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_ALIASES = {
    "DEPTH_MD": {"DEPTH_MD", "MD", "DEPT", "DEPTH", "MD_M"},
    "GR": {"GR", "GAMMA", "GAMMARAY", "GR_API"},
    "RHOB": {"RHOB", "RHOZ", "DENS", "DENSITY", "BULK_DENSITY"},
    "NPHI": {"NPHI", "NEUT", "NEU", "TNPH", "NPOR"},
    "RDEP": {"RDEP", "RES_DEEP", "ILD", "LLD", "AT90", "RDEEP"},
    "RMED": {"RMED", "RES_MEDIUM", "ILM", "LLS", "AT60"},
    "RSHA": {"RSHA", "RES_SHALLOW", "SFLU", "SFL", "AT10"},
    "PEF": {"PEF", "PE", "PHOTOELECTRIC"},
    "DTC": {"DTC", "DT", "AC", "DTCO", "SONIC"},
    "DTS": {"DTS", "DTSM", "SHEAR_SONIC"},
    "SP": {"SP", "SPONTANEOUS_POTENTIAL"},
    "CALI": {"CALI", "CALIPER", "HCAL"},
    "BS": {"BS", "BIT_SIZE", "BIT"},
    "ROP": {"ROP", "RATE_OF_PENETRATION"},
    "MUDWEIGHT": {"MUDWEIGHT", "MW", "MUD_WEIGHT"},
}
TARGET_ALIASES = {
    "FORCE_2020_LITHOFACIES_LITHOLOGY", "LITHOLOGY", "LITHOFACIES",
    "LITHO", "LITHOLOGY_CODE", "LITHOFACIES_CODE", "LITH_CODE"
}


def _clean_mnemonic(value: object) -> str:
    return re.sub(r"[^A-Z0-9_]", "", str(value).upper().strip())


def standardize_columns(frame: pd.DataFrame, well_name: str) -> pd.DataFrame:
    """Map common FORCE/LAS mnemonics to stable names; retain target and well."""
    lookup: dict[str, str] = {}
    for standard, aliases in FEATURE_ALIASES.items():
        for alias in aliases | {standard}:
            lookup[_clean_mnemonic(alias)] = standard
    lookup.update({_clean_mnemonic(a): "LITHOLOGY" for a in TARGET_ALIASES})

    rename = {}
    for col in frame.columns:
        cleaned = _clean_mnemonic(col)
        if cleaned in lookup:
            rename[col] = lookup[cleaned]
    out = frame.rename(columns=rename).copy()
    # Keep the first instance when a LAS file contains duplicate aliases.
    out = out.loc[:, ~out.columns.duplicated()]
    out["WELL"] = well_name

    for col in out.columns:
        if col not in {"WELL"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "LITHOLOGY" in out.columns:
        out["LITHOLOGY"] = out["LITHOLOGY"].round().astype("Int64")
    return out


def read_las_folder(folder: Path) -> pd.DataFrame:
    try:
        import lasio
    except ImportError as exc:
        raise RuntimeError("LAS input requires lasio. Run: pip install lasio") from exc

    files = sorted(folder.rglob("*.las")) + sorted(folder.rglob("*.LAS"))
    if not files:
        raise FileNotFoundError(f"No .las files found under {folder}")

    frames = []
    for path in files:
        try:
            las = lasio.read(path, ignore_data_comments=True)
            df = las.df().reset_index()
            if len(df.columns) and str(df.columns[0]).upper() in {"DEPT", "DEPTH", "MD"}:
                df = df.rename(columns={df.columns[0]: "DEPTH_MD"})
            prepared = standardize_columns(df, path.stem)
            if "DEPTH_MD" not in prepared.columns and "DEPT" in df.columns:
                prepared["DEPTH_MD"] = pd.to_numeric(df["DEPT"], errors="coerce")
            if "LITHOLOGY" in prepared.columns and prepared["LITHOLOGY"].notna().any():
                frames.append(prepared)
        except Exception as exc:  # continue through files; report skipped wells
            print(f"SKIP {path.name}: {exc}", file=sys.stderr)

    if not frames:
        raise ValueError("No LAS files with a recognized lithology target were found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_competition_csv(path: Path) -> pd.DataFrame:
    # FORCE competition files are commonly comma- or semicolon-delimited.
    frame = pd.read_csv(path, sep=None, engine="python")
    well_col = next((c for c in frame.columns if _clean_mnemonic(c) == "WELL"), None)
    well_name = frame[well_col].astype(str) if well_col else path.stem
    out = standardize_columns(frame, "__placeholder__")
    if well_col:
        out["WELL"] = well_name.values
    return out


def prepare(input_path: Path, output_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        frame = read_las_folder(input_path)
    elif input_path.suffix.lower() in {".csv", ".txt"}:
        frame = read_competition_csv(input_path)
    else:
        raise ValueError("Input must be a folder containing LAS files or a CSV/TXT file.")

    if "LITHOLOGY" not in frame.columns:
        raise ValueError(
            "No lithology target found. Expected FORCE_2020_LITHOFACIES_LITHOLOGY "
            "or a recognized LITHOLOGY/LITHOFACIES column."
        )
    frame = frame.dropna(subset=["LITHOLOGY"]).copy()
    frame["LITHOLOGY"] = frame["LITHOLOGY"].astype(int)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FORCE 2020 LAS/CSV data for LogLens")
    parser.add_argument("input", type=Path, help="LAS folder or competition CSV file")
    parser.add_argument("--output", type=Path, default=Path("data/force2020_prepared.csv"))
    args = parser.parse_args()
    data = prepare(args.input, args.output)
    print(f"Saved {len(data):,} labeled depth samples from {data['WELL'].nunique()} wells to {args.output}")
    print("Lithology counts:")
    print(data["LITHOLOGY"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
