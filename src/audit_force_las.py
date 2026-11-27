"""Read-only audit of the extracted FORCE 2020 LAS files.

What it does
------------
* Recursively finds every .las file under the input folder.
* Parses each file's header (~Well, ~Curve) and data (~Ascii) WITHOUT
  modifying the file (files are opened in read-only text mode).
* Writes derived summary tables to an output folder (default data/audit/).

What it does NOT do: train a model, split data, or change any LAS file.

Usage (from LogLens_AI/):
    python -m src.audit_force_las data/force_las --output data/audit
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_CURVE = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONF_CURVE = "FORCE_2020_LITHOFACIES_CONFIDENCE"
DEPTH_CURVES = ("DEPT", "DEPTH", "DEPTH_MD")


def parse_las(path: Path):
    """Return (header_dict, curves_list, dataframe). Read-only."""
    text = path.read_text(errors="replace").splitlines()
    section = None
    well_hdr, curves, data_lines = {}, [], []
    for line in text:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            section = s[1:2].upper()  # V, W, C, P, A
            continue
        if section == "A":
            data_lines.append(s)
        elif section == "W":
            m = re.match(r"^([^.\s]+)\s*\.([^\s]*)\s+(.*?)\s*:(.*)$", s)
            if m:
                well_hdr[m.group(1).strip()] = (m.group(3).strip(), m.group(2).strip())
        elif section == "C":
            m = re.match(r"^([^.\s]+(?:\s[^.\s]+)*)\s*\.(\S*)\s*(.*?):(.*)$", s)
            if m:
                curves.append((m.group(1).strip(), m.group(2).strip()))
    names = [c[0] for c in curves]
    null_val = -999.25
    if "NULL" in well_hdr:
        try:
            null_val = float(well_hdr["NULL"][0] or well_hdr["NULL"][1])
        except ValueError:
            pass
    arr = np.array([l.split() for l in data_lines], dtype=float) if data_lines else np.empty((0, len(names)))
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(f"column count {arr.shape} != curve count {len(names)}")
    df = pd.DataFrame(arr, columns=names)
    return well_hdr, curves, df, null_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("data/audit"))
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("*.las")) + sorted(args.input_dir.rglob("*.LAS"))
    files = sorted(set(files))
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"LAS files found (recursive): {len(files)}")

    rows, class_rows, curve_units = [], [], defaultdict(Counter)
    curve_presence = {}
    null_rates = {}
    all_class_counts, errors = Counter(), []
    total_rows = labelled_rows = 0

    for f in files:
        try:
            hdr, curves, df, null_val = parse_las(f)
        except Exception as e:  # keep going, report it
            errors.append((f.name, str(e)))
            continue
        names = [c[0] for c in curves]
        for n, u in curves:
            curve_units[n][u] += 1
        well_name = f.stem
        hdr_well = hdr.get("WELL", ("", ""))[0]
        # Replace declared NULL with NaN for analysis (derived copy in memory only)
        d = df.replace(null_val, np.nan)
        n = len(d)
        total_rows += n
        has_label = LABEL_CURVE in names
        lab = d[LABEL_CURVE] if has_label else pd.Series(dtype=float)
        n_lab = int(lab.notna().sum()) if has_label else 0
        labelled_rows += n_lab
        if n_lab:
            vc = lab.dropna().astype(int).value_counts()
            for k, v in vc.items():
                all_class_counts[int(k)] += int(v)
                class_rows.append({"well": well_name, "class_code": int(k), "n_samples": int(v)})
        dep = df[names[0]].to_numpy()
        dd = np.diff(dep) if n > 1 else np.array([])
        log_curves = [c for c in names if c not in DEPTH_CURVES + (LABEL_CURVE, CONF_CURVE)
                      and c not in ("x_loc", "y_loc", "z_loc")]
        rec = {
            "file": str(f.relative_to(args.input_dir)),
            "well_from_filename": well_name,
            "well_from_header": hdr_well,
            "name_matches_header": well_name.replace("_", "/") == hdr_well,
            "null_value_declared": null_val,
            "n_samples": n,
            "depth_start_header": hdr.get("STRT", ("", ""))[0],
            "depth_stop_header": hdr.get("STOP", ("", ""))[0],
            "depth_min_data": float(dep.min()) if n else np.nan,
            "depth_max_data": float(dep.max()) if n else np.nan,
            "depth_step_median": float(np.median(dd)) if len(dd) else np.nan,
            "depth_monotonic_increasing": bool((dd > 0).all()) if len(dd) else True,
            "has_label_curve": has_label,
            "n_labelled_samples": n_lab,
            "pct_labelled": round(100 * n_lab / n, 2) if n else 0.0,
            "n_classes_present": int(lab.dropna().nunique()) if n_lab else 0,
            "n_curves_total": len(names),
            "n_log_curves": len(log_curves),
            "n_log_curves_with_any_data": int(sum(d[c].notna().any() for c in log_curves)),
            "has_all_core_curves(GR,RHOB,NPHI,RDEP,DTC)": all(c in names and d[c].notna().any()
                                                             for c in ("GR", "RHOB", "NPHI", "RDEP", "DTC")),
            "curves": ";".join(names),
        }
        rows.append(rec)
        curve_presence[well_name] = {c: int(c in names) for c in names}
        null_rates[well_name] = {c: round(float(d[c].isna().mean()) * 100, 2) for c in names}

    man = pd.DataFrame(rows)
    man.to_csv(args.output / "well_manifest.csv", index=False)

    pres = pd.DataFrame(curve_presence).T.fillna(0).astype(int)
    pres.index.name = "well"
    pres.to_csv(args.output / "curve_presence_matrix.csv")
    nr = pd.DataFrame(null_rates).T
    nr.index.name = "well"
    nr.to_csv(args.output / "curve_missing_pct_by_well.csv")

    pd.DataFrame(class_rows).to_csv(args.output / "class_counts_by_well.csv", index=False)
    cls = pd.DataFrame(sorted(all_class_counts.items()), columns=["class_code", "n_samples"])
    cls["pct"] = (100 * cls.n_samples / cls.n_samples.sum()).round(2)
    cls["n_wells_containing"] = cls.class_code.map(
        pd.DataFrame(class_rows).groupby("class_code").well.nunique())
    cls.to_csv(args.output / "class_distribution.csv", index=False)

    cu = [{"curve": c, "units_seen": " | ".join(f"{u or '(none)'}×{k}" for u, k in cnt.items()),
           "n_wells_with_curve": sum(cnt.values()), "unit_inconsistent": len(cnt) > 1}
          for c, cnt in curve_units.items()]
    cu = pd.DataFrame(cu).sort_values("n_wells_with_curve", ascending=False)
    cu.to_csv(args.output / "curve_inventory.csv", index=False)

    summary = {
        "n_las_files": len(files), "n_parsed": len(rows), "parse_errors": errors,
        "n_wells_with_label_curve": int(man.has_label_curve.sum()),
        "n_wells_without_label_curve": int((~man.has_label_curve).sum()),
        "n_wells_with_label_curve_but_zero_labelled_rows": int(((man.has_label_curve) & (man.n_labelled_samples == 0)).sum()),
        "total_depth_samples": int(total_rows), "total_labelled_samples": int(labelled_rows),
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nOutputs written to", args.output)


if __name__ == "__main__":
    main()
