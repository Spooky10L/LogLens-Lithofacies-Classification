"""Generate a tiny, explicitly synthetic dataset to smoke-test the workflow."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_demo_data(n_wells: int = 8, rows_per_well: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    classes = [30000, 65000, 65030, 74000]
    centers = {
        30000: (35, 2.25, 0.16, 18.0),
        65000: (105, 2.55, 0.32, 2.0),
        65030: (72, 2.42, 0.25, 7.0),
        74000: (18, 2.68, 0.12, 45.0),
    }
    for w in range(n_wells):
        # Use repeated facies blocks down each well; transition zones are noisy.
        labels = np.resize(np.array(classes), rows_per_well)
        rng.shuffle(labels)
        for i, code in enumerate(labels):
            gr, rhob, nphi, rdep = centers[int(code)]
            rows.append({
                "WELL": f"SYNTH_WELL_{w+1:02d}",
                "DEPTH_MD": 1500 + i * 0.15,
                "GR": rng.normal(gr + (w % 3) * 2, 10),
                "RHOB": rng.normal(rhob, 0.08),
                "NPHI": rng.normal(nphi, 0.04),
                "RDEP": max(0.1, rng.normal(rdep, rdep * 0.25 + 1)),
                "PEF": rng.normal(2.5 if code == 30000 else 3.1, 0.4),
                "DTC": rng.normal(85 if code == 30000 else 105, 10),
                "LITHOLOGY": int(code),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = ROOT / "data" / "synthetic_smoke_test.csv"
    out.parent.mkdir(exist_ok=True)
    make_demo_data().to_csv(out, index=False)
    print(f"Created clearly synthetic smoke-test data: {out}")
