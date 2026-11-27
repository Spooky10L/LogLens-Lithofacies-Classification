Dataset intentionally not bundled.

Download the FORCE 2020 well-log and lithofacies dataset from:
https://zenodo.org/records/4351156

Place extracted LAS files in this folder (or any folder you choose), then run:
python -m src.prepare_force data/force_las --output data/force2020_prepared.csv

The included synthetic_smoke_test.csv is created by `python -m src.make_demo_data` and is only for software testing.
