#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.make_demo_data
pytest -q
printf '\nSetup complete. See README.md for FORCE 2020 data preparation and app commands.\n'
