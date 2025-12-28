#!/usr/bin/env bash
set -euo pipefail

# Bootstrap developer environment: create venv, install deps, install pre-commit
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r dev-requirements.txt
python -m pre_commit install
echo "Bootstrap complete. Activate the venv with: source .venv/bin/activate"
