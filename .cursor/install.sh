#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for mtg-benchmark.
# Safe to re-run: apt install is a no-op when present, the venv is reused,
# and pip -e/-r reconcile to the locked dependency set.
set -euo pipefail

cd "$(dirname "$0")/.."

# venv creation needs ensurepip, provided by python3.12-venv on Ubuntu.
if ! dpkg -s python3.12-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends python3.12-venv
fi

# The repo documents a .venv workflow; keep it so `source .venv/bin/activate`
# from the README works for future agents. .venv/ is already gitignored.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip

# Editable install exposes the `bench` entry point; requirements.txt adds
# pillow (used by dataset scripts) which is not a pyproject dependency.
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt

# pytest is not a project dependency but is required to run the test suite.
.venv/bin/pip install pytest

echo "mtg-benchmark environment ready. Activate with: source .venv/bin/activate"
