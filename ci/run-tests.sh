#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

pip install --quiet uv
uv venv "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"

uv pip install pytest "cuda-tile[tileiras]"

PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/" -v "$@"
