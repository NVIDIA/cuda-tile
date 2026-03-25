#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python python3 "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"

uv pip install pytest "cuda-tile[tileiras]" cuda-python cuda-core cupy-cuda13x

nvidia-smi

PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/" -v "$@"
