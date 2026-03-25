#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

apt-get update -qq && apt-get install -y -qq python3-full python3-dev curl > /dev/null

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear --python python3 "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"

uv pip install -r "$REPO_ROOT/requirements.txt"

nvidia-smi

PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/" -v "$@"
