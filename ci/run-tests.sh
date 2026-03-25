#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python python3 "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"

uv pip install pytest "cuda-tile[tileiras]"

# Ensure CUDA driver libs and tools are discoverable
CUDA_COMPAT="/usr/local/cuda/compat"
if [ -d "$CUDA_COMPAT" ]; then
    export LD_LIBRARY_PATH="${CUDA_COMPAT}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
for d in /usr/local/cuda/bin /usr/local/cuda/compat; do
    if [ -d "$d" ]; then
        export PATH="$d:$PATH"
    fi
done

PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/" -v "$@"
