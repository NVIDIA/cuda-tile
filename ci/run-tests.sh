#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python python3 "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"

uv pip install pytest "cuda-tile[tileiras]"

# Find libcuda.so and add its directory to LD_LIBRARY_PATH
LIBCUDA=$(find / -name 'libcuda.so*' -print -quit 2>/dev/null || true)
if [ -n "$LIBCUDA" ]; then
    export LD_LIBRARY_PATH="$(dirname "$LIBCUDA")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Find nvidia-smi and add its directory to PATH
NVSMI=$(find / -name 'nvidia-smi' -type f -print -quit 2>/dev/null || true)
if [ -n "$NVSMI" ]; then
    export PATH="$(dirname "$NVSMI"):$PATH"
fi

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "libcuda: $LIBCUDA"
echo "nvidia-smi: $NVSMI"

PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/" -v "$@"
