#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python python3 "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"

uv pip install pytest "cuda-tile[tileiras]" cuda-python cuda-core cupy-cuda13x

# Locate libcuda.so.1 and add it to LD_LIBRARY_PATH
echo "--- Searching for libcuda.so.1 ---"
LIBCUDA_PATH=$(find /usr /lib /lib64 /usr/local 2>/dev/null -name 'libcuda.so.1' -print -quit || true)
if [ -n "$LIBCUDA_PATH" ]; then
    echo "Found: $LIBCUDA_PATH"
    export LD_LIBRARY_PATH="$(dirname "$LIBCUDA_PATH")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    echo "Not found in standard paths, running ldconfig:"
    ldconfig -p 2>/dev/null | grep -i cuda || true
    echo "Checking /usr/local/cuda:"
    ls -la /usr/local/cuda*/lib64/libcuda* /usr/local/cuda*/compat/libcuda* 2>/dev/null || true
fi
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo "---"

PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/" -v "$@"
