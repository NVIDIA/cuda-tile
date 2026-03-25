#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python python3 "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"

uv pip install pytest "cuda-tile[tileiras]" cuda-python cuda-core cupy-cuda13x

echo "=== CI Environment Diagnostics ==="
echo "--- GPU ---"
lspci 2>/dev/null | grep -i nvidia || echo "lspci: no NVIDIA devices (or lspci not available)"
ls /dev/nvidia* 2>/dev/null || echo "/dev/nvidia*: not found"
echo "--- NVIDIA driver ---"
cat /proc/driver/nvidia/version 2>/dev/null || echo "/proc/driver/nvidia/version: not found"
nvidia-smi 2>/dev/null || echo "nvidia-smi: not found"
echo "--- libcuda ---"
ldconfig -p 2>/dev/null | grep -i libcuda || echo "ldconfig: no libcuda entries"
find / -name 'libcuda.so*' 2>/dev/null || echo "find: no libcuda.so* found"
echo "--- CUDA toolkit (pip) ---"
python3 -c "import nvidia.cu13.lib; print('nvidia.cu13.lib:', nvidia.cu13.lib.__path__)" 2>/dev/null || true
python3 -c "import nvidia.cu13.bin; print('nvidia.cu13.bin:', nvidia.cu13.bin.__path__)" 2>/dev/null || true
ls -la "$(python3 -c 'import nvidia.cu13.lib; print(nvidia.cu13.lib.__path__[0])' 2>/dev/null)" 2>/dev/null || true
echo "--- LD_LIBRARY_PATH ---"
echo "${LD_LIBRARY_PATH:-<unset>}"
echo "=== End Diagnostics ==="

PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/" -v "$@"
