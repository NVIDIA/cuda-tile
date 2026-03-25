#!/usr/bin/env bash
# Run the test suite inside the same Docker container used by CI.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04"

exec docker run --rm --gpus all \
    -v "$REPO_ROOT":/workspace \
    -w /workspace \
    "$IMAGE" \
    bash ci/run-tests.sh "$@"
