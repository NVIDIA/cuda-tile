#!/usr/bin/env bash
#
# Regenerate every encoding-stability golden in this directory.
#
# Usage:
#   regenerate_goldens.sh [path/to/cuda-tile-translate]
#
# If the translate tool isn't on $PATH, pass it as the first argument.
#
# Run this script ONLY when an encoding shift is intentional (for example,
# a deliberate bytecode-format version bump). A regression that shows up
# in this suite usually means either:
#   (a) MLIR's canonical raw-buffer / float / int encoding silently shifted
#       under us (the i1 0xff bug pattern); add a backward-compat shim in
#       BytecodeReader.cpp before regenerating, OR
#   (b) something on our writer side changed unintentionally; revert it.
#
# Do not run this to "make the failure go away" without first understanding
# which encoding moved and why.
#
# The -bytecode-version below is pinned and is duplicated in the RUN line of
# every *.mlir in this directory. If you intentionally advance the pinned
# version here, update those RUN lines in lockstep; otherwise the goldens
# will be regenerated at the new version while the tests still emit at the
# old one, and every cmp will fail.

set -euo pipefail

TRANSLATE="${1:-cuda-tile-translate}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for src in "${DIR}"/*.mlir; do
  name="$(basename "${src}" .mlir)"
  out="${DIR}/goldens/${name}.golden.tileirbc"
  # Not every test here compares bytes: the round-trip tests write and read
  # back instead, and some of those are split-file inputs that do not translate
  # on their own. A test with no golden has nothing to regenerate. To add a new
  # golden, create the empty file first so this picks it up.
  if [[ ! -f "${out}" ]]; then
    echo "Skipping ${src}: no golden"
    continue
  fi
  echo "Regenerating ${out}"
  "${TRANSLATE}" -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 "${src}" -o "${out}"
done
