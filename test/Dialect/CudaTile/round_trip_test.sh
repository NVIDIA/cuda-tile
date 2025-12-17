#!/bin/bash

# Get additional flags (everything after the first two arguments)
EXTRA_FLAGS="${@:3}"

cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module $1 -o $2.out.tilebc
cuda-tile-translate -cudatilebc-to-mlir $2.out.tilebc -o $2.roundtrip.mlir $EXTRA_FLAGS
cuda-tile-opt $1 -no-implicit-module -o $2.ref.mlir $EXTRA_FLAGS

diff $2.ref.mlir $2.roundtrip.mlir -B # expect perfect round-trip
