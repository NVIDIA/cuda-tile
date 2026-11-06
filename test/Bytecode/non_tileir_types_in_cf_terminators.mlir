// RUN: not cuda-tile-translate -mlir-to-cudatilebc %s -no-implicit-module -bytecode-version=250.0 2>&1 | FileCheck %s

// Verify that programs using non-TileIR types in control-flow terminators
// (yield, break, continue) are rejected by the bytecode writer. The MLIR
// verifier accepts AnyType in these ops (to support foreign frontends that
// embed non-CudaTile ops), but the bytecode format is a closed format that
// only supports CudaTile and builtin scalar/float types — so serialization
// must fail rather than silently produce an unreadable file.

// CHECK: unsupported type in bytecode writer

cuda_tile.module @kernels {

  // yield: foreign type flows from a function arg through an if-result.
  cuda_tile.testing$func @yield_foreign_type(%val: vector<4xf32>, %cond: !cuda_tile.tile<i1>) {
    %r = cuda_tile.if %cond -> (vector<4xf32>) {
      cuda_tile.yield %val : vector<4xf32>
    } else {
      cuda_tile.yield %val : vector<4xf32>
    }
    cuda_tile.return
  }

  // break: foreign type flows from a function arg out of a loop via break.
  cuda_tile.testing$func @break_foreign_type(%val: vector<4xf32>) {
    %r = cuda_tile.loop : vector<4xf32> {
      cuda_tile.break %val : vector<4xf32>
    }
    cuda_tile.return
  }

  // continue: foreign type flows from a function arg as a loop-carried value.
  cuda_tile.testing$func @continue_foreign_type(%val: vector<4xf32>) {
    cuda_tile.loop iter_values(%arg = %val) : vector<4xf32> {
      cuda_tile.continue %arg : vector<4xf32>
    }
    cuda_tile.return
  }

}
