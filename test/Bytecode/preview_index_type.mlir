// Tests that the builtin `index` type is treated as public-preview content: it
// round-trips only with --preview=simt, is version-locked to the current
// bytecode version, and is rejected by both the writer and the reader without
// the explicit opt-in. Covers plain values and the control-flow terminators
// (yield, break, continue) that carry them.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 --preview=simt %s 2>&1 | FileCheck %s --check-prefix=WRITE-VERSION-CONFLICT
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt %s -o %t.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.cutile 2>&1 | FileCheck %s --check-prefix=READ-NOFLAG
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t.cutile | FileCheck %s --check-prefix=ROUNDTRIP

// WRITE-NOFLAG: uses preview feature 'simt', which is not enabled
// WRITE-VERSION-CONFLICT: only valid at the current bytecode version {{[0-9.]+}}, but version 13.4 was explicitly requested
// READ-NOFLAG: public-preview bytecode requires at least one enabled preview feature

cuda_tile.module @index_type_test {

  // ROUNDTRIP: testing$bytecode_test_preview_type : index
  entry @f() {
    %v = testing$bytecode_test_preview_type : index
    return
  }

  // ROUNDTRIP-LABEL: entry @yield_index_type
  cuda_tile.entry @yield_index_type(%cond: !cuda_tile.tile<i1>) {
    %val = cuda_tile.testing$bytecode_test_preview_type : index
    %r = cuda_tile.if %cond -> (index) {
      // ROUNDTRIP: yield %{{.*}} : index
      cuda_tile.yield %val : index
    } else {
      // ROUNDTRIP: yield %{{.*}} : index
      cuda_tile.yield %val : index
    }
    cuda_tile.return
  }

  // ROUNDTRIP-LABEL: entry @break_index_type
  cuda_tile.entry @break_index_type() {
    %val = cuda_tile.testing$bytecode_test_preview_type : index
    %r = cuda_tile.loop : index {
      // ROUNDTRIP: break %{{.*}} : index
      cuda_tile.break %val : index
    }
    cuda_tile.return
  }

  // ROUNDTRIP-LABEL: entry @continue_index_type
  cuda_tile.entry @continue_index_type() {
    %val = cuda_tile.testing$bytecode_test_preview_type : index
    cuda_tile.loop iter_values(%arg = %val) : index {
      // ROUNDTRIP: continue %{{.*}} : index
      cuda_tile.continue %arg : index
    }
    cuda_tile.return
  }
}
