// Covers BytecodeWriter ConstantManager::serializeAttribute() i1-splat legacy
// path: when targeting bytecode versions below kCanonicalI1Version (i.e.
// versions 13.1, 13.2, 13.3) and the i1 dense attribute is a splat, the
// writer emits the legacy one-byte 0xff/0x00 encoding. Existing tests for
// this branch only READ from a baked .tileirbc fixture; nothing exercises
// the actual WRITE path of the legacy splat branch.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.3 %s -o %t_13_3.mlir
// RUN: FileCheck %s --check-prefix=CHECK-V13-3 < %t_13_3.mlir

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.1 %s -o %t_13_1.mlir
// RUN: FileCheck %s --check-prefix=CHECK-V13-1 < %t_13_1.mlir

cuda_tile.module @kernels {
  // Splat-true i1 vector: triggers the writer's
  // `denseAttr.getSplatValue<bool>() ? 0xff : 0x00` branch (true side).
  cuda_tile.entry @i1_splat_true_vec() {
    %0 = cuda_tile.constant <i1: 1> : !cuda_tile.tile<16xi1>
    cuda_tile.return
  }

  // Splat-false i1 vector: drives the same branch with the false side, so
  // both arms of the legacy splat ternary are executed in one test file.
  cuda_tile.entry @i1_splat_false_vec() {
    %0 = cuda_tile.constant <i1: 0> : !cuda_tile.tile<8xi1>
    cuda_tile.return
  }

  // Splat-true scalar i1: scalar shape exercises the same writer path with
  // a 1-element tile, confirming the splat byte is emitted regardless of
  // tile shape (legacy fixed-size-1 encoding).
  cuda_tile.entry @i1_splat_true_scalar() {
    %0 = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    cuda_tile.return
  }
}

// CHECK-V13-3-LABEL: cuda_tile.module @kernels
// CHECK-V13-3:         entry @i1_splat_true_vec
// CHECK-V13-3:           constant <i1: true> : tile<16xi1>
// CHECK-V13-3:         entry @i1_splat_false_vec
// CHECK-V13-3:           constant <i1: false> : tile<8xi1>
// CHECK-V13-3:         entry @i1_splat_true_scalar
// CHECK-V13-3:           constant <i1: true> : tile<i1>

// CHECK-V13-1-LABEL: cuda_tile.module @kernels
// CHECK-V13-1:         entry @i1_splat_true_vec
// CHECK-V13-1:           constant <i1: true> : tile<16xi1>
// CHECK-V13-1:         entry @i1_splat_false_vec
// CHECK-V13-1:           constant <i1: false> : tile<8xi1>
// CHECK-V13-1:         entry @i1_splat_true_scalar
// CHECK-V13-1:           constant <i1: true> : tile<i1>
