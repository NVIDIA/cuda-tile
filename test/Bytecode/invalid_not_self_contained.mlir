// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -split-input-file -verify-diagnostics -allow-unregistered-dialect %s

// expected-error @below{{only ops from the 'cuda_tile' dialect are allowed}}
cuda_tile.module @kernels {
  cuda_tile.entry @kernel() {
    // expected-remark @below{{invalid op}}
    "test.op_from_different_dialect"() : () -> ()
  }
}

// -----

// expected-error @below{{only function and global ops are allowed in the body}}
cuda_tile.module @kernels {
  // expected-remark @below{{invalid op}}
  cuda_tile.constant <f32: 5.0> : !cuda_tile.tile<f32>
}
