// Covers BytecodeWriter writeProducerSection() at the 13.3 boundary —
// the minimum version where the producer section is emitted. Pairs with
// producerTest.mlir (default = 13.4) and producer_attr_backward_compat.mlir
// (13.1, drops the attr), so this test pins the exact `version ==
// kMinProducerVersion` path explicitly.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.3 %s -o %t.mlir
// RUN: FileCheck %s < %t.mlir

// Module with explicit producer attribute — drives the
// `producerAttr != nullptr` branch of writeProducerSection() when the
// target bytecode version satisfies `>= kMinProducerVersion (13.3)`.
cuda_tile.module @kernels attributes {producer = "cov-test-producer v13.3-boundary"} {
  cuda_tile.entry @producer_emitted_at_13_3(%a: !cuda_tile.tile<f32>) {
    cuda_tile.return
  }
}

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK-SAME:  producer = "cov-test-producer v13.3-boundary"
// CHECK:       entry @producer_emitted_at_13_3
