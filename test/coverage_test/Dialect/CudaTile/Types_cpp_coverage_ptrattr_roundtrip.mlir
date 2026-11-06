// Target: Types.cpp — TensorViewType ptrAttr roundtrip, 0-D tensor_view with
// ptrAttr, and verification switch paths.
// RUN: cuda-tile-opt %s -allow-unregistered-dialect -split-input-file | FileCheck %s --check-prefix=CHECK

// -----

// Covers TensorViewType::verify — ptrAttr with NONE value (switch case).
// Covers TensorViewType::print — ptrAttr printing path.
// CHECK-LABEL: @ptrattr_none
// CHECK: "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides=[1], #cuda_tile.ptr_attr<none>>
cuda_tile.module @ptrattr_none {
  cuda_tile.testing$func @f() {
    %t = "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides=[1], #cuda_tile.ptr_attr<none>>
  }
}

// -----

// Covers TensorViewType::parse — 0-D tensor_view with valid ptrAttr and parseGreater.
// CHECK-LABEL: @zerod_ptrattr
// CHECK: "use_type"() : () -> !cuda_tile.tensor_view<f32, #cuda_tile.ptr_attr<none>>
cuda_tile.module @zerod_ptrattr {
  cuda_tile.testing$func @f() {
    %t = "use_type"() : () -> !cuda_tile.tensor_view<f32, #cuda_tile.ptr_attr<none>>
  }
}
