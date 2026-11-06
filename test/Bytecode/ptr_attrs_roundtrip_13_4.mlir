// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.4 %s -o %t.mlir
// RUN: cuda-tile-opt --no-implicit-module %s -o %t.ref.mlir
// RUN: diff %t.mlir %t.ref.mlir

// Pointer classifications are optional parameters added to PointerType and
// TensorViewType in 13.4. `none` is the only case that far back, and it is a
// distinct wire value from carrying no classification at all, so exercise it in
// both types. The `unicast` and `multicast` cases arrived in 13.5 and are
// covered by ptr_attrs_roundtrip_13_5.mlir.
cuda_tile.module @kernels {
  entry @roundtrip(
      %p: tile<ptr<i32, #cuda_tile.ptr_attr<none>>>,
      %plain: tile<ptr<f32>>) {
    %view = make_tensor_view %p, shape = [16], strides = [1]
        : tile<ptr<i32, #cuda_tile.ptr_attr<none>>>
        -> tensor_view<16xi32, strides=[1], #cuda_tile.ptr_attr<none>>
    %plain_view = make_tensor_view %plain, shape = [8, 8], strides = [8, 1]
        : tile<ptr<f32>>
        -> tensor_view<8x8xf32, strides=[8, 1]>
    return
  }
}
