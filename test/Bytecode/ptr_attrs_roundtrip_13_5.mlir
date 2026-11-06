// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.5 %s -o %t.mlir
// RUN: cuda-tile-opt --no-implicit-module %s -o %t.ref.mlir
// RUN: diff %t.mlir %t.ref.mlir

// The `unicast` and `multicast` classifications are 13.5 cases of the pointer
// classification enum, riding in the 13.4 parameter of PointerType and
// TensorViewType. Exercise both cases in both types.
cuda_tile.module @kernels {
  entry @roundtrip(
      %uc: tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %mc: tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
    %uc_view = make_tensor_view %uc, shape = [16], strides = [1]
        : tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> tensor_view<16xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>
    %mc_view = make_tensor_view %mc, shape = [8, 8], strides = [8, 1]
        : tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>
        -> tensor_view<8x8xf32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>
    return
  }
}
