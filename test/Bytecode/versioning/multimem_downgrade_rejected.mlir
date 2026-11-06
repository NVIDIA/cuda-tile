// Multimem view operations were added in bytecode version 13.5. The writer
// must reject them when targeting the preceding 13.4 format.
// A valid multimem store requires a multicast pointer, which is also 13.5-only.
// Check the exact opcode diagnostic to distinguish it from a ptr_attr failure.
//
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.5 %s -o /dev/null
//
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.4 %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: operation 'cuda_tile.multimem_store_view_tko' is not available in bytecode version 13.4

cuda_tile.module @multimem_downgrade {
  entry @kernel(%ptr: tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = make_tensor_view %ptr, shape = [8], strides = [1]
        : tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>
        -> tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %view = make_partition_view %tv
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %idx = constant <i32: 0> : tile<i32>
    %value = constant <i32: 1> : tile<4xi32>
    %token = multimem_store_view_tko weak %view, %value[%idx]
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, tile<4xi32>, tile<i32>
        -> token
    return
  }
}
