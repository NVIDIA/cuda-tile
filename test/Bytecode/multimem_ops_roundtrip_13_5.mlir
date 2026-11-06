// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.5 %s -o %t.mlir
// RUN: cuda-tile-opt --no-implicit-module %s -o %t.ref.mlir
// RUN: diff %t.mlir %t.ref.mlir

// Exercise every public multimem *_view_tko opcode and all of their 13.5
// optional fields, ordering variants, reduction values, and result shapes.
cuda_tile.module @kernels {
  entry @roundtrip(%ptr: tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = make_tensor_view %ptr, shape = [8, 8], strides = [8, 1]
        : tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>
        -> tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>
    %view = make_partition_view %tv
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>
    %idx = constant <i32: 0> : tile<i32>
    %value = constant <i32: 1> : tile<4x4xi32>

    %l0, %lt0 = multimem_load_reduce_view_tko weak %view[%idx, %idx]
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<i32>
        -> tile<4x4xi32>, token
    %l1, %lt1 = multimem_load_reduce_view_tko relaxed device %view[%idx, %idx]
        inbounds = [true, false]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<i32>
        -> tile<4x4xi32>, token
    %l2, %lt2 = multimem_load_reduce_view_tko acquire sys %view[%idx, %idx]
        token = %lt0 inbounds = [false, true]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<i32>, token
        -> tile<4x4xi32>, token
    %l3, %lt3 = multimem_load_reduce_view_tko weak %view[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<and>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<i32>
        -> tile<4x4xi32>, token
    %l4, %lt4 = multimem_load_reduce_view_tko weak %view[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<or>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<i32>
        -> tile<4x4xi32>, token
    %l5, %lt5 = multimem_load_reduce_view_tko weak %view[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<xor>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<i32>
        -> tile<4x4xi32>, token

    %rt0 = multimem_reduce_view_tko relaxed device %view, %value[%idx, %idx]
        token = %lt1 inbounds = [true, true]
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>, token
        -> token
    %rt1 = multimem_reduce_view_tko release sys %view, %value[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>
        -> token
    %rt2 = multimem_reduce_view_tko relaxed device %view, %value[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>
        -> token
    %rt3 = multimem_reduce_view_tko relaxed device %view, %value[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<and>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>
        -> token
    %rt4 = multimem_reduce_view_tko relaxed device %view, %value[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<or>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>
        -> token
    %rt5 = multimem_reduce_view_tko relaxed device %view, %value[%idx, %idx]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<xor>}
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>
        -> token

    %st0 = multimem_store_view_tko weak %view, %l0[%idx, %idx]
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>
        -> token
    %st1 = multimem_store_view_tko relaxed device %view, %l1[%idx, %idx]
        token = %rt0 inbounds = [true, false]
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>, token
        -> token
    %st2 = multimem_store_view_tko release sys %view, %l2[%idx, %idx]
        inbounds = [false, true]
        : partition_view<tile=(4x4), tensor_view<8x8xi32, strides=[8, 1], #cuda_tile.ptr_attr<multicast>>>, tile<4x4xi32>, tile<i32>
        -> token

    %sum = addi %l3, %l4 : tile<4x4xi32>
    %sum2 = addi %sum, %l5 : tile<4x4xi32>
    %joined = join_tokens %lt2, %lt3, %lt4, %lt5,
        %rt1, %rt2, %rt3, %rt4, %rt5, %st0, %st1, %st2 : token
    return
  }
}
