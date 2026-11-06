// RUN: %round_trip_test %s %t

// Round-trip the required `inbounds` attribute on the multimem load-reduce,
// reduce, and store ops through the bytecode reader/writer. The textual
// parser synthesizes all-false when the clause is omitted, and the printer
// elides all-false as the canonical spelling.

cuda_tile.module @kernels {

  //===--------------------------------------------------------------------===//
  // multimem_load_reduce_view_tko
  //===--------------------------------------------------------------------===//

  cuda_tile.entry @mlr_inbounds_partial(%mcptr: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %mcptr, shape = [8192, 128], strides = [128, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>
    %view = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>

    %a, %atok = cuda_tile.multimem_load_reduce_view_tko weak %view[%c0, %c0] inbounds = [true, false]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<64x64xf32>, !cuda_tile.token
    %b, %btok = cuda_tile.multimem_load_reduce_view_tko weak %view[%c0, %c0] inbounds = [false, true]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<64x64xf32>, !cuda_tile.token
    %c, %ctok = cuda_tile.multimem_load_reduce_view_tko weak %view[%c0, %c0] inbounds = [true, true]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<64x64xf32>, !cuda_tile.token
    // Explicit all-false: canonicalizes to the omitted textual form.
    %d, %dtok = cuda_tile.multimem_load_reduce_view_tko weak %view[%c0, %c0] inbounds = [false, false]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<64x64xf32>, !cuda_tile.token
  }

  cuda_tile.entry @mlr_no_attribute(%mcptr: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %mcptr, shape = [8192, 128], strides = [128, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>
    %view = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>

    %v, %tok = cuda_tile.multimem_load_reduce_view_tko weak %view[%c0, %c0]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<64x64xf32>, !cuda_tile.token
  }

  //===--------------------------------------------------------------------===//
  // multimem_reduce_view_tko
  //===--------------------------------------------------------------------===//

  cuda_tile.entry @mr_inbounds(%mcptr: !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %mcptr, shape = [2], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %view = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %partials = cuda_tile.constant <i32: 1> : !cuda_tile.tile<1xi32>

    %t1 = cuda_tile.multimem_reduce_view_tko relaxed device %view, %partials[%c0] inbounds = [true]
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
    %t2 = cuda_tile.multimem_reduce_view_tko relaxed device %view, %partials[%c0] inbounds = [false]
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }

  //===--------------------------------------------------------------------===//
  // multimem_store_view_tko
  //===--------------------------------------------------------------------===//

  cuda_tile.entry @ms_inbounds(%mcptr: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %mcptr, shape = [8192, 128], strides = [128, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>
    %view = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %val = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>

    %t1 = cuda_tile.multimem_store_view_tko weak %view, %val[%c0, %c0] inbounds = [true, false]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
    %t2 = cuda_tile.multimem_store_view_tko weak %view, %val[%c0, %c0] inbounds = [true, true]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
    // Explicit all-false: canonicalizes to the omitted textual form.
    %t3 = cuda_tile.multimem_store_view_tko weak %view, %val[%c0, %c0] inbounds = [false, false]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }

  cuda_tile.entry @ms_no_attribute(%mcptr: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %mcptr, shape = [8192, 128], strides = [128, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>
    %view = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %val = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>

    %t = cuda_tile.multimem_store_view_tko weak %view, %val[%c0, %c0]
        : !cuda_tile.partition_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128, 1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }

}
