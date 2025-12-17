// RUN: %round_trip_test %s %t


cuda_tile.module @kernels {
  // Test addf with flush_to_zero
  cuda_tile.entry @addf_op_ftz(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<nearest_even> flush_to_zero : tile<f32>
  }

  // Test addf with rounding_mode = rn
  cuda_tile.entry @addf_op_rn(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<nearest_even> : tile<f32>
  }

  // Test addf with rounding_mode = rz
  cuda_tile.entry @addf_op_rz(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<zero> : tile<f32>
  }

  // Test addf with rounding_mode = rm
  cuda_tile.entry @addf_op_rm(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<negative_inf> : tile<f32>
  }

  // Test addf with rounding_mode = rp
  cuda_tile.entry @addf_op_rp(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<positive_inf> : tile<f32>
  }

  // Test DenseI32ArrayAttr with permute op
  cuda_tile.entry @permute_op(%a: !cuda_tile.tile<f32>) {
    %reshape = reshape %a : tile<f32> -> tile<1x1x1xf32>
    %bcast = broadcast %reshape : tile<1x1x1xf32> -> tile<2x4x8xf32>
    %1 = cuda_tile.permute %bcast [2, 0, 1] : tile<2x4x8xf32> -> tile<8x2x4xf32>
  }

  // Test PaddingValueAttr with make_partition_view
  cuda_tile.entry @make_partition_view_op(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %a = make_tensor_view %p, shape = [128], strides = [1] : tensor_view<128xf32, strides=[1]>
    %0 = make_partition_view %a : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>
    %1 = make_partition_view %a : partition_view<tile=(8), padding_value = zero, tensor_view<128xf32, strides=[1]>>
    %2 = make_partition_view %a : partition_view<tile=(8), padding_value = neg_zero, tensor_view<128xf32, strides=[1]>>
    %3 = make_partition_view %a : partition_view<tile=(8), padding_value = nan, tensor_view<128xf32, strides=[1]>>
    %4 = make_partition_view %a : partition_view<tile=(8), padding_value = pos_inf, tensor_view<128xf32, strides=[1]>>
    %5 = make_partition_view %a : partition_view<tile=(8), padding_value = neg_inf, tensor_view<128xf32, strides=[1]>>
  }

  // Test SignednessAttr for divi
  cuda_tile.entry @divi_op_signed(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %reshape_a = reshape %a : tile<i32> -> tile<1x1x1xi32>
    %bcast_a = broadcast %reshape_a : tile<1x1x1xi32> -> tile<2x4x8xi32>
    %reshape_b = reshape %b : tile<i32> -> tile<1x1x1xi32>
    %bcast_b = broadcast %reshape_b : tile<1x1x1xi32> -> tile<2x4x8xi32>
    %0 = cuda_tile.divi %bcast_a, %bcast_b signed : !cuda_tile.tile<2x4x8xi32>
  }

  cuda_tile.entry @divi_op_unsigned(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %reshape_a = reshape %a : tile<i32> -> tile<1x1x1xi32>
    %bcast_a = broadcast %reshape_a : tile<1x1x1xi32> -> tile<2x4x8xi32>
    %reshape_b = reshape %b : tile<i32> -> tile<1x1x1xi32>
    %bcast_b = broadcast %reshape_b : tile<1x1x1xi32> -> tile<2x4x8xi32>
    %0 = cuda_tile.divi %bcast_a, %bcast_b unsigned : !cuda_tile.tile<2x4x8xi32>
  }

  // Test SignednessAttr for mma
  cuda_tile.entry @mmai_op(%a: !cuda_tile.tile<i8>, %b: !cuda_tile.tile<i8>, %c: !cuda_tile.tile<i32>) {
    %reshape_a = reshape %a : tile<i8> -> tile<1x1x1xi8>
    %bcast_a = broadcast %reshape_a : tile<1x1x1xi8> -> tile<2x4x8xi8>
    %reshape_b = reshape %b : tile<i8> -> tile<1x1x1xi8>
    %bcast_b = broadcast %reshape_b : tile<1x1x1xi8> -> tile<2x8x4xi8>
    %reshape_c = reshape %c : tile<i32> -> tile<1x1x1xi32>
    %bcast_c = broadcast %reshape_c : tile<1x1x1xi32> -> tile<2x4x4xi32>
    %0 = cuda_tile.mmai %bcast_a, %bcast_b, %bcast_c signed unsigned : !cuda_tile.tile<2x4x8xi8>, !cuda_tile.tile<2x8x4xi8>, !cuda_tile.tile<2x4x4xi32>
  }
}
