// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s

cuda_tile.module @kernels {

// CHECK-LABEL: testing$func @test_ptr_types
testing$func @test_ptr_types(
    // CHECK-SAME: ptr<i1>
    %arg0: !cuda_tile.ptr<i1>) {
  return
}

// CHECK-LABEL: testing$func @test_tile_types
testing$func @test_tile_types(
    // CHECK-SAME: tile<2xf32>
    %arg0: !cuda_tile.tile<2xf32>,
    // CHECK-SAME: tile<f32>
    %arg1: !cuda_tile.tile<f32>) {
  return
}

// CHECK-LABEL: testing$func @test_tensor_view_types
testing$func @test_tensor_view_types(
    // CHECK-SAME: tensor_view<f32>
    %arg0: !cuda_tile.tensor_view<f32>,
    // CHECK-SAME: tensor_view<2xf32, strides=[1]>
    %arg1: !cuda_tile.tensor_view<2xf32, strides=[1]>,
    // CHECK-SAME: tensor_view<?x2xf32, strides=[1,?]>
    %arg2: !cuda_tile.tensor_view<?x2xf32, strides=[1,?]>,
    // CHECK-SAME: tensor_view<?x?xf32, strides=[?,?]>
    %arg3: !cuda_tile.tensor_view<?x?xf32, strides=[?,?]>,
    // CHECK-SAME: tensor_view<4x?xf32, strides=[5,?]>
    %arg4: !cuda_tile.tensor_view<4x?xf32, strides=[5,?]>,
    // CHECK-SAME: tensor_view<4x?xf32, strides=[5,?]>
    %arg5: !cuda_tile.tensor_view<4x?xf32, strides=[5,?]>,
    // CHECK-SAME: tensor_view<f32>
    %arg6: !cuda_tile.tensor_view<f32>) {
  return
}

// FIXME: Once 0-d tiled views are supported, enable this test.
// CHECK-LABEL (DISABLED): testing$func @test_disabled_tile_partition_view_types
//testing$func @test_disabled_tile_partition_view_types(
//    // CHECK-SAME (DISABLED): partition_view<tile=(), tensor_view<f32>>
//    %arg0: !cuda_tile.partition_view<tile=(), tensor_view<f32>>,
//    // CHECK-SAME (DISABLED): partition_view<tile=(), tensor_view<f32>>
//    %arg1: !cuda_tile.partition_view<tile=(), !cuda_tile.tensor_view<f32>, dim_map=[]>) {
//  return
//}

// CHECK-LABEL: testing$func @test_tile_partition_view_types
testing$func @test_tile_partition_view_types(
    // CHECK-SAME: partition_view<tile=(2), tensor_view<16xf32, strides=[1]>>
    %arg0: !cuda_tile.partition_view<tile=(2), tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: partition_view<tile=(2), padding_value = zero, tensor_view<16xf32, strides=[1]>>
    %arg1: !cuda_tile.partition_view<tile=(2), padding_value = zero, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: partition_view<tile=(2), padding_value = nan, tensor_view<16xf32, strides=[1]>>
    %arg2: !cuda_tile.partition_view<tile=(2), padding_value = nan, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: partition_view<tile=(2), padding_value = neg_zero, tensor_view<16xf32, strides=[1]>>
    %arg3: !cuda_tile.partition_view<tile=(2), padding_value = neg_zero, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: partition_view<tile=(2), padding_value = pos_inf, tensor_view<16xf32, strides=[1]>>
    %arg4: !cuda_tile.partition_view<tile=(2), padding_value = pos_inf, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: partition_view<tile=(2), padding_value = neg_inf, tensor_view<16xf32, strides=[1]>>
    %arg5: !cuda_tile.partition_view<tile=(2), padding_value = neg_inf, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: partition_view<tile=(2), tensor_view<16xf32, strides=[1]>>
    %arg6: !cuda_tile.partition_view<tile=(2), tensor_view<16xf32, strides=[1]>, dim_map=[0]>,
    // CHECK-SAME: partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>>
    %arg7: !cuda_tile.partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>>,
    // CHECK-SAME: partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>>
    %arg8: !cuda_tile.partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>, dim_map=[0, 1]>,
    // CHECK-SAME: partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>
    %arg9: !cuda_tile.partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>) {
  return
}
}
