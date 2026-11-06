// RUN: cuda-tile-opt %s -split-input-file | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt -split-input-file | FileCheck %s

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MakeGatherScatterViewOp::parse/print roundtrip
// Covers MakeGatherScatterViewOp::getAsmResultNames

cuda_tile.module @gs_view_roundtrip {
  // CHECK-LABEL: entry @test_gs_view
  entry @test_gs_view(%p : !cuda_tile.tile<ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [128, 64], strides = [64, 1] : !cuda_tile.tensor_view<128x64xf32, strides = [64, 1]>
    // CHECK: %[[GS:.+]] = make_gather_scatter_view
    %gs = cuda_tile.make_gather_scatter_view %tv : gather_scatter_view<tile=(128x64), tensor_view<128x64xf32, strides=[64,1]>, sparse_dim=0>
    return
  }
}
