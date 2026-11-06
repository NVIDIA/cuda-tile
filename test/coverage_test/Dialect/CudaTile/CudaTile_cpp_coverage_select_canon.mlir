// RUN: cuda-tile-opt %s --canonicalize --split-input-file | FileCheck %s

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// SelectOp canonicalization: SelectConsts, SelectToExtI, fold

// SelectConsts with float operands (lines ~5455-5464)
// CHECK-LABEL: @test_select_const_float
cuda_tile.module @test {
  testing$func @test_select_const_float() -> !cuda_tile.tile<4xf32> {
    %cond = cuda_tile.constant <i1: [true, false, true, false]> : !cuda_tile.tile<4xi1>
    %a = cuda_tile.constant <f32: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf32>
    %b = cuda_tile.constant <f32: [5.0, 6.0, 7.0, 8.0]> : !cuda_tile.tile<4xf32>
    // CHECK: constant
    // CHECK-NOT: select
    %0 = cuda_tile.select %cond, %a, %b : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xf32>
    return %0 : tile<4xf32>
  }
}

// -----

// SelectConsts with integer operands (lines ~5446-5454)
// CHECK-LABEL: @test_select_const_int
cuda_tile.module @test {
  testing$func @test_select_const_int() -> !cuda_tile.tile<4xi32> {
    %cond = cuda_tile.constant <i1: [true, false, true, false]> : !cuda_tile.tile<4xi1>
    %a = cuda_tile.constant <i32: [10, 20, 30, 40]> : !cuda_tile.tile<4xi32>
    %b = cuda_tile.constant <i32: [50, 60, 70, 80]> : !cuda_tile.tile<4xi32>
    // CHECK: constant
    // CHECK-NOT: select
    %0 = cuda_tile.select %cond, %a, %b : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi32>
    return %0 : tile<4xi32>
  }
}

// -----

// SelectToExtI: select %x, c1, c0 => extui %arg (lines ~5483-5487)
// CHECK-LABEL: @test_select_to_exti
cuda_tile.module @test {
  testing$func @test_select_to_exti(%cond: !cuda_tile.tile<4xi1>) -> !cuda_tile.tile<4xi32> {
    %ones = cuda_tile.constant <i32: 1> : !cuda_tile.tile<4xi32>
    %zeros = cuda_tile.constant <i32: 0> : !cuda_tile.tile<4xi32>
    // CHECK: exti %{{.*}} unsigned
    // CHECK-NOT: select
    %0 = cuda_tile.select %cond, %ones, %zeros : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi32>
    return %0 : tile<4xi32>
  }
}

// -----

// SelectToExtI inverse: select %x, c0, c1 => extui (xor %arg, true) (lines ~5492-5497)
// CHECK-LABEL: @test_select_to_exti_inverse
cuda_tile.module @test {
  testing$func @test_select_to_exti_inverse(%cond: !cuda_tile.tile<4xi1>) -> !cuda_tile.tile<4xi32> {
    %ones = cuda_tile.constant <i32: 1> : !cuda_tile.tile<4xi32>
    %zeros = cuda_tile.constant <i32: 0> : !cuda_tile.tile<4xi32>
    // CHECK: xori
    // CHECK: exti %{{.*}} unsigned
    // CHECK-NOT: select
    %0 = cuda_tile.select %cond, %zeros, %ones : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi32>
    return %0 : tile<4xi32>
  }
}

// -----

// Select fold: select c, x, x => x (lines ~5509-5515)
// CHECK-LABEL: @test_select_same_operands
cuda_tile.module @test {
  testing$func @test_select_same_operands(%cond: !cuda_tile.tile<4xi1>, %v: !cuda_tile.tile<4xi32>) -> !cuda_tile.tile<4xi32> {
    // CHECK-NOT: select
    // CHECK: return %{{.*}}
    %0 = cuda_tile.select %cond, %v, %v : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi32>
    return %0 : tile<4xi32>
  }
}

// -----

// Select fold: select true, x, y => x (lines ~5520-5526)
// CHECK-LABEL: @test_select_true_cond
cuda_tile.module @test {
  testing$func @test_select_true_cond(%a: !cuda_tile.tile<4xi32>, %b: !cuda_tile.tile<4xi32>) -> !cuda_tile.tile<4xi32> {
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<4xi1>
    // CHECK-NOT: select
    %0 = cuda_tile.select %true, %a, %b : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi32>
    return %0 : tile<4xi32>
  }
}

// -----

// Select fold: select false, x, y => y (lines ~5524-5526)
// CHECK-LABEL: @test_select_false_cond
cuda_tile.module @test {
  testing$func @test_select_false_cond(%a: !cuda_tile.tile<4xi32>, %b: !cuda_tile.tile<4xi32>) -> !cuda_tile.tile<4xi32> {
    %false = cuda_tile.constant <i1: 0> : !cuda_tile.tile<4xi1>
    // CHECK-NOT: select
    %0 = cuda_tile.select %false, %a, %b : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi32>
    return %0 : tile<4xi32>
  }
}

// -----

// Select bool identity: select c, true, false => c (lines ~5534-5545)
// CHECK-LABEL: @test_select_bool_identity
cuda_tile.module @test {
  testing$func @test_select_bool_identity(%cond: !cuda_tile.tile<4xi1>) -> !cuda_tile.tile<4xi1> {
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<4xi1>
    %false = cuda_tile.constant <i1: 0> : !cuda_tile.tile<4xi1>
    // CHECK-NOT: select
    %0 = cuda_tile.select %cond, %true, %false : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi1>
    return %0 : tile<4xi1>
  }
}
