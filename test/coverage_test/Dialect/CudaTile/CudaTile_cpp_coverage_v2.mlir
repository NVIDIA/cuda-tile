// RUN: cuda-tile-opt %s -split-input-file | FileCheck %s

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers ForOp unsigned keyword parse/print roundtrip

// ForOp: unsigned keyword roundtrip
cuda_tile.module @unsigned_for_loop {
  // CHECK-LABEL: entry @test_unsigned_for
  entry @test_unsigned_for(%lb : !cuda_tile.tile<i32>, %ub : !cuda_tile.tile<i32>, %step : !cuda_tile.tile<i32>) {
    // CHECK: for unsigned
    cuda_tile.for unsigned %iv in (%lb to %ub, step %step) : tile<i32> {
      cuda_tile.continue
    }
    return
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers IfOp verify with else region
// Covers IfOp with results roundtrip

// IfOp: with else and results
cuda_tile.module @if_with_results {
  // CHECK-LABEL: entry @test_if_with_results
  entry @test_if_with_results(%cond : !cuda_tile.tile<i1>) {
    %ci = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %cf = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: %{{.*}} = if
    %r = cuda_tile.if %cond -> (tile<i32>) {
      cuda_tile.yield %ci : tile<i32>
    } else {
      cuda_tile.yield %cf : tile<i32>
    }
    return
  }
}

// -----

// IfOp: no results, no else
cuda_tile.module @if_no_results {
  // CHECK-LABEL: entry @test_if_no_results
  entry @test_if_no_results(%cond : !cuda_tile.tile<i1>) {
    // CHECK: if
    cuda_tile.if %cond {
      cuda_tile.yield
    }
    return
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Rounding mode parse/print paths for various ops

// IEEE rounding modes for addf/subf/mulf (lines ~966-984)
cuda_tile.module @ieee_rounding {
  // CHECK-LABEL: entry @test_ieee_rounding
  entry @test_ieee_rounding() {
    %a = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<4xf32>
    %b = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<4xf32>
    // Default nearest_even not printed - check this first
    // CHECK: %{{.*}} = addf %{{.*}}, %{{.*}} : tile<4xf32>
    %3 = cuda_tile.addf %a, %b : tile<4xf32>
    // CHECK: addf {{.*}} rounding<zero>
    %0 = cuda_tile.addf %a, %b rounding<zero> : tile<4xf32>
    // CHECK: addf {{.*}} rounding<negative_inf>
    %1 = cuda_tile.addf %a, %b rounding<negative_inf> : tile<4xf32>
    // CHECK: addf {{.*}} rounding<positive_inf>
    %2 = cuda_tile.addf %a, %b rounding<positive_inf> : tile<4xf32>
    return
  }
}

// -----

// DivF with various rounding modes (lines ~826-836)
cuda_tile.module @divf_rounding {
  // CHECK-LABEL: entry @test_divf_rounding
  entry @test_divf_rounding() {
    %a = cuda_tile.constant <f32: 10.0> : !cuda_tile.tile<4xf32>
    %b = cuda_tile.constant <f32: 3.0> : !cuda_tile.tile<4xf32>
    // CHECK: divf {{.*}} rounding<zero>
    %0 = cuda_tile.divf %a, %b rounding<zero> : tile<4xf32>
    // CHECK: divf {{.*}} rounding<full>
    %1 = cuda_tile.divf %a, %b rounding<full> : tile<4xf32>
    return
  }
}

// -----

// SqrtOp rounding modes (lines ~838-848)
cuda_tile.module @sqrt_rounding {
  // CHECK-LABEL: entry @test_sqrt_rounding
  entry @test_sqrt_rounding() {
    %a = cuda_tile.constant <f32: 4.0> : !cuda_tile.tile<4xf32>
    // CHECK: sqrt {{.*}} rounding<approx>
    %0 = cuda_tile.sqrt %a rounding<approx> : tile<4xf32>
    // CHECK: sqrt {{.*}} rounding<zero>
    %1 = cuda_tile.sqrt %a rounding<zero> : tile<4xf32>
    return
  }
}

// -----

// TanH rounding modes (lines ~850-864)
cuda_tile.module @tanh_rounding {
  // CHECK-LABEL: entry @test_tanh_rounding
  entry @test_tanh_rounding() {
    %a = cuda_tile.constant <f32: 0.5> : !cuda_tile.tile<4xf32>
    // CHECK: tanh {{.*}} rounding<approx>
    %1 = cuda_tile.tanh %a rounding<approx> : tile<4xf32>
    // Default is full for tanh, so full is not printed
    // CHECK: tanh %{{.*}} : tile<4xf32>
    %0 = cuda_tile.tanh %a : tile<4xf32>
    return
  }
}

// -----

// Exp rounding modes (lines ~866-880)
cuda_tile.module @exp_rounding {
  // CHECK-LABEL: entry @test_exp_rounding
  entry @test_exp_rounding() {
    %a = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<4xf32>
    // CHECK: exp {{.*}} rounding<approx>
    %1 = cuda_tile.exp %a rounding<approx> : tile<4xf32>
    // Default is full for exp, so full is not printed
    // CHECK: exp %{{.*}} : tile<4xf32>
    %0 = cuda_tile.exp %a : tile<4xf32>
    return
  }
}

// -----

// FToF with f8E8M0FNU target using valid rounding modes (lines ~2814-2820)
cuda_tile.module @ftof_e8m0_rounding {
  // CHECK-LABEL: entry @test_ftof_e8m0
  entry @test_ftof_e8m0() {
    %a = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<4xf32>
    // CHECK: ftof {{.*}} rounding<zero>
    %0 = cuda_tile.ftof %a rounding<zero> : tile<4xf32> -> tile<4xf8E8M0FNU>
    // CHECK: ftof {{.*}} rounding<positive_inf>
    %1 = cuda_tile.ftof %a rounding<positive_inf> : tile<4xf32> -> tile<4xf8E8M0FNU>
    return
  }
}

// -----

// DivI with valid rounding modes and signedness (lines ~2130-2143)
cuda_tile.module @divi_rounding {
  // CHECK-LABEL: entry @test_divi_rounding
  entry @test_divi_rounding() {
    %a = cuda_tile.constant <i32: 10> : !cuda_tile.tile<4xi32>
    %b = cuda_tile.constant <i32: 3> : !cuda_tile.tile<4xi32>
    // Default rounding<zero> is not printed for divi
    // CHECK: divi {{.*}} signed : tile<4xi32>
    %0 = cuda_tile.divi %a, %b signed rounding<zero> : tile<4xi32>
    // CHECK: divi {{.*}} unsigned rounding<positive_inf>
    %1 = cuda_tile.divi %a, %b unsigned rounding<positive_inf> : tile<4xi32>
    // CHECK: divi {{.*}} signed rounding<negative_inf>
    %2 = cuda_tile.divi %a, %b signed rounding<negative_inf> : tile<4xi32>
    return
  }
}

// -----

// ForOp with iter_values (lines ~2745-2762)
cuda_tile.module @for_iter_values {
  // CHECK-LABEL: @test_for_iter_values
  testing$func @test_for_iter_values() -> !cuda_tile.tile<i32> {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c10 = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>
    // CHECK: for %{{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}}) : tile<i32> iter_values
    %result = cuda_tile.for %i in (%c0 to %c10, step %c1) : tile<i32>
      iter_values(%acc = %c0) -> (tile<i32>) {
      %sum = cuda_tile.addi %acc, %i : tile<i32>
      cuda_tile.continue %sum : tile<i32>
    }
    return %result : tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MakeTensorViewOp parse/print roundtrip with dynamic shapes and strides
// Covers MakeTensorViewOp::getAsmResultNames()

// MakeTensorViewOp: fully static shape and strides
cuda_tile.module @tview_static {
  // CHECK-LABEL: entry @test_tview_static
  entry @test_tview_static(%base: !cuda_tile.tile<ptr<f32>>) {
    // CHECK: %[[TV:.+]] = make_tensor_view %{{.*}}, shape = [128, 64], strides = [64, 1] : tensor_view<128x64xf32, strides=[64,1]>
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1] : !cuda_tile.tensor_view<128x64xf32, strides = [64, 1]>
    return
  }
}

// -----

// MakeTensorViewOp: dynamic shapes
cuda_tile.module @tview_dynamic_shapes {
  // CHECK-LABEL: entry @test_tview_dynamic_shapes
  entry @test_tview_dynamic_shapes(%base: !cuda_tile.tile<ptr<f32>>, %d0: !cuda_tile.tile<i32>, %d1: !cuda_tile.tile<i32>) {
    // CHECK: make_tensor_view
    %tv = cuda_tile.make_tensor_view %base, shape = [%d0, %d1], strides = [1, 1] : tile<i32> -> !cuda_tile.tensor_view<?x?xf32, strides = [1, 1]>
    return
  }
}

