// RUN: cuda-tile-opt %s --pass-pipeline='builtin.module(cuda_tile.module(cuda_tile.testing$func(fuse-fma)))' --split-input-file | FileCheck %s

// Basic multiply-add fusion (x * y + z)
// CHECK-LABEL: testing$func @test_mul_add_fusion
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_add_fusion() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Basic multiply-add fusion (x * y + z)
// CHECK-LABEL: testing$func @test_mul_add_fusion
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_add_fusion() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Basic multiply-add fusion (x * y + z)
// CHECK-LABEL: testing$func @test_mul_add_fusion
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_add_fusion() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Multiply-add fusion with broadcast (x * y + bcast(z))
// CHECK-LABEL: testing$func @test_mul_add_bcast_fusion
// CHECK: reshape
// CHECK: broadcast
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<2x2xf32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_add_bcast_fusion() -> !cuda_tile.tile<2x2xf32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<2x2xf32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<2x2xf32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<2x2xf32>
    %4 = cuda_tile.reshape %2 : !cuda_tile.tile<f32> -> !cuda_tile.tile<1x1xf32>
    %5 = cuda_tile.broadcast %4 : !cuda_tile.tile<1x1xf32> -> !cuda_tile.tile<2x2xf32>
    %6 = cuda_tile.addf %3, %5 rounding<nearest_even> : !cuda_tile.tile<2x2xf32>
    
    return %6 : !cuda_tile.tile<2x2xf32>
  }
}


// -----

// Multiply-add fusion with no-op broadcast (x * y + bcast(z))
// CHECK-LABEL: testing$func @test_mul_add_noop_bcast_fusion
// CHECK: reshape
// CHECK: broadcast
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<1x1xf32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_add_noop_bcast_fusion() -> !cuda_tile.tile<1x1xf32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<1x1xf32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<1x1xf32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<1x1xf32>
    %4 = cuda_tile.reshape %2 : !cuda_tile.tile<f32> -> !cuda_tile.tile<1x1xf32>
    %5 = cuda_tile.broadcast %4 : !cuda_tile.tile<1x1xf32> -> !cuda_tile.tile<1x1xf32>
    %6 = cuda_tile.addf %3, %5 rounding<nearest_even> : !cuda_tile.tile<1x1xf32>
    
    return %6 : !cuda_tile.tile<1x1xf32>
  }
}

// -----

// Basic multiply-subtract fusion (x * y - z)
// CHECK-LABEL: testing$func @test_mul_sub_fusion
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: subf

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_sub_fusion() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %4 = cuda_tile.subf %3, %2 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Multiply-subtract fusion with no-op broadcast (x * y - bcast(z))
// CHECK-LABEL: testing$func @test_mul_sub_noop_bcast_fusion
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: subf
// CHECK-NOT: broadcast

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_sub_noop_bcast_fusion() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %4 = cuda_tile.broadcast %2 : !cuda_tile.tile<f32> -> !cuda_tile.tile<f32>
    %5 = cuda_tile.subf %3, %4 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %5 : !cuda_tile.tile<f32>
  }
}

// -----

// Multiply-subtract fusion with broadcast (x * y - bcast(z))
// CHECK-LABEL: testing$func @test_mul_sub_bcast_fusion
// CHECK: reshape
// CHECK: broadcast
// CHECK: negf
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<2x2xf32>
// CHECK-NOT: mulf
// CHECK-NOT: subf

cuda_tile.module @test {
  cuda_tile.testing$func @test_mul_sub_bcast_fusion() -> !cuda_tile.tile<2x2xf32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<2x2xf32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<2x2xf32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<2x2xf32>
    %4 = cuda_tile.reshape %2 : !cuda_tile.tile<f32> -> !cuda_tile.tile<1x1xf32>
    %5 = cuda_tile.broadcast %4 : !cuda_tile.tile<1x1xf32> -> !cuda_tile.tile<2x2xf32>
    %6 = cuda_tile.subf %3, %5 rounding<nearest_even> : !cuda_tile.tile<2x2xf32>
    
    return %6 : !cuda_tile.tile<2x2xf32>
  }
}

// -----

// Different rounding modes (should not fuse)
// CHECK-LABEL: testing$func @test_different_rounding
// CHECK: mulf
// CHECK: addf
// CHECK-NOT: fma

cuda_tile.module @test {
  cuda_tile.testing$func @test_different_rounding() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 rounding<zero> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Flush to zero enabled
// CHECK-LABEL: testing$func @test_ftz_enabled
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} flush_to_zero : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_ftz_enabled() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Different flush-to-zero settings (should not fuse)
// CHECK-LABEL: testing$func @test_different_ftz
// CHECK: mulf
// CHECK: addf
// CHECK-NOT: fma

cuda_tile.module @test {
  cuda_tile.testing$func @test_different_ftz() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Both rounding mode and flush-to-zero
// CHECK-LABEL: testing$func @test_rounding_and_ftz
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} rounding<zero> flush_to_zero : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_rounding_and_ftz() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<zero> flush_to_zero : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 rounding<zero> flush_to_zero : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Mismatch in both rounding mode and flush-to-zero (should not fuse)
// CHECK-LABEL: testing$func @test_mismatch_both
// CHECK: mulf
// CHECK: addf
// CHECK-NOT: fma

cuda_tile.module @test {
  cuda_tile.testing$func @test_mismatch_both() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<f32>
    %4 = cuda_tile.addf %3, %2 rounding<zero> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Multiple uses of multiply result (should not fuse)
// CHECK-LABEL: testing$func @test_multiple_uses
// CHECK: mulf
// CHECK: addf
// CHECK: subf
// CHECK-NOT: fma

cuda_tile.module @test {
  cuda_tile.testing$func @test_multiple_uses() -> (!cuda_tile.tile<f32>, !cuda_tile.tile<f32>) {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    %3 = constant <f32: 5.0> : !cuda_tile.tile<f32>
    
    %4 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %5 = cuda_tile.addf %4, %2 rounding<nearest_even> : !cuda_tile.tile<f32>
    %6 = cuda_tile.subf %4, %3 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %5, %6 : !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
  }
}

// -----

// Commutative add with multiply on RHS (z + x * y) -> should canonicalize and fuse
// The canonicalize pass should reorder operands, then FMA fusion should occur
// CHECK-LABEL: testing$func @test_commutative_add_mul_rhs
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_commutative_add_mul_rhs() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    // This should be canonicalized to put %3 on LHS, then fused into FMA
    %4 = cuda_tile.addf %2, %3 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Commutative add with no-op broadcast and multiply on RHS (bcast(z) + x * y)
// CHECK-LABEL: testing$func @test_commutative_add_bcast_mul_rhs
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf
// CHECK-NOT: broadcast

cuda_tile.module @test {
  cuda_tile.testing$func @test_commutative_add_bcast_mul_rhs() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %4 = cuda_tile.broadcast %2 : !cuda_tile.tile<f32> -> !cuda_tile.tile<f32>
    // This should be canonicalized to put %3 on LHS, then fused into FMA
    %5 = cuda_tile.addf %4, %3 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %5 : !cuda_tile.tile<f32>
  }
}

// -----

// Commutative add with different rounding modes (should canonicalize but not fuse)
// CHECK-LABEL: testing$func @test_commutative_different_rounding
// CHECK: addf %[[MUL:.*]], %{{.*}} rounding<zero>
// CHECK-NOT: fma

cuda_tile.module @test {
  cuda_tile.testing$func @test_commutative_different_rounding() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    // This should be canonicalized to put %3 on LHS, but not fused due to different rounding
    %4 = cuda_tile.addf %2, %3 rounding<zero> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Commutative add with flush-to-zero mismatch (should canonicalize but not fuse)
// CHECK-LABEL: testing$func @test_commutative_ftz_mismatch
// CHECK: addf %[[MUL:.*]], %{{.*}}
// CHECK-NOT: fma

cuda_tile.module @test {
  cuda_tile.testing$func @test_commutative_ftz_mismatch() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<f32>
    // This should be canonicalized to put %3 on LHS, but not fused due to FTZ mismatch
    %4 = cuda_tile.addf %2, %3 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %4 : !cuda_tile.tile<f32>
  }
}

// -----

// Chained operations with commutative pattern
// CHECK-LABEL: testing$func @test_chained_commutative
// CHECK: %[[FMA1:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK: %[[FMA2:.*]] = fma %{{.*}}, %{{.*}}, %[[FMA1]] : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf

cuda_tile.module @test {
  cuda_tile.testing$func @test_chained_commutative() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    %3 = constant <f32: 5.0> : !cuda_tile.tile<f32>
    %4 = constant <f32: 6.0> : !cuda_tile.tile<f32>
    
    %5 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %6 = cuda_tile.mulf %2, %3 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    // First: canonicalize and fuse z + (x * y) -> FMA(x, y, z)
    %7 = cuda_tile.addf %4, %5 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    // Second: canonicalize and fuse result + (a * b) -> FMA(a, b, result)
    %8 = cuda_tile.addf %7, %6 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %8 : !cuda_tile.tile<f32>
  }
}

// -----

// Commutative add with no-op broadcast and multiply on RHS (bcast(z) + x * y)
// CHECK-LABEL: testing$func @test_commutative_add_bcast_mul_rhs
// CHECK: %[[RESULT:.*]] = fma %{{.*}}, %{{.*}}, %{{.*}} : tile<f32>
// CHECK-NOT: mulf
// CHECK-NOT: addf
// CHECK-NOT: broadcast

cuda_tile.module @test {
  cuda_tile.testing$func @test_commutative_add_bcast_mul_rhs() -> !cuda_tile.tile<f32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<f32>
    %1 = constant <f32: 3.0> : !cuda_tile.tile<f32>
    %2 = constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %3 = cuda_tile.mulf %0, %1 rounding<nearest_even> : !cuda_tile.tile<f32>
    %4 = cuda_tile.broadcast %2 : !cuda_tile.tile<f32> -> !cuda_tile.tile<f32>
    // This should be canonicalized to put %3 on LHS, then fused into FMA
    %5 = cuda_tile.addf %4, %3 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %5 : !cuda_tile.tile<f32>
  }
}
