// RUN: cuda-tile-opt %s --canonicalize --split-input-file | FileCheck %s

// ==== AddFOp Canonicalization ====
// Test canonicalization of AddFOp operations to put multiply on LHS
// This enables better FMA fusion patterns

// CHECK-LABEL: @test_reorder_bcast_add_mul
cuda_tile.module @test {
  testing$func @test_reorder_bcast_add_mul() -> !cuda_tile.tile<f32> {
    %a = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    %b = cuda_tile.constant <f32: 3.0> : !cuda_tile.tile<f32>
    %c = cuda_tile.constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %bcast_c = cuda_tile.broadcast %c : !cuda_tile.tile<f32> -> !cuda_tile.tile<f32>
    %mul = cuda_tile.mulf %a, %b rounding<nearest_even> : !cuda_tile.tile<f32>
    
    // This should be canonicalized to put %mul on the left
    // CHECK: %[[RESULT:.*]] = addf %[[MUL:.*]], %[[BCAST:.*]] : tile<f32>
    // CHECK-NOT: addf %[[BCAST:.*]], %[[MUL:.*]]
    %result = cuda_tile.addf %bcast_c, %mul rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %result : !cuda_tile.tile<f32>
  }
}

// -----

// CHECK-LABEL: @test_reorder_bcast_add_mul
cuda_tile.module @test {
  testing$func @test_reorder_bcast_add_mul_implicit_rounding() -> !cuda_tile.tile<f32> {
    %a = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    %b = cuda_tile.constant <f32: 3.0> : !cuda_tile.tile<f32>
    %c = cuda_tile.constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %bcast_c = cuda_tile.broadcast %c : !cuda_tile.tile<f32> -> !cuda_tile.tile<f32>
    %mul = cuda_tile.mulf %a, %b : !cuda_tile.tile<f32>
    
    // This should be canonicalized to put %mul on the left
    // CHECK: %[[RESULT:.*]] = addf %[[MUL:.*]], %[[BCAST:.*]] : tile<f32>
    // CHECK-NOT: addf %[[BCAST:.*]], %[[MUL:.*]]
    %result = cuda_tile.addf %bcast_c, %mul : !cuda_tile.tile<f32>
    
    return %result : !cuda_tile.tile<f32>
  }
}
// -----

// CHECK-LABEL: @test_reorder_scalar_add_mul
cuda_tile.module @test {
  testing$func @test_reorder_scalar_add_mul() -> !cuda_tile.tile<f32> {
    %a = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    %b = cuda_tile.constant <f32: 3.0> : !cuda_tile.tile<f32>
    %c = cuda_tile.constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %mul = cuda_tile.mulf %a, %b rounding<nearest_even> : !cuda_tile.tile<f32>
    
    // This should be canonicalized to put %mul on the left
    // CHECK: %[[RESULT:.*]] = addf %[[MUL:.*]], %[[C:.*]] : tile<f32>
    // CHECK-NOT: addf %[[C:.*]], %[[MUL:.*]]
    %result = cuda_tile.addf %c, %mul rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %result : !cuda_tile.tile<f32>
  }
}

// -----

// CHECK-LABEL: @test_no_reorder_mul_already_lhs
cuda_tile.module @test {
  testing$func @test_no_reorder_mul_already_lhs() -> !cuda_tile.tile<f32> {
    %a = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    %b = cuda_tile.constant <f32: 3.0> : !cuda_tile.tile<f32>
    %c = cuda_tile.constant <f32: 4.0> : !cuda_tile.tile<f32>
    
    %mul = cuda_tile.mulf %a, %b rounding<nearest_even> : !cuda_tile.tile<f32>
    
    // This should NOT be reordered since mul is already on LHS
    // CHECK: %[[RESULT:.*]] = addf %[[MUL:.*]], %[[C:.*]] : tile<f32>
    %result = cuda_tile.addf %mul, %c rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %result : !cuda_tile.tile<f32>
  }
}

// -----

// CHECK-LABEL: @test_no_reorder_both_mul
cuda_tile.module @test {
  testing$func @test_no_reorder_both_mul() -> !cuda_tile.tile<f32> {
    %a = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    %b = cuda_tile.constant <f32: 3.0> : !cuda_tile.tile<f32>
    %c = cuda_tile.constant <f32: 4.0> : !cuda_tile.tile<f32>
    %d = cuda_tile.constant <f32: 5.0> : !cuda_tile.tile<f32>
    
    %mul1 = cuda_tile.mulf %a, %b rounding<nearest_even> : !cuda_tile.tile<f32>
    %mul2 = cuda_tile.mulf %c, %d rounding<nearest_even> : !cuda_tile.tile<f32>
    
    // This should NOT be reordered since both operands are multiply operations
    // CHECK: %[[RESULT:.*]] = addf %[[MUL1:.*]], %[[MUL2:.*]] : tile<f32>
    %result = cuda_tile.addf %mul1, %mul2 rounding<nearest_even> : !cuda_tile.tile<f32>
    
    return %result : !cuda_tile.tile<f32>
  }
}

// -----
// Canonicalization of IfOp with static condition
// CHECK-LABEL: @test_if_static_cond
cuda_tile.module @test {
  testing$func @test_if_static_cond() -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK-NOT: if
    // CHECK: %[[RESULT:.*]] = addi %[[R0]], %[[R2]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %1 = if %true -> (tile<i32>) {
      yield %a : tile<i32>
    } else {
      yield %b : tile<i32>
    }
    %2 = addi %1, %c : tile<i32>
    return %2 : tile<i32>
  }
}

// -----
// Canonicalization of IfOp with static condition & return instead of yield
// CHECK-LABEL: @test_if_static_cond_return
cuda_tile.module @test {
  testing$func @test_if_static_cond_return() -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK-NOT: if
    // CHECK-NOT: addi
    // CHECK: return %[[R0]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %1 = if %true -> (tile<i32>) {
      return %a : tile<i32>
    } else {
      yield %b : tile<i32>
    }
    %2 = addi %1, %c : tile<i32>
    return %2 : tile<i32>
  }
}

// -----
// Canonicalization of IfOp with static condition & continue instead of yield
// CHECK-LABEL: @test_if_static_cond_continue
cuda_tile.module @test {
  testing$func @test_if_static_cond_continue() -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[FOR:.*]] = for {{.*}}
    // CHECK-NOT: if
    // CHECK-NOT: add
    // CHECK: continue %[[R0]]
    // CHECK: return %[[FOR]]
    %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
    %0 = constant <i64: 128> : !cuda_tile.tile<i64>
    %1 = constant <i64: 0> : !cuda_tile.tile<i64>
    %2 = constant <i64: 1> : !cuda_tile.tile<i64>
    %3 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%4 = %c1) -> (tile<i32>) {
      %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
      %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
      %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
      %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
      %5 = if %true -> (tile<i32>) {
        continue %a : tile<i32>
      } else {
        yield %b : tile<i32>
      }
      %6 = addi %5, %c : tile<i32>
      continue %6 : tile<i32>
    }
    return %3 : tile<i32>
  }
}

// -----
// Canonicalization of IfOp with static condition & break instead of yield
// CHECK-LABEL: @test_if_static_cond_break
cuda_tile.module @test {
  testing$func @test_if_static_cond_break() -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[LOOP:.*]] = loop {{.*}}
    // CHECK-NOT: if
    // CHECK-NOT: add
    // CHECK: break %[[R0]]
    // CHECK: return %[[LOOP]]
    %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
    %0 = loop iter_values(%4 = %c1) : tile<i32> -> tile<i32> {
      %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
      %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
      %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
      %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
      %5 = if %true -> (tile<i32>) {
        break %a : tile<i32>
      } else {
        yield %b : tile<i32>
      }
      %6 = addi %5, %c : tile<i32>
      continue %6 : tile<i32>
    }
    return %0 : tile<i32>
  }
}

// -----
// Canonicalization of Trivial IfOp - conversion to SelectOp
// CHECK-LABEL: @test_if_select
cuda_tile.module @test {
  testing$func @test_if_select(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK-NOT: if
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R2]]
    // CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[R0]], %[[R1]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %c, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i32>) {
      yield %a : tile<i32>
    } else {
      yield %b : tile<i32>
    }
    %2 = addi %1, %c : tile<i32>
    return %2 : tile<i32>
  }
}
// -----
// Canonicalization of Trivial IfOp - conversion to SelectOp in the case of multiple yield arguments
// Only one is converted, as another is unsupported, as defined within then-block
// CHECK-LABEL: @test_if_select_many
cuda_tile.module @test {
  testing$func @test_if_select_many(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R2]]
    // CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[R0]], %[[R1]]
    // CHECK: %[[IF:.*]] = if %[[CMP]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %c, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1, %2 = if %cond -> (tile<i32>, tile<i32>) {
      %add = addi %b, %arg1 : tile<i32>
      yield %a, %add : tile<i32>, tile<i32>
    } else {
      yield %b, %a : tile<i32>, tile<i32>
    }
    %3 = addi %1, %2 : tile<i32>
    return %3 : tile<i32>
  }
}
// -----
// Canonicalization of Trivial IfOp - conversion of all YieldOp arguments to multiple SelectOps
// CHECK-LABEL: @test_if_select_all
cuda_tile.module @test {
  testing$func @test_if_select_all(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R2]]
    // CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[R0]], %[[R1]]
    // CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[R1]], %[[R0]]
    // CHECK-NOT: if
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %c, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1, %2 = if %cond -> (tile<i32>, tile<i32>) {
      yield %a, %b : tile<i32>, tile<i32>
    } else {
      yield %b, %a : tile<i32>, tile<i32>
    }
    %3 = addi %1, %2 : tile<i32>
    return %3 : tile<i32>
  }
}
// -----
// Folding of the following sequence "%inv = XorIOp %cond, 1", "if %inv"
// CHECK-LABEL: @test_if_fold
cuda_tile.module @test {
  testing$func @test_if_fold(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R2]]
    // CHECK-NOT: xori
    // CHECK: %{{.*}} = if %[[CMP]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %cond = cmpi equal %arg1, %c, signed : !cuda_tile.tile<i32> -> tile<i1>
    %inv = xori %cond, %c1 : tile<i1>
    %1 = if %inv -> (tile<i32>) {
      %3 = addi %a, %arg1 : tile<i32>
      yield %3 : tile<i32>
    } else {
      yield %b : tile<i32>
    }
    %2 = addi %1, %c : tile<i32>
    return %2 : tile<i32>
  }
}

// -----
// Canonicalization of IfOp with Yield of values defined outside of then-block
// & ReturnOp inside the else-block.
// When return doesn't happen we always yield the same values, SelectOp is not needed
// CHECK-LABEL: @test_if_yield_return
cuda_tile.module @test {
  testing$func @test_if_yield_return(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R2]]
    // CHECK: if %[[CMP]]
    // CHECK-NOT: yield
    // CHECK: return %[[R2]]
    // CHECK %[[RESULT:.*]] = addi %[[R0]], %[[R1]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %c, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1, %2 = if %cond -> (tile<i32>, tile<i32>) {
      yield %a, %b : tile<i32>, tile<i32>
    } else {
      return %c : tile<i32>
    }
    %3 = addi %1, %2 : tile<i32>
    return %3 : tile<i32>
  }
}

// -----
// Canonicalization of IfOp with Yield of values defined outside of else-block
// & ReturnOp inside the then-block.
// When return doesn't happen we always yield the same values, SelectOp is not needed
// Difference from above is that else-block will be empty and should be deleted
// CHECK-LABEL: @test_if_return_yield
cuda_tile.module @test {
  testing$func @test_if_return_yield(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R2]]
    // CHECK: if %[[CMP]]
    // CHECK: return %[[R2]]
    // CHECK-NOT: else
    // CHECK-NOT: yield
    // CHECK %[[RESULT:.*]] = addi %[[R0]], %[[R1]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %c, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1, %2 = if %cond -> (tile<i32>, tile<i32>) {
      return %c : tile<i32>
    } else {
      yield %a, %b : tile<i32>, tile<i32>
    }
    %3 = addi %1, %2 : tile<i32>
    return %3 : tile<i32>
  }
}

// -----
// Canonicalization of IfOp with True/False result
// CHECK-LABEL: @test_if_yield
cuda_tile.module @test {
  testing$func @test_if_yield(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i1> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK-NOT: if
    // CHECK-NOT: else
    // CHECK-NOT: yield
    // CHECK return %[[CMP]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i1>) {
      %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
      yield %true : tile<i1>
    } else {
      %false = cuda_tile.constant <i1: 0> : !cuda_tile.tile<i1>
      yield %false : tile<i1>
    }
    return %1 : tile<i1>
  }
}

// -----
// Canonicalization of IfOp with False/True result
// CHECK-LABEL: @test_if_yield_xor
cuda_tile.module @test {
  testing$func @test_if_yield_xor(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i1> {
    // CHECK: %[[TRUE:.*]] = constant <i1: true>
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[RESULT:.*]] = xori %[[CMP]], %[[TRUE]]
    // CHECK-NOT: if
    // CHECK-NOT: else
    // CHECK-NOT: yield
    // CHECK return %[[RESULT]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i1>) {
      %false = cuda_tile.constant <i1: 0> : !cuda_tile.tile<i1>
      yield %false : tile<i1>
    } else {
      %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
      yield %true : tile<i1>
    }
    return %1 : tile<i1>
  }
}

// -----
// Canonicalization of two IfOps with same predicate
// CHECK-LABEL: @test_if_merge
cuda_tile.module @test {
  testing$func @test_if_merge(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[RES:[^:]+]]:2 = if %[[CMP]]
    // CHECK-NOT: if
    // CHECK: %[[RESULT:.*]] = addi %[[RES]]#0, %[[RES]]#1
    // CHECK return %[[RESULT]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i32>) {
      %2 = addi %arg1, %b : tile<i32>
      yield %2 : tile<i32>
    } else {
      %2 = addi %arg1, %c : tile<i32>
      yield %2 : tile<i32>
    }
    %3 = if %cond -> (tile<i32>) {
      %4 = addi %1, %c : tile<i32>
      yield %4 : tile<i32>
    } else {
      %4 = addi %1, %b : tile<i32>
      yield %4 : tile<i32>
    }
    %5 = addi %1, %3 : tile<i32>
    return %5 : tile<i32>
  }
}

// -----
// Canonicalization of two IfOps with same predicate
// CHECK-LABEL: @test_if_merge_then_return_first
cuda_tile.module @test {
  testing$func @test_if_merge_then_return_first(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[RES:[^:]+]]:2 = if %[[CMP]]
    // CHECK: return
    // CHECK-NEXT: } else {
    // CHECK: %[[RESULT:.*]] = addi %[[RES]]#0, %[[RES]]#1
    // CHECK return %[[RESULT]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i32>) {
      %2 = addi %arg1, %b : tile<i32>
      return %2 : tile<i32>
    } else {
      %2 = addi %arg1, %c : tile<i32>
      yield %2 : tile<i32>
    }
    %3 = if %cond -> (tile<i32>) {
      %4 = addi %arg1, %c : tile<i32>
      yield %4 : tile<i32>
    } else {
      %4 = addi %arg1, %b : tile<i32>
      yield %4 : tile<i32>
    }
    %5 = addi %1, %3 : tile<i32>
    return %5 : tile<i32>
  }
}

// -----
// Canonicalization of two IfOps with same predicate
// CHECK-LABEL: @test_if_merge_else_return_first
cuda_tile.module @test {
  testing$func @test_if_merge_else_return_first(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[RES:[^:]+]]:2 = if %[[CMP]]
    // CHECK: } else {
    // CHECK:   return
    // CHECK: %[[RESULT:.*]] = addi %[[RES]]#0, %[[RES]]#1
    // CHECK return %[[RESULT]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i32>) {
      %2 = addi %arg1, %b : tile<i32>
      yield %2 : tile<i32>
    } else {
      %2 = addi %arg1, %c : tile<i32>
      return %2 : tile<i32>
    }
    %3 = if %cond -> (tile<i32>) {
      %4 = addi %arg1, %c : tile<i32>
      yield %4 : tile<i32>
    } else {
      %4 = addi %arg1, %b : tile<i32>
      yield %4 : tile<i32>
    }
    %5 = addi %1, %3 : tile<i32>
    return %5 : tile<i32>
  }
}

// -----
// Canonicalization of two IfOps with same predicate
// CHECK-LABEL: @test_if_merge_then_return_second
cuda_tile.module @test {
  testing$func @test_if_merge_then_return_second(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[RES:[^:]+]]:2 = if %[[CMP]]
    // CHECK:   return
    // CHECK-NEXT: } else {
    // CHECK: %[[RESULT:.*]] = addi %[[RES]]#0, %[[RES]]#1
    // CHECK return %[[RESULT]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i32>) {
      %2 = addi %arg1, %b : tile<i32>
      yield %2 : tile<i32>
    } else {
      %2 = addi %arg1, %c : tile<i32>
      yield %2 : tile<i32>
    }
    %3 = if %cond -> (tile<i32>) {
      %4 = addi %1, %c : tile<i32>
      return %4 : tile<i32>
    } else {
      %4 = addi %1, %b : tile<i32>
      yield %4 : tile<i32>
    }
    %5 = addi %1, %3 : tile<i32>
    return %5 : tile<i32>
  }
}

// -----
// Canonicalization of two IfOps with same predicate
// CHECK-LABEL: @test_if_merge_else_return_second
cuda_tile.module @test {
  testing$func @test_if_merge_else_return_second(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[RES:[^:]+]]:2 = if %[[CMP]]
    // CHECK: } else {
    // CHECK:   return
    // CHECK: %[[RESULT:.*]] = addi %[[RES]]#0, %[[RES]]#1
    // CHECK return %[[RESULT]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %1 = if %cond -> (tile<i32>) {
      %2 = addi %arg1, %b : tile<i32>
      yield %2 : tile<i32>
    } else {
      %2 = addi %arg1, %c : tile<i32>
      yield %2 : tile<i32>
    }
    %3 = if %cond -> (tile<i32>) {
      %4 = addi %arg1, %c : tile<i32>
      yield %4 : tile<i32>
    } else {
      %4 = addi %arg1, %b : tile<i32>
      return %4 : tile<i32>
    }
    %5 = addi %1, %3 : tile<i32>
    return %5 : tile<i32>
  }
}

// -----
// Canonicalization of nested IfOps
// CHECK-LABEL: @test_if_nested
cuda_tile.module @test {
  testing$func @test_if_nested(%arg1 : !cuda_tile.tile<i32>, %arg2 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP1:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[CMP2:.*]] = cmpi equal %{{.*}}, %[[R1]]
    // CHECK: %[[AND:.*]] = andi %[[CMP1]], %[[CMP2]]
    // CHECK: if %[[AND]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond1 = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %cond2 = cmpi equal %arg2, %b, signed : !cuda_tile.tile<i32> -> tile<i1>
    if %cond1 {
      if %cond2 {
        print "%d", %c : tile<i32>
      }
    }
    return %a : tile<i32>
  }
}

// -----
// Canonicalization of nested IfOps
// CHECK-LABEL: @test_if_nested_return
cuda_tile.module @test {
  testing$func @test_if_nested_return(%arg1 : !cuda_tile.tile<i32>, %arg2 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[R2:.*]] = constant <i32: 2>
    // CHECK: %[[CMP1:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: %[[CMP2:.*]] = cmpi equal %{{.*}}, %[[R1]]
    // CHECK: %[[AND:.*]] = andi %[[CMP1]], %[[CMP2]]
    // CHECK: if %[[AND]]
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond1 = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %cond2 = cmpi equal %arg2, %b, signed : !cuda_tile.tile<i32> -> tile<i1>
    if %cond1 {
      if %cond2 {
        print "%d", %c : tile<i32>
        return %b : tile<i32>
      }
    }
    return %a : tile<i32>
  }
}

// -----
// Canonicalization of IfOps with two ReturnOps both in Then-Block & Else-Block
// In this case everything below the IfOp is unreachable,
// So Else-block will be moved to parent & replace everything below IfOp
// CHECK-LABEL: @test_if_both_return
cuda_tile.module @test {
  testing$func @test_if_both_return(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: if %[[CMP]] {
    // CHECK:   return %[[R0]]
    // CHECK-NOT: else
    // CHECK: return %[[R1]]
    // CHECK-NOT: return
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond1 = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    if %cond1 {
      print "%d", %a : tile<i32>
      return %a : tile<i32>
    } else {
      print "%d", %b : tile<i32>
      return %b : tile<i32>
    }
    print "%d", %c : tile<i32>
    return %c : tile<i32>
  }
}

// -----
// Canonicalization of IfOps with two ReturnOps both in Then-Block & Else-Block
// In this case everything below the IfOp is unreachable,
// So Else-block will be moved to parent & replace everything below IfOp
// CHECK-LABEL: @test_if_def_both_return
cuda_tile.module @test {
  testing$func @test_if_def_both_return(%arg1 : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // CHECK: %[[R0:.*]] = constant <i32: 0>
    // CHECK: %[[R1:.*]] = constant <i32: 1>
    // CHECK: %[[CMP:.*]] = cmpi equal %{{.*}}, %[[R0]]
    // CHECK: if %[[CMP]] {
    // CHECK:   return %[[R0]]
    // CHECK-NOT: else
    // CHECK: return %[[R1]]
    // CHECK-NOT: return
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cond1 = cmpi equal %arg1, %a, signed : !cuda_tile.tile<i32> -> tile<i1>
    %if = if %cond1 -> (tile<i32>) {
      print "%d", %a : tile<i32>
      return %a : tile<i32>
    } else {
      print "%d", %b : tile<i32>
      return %b : tile<i32>
    }
    print "%d", %if : tile<i32>
    return %if : tile<i32>
  }
}

// -----
// Test ConvertToSelect with token types - should NOT convert to select
// This tests the fix that checks all yielded values are TileType before converting
// CHECK-LABEL: entry @test_if_token_yield
cuda_tile.module @cuda_module {
  entry @test_if_token_yield(%arg0: tile<i1>, %arg1: tile<ptr<i32>>) {
    // CHECK: make_token
    // CHECK: make_token
    // CHECK: if %arg0
    // CHECK-NOT: select
    %cst_0_i32 = constant <i32: 0> : tile<i32>
    %0 = make_token : token
    %1 = make_token : token
    %2 = if %arg0 -> (token) {
      yield %0 : token
    } else {
      yield %1 : token
    }
    %3 = store_ptr_tko weak %arg1, %cst_0_i32 token=%2 : tile<ptr<i32>>, tile<i32> -> token
    return
  }
}

// -----
// Test ConvertToSelect with non-0 dim tile types - should NOT convert to select
// This tests the fix that checks all yielded values are 0-dim TileType before converting
// CHECK-LABEL: entry @test_if_tile_yield
cuda_tile.module @cuda_module {
  entry @test_if_tile_yield(%arg0: tile<i1>, %arg1: tile<ptr<i32>>) {
    // CHECK: if %arg0
    // CHECK-NOT: select
    %cst_0_i32 = constant <i32: 0> : tile<1xi32>
    %cst_1_i32 = constant <i32: 1> : tile<1xi32>
    %if = if %arg0 -> (tile<1xi32>) {
      yield %cst_0_i32 : tile<1xi32>
    } else {
      yield %cst_1_i32 : tile<1xi32>
    }
    %off = reshape %arg1 : tile<ptr<i32>> -> tile<1xptr<i32>>
    %3 = store_ptr_tko weak %off, %if: tile<1xptr<i32>>, tile<1xi32> -> token
    return
  }
}

// -----
// Test CombineIfs fix - ensures yielded values are properly retrieved
// This tests the fix that removed nextThen/nextElse conditions
// CHECK-LABEL: entry @test_combine_ifs_with_tokens
cuda_tile.module @cuda_module {
  global @exitval alignment = 4 <i32: 0> : tile<1xi32>
  entry @test_combine_ifs_with_tokens(%arg0: tile<i1>, %arg1: tile<ptr<i32>>) {
    %cst_1_i32 = constant <i32: 1> : tile<i32>
    %cst_0_i32 = constant <i32: 0> : tile<i32>
    %0 = make_token : token
    %1 = cmpi not_equal %cst_0_i32, %cst_0_i32, signed : tile<i32> -> tile<i1>
    // First if statement
    %2:2 = if %1 -> (token, token) {
      %3 = get_global @exitval : tile<ptr<i32>>
      %result, %result_token = load_ptr_tko weak %3 token=%0 : tile<ptr<i32>> -> tile<i32>, token
      %4 = join_tokens %0, %result_token : token
      %5 = addi %result, %cst_1_i32 overflow<no_signed_wrap> : tile<i32>
      %6 = store_ptr_tko weak %3, %5 token=%4 : tile<ptr<i32>>, tile<i32> -> token
      yield %6, %4 : token, token
    } else {
      yield %0, %0 : token, token
    }
    // Second if statement that uses results from first if
    // This tests that prevThenYielded and prevElseYielded are retrieved correctly
    if %1 {
      %3 = get_global @exitval : tile<ptr<i32>>
      %result, %result_token = load_ptr_tko weak %3 token=%2#0 : tile<ptr<i32>> -> tile<i32>, token
      %4 = join_tokens %2#1, %result_token : token
      %5 = addi %result, %cst_1_i32 overflow<no_signed_wrap> : tile<i32>
      %6 = join_tokens %4, %2#0 : token
      %7 = store_ptr_tko weak %3, %5 token=%6 : tile<ptr<i32>>, tile<i32> -> token
    }
    return
  }
}

// -----
// Test CombineIfs fix - ensures yielded values are properly retrieved
// This tests the fix that removed nextThen/nextElse conditions
// CHECK-LABEL: entry @test_combine_ifs_with_tokens_and_return
cuda_tile.module @cuda_module {
  global @exitval alignment = 4 <i32: 0> : tile<1xi32>
  entry @test_combine_ifs_with_tokens_and_return(%arg0: tile<i1>, %arg1: tile<ptr<i32>>) {
    %cst_1_i32 = constant <i32: 1> : tile<i32>
    %cst_0_i32 = constant <i32: 0> : tile<i32>
    %0 = make_token : token
    %1 = cmpi not_equal %cst_0_i32, %cst_0_i32, signed : tile<i32> -> tile<i1>
    // First if statement
    %2:2 = if %1 -> (token, token) {
      %3 = get_global @exitval : tile<ptr<i32>>
      %result, %result_token = load_ptr_tko weak %3 token=%0 : tile<ptr<i32>> -> tile<i32>, token
      %4 = join_tokens %0, %result_token : token
      %5 = addi %result, %cst_1_i32 overflow<no_signed_wrap> : tile<i32>
      %6 = store_ptr_tko weak %3, %5 token=%4 : tile<ptr<i32>>, tile<i32> -> token
      yield %6, %4 : token, token
    } else {
      return
    }
    // Second if statement that uses results from first if
    // This tests that prevThenYielded and prevElseYielded are retrieved correctly
    if %1 {
      %3 = get_global @exitval : tile<ptr<i32>>
      %result, %result_token = load_ptr_tko weak %3 token=%2#0 : tile<ptr<i32>> -> tile<i32>, token
      %4 = join_tokens %2#1, %result_token : token
      %5 = addi %result, %cst_1_i32 overflow<no_signed_wrap> : tile<i32>
      %6 = join_tokens %4, %2#0 : token
      %7 = store_ptr_tko weak %3, %5 token=%6 : tile<ptr<i32>>, tile<i32> -> token
    }
    return
  }
}
