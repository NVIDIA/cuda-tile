// RUN: cuda-tile-opt %s --canonicalize -allow-unregistered-dialect --split-input-file | FileCheck %s

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Additional IfOp canonicalization: CombineNestedIfs with yields,
// CombineConsecutiveIfs negated, MoveTerminatorToParent,
// RemoveStaticCondition with return terminator

// CombineNestedIfs with yielded results from inner if (lines ~3607-3661)
// CHECK-LABEL: @test_combine_nested_ifs_with_yield
cuda_tile.module @test {
  testing$func @test_combine_nested_ifs_with_yield(%c1: !cuda_tile.tile<i1>, %c2: !cuda_tile.tile<i1>) -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 20> : !cuda_tile.tile<i32>
    // Nested ifs with yields combine and then fold to select
    // CHECK: andi
    // CHECK: select
    %0 = cuda_tile.if %c1 -> (tile<i32>) {
      %1 = cuda_tile.if %c2 -> (tile<i32>) {
        yield %a : tile<i32>
      } else {
        yield %b : tile<i32>
      }
      yield %1 : tile<i32>
    } else {
      yield %b : tile<i32>
    }
    return %0 : tile<i32>
  }
}

// -----

// CombineConsecutiveIfs with negated condition via xor (lines ~3406-3419)
// CHECK-LABEL: @test_combine_consecutive_negated
cuda_tile.module @test {
  testing$func @test_combine_consecutive_negated(%cond: !cuda_tile.tile<i1>) {
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %not_cond = cuda_tile.xori %cond, %true : !cuda_tile.tile<i1>
    // Second if uses negated condition => should combine
    // CHECK: if %{{.*}} {
    // CHECK:   "op_a"
    // CHECK:   "op_d"
    // CHECK: } else {
    // CHECK:   "op_b"
    // CHECK:   "op_c"
    cuda_tile.if %cond {
      "op_a"() : () -> ()
    } else {
      "op_b"() : () -> ()
    }
    cuda_tile.if %not_cond {
      "op_c"() : () -> ()
    } else {
      "op_d"() : () -> ()
    }
    return
  }
}

// -----

// CombineConsecutiveIfs where prev condition is negated (lines ~3414-3419)
// CHECK-LABEL: @test_combine_consecutive_prev_negated
cuda_tile.module @test {
  testing$func @test_combine_consecutive_prev_negated(%cond: !cuda_tile.tile<i1>) {
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %not_cond = cuda_tile.xori %cond, %true : !cuda_tile.tile<i1>
    // First if uses negated, second uses original
    // CHECK: if %{{.*}} {
    cuda_tile.if %not_cond {
      "op_a"() : () -> ()
    } else {
      "op_b"() : () -> ()
    }
    cuda_tile.if %cond {
      "op_c"() : () -> ()
    } else {
      "op_d"() : () -> ()
    }
    return
  }
}

// -----

// MoveTerminatorToParent: both branches end with return (lines ~3668-3700)
// CHECK-LABEL: @test_move_terminator_to_parent
cuda_tile.module @test {
  testing$func @test_move_terminator_to_parent(%cond: !cuda_tile.tile<i1>) -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    // When both branches have return, the else can be moved out
    // CHECK: if %{{.*}} {
    // CHECK:   return
    // CHECK: }
    // CHECK: return
    cuda_tile.if %cond {
      return %a : tile<i32>
    } else {
      return %b : tile<i32>
    }
    // unreachable
    return %a : tile<i32>
  }
}

// -----

// RemoveStaticCondition with true condition (lines ~3142-3143)
// containing a return (replaceOpWithRegion with non-yield terminator)
// CHECK-LABEL: @test_static_true_with_return
cuda_tile.module @test {
  testing$func @test_static_true_with_return() -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 99> : !cuda_tile.tile<i32>
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    // Static true condition => inline then branch
    // The return terminates early, so addi should be eliminated
    // CHECK: constant <i32: 42>
    // CHECK-NOT: addi
    // CHECK: return
    cuda_tile.if %true {
      return %a : tile<i32>
    } else {
      cuda_tile.yield
    }
    %c = cuda_tile.addi %a, %b : tile<i32>
    return %c : tile<i32>
  }
}

// -----

// RemoveStaticCondition with false => else branch has return
// CHECK-LABEL: @test_static_false_with_else_return
cuda_tile.module @test {
  testing$func @test_static_false_with_else_return() -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 99> : !cuda_tile.tile<i32>
    %false = cuda_tile.constant <i1: 0> : !cuda_tile.tile<i1>
    // Static false => inline else branch which has return
    // CHECK: constant <i32: 99>
    // CHECK-NOT: addi
    // CHECK: return
    cuda_tile.if %false {
      cuda_tile.yield
    } else {
      return %b : tile<i32>
    }
    %c = cuda_tile.addi %a, %b : tile<i32>
    return %c : tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// IfOp canonicalization patterns

// RemoveStaticCondition: false condition without else => erase (lines ~3144-3148)
// CHECK-LABEL: @test_static_false_no_else
cuda_tile.module @test {
  testing$func @test_static_false_no_else() {
    // CHECK-NOT: if
    %false = cuda_tile.constant <i1: 0> : !cuda_tile.tile<i1>
    cuda_tile.if %false {
      "side_effect_op"() : () -> ()
    }
    return
  }
}

// -----

// IfOp::fold: if (!c) then A else B => if c then B else A (lines ~3110-3125)
// CHECK-LABEL: @test_if_fold_xor
cuda_tile.module @test {
  testing$func @test_if_fold_xor(%cond: !cuda_tile.tile<i1>) {
    %true = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %not_cond = cuda_tile.xori %cond, %true : !cuda_tile.tile<i1>
    // After fold, %cond replaces %not_cond, then/else swap:
    // CHECK: if %{{.*}} {
    // CHECK:   "op_b"
    // CHECK: } else {
    // CHECK:   "op_a"
    cuda_tile.if %not_cond {
      "op_a"() : () -> ()
    } else {
      "op_b"() : () -> ()
    }
    return
  }
}

// -----

// RemoveEmptyElseBranch (lines ~3535-3551)
// CHECK-LABEL: @test_remove_empty_else
cuda_tile.module @test {
  testing$func @test_remove_empty_else(%cond: !cuda_tile.tile<i1>) {
    // CHECK: if %{{.*}} {
    // CHECK-NEXT: "op_a"
    // CHECK-NEXT: }
    // CHECK-NOT: else
    cuda_tile.if %cond {
      "op_a"() : () -> ()
    } else {
    }
    return
  }
}

// -----

// CombineNestedIfs (lines ~3556-3650)
// CHECK-LABEL: @test_combine_nested_ifs
cuda_tile.module @test {
  testing$func @test_combine_nested_ifs(%c1: !cuda_tile.tile<i1>, %c2: !cuda_tile.tile<i1>) {
    // Nested ifs should be combined using andi
    // CHECK: %[[AND:.+]] = andi
    // CHECK: if %[[AND]]
    cuda_tile.if %c1 {
      cuda_tile.if %c2 {
        "side_effect_op"() : () -> ()
      }
    }
    return
  }
}

// -----

// ConvertToSelect (lines ~3160-3232)
// CHECK-LABEL: @test_convert_to_select
cuda_tile.module @test {
  testing$func @test_convert_to_select(%cond: !cuda_tile.tile<i1>, %a: !cuda_tile.tile<4xi32>, %b: !cuda_tile.tile<4xi32>) -> !cuda_tile.tile<4xi32> {
    // CHECK: select
    // CHECK-NOT: if
    %0 = cuda_tile.if %cond -> (tile<4xi32>) {
      yield %a : tile<4xi32>
    } else {
      yield %b : tile<4xi32>
    }
    return %0 : tile<4xi32>
  }
}

// -----

// IfOp: identical yield values => replace uses (lines ~3349-3354)
// CHECK-LABEL: @test_identical_yields
cuda_tile.module @test {
  testing$func @test_identical_yields(%cond: !cuda_tile.tile<i1>, %v: !cuda_tile.tile<4xi32>) -> !cuda_tile.tile<4xi32> {
    // CHECK-NOT: if
    // CHECK: return %{{.*}}
    %0 = cuda_tile.if %cond -> (tile<4xi32>) {
      yield %v : tile<4xi32>
    } else {
      yield %v : tile<4xi32>
    }
    return %0 : tile<4xi32>
  }
}

// -----

// CombineConsecutiveIfs with same condition (lines ~3401-3527)
// CHECK-LABEL: @test_combine_consecutive_ifs
cuda_tile.module @test {
  testing$func @test_combine_consecutive_ifs(%cond: !cuda_tile.tile<i1>) {
    // Two consecutive ifs with same condition should be merged
    // CHECK: if %{{.*}} {
    // CHECK:   "op_a"
    // CHECK:   "op_c"
    // CHECK: } else {
    // CHECK:   "op_b"
    // CHECK:   "op_d"
    cuda_tile.if %cond {
      "op_a"() : () -> ()
    } else {
      "op_b"() : () -> ()
    }
    cuda_tile.if %cond {
      "op_c"() : () -> ()
    } else {
      "op_d"() : () -> ()
    }
    return
  }
}

