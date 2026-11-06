// RUN: cuda-tile-opt %s --pass-pipeline='builtin.module(cuda_tile.module(cuda_tile.testing$func(cuda-tile-licm)))' --split-input-file | FileCheck %s

// ============================================================================
// Group 1: Basic Hoisting from ForOp
// ============================================================================

// CHECK-LABEL: testing$func @test_hoist_constant
// CHECK:         %[[CST:.*]] = constant <i32: 42>
// CHECK-NEXT:    %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_constant(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %c = constant <i32: 42> : tile<i32>
      %r = addi %acc, %c : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_hoist_pure_arithmetic
// CHECK:         %[[A:.*]] = addi
// CHECK-NEXT:    %[[M:.*]] = muli
// CHECK-NEXT:    %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_pure_arithmetic(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>, %y: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %a = addi %x, %y : tile<i32>
      %m = muli %a, %x : tile<i32>
      %r = addi %acc, %m : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_no_hoist_iv_dependent
// CHECK:         %{{.*}} = for %[[IV:.*]] in
// CHECK:           addi %[[IV]]
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_iv_dependent(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %a = addi %iv, %x : tile<i32>
      %r = addi %acc, %a : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_no_hoist_iter_arg_dependent
// CHECK:         %{{.*}} = for
// CHECK:           addi %{{.*}}, %{{.*}}
// CHECK:           continue
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_iter_arg_dependent(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %r = addi %acc, %x : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// ============================================================================
// Group 2: Transitive Invariance
// ============================================================================

// CHECK-LABEL: testing$func @test_transitive_hoist
// CHECK:         %[[B:.*]] = addi %{{.*}}, %{{.*}}
// CHECK-NEXT:    %[[C:.*]] = muli %[[B]], %[[B]]
// CHECK-NEXT:    %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_transitive_hoist(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %a: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %b = addi %a, %a : tile<i32>
      %c = muli %b, %b : tile<i32>
      %r = addi %acc, %c : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_partial_transitive
// CHECK:         %[[B:.*]] = addi
// CHECK-NEXT:    %{{.*}} = for %[[IV:.*]] in
// CHECK:           muli %[[B]], %[[IV]]
cuda_tile.module @test {
  cuda_tile.testing$func @test_partial_transitive(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %a: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %b = addi %a, %a : tile<i32>
      %c = muli %b, %iv : tile<i32>
      %r = addi %acc, %c : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// ============================================================================
// Group 3: Nested Loops
// ============================================================================

// CHECK-LABEL: testing$func @test_nested_for_for
// CHECK:         %[[CST:.*]] = constant <i32: 7>
// CHECK:         %{{.*}} = for
// CHECK:           %{{.*}} = for
// CHECK-NOT:         constant <i32: 7>
cuda_tile.module @test {
  cuda_tile.testing$func @test_nested_for_for(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %inner = for %iv2 in (%lb to %ub, step %step) : tile<i32> iter_values(%acc2 = %acc) -> (tile<i32>) {
        %c = constant <i32: 7> : tile<i32>
        %r = addi %acc2, %c : tile<i32>
        continue %r : tile<i32>
      }
      continue %inner : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_nested_loop_in_for
// CHECK:         %[[CST:.*]] = constant <i32: 5>
// CHECK:         %{{.*}} = for
// CHECK:           %{{.*}} = loop
// CHECK-NOT:         constant <i32: 5>
cuda_tile.module @test {
  cuda_tile.testing$func @test_nested_loop_in_for(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %cond: tile<i1>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %lp = loop iter_values(%lacc = %acc) : tile<i32> -> tile<i32> {
        %c = constant <i32: 5> : tile<i32>
        %r = addi %lacc, %c : tile<i32>
        %done = if %cond -> (tile<i32>) {
          yield %r : tile<i32>
        } else {
          yield %lacc : tile<i32>
        }
        break %done : tile<i32>
      }
      continue %lp : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_nested_for_in_loop
// CHECK:         %[[CST:.*]] = constant <i32: 3>
// CHECK:         %{{.*}} = loop
// CHECK:           %{{.*}} = for
// CHECK-NOT:         constant <i32: 3>
cuda_tile.module @test {
  cuda_tile.testing$func @test_nested_for_in_loop(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %cond: tile<i1>) -> tile<i32> {
    %lp = loop iter_values(%acc = %lb) : tile<i32> -> tile<i32> {
      %inner = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%iacc = %acc) -> (tile<i32>) {
        %c = constant <i32: 3> : tile<i32>
        %r = addi %iacc, %c : tile<i32>
        continue %r : tile<i32>
      }
      %done = if %cond -> (tile<i32>) {
        yield %inner : tile<i32>
      } else {
        yield %acc : tile<i32>
      }
      break %done : tile<i32>
    }
    return %lp : tile<i32>
  }
}

// -----

// Invariant to inner loop (uses outer IV) but not outer loop.
// CHECK-LABEL: testing$func @test_invariant_at_different_levels
// CHECK:         %{{.*}} = for %[[OIV:.*]] in
// CHECK:           %[[A:.*]] = addi %[[OIV]], %{{.*}}
// CHECK:           %{{.*}} = for
// CHECK-NOT:         addi %[[OIV]]
cuda_tile.module @test {
  cuda_tile.testing$func @test_invariant_at_different_levels(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %oiv in (%lb to %ub, step %step) : tile<i32> iter_values(%oacc = %lb) -> (tile<i32>) {
      %inner = for %iiv in (%lb to %ub, step %step) : tile<i32> iter_values(%iacc = %oacc) -> (tile<i32>) {
        %a = addi %oiv, %x : tile<i32>
        %r = addi %iacc, %a : tile<i32>
        continue %r : tile<i32>
      }
      continue %inner : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// ============================================================================
// Group 4: Hoisting from IfOp Regions
// ============================================================================

// CHECK-LABEL: testing$func @test_hoist_from_then_region
// CHECK:         %[[M:.*]] = muli
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_from_then_region(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>, %y: tile<i32>, %cond: tile<i1>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %res = if %cond -> (tile<i32>) {
        %m = muli %x, %y : tile<i32>
        yield %m : tile<i32>
      } else {
        yield %acc : tile<i32>
      }
      continue %res : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_hoist_from_else_region
// CHECK:         %[[M:.*]] = muli
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_from_else_region(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>, %y: tile<i32>, %cond: tile<i1>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %res = if %cond -> (tile<i32>) {
        yield %acc : tile<i32>
      } else {
        %m = muli %x, %y : tile<i32>
        yield %m : tile<i32>
      }
      continue %res : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_hoist_from_nested_if
// CHECK:         %[[M:.*]] = muli
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_from_nested_if(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>, %y: tile<i32>, %c1: tile<i1>, %c2: tile<i1>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %res = if %c1 -> (tile<i32>) {
        %inner = if %c2 -> (tile<i32>) {
          %m = muli %x, %y : tile<i32>
          yield %m : tile<i32>
        } else {
          yield %acc : tile<i32>
        }
        yield %inner : tile<i32>
      } else {
        yield %acc : tile<i32>
      }
      continue %res : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// Op inside if depends on value defined in same if block -- not invariant.
// CHECK-LABEL: testing$func @test_no_hoist_if_dependent
// CHECK:         %{{.*}} = for %[[IV:.*]] in
// CHECK:           if
// CHECK:             addi %[[IV]]
// CHECK:             muli
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_if_dependent(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>, %cond: tile<i1>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %res = if %cond -> (tile<i32>) {
        %a = addi %iv, %x : tile<i32>
        %m = muli %a, %x : tile<i32>
        yield %m : tile<i32>
      } else {
        yield %acc : tile<i32>
      }
      continue %res : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// ============================================================================
// Group 5: Speculatability (divi / remi)
// ============================================================================

// CHECK-LABEL: testing$func @test_no_hoist_divi_dynamic_divisor
// CHECK:         %{{.*}} = for
// CHECK:           divi
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_divi_dynamic_divisor(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>, %d: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %q = divi %x, %d signed : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_hoist_divi_const_nonzero
// CHECK:         %[[Q:.*]] = divi
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_divi_const_nonzero(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %c3 = constant <i32: 3> : tile<i32>
      %q = divi %x, %c3 signed : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_no_hoist_divi_const_zero
// CHECK:         %{{.*}} = for
// CHECK:           divi
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_divi_const_zero(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %c0 = constant <i32: 0> : tile<i32>
      %q = divi %x, %c0 signed : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// Signed divi with -1: INT_MIN / -1 is UB, so not speculatable.
// CHECK-LABEL: testing$func @test_no_hoist_divi_signed_neg_one
// CHECK:         %{{.*}} = for
// CHECK:           divi
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_divi_signed_neg_one(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %cn1 = constant <i32: -1> : tile<i32>
      %q = divi %x, %cn1 signed : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// Unsigned divi with all-ones: no INT_MIN concern, safe to hoist.
// CHECK-LABEL: testing$func @test_hoist_divi_unsigned_neg_one
// CHECK:         %[[Q:.*]] = divi
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_divi_unsigned_neg_one(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %cn1 = constant <i32: -1> : tile<i32>
      %q = divi %x, %cn1 unsigned : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_hoist_remi_const_nonzero
// CHECK:         %[[R:.*]] = remi
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_remi_const_nonzero(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %c5 = constant <i32: 5> : tile<i32>
      %q = remi %x, %c5 signed : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// Signed remi with -1: X % -1 = 0 is always well-defined, safe to hoist.
// CHECK-LABEL: testing$func @test_hoist_remi_signed_neg_one
// CHECK:         %[[R:.*]] = remi
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_remi_signed_neg_one(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %cn1 = constant <i32: -1> : tile<i32>
      %q = remi %x, %cn1 signed : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_no_hoist_remi_dynamic
// CHECK:         %{{.*}} = for
// CHECK:           remi
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_remi_dynamic(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %x: tile<i32>, %d: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %q = remi %x, %d signed : tile<i32>
      %r = addi %acc, %q : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// ============================================================================
// Group 6: LoopOp
// ============================================================================

// CHECK-LABEL: testing$func @test_hoist_from_loop_op
// CHECK:         %[[CST:.*]] = constant <i32: 10>
// CHECK:         %{{.*}} = loop
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_from_loop_op(%init: tile<i32>, %cond: tile<i1>) -> tile<i32> {
    %lp = loop iter_values(%acc = %init) : tile<i32> -> tile<i32> {
      %c = constant <i32: 10> : tile<i32>
      %r = addi %acc, %c : tile<i32>
      break %r : tile<i32>
    }
    return %lp : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_loop_with_continue_and_break
// CHECK:         %[[CST:.*]] = constant <i32: 2>
// CHECK:         %{{.*}} = loop
cuda_tile.module @test {
  cuda_tile.testing$func @test_loop_with_continue_and_break(%init: tile<i32>, %cond: tile<i1>) -> tile<i32> {
    %lp = loop iter_values(%acc = %init) : tile<i32> -> tile<i32> {
      %c = constant <i32: 2> : tile<i32>
      %r = addi %acc, %c : tile<i32>
      if %cond {
        continue %r : tile<i32>
      }
      break %r : tile<i32>
    }
    return %lp : tile<i32>
  }
}

// -----

// ============================================================================
// Group 7: View Operations
// ============================================================================

// CHECK-LABEL: testing$func @test_hoist_make_tensor_view
// CHECK:         %[[TV:.*]] = make_tensor_view
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_make_tensor_view(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %ptr: tile<ptr<f32>>, %sh0: tile<i32>, %sh1: tile<i32>, %st0: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %tv = make_tensor_view %ptr, shape = [%sh0, %sh1], strides = [%st0, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
      %d0, %d1 = get_tensor_shape %tv : tensor_view<?x?xf32, strides=[?,1]> -> tile<i32>
      %r = addi %acc, %d0 : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// Transitive: make_partition_view depends on hoistable make_tensor_view.
// CHECK-LABEL: testing$func @test_hoist_make_partition_view
// CHECK:         %[[TV:.*]] = make_tensor_view
// CHECK:         %[[PV:.*]] = make_partition_view %[[TV]]
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_make_partition_view(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %ptr: tile<ptr<f32>>, %sh0: tile<i32>, %sh1: tile<i32>, %st0: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %tv = make_tensor_view %ptr, shape = [%sh0, %sh1], strides = [%st0, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
      %cst32 = constant <i32: 32> : tile<i32>
      %cst32b = constant <i32: 32> : tile<i32>
      %pv = make_partition_view %tv : partition_view<tile=(32x32), tensor_view<?x?xf32, strides=[?,1]>>
      continue %acc : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_hoist_get_tensor_shape
// CHECK:         %{{.*}}:2 = get_tensor_shape
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_get_tensor_shape(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %tv: tensor_view<?x?xf32, strides=[?,1]>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %d0:2 = get_tensor_shape %tv : tensor_view<?x?xf32, strides=[?,1]> -> tile<i32>
      %r = addi %acc, %d0#0 : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// load_view_tko has memory effects and must not be hoisted.
// CHECK-LABEL: testing$func @test_no_hoist_load_view_tko
// CHECK:         %[[TV:.*]] = make_tensor_view
// CHECK:         %[[PV:.*]] = make_partition_view %[[TV]]
// CHECK:         %{{.*}} = for
// CHECK:           load_view_tko
cuda_tile.module @test {
  cuda_tile.testing$func @test_no_hoist_load_view_tko(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %ptr: tile<ptr<f32>>, %sh0: tile<i32>, %sh1: tile<i32>, %st0: tile<i32>, %idx0: tile<i32>, %idx1: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %tv = make_tensor_view %ptr, shape = [%sh0, %sh1], strides = [%st0, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
      %pv = make_partition_view %tv : partition_view<tile=(32x32), tensor_view<?x?xf32, strides=[?,1]>>
      %tile, %res_token = load_view_tko weak %pv[%idx0, %idx1] : partition_view<tile=(32x32), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> tile<32x32xf32>, token
      continue %acc : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// ============================================================================
// Group 8: Scan and Reduce Regions
// ============================================================================

// The scan depends on a loop-carried value and cannot be hoisted, but the
// invariant constant in its non-isolated region can be hoisted and captured.
// CHECK-LABEL: testing$func @test_hoist_from_scan_region
// CHECK-NEXT:    %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK-NEXT:    %{{.*}} = for
// CHECK:           %{{.*}} = scan
// CHECK-NEXT:      (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>) {
// CHECK-NEXT:        %[[LHS_WITH_ZERO:.*]] = addi %[[LHS]], %[[ZERO]] : tile<i32>
// CHECK-NEXT:        %[[SUM:.*]] = addi %[[LHS_WITH_ZERO]], %[[RHS]] : tile<i32>
// CHECK-NEXT:        yield %[[SUM]] : tile<i32>
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_from_scan_region(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %input: tile<8xi32>) -> tile<8xi32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %input) -> (tile<8xi32>) {
      %scan = scan %acc dim=0 reverse=false identities=[0 : i32] : tile<8xi32> -> tile<8xi32>
      (%lhs: tile<i32>, %rhs: tile<i32>) {
        %zero = constant <i32: 0> : tile<i32>
        %lhs_with_zero = addi %lhs, %zero : tile<i32>
        %sum = addi %lhs_with_zero, %rhs : tile<i32>
        yield %sum : tile<i32>
      }
      continue %scan : tile<8xi32>
    }
    return %for : tile<8xi32>
  }
}

// -----

// The reduce depends on a loop-carried value and cannot be hoisted, but the
// invariant constant in its non-isolated region can be hoisted and captured.
// CHECK-LABEL: testing$func @test_hoist_from_reduce_region
// CHECK-NEXT:    %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK-NEXT:    %{{.*}} = for
// CHECK:           %{{.*}} = reduce
// CHECK-NEXT:      (%[[LHS:.*]]: tile<i32>, %[[RHS:.*]]: tile<i32>) {
// CHECK-NEXT:        %[[LHS_WITH_ZERO:.*]] = addi %[[LHS]], %[[ZERO]] : tile<i32>
// CHECK-NEXT:        %[[SUM:.*]] = addi %[[LHS_WITH_ZERO]], %[[RHS]] : tile<i32>
// CHECK-NEXT:        yield %[[SUM]] : tile<i32>
cuda_tile.module @test {
  cuda_tile.testing$func @test_hoist_from_reduce_region(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>, %input: tile<8xi32>) -> tile<i32> {
    %for:2 = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%tile_acc = %input, %scalar_acc = %lb) -> (tile<8xi32>, tile<i32>) {
      %reduced = reduce %tile_acc dim=0 identities=[0 : i32] : tile<8xi32> -> tile<i32>
      (%lhs: tile<i32>, %rhs: tile<i32>) {
        %zero = constant <i32: 0> : tile<i32>
        %lhs_with_zero = addi %lhs, %zero : tile<i32>
        %sum = addi %lhs_with_zero, %rhs : tile<i32>
        yield %sum : tile<i32>
      }
      %next = addi %scalar_acc, %reduced : tile<i32>
      continue %tile_acc, %next : tile<8xi32>, tile<i32>
    }
    return %for#1 : tile<i32>
  }
}

// -----

// ============================================================================
// Group 9: Edge Cases
// ============================================================================

// CHECK-LABEL: testing$func @test_empty_loop_body
// CHECK:         %{{.*}} = for
// CHECK:           continue
cuda_tile.module @test {
  cuda_tile.testing$func @test_empty_loop_body(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      continue %acc : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// CHECK-LABEL: testing$func @test_already_optimal
// CHECK:         %{{.*}} = for %[[IV:.*]] in
// CHECK:           addi %[[IV]]
// CHECK:           muli
// CHECK:           continue
cuda_tile.module @test {
  cuda_tile.testing$func @test_already_optimal(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %a = addi %iv, %iv : tile<i32>
      %m = muli %a, %iv : tile<i32>
      %r = addi %acc, %m : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}

// -----

// get_tile_block_id has no operands -- always invariant.
// CHECK-LABEL: testing$func @test_multiple_results_op
// CHECK:         %{{.*}}, %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK:         %{{.*}} = for
cuda_tile.module @test {
  cuda_tile.testing$func @test_multiple_results_op(%lb: tile<i32>, %ub: tile<i32>, %step: tile<i32>) -> tile<i32> {
    %for = for %iv in (%lb to %ub, step %step) : tile<i32> iter_values(%acc = %lb) -> (tile<i32>) {
      %bx, %by, %bz = get_tile_block_id : tile<i32>
      %r = addi %acc, %bx : tile<i32>
      continue %r : tile<i32>
    }
    return %for : tile<i32>
  }
}
