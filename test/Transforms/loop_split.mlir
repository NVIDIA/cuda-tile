// RUN: cuda-tile-opt %s --pass-pipeline='builtin.module(cuda_tile.module(cuda_tile.entry(loop-split)))'  --split-input-file | FileCheck %s

// LoopSplit is enabled for loop - unsupported due to comparison of non-iv with invariant
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @unsupported_cmp_non_iv
  cuda_tile.module @unsupported_cmp_non_iv {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %70, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - unsupported due to comparison of iv with non-invariant
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @unsupported_cmp_non_inv
  cuda_tile.module @unsupported_cmp_non_inv {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %70, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop - sge predicate split
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_sge
  cuda_tile.module @split_sge {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - slt predicate split
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_slt
  cuda_tile.module @split_slt {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - sle predicate split
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_sle
  cuda_tile.module @split_sle {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop - continue inside if-block
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_continue
  cuda_tile.module @split_continue {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT:    if
      // CHECK:        %[[MUL:.*]] = muli {{.*}}, {{.*}} : tile<i32>
      // CHECK-NEXT:   continue %[[MUL]] : tile<i32>
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than_or_equal %3, %arg1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          continue %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          continue %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop - CmpOp with uses
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_cmp_uses
  cuda_tile.module @split_cmp_uses {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:      {{.*}} = negi %[[FALSE]] : tile<i1>
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:      %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:      {{.*}} = negi %[[TRUE]] : tile<i1>
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %n = negi %5: tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop, IfOp requesting split is inside another IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @supported_split_inner_if
  cuda_tile.module @supported_split_inner_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:      {{.*}} = if {{.*}} {
      // CHECK:        {{.*}} = if %[[FALSE]] -> (tile<i32>) {
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:      %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:      {{.*}} = if {{.*}} {
      // CHECK:        {{.*}} = if %[[TRUE]] -> (tile<i32>) {
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %70, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %100 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          %96 = if %100 -> (tile<i32>) {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            %99 = muli %c7, %920 : tile<i32>
            yield %99 : tile<i32>
          } else {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            yield %920 : tile<i32>
          }
          yield %96 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop, splitting with IfOp inside IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_supported_nested_if
  cuda_tile.module @split_supported_nested_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:        {{.*}} = if {{.*}}
      // CHECK-NOT:    {{.*}} = if {{.*}}
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          %96 = if %5 -> (tile<i32>) {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            %99 = muli %c7, %920 : tile<i32>
            yield %99 : tile<i32>
          } else {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            yield %920 : tile<i32>
          }
          yield %96 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// Loop split enabled - branch is inside inner loop
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @supported_if_inside_nested_for
  cuda_tile.module @supported_if_inside_nested_for {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:        %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:        {{.*}} = for {{.*}}
      // CHECK:          {{.*}} = if %[[FALSE]] -> (tile<i32>) {
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:        %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:        {{.*}} = for {{.*}}
      // CHECK:          {{.*}} = if %[[TRUE]] -> (tile<i32>) {
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %99 = for %arg2 in (%1 to %0, step %2) : tile<i64> iter_values(%100 = %7) -> (tile<i32>) {
          %6 = if %5 -> (tile<i32>) {
            %9 = muli %c7, %c8 : tile<i32>
            yield %9 : tile<i32>
          } else {
            %96 = if %5 -> (tile<i32>) {
              %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
              %99 = muli %c7, %920 : tile<i32>
              yield %99 : tile<i32>
            } else {
              %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
              yield %920 : tile<i32>
            }
            yield %96 : tile<i32>
          }
          %80 = addi %6, %100 : tile<i32>
          continue %80 : tile <i32>
        }
        %8 = addi %7, %99 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// Check supported splitting of inner ForOp inside IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @supported_for_inside_if
  cuda_tile.module @supported_for_inside_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[ADD:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK-NEXT: %[[SPLITU:.*]] = mini %[[ADD]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[ADD]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      %[[SPLITIU:.*]] = mini %[[SPLITI:.*]], {{.*}} signed : tile<i64>
      // CHECK:      %[[SPLITIL:.*]] = maxi %[[SPLITI]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITIU]], step {{.*}})
      // CHECK: {{.*}} = for {{.*}} in (%[[SPLITIL]] to {{.*}}, step {{.*}})
      // CHECK: {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %77 = if %5 -> (tile<i32>) {
          yield %c8 : tile<i32>
        } else {
          %99 = for %arg2 in (%1 to %0, step %2) : tile<i64> iter_values(%100 = %7) -> (tile<i32>) {
            %11 = cmpi greater_than %arg2, %3, signed : tile<i64> -> tile<i1>
            %6 = if %11 -> (tile<i32>) {
              %9 = muli %c7, %c8 : tile<i32>
              yield %9 : tile<i32>
            } else {
              %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
              %99 = muli %c7, %920 : tile<i32>
              yield %99 : tile<i32>
            }
            %80 = addi %6, %100 : tile<i32>
            continue %80 : tile <i32>
          }
          yield %99 : tile<i32>
        }
        %8 = addi %7, %77 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit disabled by hint for IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_if
  cuda_tile.module @hint_disable_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%3 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        } {cuda_tile.loop_split = 0}
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        } {cuda_tile.loop_split = 0}
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit disabled by hint for ForOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_for
  cuda_tile.module @hint_disable_for {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%3 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } {cuda_tile.loop_split = 0}
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit disabled by hint for EntryOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_entry
  cuda_tile.module @hint_disable_entry {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%3 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } 
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    } {cuda_tile.loop_split = 0}
  }
}

// -----

// LoopSplit disabled by hint for EntryOp but enabled by hint for ForOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_entry_enable_for
  cuda_tile.module @hint_disable_entry_enable_for {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } {cuda_tile.loop_split = 1}
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    } {cuda_tile.loop_split = 0}
  }
}

// -----

// LoopSplit disabled by hint for ForOp but enabled by hint for IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_for_enable_if
  cuda_tile.module @hint_disable_for_enable_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        } {cuda_tile.loop_split = 1}
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } {cuda_tile.loop_split = 0}
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    } {cuda_tile.loop_split = 0}
  }
}

// -----
// LoopSplit is enabled for loop - unsigned comparison unsupported
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_unsigned
  cuda_tile.module @split_unsigned {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than %arg1, %3, unsigned : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - split with non-1 step
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_step
  cuda_tile.module @split_step {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 4> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[ADDI:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK-NEXT: %[[SUBI:.*]] = subi %[[ADDI]], %[[LB:.*]] : tile<i64>
      // CHECK-NEXT: %[[DIVI:.*]] = divi %[[SUBI]], %[[STEP:.*]] signed rounding<positive_inf> : tile<i64>
      // CHECK-NEXT: %[[MULI:.*]] = muli %[[DIVI]], %[[STEP]] : tile<i64>
      // CHECK-NEXT: %[[SPLIT:.*]] = addi %[[LB]], %[[MULI]] : tile<i64>
      // CHECK-NEXT: %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], %[[LB]] signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_if_inside_while_loop
  cuda_tile.module @split_if_inside_while_loop {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:        %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:        {{.*}} = loop {{.*}}
      // CHECK:          {{.*}} = if %[[FALSE]] -> (tile<i32>) {
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:        %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:        {{.*}} = loop {{.*}}
      // CHECK:          {{.*}} = if %[[TRUE]] -> (tile<i32>) {
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %loop = loop iter_values(%arg2 = %c1) : tile<i32> -> tile<i32> {
          %6 = if %5 -> (tile<i32>) {
            %9 = muli %c7, %c8 : tile<i32>
            yield %9 : tile<i32>
          } else {
            yield %c7 : tile<i32>
          }
          break %6 : tile<i32>
        }
        %8 = addi %7, %loop : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}
