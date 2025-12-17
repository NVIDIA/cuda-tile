// RUN: cuda-tile-opt %s --pass-pipeline="builtin.module(cuda_tile.module(synthesize-debug-info-scopes))" --split-input-file --mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: testing$func @func_no_debug()
// CHECK: loc(#loc[[LOC:[0-9]+]])
// CHECK: #[[FILE:.*]] = #cuda_tile.di_file<"<unknown>" in "">
// CHECK: #[[COMPILE_UNIT:.*]] = #cuda_tile.di_compile_unit<file = #[[FILE]]>
// CHECK: #[[SUBPROGRAM:.*]] = #cuda_tile.di_subprogram<file = #[[FILE]], line = 1, name = "func_no_debug", linkageName = "func_no_debug", compileUnit = #[[COMPILE_UNIT]], scopeLine = 1>
// CHECK: #loc[[LOC]] = #cuda_tile.di_loc<{{.*}} in #[[SUBPROGRAM]]>

cuda_tile.module @test {
  testing$func @func_no_debug() {
    return loc(unknown)
  } loc(unknown)
} loc(unknown)

// -----

// Test that existing debug info is not overwritten.
// CHECK-LABEL: testing$func @func_with_debug()
// CHECK: return loc(#loc
// CHECK: loc(#loc[[LOC:[0-9]+]])
// CHECK: #[[FILE:.*]] = #cuda_tile.di_file<"<unknown>" in "">
// CHECK: #[[COMPILE_UNIT]] = #cuda_tile.di_compile_unit<file = #[[FILE]]>
// CHECK: #[[SUBPROGRAM]] = #cuda_tile.di_subprogram<file = #[[FILE]], line = 15, name = "func_with_debug", linkageName = "func_with_debug", compileUnit = #[[COMPILE_UNIT]]>
// CHECK: #loc[[LOC]] = #cuda_tile.di_loc<{{.*}} in #[[SUBPROGRAM]]>

#di_file = #cuda_tile.di_file<"<unknown>" in "">
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file>
#di_subprogram = #cuda_tile.di_subprogram<file = #di_file, line = 15, name = "func_with_debug", linkageName = "func_with_debug", compileUnit = #di_compile_unit>

cuda_tile.module @test {
  testing$func @func_with_debug() {
    return loc(unknown)
  } loc(#cuda_tile.di_loc<loc("unknown":1:1) in #di_subprogram>)
}

// -----

// Test that we use existing file locations.
// CHECK-LABEL: testing$func @func_with_filelocs()
// CHECK: return loc(#[[LOC_RETURN:.*]])
// CHECK: } loc(#[[LOC_FN:.*]])

// CHECK-DAG: #[[FILE:.*]] = #cuda_tile.di_file<"file.py" in "">
// CHECK-DAG: #[[CU_FILE:.*]] = #cuda_tile.di_file<"other_file.py" in "">
// CHECK-DAG: #[[LOC_FN_FILE:.*]] = loc("file.py":10:4)
// CHECK-DAG: #[[LOC_RETURN_FILE:.*]] = loc("file.py":12:4)
// CHECK-DAG: #[[COMPILE_UNIT]] = #cuda_tile.di_compile_unit<file = #[[CU_FILE]]>
// CHECK-DAG: #[[SUBPROGRAM]] = #cuda_tile.di_subprogram<file = #[[FILE]], line = 10, name = "func_with_filelocs", linkageName = "func_with_filelocs", compileUnit = #[[COMPILE_UNIT]], scopeLine = 10>
// CHECK-DAG: #[[LOC_RETURN]] = #cuda_tile.di_loc<#[[LOC_RETURN_FILE]] in #[[SUBPROGRAM]]>
// CHECK-DAG: #[[LOC_FN]] = #cuda_tile.di_loc<#[[LOC_FN_FILE]] in #[[SUBPROGRAM]]>

cuda_tile.module @test {
  testing$func @func_with_filelocs() {
    return loc("file.py":12:4)
  } loc("file.py":10:4)
} loc("other_file.py":1:1)
