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

// -----

// Test that we handle OpaqueLoc, NameLoc, and CallSiteLoc
// CHECK-LABEL: testing$func @func_with_other_locs()
// CHECK: return loc(#[[LOC_RETURN:.*]])
// CHECK: } loc(#[[LOC_FN:.*]])

// CHECK-DAG: #[[FILE:.*]] = #cuda_tile.di_file<"file.py" in "">
// CHECK-DAG: #[[CU_FILE:.*]] = #cuda_tile.di_file<"other_file.py" in "">
// CHECK-DAG: #[[LOC_FN_FILE:.*]] = loc("file.py":10:4)
// CHECK-DAG: #[[LOC_RETURN_FILE:.*]] = loc("file.py":12:4)
// CHECK-DAG: #[[COMPILE_UNIT]] = #cuda_tile.di_compile_unit<file = #[[CU_FILE]]>
// CHECK-DAG: #[[SUBPROGRAM]] = #cuda_tile.di_subprogram<file = #[[FILE]], line = 10, name = "func_with_other_locs", linkageName = "func_with_other_locs", compileUnit = #[[COMPILE_UNIT]], scopeLine = 10>
// CHECK-DAG: #[[LOC_RETURN]] = #cuda_tile.di_loc<#[[LOC_RETURN_FILE]] in #[[SUBPROGRAM]]>
// CHECK-DAG: #[[LOC_FN]] = #cuda_tile.di_loc<#[[LOC_FN_FILE]] in #[[SUBPROGRAM]]>

cuda_tile.module @test {
  testing$func @func_with_other_locs() {
    return loc(callsite(unknown at "file.py":12:4))
  } loc(fused["file.py":10:4, unknown])
} loc("blah"("other_file.py":1:1))

// -----

// FusedLoc with no extractable FileLineCol: FusedLoc::get drops UnknownLoc children, so
// fused[unknown, unknown] collapses to UnknownLoc and never hits the FusedLoc TypeSwitch arm.
// Use non-Unknown children (NameLoc with unknown child) so the IR stays a real FusedLoc while
// extractFileLoc still walks subs and returns empty FileLineColLoc (then scope fallback).
// CHECK-LABEL: testing$func @func_fused_no_fileline()
// CHECK: return loc(#loc[[LOC_RET_FUSED:[0-9]+]])
// CHECK: } loc(#loc[[LOC_FN_FUSED:[0-9]+]])

// CHECK-DAG: #[[FUSED_FILE:.*]] = #cuda_tile.di_file<"<unknown>" in "">
// CHECK-DAG: #[[FUSED_CU:.*]] = #cuda_tile.di_compile_unit<file = #[[FUSED_FILE]]>
// CHECK-DAG: #[[FUSED_SUB:.*]] = #cuda_tile.di_subprogram<file = #[[FUSED_FILE]], line = 1, name = "func_fused_no_fileline", linkageName = "func_fused_no_fileline", compileUnit = #[[FUSED_CU]], scopeLine = 1>

#fused_nf_a = loc("nf_a"(unknown))
#fused_nf_b = loc("nf_b"(unknown))
#fused_nf = loc(fused[#fused_nf_a, #fused_nf_b])

cuda_tile.module @test {
  testing$func @func_fused_no_fileline() {
    return loc(unknown)
  } loc(#fused_nf)
} loc(unknown)

// -----

// FusedLoc whose first child is a direct FileLineColLoc: extractFileLoc takes the dyn_cast
// success path and returns from inside the loop. Use fused[#loc, #loc] aliases (not only
// fused["f":l:c, unknown]) so that child is a plain FileLineCol in the fused bag.
// CHECK-LABEL: testing$func @func_fused_first_is_fileline()
// CHECK: return loc(#loc[[LOC_FF:[0-9]+]])
// CHECK: } loc(#loc[[LOC_FF]])

// CHECK-DAG: #[[FF_FILE:.*]] = #cuda_tile.di_file<"fused_branch_file.py" in "">
// CHECK-DAG: #[[FF_CU_UNK:.*]] = #cuda_tile.di_file<"<unknown>" in "">
// CHECK-DAG: #[[FF_FN_SRC:.*]] = loc("fused_branch_file.py":11:5)
// CHECK-DAG: #[[FF_CU:.*]] = #cuda_tile.di_compile_unit<file = #[[FF_CU_UNK]]>
// CHECK-DAG: #[[FF_SUB:.*]] = #cuda_tile.di_subprogram<file = #[[FF_FILE]], line = 11, name = "func_fused_first_is_fileline", linkageName = "func_fused_first_is_fileline", compileUnit = #[[FF_CU]], scopeLine = 11>
// CHECK-DAG: #loc[[LOC_FF]] = #cuda_tile.di_loc<#[[FF_FN_SRC]] in #[[FF_SUB]]>

#ff_file = loc("fused_branch_file.py":11:5)
#ff_other = loc("ff_o"(unknown))
#ff_fused = loc(fused[#ff_file, #ff_other])

cuda_tile.module @test {
  testing$func @func_fused_first_is_fileline() {
    return loc(unknown)
  } loc(#ff_fused)
} loc(unknown)

// OpaqueLoc: not in the standard location assembly parsed by cuda-tile-opt here; exercise via
// C++ if line coverage for that arm is required.
