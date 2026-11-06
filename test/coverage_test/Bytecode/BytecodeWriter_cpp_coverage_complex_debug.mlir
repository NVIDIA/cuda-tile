// Test bytecode roundtrip with complex debug info patterns.
// Exercises debug info registration paths for various location types
// including nested CallSiteLoc, multiple DILexicalBlocks, and DILocAttr
// with different scopes.
// RUN: %round_trip_test %s %t --mlir-print-debuginfo

cuda_tile.module @kernels {
  entry @complex_debug(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    // Multiple operations with different debug locations to fill
    // the debug info section and exercise section writing paths
    %c0 = constant <i32: 0> : tile<i32> loc(#loc_inner1)
    %c1 = constant <i32: 1> : tile<i32> loc(#loc_inner2)
    %c2 = constant <i32: 2> : tile<i32> loc(#loc_inner3)
    %sum = addf %a, %b rounding<nearest_even> : tile<f32> loc(#loc_lexblock)
    // Nested callsite locations to exercise CallSiteLoc
    %c3 = constant <i32: 3> : tile<i32> loc(#loc_nested_callsite)
    return loc(#loc_inner1)
  } loc(#loc_func)
} loc(#loc)

#di_file = #cuda_tile.di_file<"complex_debug.cu" in "/home/test">
#di_file2 = #cuda_tile.di_file<"header.cuh" in "/home/test/include">
#loc = loc(unknown)
#loc1 = loc("complex_debug.cu":10:5)
#loc2 = loc("complex_debug.cu":20:3)
#loc3 = loc("complex_debug.cu":30:7)
#loc4 = loc("complex_debug.cu":40:1)
#loc5 = loc("header.cuh":5:10)
#loc6 = loc("complex_debug.cu":50:2)
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file>
#di_subprogram = #cuda_tile.di_subprogram<file = #di_file, line = 10, name = "complex_debug", linkageName = "complex_debug", compileUnit = #di_compile_unit, scopeLine = 10>
// Lexical block scope
#di_lexical_block = #cuda_tile.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 15, column = 3>
// Subprogram in a different file (header)
#di_subprogram2 = #cuda_tile.di_subprogram<file = #di_file2, line = 5, name = "helper", linkageName = "helper", compileUnit = #di_compile_unit, scopeLine = 5>

#loc_func = #cuda_tile.di_loc<#loc1 in #di_subprogram>
#loc_inner1 = #cuda_tile.di_loc<#loc2 in #di_subprogram>
#loc_inner2 = #cuda_tile.di_loc<#loc3 in #di_subprogram>
#loc_inner3 = #cuda_tile.di_loc<#loc4 in #di_lexical_block>
#loc_lexblock = #cuda_tile.di_loc<#loc6 in #di_lexical_block>
// CallSiteLoc with callee in header
#loc_callee = #cuda_tile.di_loc<#loc5 in #di_subprogram2>
#loc_caller = #cuda_tile.di_loc<#loc4 in #di_subprogram>
// Nested callsite: callsite at a callsite
#loc_inner_callsite = loc(callsite(#loc_caller at #loc_func))
#loc_nested_callsite = loc(callsite(#loc_callee at #loc_inner_callsite))
