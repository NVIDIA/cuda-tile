// Roundtrip test with CallSiteLoc and DILexicalBlock debug info.
// This exercises the registerDebugInfo paths for CallSiteLoc and
// DILexicalBlockAttr, and the serialize methods for those debug types.
// RUN: %round_trip_test %s %t --mlir-print-debuginfo

cuda_tile.module @kernels {
  entry @callsite_loc_test() {
    %cst = constant <i32: 42> : tile<i32> loc(#loc_callsite)
    return loc(#loc_inner_callsite)
  } loc(#loc_func)
} loc(#loc)

#di_file = #cuda_tile.di_file<"callsite_test.cu" in "/tmp">
#loc = loc(unknown)
#loc1 = loc("callsite_test.cu":10:5)
#loc2 = loc("callsite_test.cu":20:3)
#loc3 = loc("callsite_test.cu":30:7)
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file>
#di_subprogram = #cuda_tile.di_subprogram<file = #di_file, line = 10, name = "callsite_loc_test", linkageName = "callsite_loc_test", compileUnit = #di_compile_unit, scopeLine = 10>
#di_lexical_block = #cuda_tile.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 15, column = 3>
#loc_inner = #cuda_tile.di_loc<#loc1 in #di_subprogram>
#loc_callee = #cuda_tile.di_loc<#loc2 in #di_lexical_block>
#loc_caller = #cuda_tile.di_loc<#loc3 in #di_subprogram>
#loc_callsite = loc(callsite(#loc_callee at #loc_caller))
#loc_inner_callsite = loc(callsite(#loc_inner at #loc_caller))
#loc_func = #cuda_tile.di_loc<#loc1 in #di_subprogram>
