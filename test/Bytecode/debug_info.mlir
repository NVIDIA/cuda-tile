// Roundtrip test with DebugInfo section
// RUN: %round_trip_test %s %t --mlir-print-debuginfo

cuda_tile.module @kernels {
  entry @no_parameters() {
    %cst_42_i32 = constant <i32: 42> : tile<i32> loc(#loc5)
    return loc(#loc6)
  } loc(#loc4)
} loc(#loc)
#di_file = #cuda_tile.di_file<"debug_info.mlir" in "foo">
#loc = loc(unknown)
#loc1 = loc("debug_info.mlir":8:3)
#loc2 = loc("debug_info.mlir":10:10)
#loc3 = loc("debug_info.mlir":12:5)
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file>
#di_subprogram = #cuda_tile.di_subprogram<file = #di_file, line = 8, name = "no_parameters", linkageName = "no_parameters", compileUnit = #di_compile_unit, scopeLine = 8>
#loc4 = #cuda_tile.di_loc<#loc1 in #di_subprogram>
#loc5 = #cuda_tile.di_loc<#loc2 in #di_subprogram>
#loc6 = #cuda_tile.di_loc<#loc3 in #di_subprogram>
