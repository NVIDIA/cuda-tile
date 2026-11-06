// Test that unsupported location types (NameLoc, FusedLoc) are rejected
// by the bytecode writer with meaningful error messages.
// Exercises validateDebugInfo and invalidLocError paths.

// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -split-input-file %s 2>&1 | FileCheck %s

// Test: NameLoc on operation triggers invalidLocError with "NameLoc" message
// CHECK: unsupported location, got NameLoc
cuda_tile.module @test_nameloc {
  cuda_tile.entry @nameloc_op() {
    %c = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32> loc("myname")
    cuda_tile.return
  } loc(#loc_func)
} loc(#loc)

#di_file = #cuda_tile.di_file<"nameloc_test.cu" in "/tmp">
#loc = loc(unknown)
#loc1 = loc("nameloc_test.cu":10:5)
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file>
#di_subprogram = #cuda_tile.di_subprogram<file = #di_file, line = 10, name = "nameloc_op", linkageName = "nameloc_op", compileUnit = #di_compile_unit, scopeLine = 10>
#loc_func = #cuda_tile.di_loc<#loc1 in #di_subprogram>

// -----

// Test: FusedLoc on operation triggers invalidLocError with "FusedLoc" message
// CHECK: unsupported location, got FusedLoc
cuda_tile.module @test_fusedloc {
  cuda_tile.entry @fusedloc_op() {
    %c = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32> loc(fused["test.cu":1:1, "test.cu":2:2])
    cuda_tile.return
  } loc(#loc2_func)
} loc(#loc2)

#di_file2 = #cuda_tile.di_file<"fusedloc_test.cu" in "/tmp">
#loc2 = loc(unknown)
#loc2_1 = loc("fusedloc_test.cu":10:5)
#di_compile_unit2 = #cuda_tile.di_compile_unit<file = #di_file2>
#di_subprogram2 = #cuda_tile.di_subprogram<file = #di_file2, line = 10, name = "fusedloc_op", linkageName = "fusedloc_op", compileUnit = #di_compile_unit2, scopeLine = 10>
#loc2_func = #cuda_tile.di_loc<#loc2_1 in #di_subprogram2>
