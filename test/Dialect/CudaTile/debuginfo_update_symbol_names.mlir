// RUN: cuda-tile-opt --test-debuginfo-update-symbol-name="symbol-to-rename=kernels::test_func new-name=_fqZtest_func_f32 update-uses=true" --mlir-print-debuginfo %s | FileCheck %s

#file = #cuda_tile.di_file<"foo.py" in "/tmp/">

#compile_unit = #cuda_tile.di_compile_unit<
  file = #file
>

// Ensure the linkage name was updated, and that the rename (which rebuilds the
// attribute through cloneWithNewLinkageName) carries every other field over --
// `flags` is set here only so a dropped field would show up as a diff.
// CHECK-DAG: #cuda_tile.di_subprogram<{{.*}}, name = "test_func", linkageName = "_fqZtest_func_f32"{{.*}}, scopeLine = 2, flags = declaration>
#func = #cuda_tile.di_subprogram<
  file = #file,
  line = 1,
  name = "test_func",
  linkageName = "test_func",
  compileUnit = #compile_unit,
  scopeLine = 2,
  flags = declaration
>

#block_func = #cuda_tile.di_lexical_block<
  scope = #func,
  file = #file,
  line = 3,
  column = 4
>

#inner_block_func = #cuda_tile.di_lexical_block<
  scope = #block_func,
  file = #file,
  line = 5,
  column = 6
>

#loc_func = loc("/tmp/foo.py":7:8)
#loc_block = loc("/tmp/foo.py":9:10)
#loc_inner_block = loc("/tmp/foo.py":11:12)

#di_loc_func = #cuda_tile.di_loc<#loc_func in #func>
#di_loc_block_func = #cuda_tile.di_loc<#loc_block in #block_func>
#di_loc_inner_block_func = #cuda_tile.di_loc<#loc_inner_block in #inner_block_func>

cuda_tile.module @kernels {
  // CHECK-DAG: @_fqZtest_func_f32()
  entry @test_func() {
    %c1 = constant <i32: 1> : !cuda_tile.tile<i32> loc(#di_loc_func)
    %c2 = constant <i32: 2> : !cuda_tile.tile<i32> loc(#di_loc_block_func)
    %c3 = constant <i32: 3> : !cuda_tile.tile<i32> loc(#di_loc_inner_block_func)
    return loc(unknown)
  } loc(#di_loc_func)
} loc(unknown)
