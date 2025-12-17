// RUN: cuda-tile-opt --mlir-print-debuginfo %s | FileCheck %s

// CHECK-DAG: #[[FILE:[_a-zA-Z0-9]*]] = #cuda_tile.di_file<"foo.py" in "/tmp/">
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">

// CHECK-DAG: #[[COMPILE_UNIT:[_a-zA-Z0-9]*]] = #cuda_tile.di_compile_unit<file = #[[FILE]]>
#compile_unit = #cuda_tile.di_compile_unit<
  file = #file
>

// CHECK-DAG: #[[FUNC:[_a-zA-Z0-9]*]] = #cuda_tile.di_subprogram<file = #[[FILE]], line = 1, name = "test_func", linkageName = "test_func", compileUnit = #[[COMPILE_UNIT]], scopeLine = 2>
#func = #cuda_tile.di_subprogram<
  file = #file,
  line = 1,
  name = "test_func",
  linkageName = "test_func",
  compileUnit = #compile_unit,
  scopeLine = 2
>

// CHECK-DAG: #[[ENTRY:[_a-zA-Z0-9]*]] = #cuda_tile.di_subprogram<file = #[[FILE]], line = 1, name = "test_entry", linkageName = "test_entry", compileUnit = #[[COMPILE_UNIT]], scopeLine = 2>
#entry = #cuda_tile.di_subprogram<
  file = #file,
  line = 1,
  name = "test_entry",
  linkageName = "test_entry",
  compileUnit = #compile_unit,
  scopeLine = 2
>

// CHECK-DAG: #[[BLOCK_FUNC:[_a-zA-Z0-9]*]] = #cuda_tile.di_lexical_block<scope = #[[FUNC]], file = #[[FILE]], line = 3, column = 4>
#block_func = #cuda_tile.di_lexical_block<
  scope = #func,
  file = #file,
  line = 3,
  column = 4
>

// CHECK-DAG: #[[BLOCK_ENTRY:[_a-zA-Z0-9]*]] = #cuda_tile.di_lexical_block<scope = #[[ENTRY]], file = #[[FILE]], line = 3, column = 4>
#block_entry = #cuda_tile.di_lexical_block<
  scope = #entry,
  file = #file,
  line = 3,
  column = 4
>

// CHECK-DAG: #[[INNER_BLOCK_FUNC:[_a-zA-Z0-9]*]] = #cuda_tile.di_lexical_block<scope = #[[BLOCK_FUNC]], file = #[[FILE]], line = 5, column = 6>
#inner_block_func = #cuda_tile.di_lexical_block<
  scope = #block_func,
  file = #file,
  line = 5,
  column = 6
>

// CHECK-DAG: #[[INNER_BLOCK_ENTRY:[_a-zA-Z0-9]*]] = #cuda_tile.di_lexical_block<scope = #[[BLOCK_ENTRY]], file = #[[FILE]], line = 5, column = 6>
#inner_block_entry = #cuda_tile.di_lexical_block<
  scope = #block_entry,
  file = #file,
  line = 5,
  column = 6
>

// CHECK-DAG: [[LOC_FUNC:#loc[0-9]*]] = loc("/tmp/foo.py":7:8)
// CHECK-DAG: [[LOC_BLOCK:#loc[0-9]*]] = loc("/tmp/foo.py":9:10)
// CHECK-DAG: [[LOC_INNER_BLOCK:#loc[0-9]*]] = loc("/tmp/foo.py":11:12)
#loc_func = loc("/tmp/foo.py":7:8)
#loc_block = loc("/tmp/foo.py":9:10)
#loc_inner_block = loc("/tmp/foo.py":11:12)

// CHECK-DAG: [[DI_LOC_FUNC:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC_FUNC]] in #[[FUNC]]>
// CHECK-DAG: [[DI_LOC_BLOCK_FUNC:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC_BLOCK]] in #[[BLOCK_FUNC]]>
// CHECK-DAG: [[DI_LOC_INNER_BLOCK_FUNC:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC_INNER_BLOCK]] in #[[INNER_BLOCK_FUNC]]>
#di_loc_func = #cuda_tile.di_loc<#loc_func in #func>
#di_loc_block_func = #cuda_tile.di_loc<#loc_block in #block_func>
#di_loc_inner_block_func = #cuda_tile.di_loc<#loc_inner_block in #inner_block_func>

// CHECK-DAG: [[DI_LOC_ENTRY:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC_FUNC]] in #[[ENTRY]]>
// CHECK-DAG: [[DI_LOC_BLOCK_ENTRY:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC_BLOCK]] in #[[BLOCK_ENTRY]]>
// CHECK-DAG: [[DI_LOC_INNER_BLOCK_ENTRY:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC_INNER_BLOCK]] in #[[INNER_BLOCK_ENTRY]]>
#di_loc_entry = #cuda_tile.di_loc<#loc_func in #entry>
#di_loc_block_entry = #cuda_tile.di_loc<#loc_block in #block_entry>
#di_loc_inner_block_entry = #cuda_tile.di_loc<#loc_inner_block in #inner_block_entry>

cuda_tile.module @kernels {
  // CHECK-DAG: @test_func()
  // CHECK-DAG:   constant <i32: 1> : tile<i32> loc([[DI_LOC_FUNC]])
  // CHECK-DAG:   constant <i32: 2> : tile<i32> loc([[DI_LOC_BLOCK_FUNC]])
  // CHECK-DAG:   constant <i32: 3> : tile<i32> loc([[DI_LOC_INNER_BLOCK_FUNC]])
  // CHECK-DAG: } loc([[DI_LOC_FUNC]])
  entry @test_func() {
    %c1 = constant <i32: 1> : !cuda_tile.tile<i32> loc(#di_loc_func)
    %c2 = constant <i32: 2> : !cuda_tile.tile<i32> loc(#di_loc_block_func)
    %c3 = constant <i32: 3> : !cuda_tile.tile<i32> loc(#di_loc_inner_block_func)
    return loc(unknown)
  } loc(#di_loc_func)

  // CHECK-DAG: entry @test_entry()
  // CHECK-DAG:   constant <i32: 1> : tile<i32> loc([[DI_LOC_ENTRY]])
  // CHECK-DAG:   constant <i32: 2> : tile<i32> loc([[DI_LOC_BLOCK_ENTRY]])
  // CHECK-DAG:   constant <i32: 3> : tile<i32> loc([[DI_LOC_INNER_BLOCK_ENTRY]])
  // CHECK-DAG: } loc([[DI_LOC_ENTRY]])
  entry @test_entry() {
    %c1 = constant <i32: 1> : !cuda_tile.tile<i32> loc(#di_loc_entry)
    %c2 = constant <i32: 2> : !cuda_tile.tile<i32> loc(#di_loc_block_entry)
    %c3 = constant <i32: 3> : !cuda_tile.tile<i32> loc(#di_loc_inner_block_entry)
    return loc(unknown)
  } loc(#di_loc_entry)
} loc(unknown)
