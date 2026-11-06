// RUN: cuda-tile-opt --test-debuginfo-update-symbol-name="symbol-to-rename=kernels::my_func new-name=_renamed_func update-uses=false" --mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-NO-SYMTAB
// RUN: cuda-tile-opt --test-debuginfo-update-symbol-name="symbol-to-rename=kernels::my_func new-name=_renamed_func update-uses=true" --mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-WITH-SYMTAB

// Test 1 (CHECK-NO-SYMTAB): update-uses=false means symtab=nullptr is passed
// to updateSymbolName. Covers (symtab is null).
//   The entry has a DILocAttr with a subprogram, so we also go through the
//   replacer path. The replacer walks the whole module scope and encounters
// #other_sp (from other_func), exercising (subprogram !=
//   oldSP -> return subprogram unchanged).

// Test 2 (CHECK-WITH-SYMTAB): update-uses=true exercises the symtab rename.

#file = #cuda_tile.di_file<"test.py" in "/tmp/">

#compile_unit = #cuda_tile.di_compile_unit<
  file = #file
>

#func_sp = #cuda_tile.di_subprogram<
  file = #file,
  line = 10,
  name = "my_func",
  linkageName = "my_func",
  compileUnit = #compile_unit,
  scopeLine = 11
>

// A second, different subprogram. The replacer lambda walks the entire scope
// and will encounter this subprogram (on other_func's loc). Since it does not
// match #func_sp, the ternary returns it unchanged -- covering.
#other_sp = #cuda_tile.di_subprogram<
  file = #file,
  line = 20,
  name = "other_func",
  linkageName = "other_func",
  compileUnit = #compile_unit,
  scopeLine = 21
>

#loc_func = loc("/tmp/test.py":10:0)
#loc_other = loc("/tmp/test.py":20:0)

#di_loc_func = #cuda_tile.di_loc<#loc_func in #func_sp>
#di_loc_other = #cuda_tile.di_loc<#loc_other in #other_sp>

cuda_tile.module @kernels {
  // CHECK-NO-SYMTAB: @_renamed_func()
  // CHECK-WITH-SYMTAB: @_renamed_func()
  entry @my_func() {
    %c1 = constant <i32: 1> : !cuda_tile.tile<i32> loc(#di_loc_func)
    return loc(unknown)
  } loc(#di_loc_func)

  // This entry has a different subprogram in its loc. The replacer will see
  // #other_sp here and take the "not equal" branch of the ternary.
  entry @other_func() {
    %c2 = constant <i32: 2> : !cuda_tile.tile<i32> loc(#di_loc_other)
    return loc(unknown)
  } loc(#di_loc_other)
} loc(unknown)

// CHECK-NO-SYMTAB-DAG: #cuda_tile.di_subprogram<{{.*}}, name = "my_func", linkageName = "_renamed_func"
// CHECK-WITH-SYMTAB-DAG: #cuda_tile.di_subprogram<{{.*}}, name = "my_func", linkageName = "_renamed_func"
