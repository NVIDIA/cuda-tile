// RUN: not cuda-tile-opt --split-input-file --mlir-print-debuginfo --allow-unregistered-dialect %s 2>&1 | FileCheck %s
// RUN: not cuda-tile-translate --test-cudatile-roundtrip --no-implicit-module --split-input-file --mlir-print-debuginfo --allow-unregistered-dialect %s 2>&1 | FileCheck %s

// NOTE: This test generates invalid debug info. The presence of invalid debug
// info means that the typical --verify-diagnostics flow used for invalid tests
// will not work for this test as that flow relies on valid debug info. The
// inability to use the --verify-diagnostics flow means that this test is
// expected to fail. The expected failure means that the bytecode
// round_trip_test.sh script will also not work for this test.

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
// end common test setup

// Rule 1: If a function has scope, it must have subprogram scope.
// Test C: Using entry with NameLoc wrapper
// CHECK: invalid function debug info scope
// CHECK: Function location must have cuda_tile.di_subprogram debug info scope
cuda_tile.module @kernels {
  entry @test() {
    return loc(#di_loc_func)
  } loc("entry_loc"(#di_loc_block))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
// end common test setup

// Rule 1: If a function has scope, it must have subprogram scope.
// Test D: Using entry with FusedLoc wrapper
// CHECK: invalid function debug info scope
// CHECK: Function location must have cuda_tile.di_subprogram debug info scope
cuda_tile.module @kernels {
  entry @test() {
    return loc(#di_loc_func)
  } loc(fused[#loc_func, #di_loc_block])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
// end common test setup

// Rule 2: If a function has subprogram scope, the function name must match the subprogram scope linkage name.
// Test C: Using entry with NameLoc wrapper
// CHECK: invalid function debug info scope
// CHECK: Function name "foo" does not match subprogram scope linkage name "test"
cuda_tile.module @kernels {
  entry @foo() {
    return loc(#di_loc_func)
  } loc("entry_loc"(#di_loc_func))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
// end common test setup

// Rule 2: If a function has subprogram scope, the function name must match the subprogram scope linkage name.
// Test D: Using entry with FusedLoc wrapper
// CHECK: invalid function debug info scope
// CHECK: Function name "foo" does not match subprogram scope linkage name "test"
cuda_tile.module @kernels {
  entry @foo() {
    return loc(#di_loc_func)
  } loc(fused[#loc_func, #di_loc_func])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test D: Using entry with operation having NameLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    return loc("op_loc"(#di_loc_func))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test E: Using entry with operation having FusedLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    return loc(fused[#loc_func, #di_loc_func])
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test F: Using entry with operation having CallSiteLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    return loc(callsite(#loc_func at #di_loc_func))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test G: Using entry with block scope operation having NameLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    return loc("op_loc"(#di_loc_block))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test H: Using entry with block scope operation having FusedLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    return loc(fused[#loc_func, #di_loc_block])
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test I: Using entry with block scope operation having CallSiteLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    return loc(callsite(#loc_func at #di_loc_block))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test P: Using entry with if-else operation having NameLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield loc("op_loc"(#di_loc_func))
    } else {
      cuda_tile.yield
    }
    return
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test Q: Using entry with if-else operation having FusedLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield loc(fused[#loc_func, #di_loc_func])
    } else {
      cuda_tile.yield
    }
    return
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test R: Using entry with if-else operation having CallSiteLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  entry @test() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield loc(callsite(#loc_func at #di_loc_func))
    } else {
      cuda_tile.yield
    }
    return
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test B1: entry + subprogram scope (function NameLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc("op_loc"(#di_loc_func))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test B2: entry + subprogram scope (function NameLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(fused[#loc_func, #di_loc_func])
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test B3: entry + subprogram scope (function NameLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(callsite(#loc_func at #di_loc_func))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test B4: entry + subprogram scope (function FusedLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc("op_loc"(#di_loc_func))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test B5: entry + subprogram scope (function FusedLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(fused[#loc_func, #di_loc_func])
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test B6: entry + subprogram scope (function FusedLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(callsite(#loc_func at #di_loc_func))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test D1: entry + block scope (function NameLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc("op_loc"(#di_loc_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test D2: entry + block scope (function NameLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(fused[#loc_func, #di_loc_block])
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test D3: entry + block scope (function NameLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(callsite(#loc_func at #di_loc_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test D4: entry + block scope (function FusedLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc("op_loc"(#di_loc_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test D5: entry + block scope (function FusedLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(fused[#loc_func, #di_loc_block])
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test D6: entry + block scope (function FusedLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(callsite(#loc_func at #di_loc_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test F1: entry + inner block scope (function NameLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc("op_loc"(#di_loc_inner_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test F2: entry + inner block scope (function NameLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(fused[#loc_func, #di_loc_inner_block])
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test F3: entry + inner block scope (function NameLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(callsite(#loc_func at #di_loc_inner_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test F4: entry + inner block scope (function FusedLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc("op_loc"(#di_loc_inner_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test F5: entry + inner block scope (function FusedLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(fused[#loc_func, #di_loc_inner_block])
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test F6: entry + inner block scope (function FusedLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  entry @invalid() {
    return loc(callsite(#loc_func at #di_loc_inner_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
// end common test setup

// Rule 5: Global variables must not have scope.
// Test A: Using NameLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Global variables must not have scope
cuda_tile.module @kernels {
  "some.op"() : () -> () loc("global_op"(#di_loc_func))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
// end common test setup

// Rule 5: Global variables must not have scope.
// Test B: Using FusedLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Global variables must not have scope
cuda_tile.module @kernels {
  "some.op"() : () -> () loc(fused[#loc_func, #di_loc_func])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
// end common test setup

// Rule 5: Global variables must not have scope.
// Test C: Using CallSiteLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Global variables must not have scope
cuda_tile.module @kernels {
  "some.op"() : () -> () loc(callsite(#loc_func at #di_loc_func))
}


// **************************** Non-verifier Tests ******************************

// -----

#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
// CHECK: expected a parameter name in struct
#compile_unit = #cuda_tile.di_compile_unit<>

// -----
// CHECK: struct is missing required parameter: name 
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram1 = #cuda_tile.di_subprogram<file = #file, line = 1, linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>

// -----
// CHECK: struct is missing required parameter: linkageName
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram2 = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", compileUnit = #compile_unit, scopeLine = 2>
