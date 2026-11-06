// Roundtrip test with complex debug info to exercise more BytecodeReader
// debug info parsing paths: DILexicalBlock, CallSiteLoc, multiple functions
// with debug info, and nested callsites.
// RUN: %round_trip_test %s %t --mlir-print-debuginfo

cuda_tile.module @kernels {
  // Function with DILexicalBlock locations
  entry @func_with_lexical_blocks(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %0 = addi %a, %b : tile<i32> loc(#loc_lex1)
    %1 = subi %a, %b : tile<i32> loc(#loc_lex2)
    return loc(#loc_lex3)
  } loc(#loc_func1)

  // Function with CallSiteLoc (inlined call site)
  entry @func_with_callsite(%x: !cuda_tile.tile<f32>) {
    %0 = absf %x : tile<f32> loc(#loc_callsite1)
    %1 = negf %x : tile<f32> loc(#loc_callsite2)
    return loc(#loc_callsite3)
  } loc(#loc_func2)

  // Function with nested callsite locations
  entry @func_with_nested_callsite(%y: !cuda_tile.tile<i32>) {
    %0 = constant <i32: 42> : tile<i32> loc(#loc_nested_callsite)
    return loc(#loc_deep)
  } loc(#loc_func3)
} loc(#loc_module)

// File and compile unit
#di_file1 = #cuda_tile.di_file<"main.cu" in "/home/user/project">
#di_file2 = #cuda_tile.di_file<"helper.cu" in "/home/user/project">
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file1>

// Subprograms
#di_subprogram1 = #cuda_tile.di_subprogram<file = #di_file1, line = 10, name = "func_with_lexical_blocks", linkageName = "func_with_lexical_blocks", compileUnit = #di_compile_unit, scopeLine = 10>
#di_subprogram2 = #cuda_tile.di_subprogram<file = #di_file1, line = 30, name = "func_with_callsite", linkageName = "func_with_callsite", compileUnit = #di_compile_unit, scopeLine = 30>
#di_subprogram3 = #cuda_tile.di_subprogram<file = #di_file2, line = 50, name = "func_with_nested_callsite", linkageName = "func_with_nested_callsite", compileUnit = #di_compile_unit, scopeLine = 50>
#di_callee_subprogram = #cuda_tile.di_subprogram<file = #di_file2, line = 100, name = "callee_func", linkageName = "callee_func", compileUnit = #di_compile_unit, scopeLine = 100>

// Lexical blocks
#di_lex_block1 = #cuda_tile.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 12, column = 5>
#di_lex_block2 = #cuda_tile.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 15, column = 3>

// Module and basic locations
#loc_module = loc(unknown)
#loc_raw1 = loc("main.cu":10:1)
#loc_raw2 = loc("main.cu":12:5)
#loc_raw3 = loc("main.cu":15:3)
#loc_raw4 = loc("main.cu":18:1)
#loc_raw5 = loc("main.cu":30:1)
#loc_raw6 = loc("main.cu":32:5)
#loc_raw7 = loc("main.cu":34:5)
#loc_raw8 = loc("main.cu":36:1)
#loc_raw9 = loc("helper.cu":100:5)
#loc_raw10 = loc("helper.cu":102:3)
#loc_raw11 = loc("helper.cu":50:1)
#loc_raw12 = loc("helper.cu":52:5)
#loc_raw13 = loc("helper.cu":54:1)

// Function locations
#loc_func1 = #cuda_tile.di_loc<#loc_raw1 in #di_subprogram1>
#loc_func2 = #cuda_tile.di_loc<#loc_raw5 in #di_subprogram2>
#loc_func3 = #cuda_tile.di_loc<#loc_raw11 in #di_subprogram3>

// Lexical block locations
#loc_lex1 = #cuda_tile.di_loc<#loc_raw2 in #di_lex_block1>
#loc_lex2 = #cuda_tile.di_loc<#loc_raw3 in #di_lex_block2>
#loc_lex3 = #cuda_tile.di_loc<#loc_raw4 in #di_subprogram1>

// CallSiteLoc locations (callee at caller)
#loc_callee1 = #cuda_tile.di_loc<#loc_raw9 in #di_callee_subprogram>
#loc_caller1 = #cuda_tile.di_loc<#loc_raw6 in #di_subprogram2>
#loc_callsite1 = loc(callsite(#loc_callee1 at #loc_caller1))
#loc_callee2 = #cuda_tile.di_loc<#loc_raw10 in #di_callee_subprogram>
#loc_caller2 = #cuda_tile.di_loc<#loc_raw7 in #di_subprogram2>
#loc_callsite2 = loc(callsite(#loc_callee2 at #loc_caller2))
#loc_callsite3 = loc(callsite(#loc_callee1 at #cuda_tile.di_loc<#loc_raw8 in #di_subprogram2>))

// Nested callsite: callsite(callsite(A at B) at C)
#loc_inner = #cuda_tile.di_loc<#loc_raw9 in #di_callee_subprogram>
#loc_outer_caller = #cuda_tile.di_loc<#loc_raw12 in #di_subprogram3>
#loc_top_caller = #cuda_tile.di_loc<#loc_raw13 in #di_subprogram3>
#loc_inner_callsite = loc(callsite(#loc_inner at #loc_outer_caller))
#loc_nested_callsite = loc(callsite(#loc_inner_callsite at #loc_top_caller))
#loc_deep = #cuda_tile.di_loc<#loc_raw13 in #di_subprogram3>
