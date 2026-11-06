// Coverage test for BytecodeReader.cpp deep error handling paths.
// Uses surgically corrupted valid bytecodes to exercise section-specific parsing errors.

//===--------------------------------------------------------------------===//
// Function section: body length exceeds remaining data
// Exercises parseFunctionTableSection
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/func_body_too_long.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_BODYLEN
// FUNC_BODYLEN: function body length {{[0-9]+}} exceeds remaining bytecode data

//===--------------------------------------------------------------------===//
// Function section: truncated metadata (100 functions but tiny payload)
// Exercises parseFunctionTableSection
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/func_truncated_metadata.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_TRUNC
// FUNC_TRUNC: number of functions (100) exceeds payload size

//===--------------------------------------------------------------------===//
// createFunction: signature index out of bounds
// Exercises createFunction
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/func_bad_sig_idx.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_SIGIDX
// FUNC_SIGIDX: function signature index {{[0-9]+}} out of bounds

//===--------------------------------------------------------------------===//
// createFunction: name index out of bounds (string lookup fails)
// Exercises createFunction and getString
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/func_bad_name_idx.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_NAMEIDX
// FUNC_NAMEIDX: string index {{[0-9]+}} out of bounds

//===--------------------------------------------------------------------===//
// createGlobal: constant value index out of bounds
// Exercises createGlobal
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/global_bad_const_idx.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=GLOBAL_CONSTIDX
// GLOBAL_CONSTIDX: global value constant index out of bounds

//===--------------------------------------------------------------------===//
// createGlobal: invalid symbol visibility value
// Exercises createGlobal
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/global_bad_visibility.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=GLOBAL_VIS
// GLOBAL_VIS: invalid symbol visibility value: 99

//===--------------------------------------------------------------------===//
// createGlobal: name index out of bounds
// Exercises createGlobal
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/global_bad_name_idx.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=GLOBAL_NAMEIDX
// GLOBAL_NAMEIDX: failed to get global symbol name string for index

//===--------------------------------------------------------------------===//
// parseTypeSection: excessive type count relative to payload
// Exercises parseTypeSection
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/type_insufficient_data.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TYPE_EXCESS
// TYPE_EXCESS: number of types (100) exceeds the maximum

//===--------------------------------------------------------------------===//
// parseTypeSection: empty type table then createFunction fails
// Exercises parseTypeSection and createFunction 2317
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/type_empty.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TYPE_EMPTY
// TYPE_EMPTY: function signature index {{[0-9]+}} out of bounds

//===--------------------------------------------------------------------===//
// parseStringSection: excessive string count
// Exercises parseStringSection
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/string_excessive_count.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=STR_EXCESS
// STR_EXCESS: number of strings (100) exceeds the maximum

//===--------------------------------------------------------------------===//
// parseStringSection: empty string table then createFunction fails
// Exercises getString
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/string_empty.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=STR_EMPTY
// STR_EMPTY: string index {{[0-9]+}} out of bounds

//===--------------------------------------------------------------------===//
// parseGlobalSection: excessive global count
// Exercises parseGlobalSection
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/global_truncated_entry.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=GLOBAL_EXCESS
// GLOBAL_EXCESS: number of globals (5) exceeds the maximum

//===--------------------------------------------------------------------===//
// createGlobal: constant table empty, so constant index out of bounds
// Exercises createGlobal via empty constant section
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/constant_empty.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=CONST_EMPTY
// CONST_EMPTY: global value constant index out of bounds
