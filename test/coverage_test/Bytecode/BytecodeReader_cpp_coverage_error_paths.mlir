// Coverage test for BytecodeReader.cpp error handling paths.
// Tests use crafted invalid bytecode files to exercise error branches.

//===--------------------------------------------------------------------===//
// Truncated header (magic only, no version bytes)
// Exercises parseHeader failure path when readLE(verMajor) fails
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/truncated_header.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TRUNC_HDR
// TRUNC_HDR: failed to parse bytecode header

//===--------------------------------------------------------------------===//
// Truncated version (magic + major but missing minor/tag)
// Exercises parseHeader failure path when readLE(verMinor)/readLE(tag) fails
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/truncated_version.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TRUNC_VER
// TRUNC_VER: failed to parse bytecode header

//===--------------------------------------------------------------------===//
// End section with alignment flag set
// Exercises parseSectionHeader end-section alignment check
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/end_section_aligned.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=END_ALIGN
// END_ALIGN: end section should not have alignment flag set

//===--------------------------------------------------------------------===//
// Section with non-power-of-2 alignment value
// Exercises parseSectionHeader alignment validation
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/non_power2_alignment.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=BAD_ALIGN
// BAD_ALIGN: invalid alignment value: 3

//===--------------------------------------------------------------------===//
// Section with zero alignment value
// Exercises parseSectionHeader alignment==0 check
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/zero_alignment.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=ZERO_ALIGN
// ZERO_ALIGN: invalid alignment value: 0

//===--------------------------------------------------------------------===//
// Section with length exceeding remaining data
// Exercises parseSectionHeader length check
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/section_length_exceeds.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=SEC_LEN
// SEC_LEN: section length {{[0-9]+}} exceeds remaining data size

//===--------------------------------------------------------------------===//
// Truncated data before end section (parseSectionHeader can't read ID)
// Exercises parseSectionHeader remaining<1 check
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/truncated_end.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TRUNC_END
// TRUNC_END: unexpected end of data while reading section header

//===--------------------------------------------------------------------===//
// Missing string section (only end marker)
// Exercises readBytecode mandatory string section check
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/missing_string_section.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=NO_STRING
// NO_STRING: string section is mandatory

//===--------------------------------------------------------------------===//
// Missing function section (string section present but no func)
// Exercises readBytecode mandatory func section check
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/missing_func_section.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=NO_FUNC
// NO_FUNC: function section is mandatory

//===--------------------------------------------------------------------===//
// Trailing data after end section
// Exercises readBytecode end-section remaining!=0 check
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/trailing_data_after_end.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TRAIL
// TRAIL: end section is not the last section

//===--------------------------------------------------------------------===//
// String section without required alignment
// Exercises readBytecode validateSectionAlignment for String
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/string_no_align.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=STR_NOALIGN
// STR_NOALIGN: section 1 must have alignment that is a multiple of 4

//===--------------------------------------------------------------------===//
// Type section without required alignment
// Exercises readBytecode validateSectionAlignment for Type
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/type_no_align.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TYPE_NOALIGN
// TYPE_NOALIGN: section 5 must have alignment that is a multiple of 4

//===--------------------------------------------------------------------===//
// Constant section without required alignment
// Exercises readBytecode validateSectionAlignment for Constant
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/constant_no_align.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=CONST_NOALIGN
// CONST_NOALIGN: section 4 must have alignment that is a multiple of 8

//===--------------------------------------------------------------------===//
// Debug section without required alignment
// Exercises readBytecode validateSectionAlignment for Debug
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/debug_no_align.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=DEBUG_NOALIGN
// DEBUG_NOALIGN: section 3 must have alignment that is a multiple of 8

//===--------------------------------------------------------------------===//
// Func section without required alignment
// Exercises readBytecode validateSectionAlignment for Func
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/func_no_align.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_NOALIGN
// FUNC_NOALIGN: section 2 must have alignment that is a multiple of 8

//===--------------------------------------------------------------------===//
// Excessive string count in string section payload
// Exercises parseStringSection numStrings validation
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/excessive_string_count.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=STR_COUNT
// STR_COUNT: number of strings (9999999) exceeds the maximum

//===--------------------------------------------------------------------===//
// Excessive type count in type section payload
// Exercises parseTypeSection numTypes validation
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/excessive_type_count.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TYPE_COUNT
// TYPE_COUNT: number of types (9999999) exceeds the maximum

//===--------------------------------------------------------------------===//
// Excessive function count in function section payload
// Exercises parseFunctionTableSection numFunctions validation
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/excessive_func_count.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_COUNT
// FUNC_COUNT: number of functions (9999999) exceeds payload size

//===--------------------------------------------------------------------===//
// Excessive global count in global section payload
// Exercises parseGlobalSection numGlobals validation
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid/excessive_global_count.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=GLOBAL_COUNT
// GLOBAL_COUNT: number of globals (9999999) exceeds the maximum
