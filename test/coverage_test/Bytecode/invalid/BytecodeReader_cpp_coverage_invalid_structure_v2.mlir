// Additional crafted-bytecode failure cases that exercise reader error paths
// not covered by invalid_structure.mlir. Each .tileirbc fixture is a small,
// deliberately malformed bytecode file that triggers a single emitError() in
// BytecodeReader.cpp.

//===--------------------------------------------------------------------===//
// Truncated bytecode header — the 2-byte version tag is cut to 1 byte, so
// parseHeader's readLE(tag) fails (third clause of the header readLE chain).
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/truncated_version_tag.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TRUNC_VER
// TRUNC_VER: failed to parse bytecode header

//===--------------------------------------------------------------------===//
// Section header with alignment flag but alignment value not a power of 2.
// Triggers parseSectionHeader's "invalid alignment value" branch.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/bad_alignment_value.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=BAD_ALIGN
// BAD_ALIGN: invalid alignment value: 3

//===--------------------------------------------------------------------===//
// Section header with alignment flag and alignment value of zero.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/zero_alignment_value.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=ZERO_ALIGN
// ZERO_ALIGN: invalid alignment value: 0

//===--------------------------------------------------------------------===//
// Producer section with empty payload — readVarInt on empty buffer fails.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/truncated_producer.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TRUNC_PROD
// TRUNC_PROD: failed to read producer string index

//===--------------------------------------------------------------------===//
// Producer section references string index 5 in an empty string table.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/producer_bad_index.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=PROD_IDX
// PROD_IDX: failed to get producer string at index 5

//===--------------------------------------------------------------------===//
// Bytecode with only an end marker after the header — no string section.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/missing_string_section.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=NO_STR
// NO_STR: string section is mandatory

//===--------------------------------------------------------------------===//
// Bytecode with a string section but no function section.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/missing_function_section.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=NO_FUNC
// NO_FUNC: function section is mandatory

//===--------------------------------------------------------------------===//
// String section written without the alignment flag. readBytecode's
// validateSectionAlignment helper rejects sections that lack the flag.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/string_section_no_alignment.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=STR_NOALIGN
// STR_NOALIGN: section 1 must have alignment that is a multiple of 4 bytes

//===--------------------------------------------------------------------===//
// String section aligned to 1 (a power of two but not /4). Same alignment
// validator branch but exercises the alignment-value-non-multiple path.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/string_section_align_one.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=STR_ALIGN1
// STR_ALIGN1: section 1 must have alignment that is a multiple of 4 bytes

//===--------------------------------------------------------------------===//
// Type section: numTypes encoded as 2^64-1 → exceeds available payload.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/type_section_huge_numtypes.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TYPE_HUGE
// TYPE_HUGE: number of types (18446744073709551615) exceeds the maximum

//===--------------------------------------------------------------------===//
// Global section: numGlobals encoded as 2^64-1 → exceeds available payload.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/global_section_huge_numglobals.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=GLOB_HUGE
// GLOB_HUGE: number of globals (18446744073709551615) exceeds the maximum

//===--------------------------------------------------------------------===//
// Section length field larger than remaining stream bytes.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/section_length_exceeds_remaining.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=LEN_EXCEED
// LEN_EXCEED: section length 100 exceeds remaining data size

//===--------------------------------------------------------------------===//
// String section that claims 10 strings but offers nowhere near enough
// payload to hold the 10*sizeof(uint32_t) offset array.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/string_section_too_many_strings.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=STR_OVF
// STR_OVF: number of strings (10) exceeds the maximum

//===--------------------------------------------------------------------===//
// Function table section that claims 100 functions but payload is 1 byte.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/function_section_too_many_functions.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_OVF
// FUNC_OVF: number of functions (100) exceeds payload size

//===--------------------------------------------------------------------===//
// End-of-bytecode marker carrying the high alignment bit — explicitly
// forbidden by parseSectionHeader.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/end_section_with_alignment_flag.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=END_ALIGN
// END_ALIGN: end section should not have alignment flag set

//===--------------------------------------------------------------------===//
// Unknown section ID byte (0x42, which is >= NumSections).
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/unknown_section_id_42.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=UNK_ID
// UNK_ID: unknown section ID: 66

//===--------------------------------------------------------------------===//
// Constant section requires alignment multiple of 8. Provide 2.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/constant_section_bad_alignment.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=CONST_ALIGN
// CONST_ALIGN: section 4 must have alignment that is a multiple of 8 bytes

//===--------------------------------------------------------------------===//
// Debug section requires alignment multiple of 8. Provide 4.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/debug_section_bad_alignment.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=DEBUG_ALIGN
// DEBUG_ALIGN: section 3 must have alignment that is a multiple of 8 bytes

//===--------------------------------------------------------------------===//
// Type section requires alignment multiple of 4. Provide 2.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/type_section_bad_alignment.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TYPE_ALIGN
// TYPE_ALIGN: section 5 must have alignment that is a multiple of 4 bytes

//===--------------------------------------------------------------------===//
// Trailing bytes after the end-of-bytecode marker. Reader requires the
// stream to be fully consumed at EndOfBytecode.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/end_section_not_last.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=END_NOT_LAST
// END_NOT_LAST: end section is not the last section

//===--------------------------------------------------------------------===//
// Section header has alignment flag but no alignment varint follows.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/section_no_alignment_byte.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=NO_ALIGN_BYTE
// NO_ALIGN_BYTE: failed to parse section header

//===--------------------------------------------------------------------===//
// Completely empty file — magic check fails on the first byte read.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/empty_file.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=EMPTY_FILE
// EMPTY_FILE: truncated bytecode: missing magic byte at position 0

//===--------------------------------------------------------------------===//
// Header parsed successfully, but no bytes remain for the first section
// header. parseSectionHeader's "unexpected end of data" branch.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/eof_at_section_header.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=EOF_SEC_HDR
// EOF_SEC_HDR: unexpected end of data while reading section header

//===--------------------------------------------------------------------===//
// Constant section: numConstants=2 but no room for the 8-byte-aligned
// offset table — falls into the constant section failure path.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/constant_section_insufficient_data.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=CONST_INSUFF
// CONST_INSUFF: failed to parse constant section

//===--------------------------------------------------------------------===//
// Debug section payload empty → first varint read fails.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/debug_section_empty_payload.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=DEBUG_EMPTY
// DEBUG_EMPTY: failed to read total number of operations with debug info

//===--------------------------------------------------------------------===//
// A 13.5 di-subprogram record whose flags varint sets bit 1, which no public
// build defines, so the reader reports "unrecognized flag bits". Guards against
// a writer that knows flag bits this build does not.
//
// Unlike the fixtures above, this one is a *valid* module with a single byte
// patched, so its offset is tied to the current record layout. To rebuild it,
// serialize a module holding one `di_subprogram` with `flags = declaration` at
// -bytecode-version=13.5, then change the flags varint (the 0x01 byte, at
// offset 104 in the module used here) to 0x02.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/di_subprogram_unrecognized_flag.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=DI_BAD_FLAG
// DI_BAD_FLAG: unrecognized flag bits (0x2) when parsing DISubprogramAttr

//===--------------------------------------------------------------------===//
// Function-table entry with bit 3 (0x08) set in entryFlag on a 13.4 buffer →
// "function entry has unrecognized flag bits". Pre-13.5 only bits 0-2 are
// recognized; 13.5 Form claims bits [3:2], so this fixture stays at 13.4.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/function_unrecognized_flag.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FN_BAD_FLAG
// FN_BAD_FLAG: function entry has unrecognized flag bits

//===--------------------------------------------------------------------===//
// Reuse the minimal function-table fixture above with a 13.5 header and
// FormReserved (0b11 in bits [3:2]).
//===--------------------------------------------------------------------===//
// RUN: %PYTHON -c "p=bytearray(open(r'%S/function_unrecognized_flag.tileirbc','rb').read()); p[9]=5; p[27]=0x0c; open(r'%t.reserved-form.tileirbc','wb').write(p)"
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %t.reserved-form.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FN_RESERVED_FORM
// FN_RESERVED_FORM: function entry has reserved form value

//===--------------------------------------------------------------------===//
// A 13.5 function-table entry cannot combine KindKernel with FormTesting.
//===--------------------------------------------------------------------===//
// RUN: %PYTHON -c "p=bytearray(open(r'%S/function_unrecognized_flag.tileirbc','rb').read()); p[9]=5; p[27]=0x0a; open(r'%t.testing-kernel.tileirbc','wb').write(p)"
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %t.testing-kernel.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FN_TESTING_KERNEL
// FN_TESTING_KERNEL: function cannot be both testing func and kernel entry

//===--------------------------------------------------------------------===//
// Function-table entry whose body length exceeds remaining section bytes.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/function_body_length_too_large.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FN_BAD_LEN
// FN_BAD_LEN: function body length 100 exceeds remaining bytecode data

//===--------------------------------------------------------------------===//
// Truncated header where verMajor read fails at EOF (only magic present).
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/eof_at_version.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=EOF_VER
// EOF_VER: failed to parse bytecode header

//===--------------------------------------------------------------------===//
// Producer section payload contains a varint whose continuation bits run
// past 64 bits — exercises readVarInt's "shift > 63" guard.
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/producer_overlong_varint.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=PROD_OVERLONG
// PROD_OVERLONG: failed to read producer string index
