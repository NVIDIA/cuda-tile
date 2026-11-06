// This file contains various failure test cases related to the structure of
// a bytecode file.

//===--------------------------------------------------------------------===//
// Magic Number
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_magic_number.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=MAGIC
// MAGIC: invalid magic number

//===--------------------------------------------------------------------===//
// Version
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/unsupported_version.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=VERSION
// VERSION: unsupported Tile version 18.0.0, this reader supports versions [13.1 - 13.5]

//===--------------------------------------------------------------------===//
// Version Tag
//===--------------------------------------------------------------------===//
// Only bit 0 of the 16-bit tag is defined; the fixture sets a reserved low bit
// and the whole reserved high byte.
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/unknown_version_tag_bits.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=VERSION_TAG
// VERSION_TAG: header has unrecognized version tag bits (0xFF02)

//===--------------------------------------------------------------------===//
// Section ID
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_section_id.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=SECTION_ID
// SECTION_ID: unknown section ID: 127

//===--------------------------------------------------------------------===//
// Section Length
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/excessive_section_length.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=SECTION_LENGTH
// SECTION_LENGTH: end section is not the last section

//===--------------------------------------------------------------------===//
// Invalid Dense Map Value
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_dense_map_value.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=DENSE_MAP
// DENSE_MAP: array contains unsupported value -2147483648

//===--------------------------------------------------------------------===//
// Invalid Attribute Name
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_attribute_name.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=ATTR_NAME
// ATTR_NAME: invalid empty attribute name for DictionaryAttr element 0

//===--------------------------------------------------------------------===//
// Constant rawDataSize additive overflow
//===--------------------------------------------------------------------===//
// A constant blob whose DenseElementsAttr `rawDataSize` varint is ~2^64.
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/constant_rawdatasize_overflow.tileirbc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=RAW_OVERFLOW
// RAW_OVERFLOW: failed to read the raw byte data
