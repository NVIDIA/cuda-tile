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
// VERSION: unsupported Tile version 18.0.0, this reader supports versions [13.1, 13.1]

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

