//===- BytecodeAttrCodeGen.h ------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BytecodeAttrCodeGen.h - Attribute Bytecode Code Generation
//
// This file declares functions for generating C++ code from analyzed
// attribute bytecode structures.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_CODEGEN_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_CODEGEN_H_

#include "llvm/Support/raw_ostream.h"

namespace mlir::tblgen {

struct BytecodeAttrStructure;

//===----------------------------------------------------------------------===//
// Code Generation Entry Points
//===----------------------------------------------------------------------===//

/// Generate the AttributeTag enum.
void generateAttrTagEnum(const BytecodeAttrStructure &structure,
                         llvm::raw_ostream &os);

/// Generate runtime version checking function for attribute tags.
void generateAttrVersionCheck(const BytecodeAttrStructure &structure,
                              llvm::raw_ostream &os);

/// Generate runtime classifiers for public-preview attribute tags and values.
void generatePreviewAttrClassification(const BytecodeAttrStructure &structure,
                                       llvm::raw_ostream &os);

/// Generate runtime classification of public-preview attribute values without
/// bytecode-tag helpers.
void generatePreviewAttrValueClassification(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os);

/// Generate is_cuda_tile_enum type trait for BytecodeWriter.
void generateEnumTypeTrait(const BytecodeAttrStructure &structure,
                           llvm::raw_ostream &os);

/// Generate is_cuda_tile_enum_attr type trait and symbolizeEnum for
/// BytecodeReader.
void generateEnumAttrTypeTrait(const BytecodeAttrStructure &structure,
                               llvm::raw_ostream &os);

/// Generate runtime version checking for enum attributes.
void generateEnumAttrVersionCheck(const BytecodeAttrStructure &structure,
                                  llvm::raw_ostream &os);

/// Generate per-value version checking for enum values.
void generateEnumValueVersionCheck(const BytecodeAttrStructure &structure,
                                   llvm::raw_ostream &os);

/// Generate attribute version map for MinVersionAnalyzer.
void generateAttrVersionMap(const BytecodeAttrStructure &structure,
                            llvm::raw_ostream &os);

/// Generate enum value version map for MinVersionAnalyzer.
void generateEnumValueVersionMap(const BytecodeAttrStructure &structure,
                                 llvm::raw_ostream &os);

/// Generate attribute parameter version map for MinVersionAnalyzer.
void generateAttrParamVersionMap(const BytecodeAttrStructure &structure,
                                 llvm::raw_ostream &os);
//===----------------------------------------------------------------------===//
// Attribute Serialization/Deserialization Code Generation
//===----------------------------------------------------------------------===//

/// Generate serializers for CudaTile attributes with parameters.
/// Mirrors generateTypeSerializers in BytecodeTypeCodeGen.
void generateAttrSerializers(const BytecodeAttrStructure &structure,
                             llvm::raw_ostream &os);

/// Generate deserializers for CudaTile attributes with parameters.
/// Mirrors generateTypeDeserializers in BytecodeTypeCodeGen.
void generateAttrDeserializers(const BytecodeAttrStructure &structure,
                               llvm::raw_ostream &os);

/// Generate dispatch switch for attribute serialization.
void generateAttrSerializerDispatch(const BytecodeAttrStructure &structure,
                                    llvm::raw_ostream &os);

/// Generate dispatch switch for attribute deserialization.
void generateAttrDeserializerDispatch(const BytecodeAttrStructure &structure,
                                      llvm::raw_ostream &os);

/// Generate explicit case labels for CudaTile attrs in the reader's
/// self-contained attribute switch. Emits only the `case Tag:` lines.
void generateAttrReaderSwitchCases(const BytecodeAttrStructure &structure,
                                   llvm::raw_ostream &os);

/// Generate is_cuda_tile_serializable_attr type trait for compile-time
/// dispatch in the BytecodeReader's parseOpAttribute template.
void generateSerializableAttrTypeTrait(const BytecodeAttrStructure &structure,
                                       llvm::raw_ostream &os);

/// Generate parseCudaTileAttrInline<T> template specializations that dispatch
/// to the generated parse*Data functions. Used by parseOpAttribute to handle
/// all CudaTile serializable attrs.
void generateAttrInlineReaderDispatch(const BytecodeAttrStructure &structure,
                                      llvm::raw_ostream &os);

/// Generate dependent type registration logic for CudaTile attributes.
/// For each CudaTileAttr with type-bearing parameters (TypeAttr, NestedAttr,
/// AttributeArray), emits dispatch that registers nested types or recurses
/// into nested attrs. Mirrors generateDependentTypeRegistration for types.
void generateDependentAttrTypeRegistration(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os);

} // namespace mlir::tblgen

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_CODEGEN_H_
