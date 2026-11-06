//===- Remarks.td - CUDA Tile Remark Definitions -----------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_REMARKS_TD
#define CUDATILE_DIALECT_REMARKS_TD

include "cuda_tile/Dialect/CudaTile/IR/RemarkUtils.td"

//===--------------------------------------------------------------------===//
// Diagnostic kinds
//===--------------------------------------------------------------------===//

def DK_Pass     : TileIRDiagnosticKind<"Pass",     0, "Succeeded">;
def DK_Miss     : TileIRDiagnosticKind<"Miss",     1, "Missed">;
def DK_Fail     : TileIRDiagnosticKind<"Fail",     2, "Failed">;
def DK_Analysis : TileIRDiagnosticKind<"Analysis", 3, "Analysis">;
def DK_Error    : TileIRDiagnosticKind<"Error",    4, "Error">;
def DK_Warning  : TileIRDiagnosticKind<"Warning",  5, "Warning">;

//===--------------------------------------------------------------------===//
// Remark Categories
//===--------------------------------------------------------------------===//

def CategoryTensorCore : TileIRCategory<"TensorCore", 1, "Tensor-core">;
def CategoryMemory     : TileIRCategory<"Memory",     2, "Memory">;
def CategoryHints      : TileIRCategory<"Hints",      3, "Hints">;
//===--------------------------------------------------------------------===//
// Remarks
//===--------------------------------------------------------------------===//

// Tensor Core Remarks
def RemarkTensorCoreMMA
  : TileIRRemark<"RemarkTensorCoreMMA", DK_Pass, CategoryTensorCore, 1, "MMA operation successfully optimized to use Tensor Cores">;

def RemarkTensorCoreMMAFailed
  : TileIRRemark<"RemarkTensorCoreMMA", DK_Fail, CategoryTensorCore, 2, "MMA operation failed to optimize to use Tensor Cores, it is using FMA instructions instead">;

// Memory Remarks
def RemarkMemoryLoadInstructionSelected
  : TileIRRemark<"RemarkMemoryLoadInstructionSelected", DK_Pass, CategoryMemory, 1, "Load instruction selected">;

def RemarkMemoryStoreInstructionSelected
  : TileIRRemark<"RemarkMemoryStoreInstructionSelected", DK_Pass, CategoryMemory, 2, "Store instruction selected">;

def RemarkMemoryTmaLoadFailed
  : TileIRRemark<"RemarkMemoryTmaLoadFailed", DK_Fail, CategoryMemory, 1, "Failed to select TMA load instruction">;

def RemarkMemoryTmaStoreFailed
  : TileIRRemark<"RemarkMemoryTmaStoreFailed", DK_Fail, CategoryMemory, 2, "Failed to select TMA store instruction">;

def RemarkTensorViewIndexTypeNarrowing
  : TileIRRemark<"RemarkTensorViewIndexTypeNarrowing", DK_Miss, CategoryMemory, 5,
                 "Could not narrow tensor_view indexing to i32: {0}">;

def RemarkHint
  : TileIRRemark<"RemarkHint", DK_Pass, CategoryHints, 1, "OptimizationHint used {0}">;
def RemarkHintFailed
  : TileIRRemark<"RemarkHint", DK_Fail, CategoryHints, 2, "OptimizationHint incorrect {0}">;
def RemarkHintDefault
  : TileIRRemark<"RemarkHintDefault", DK_Pass, CategoryHints, 3, "OptimizationHint used default {0}">;
def RemarkHintOutOfRange
  : TileIRRemark<"RemarkHintOutOfRange", DK_Miss, CategoryHints, 4, "OptimizationHint value out of range {0}">;

//===--------------------------------------------------------------------===//
// Remark Attribute Definitions
//===--------------------------------------------------------------------===//

def RemarkID
    : I32EnumAttr<"RemarkID", "All TileIR remark IDs",
                  [RemarkTensorCoreMMA, 
                   RemarkMemoryLoadInstructionSelected,
                   RemarkMemoryStoreInstructionSelected,
                   RemarkTensorCoreMMAFailed,
                   RemarkMemoryTmaLoadFailed,
                   RemarkMemoryTmaStoreFailed,
                   RemarkTensorViewIndexTypeNarrowing,
                   RemarkHint,
                   RemarkHintFailed,
                   RemarkHintDefault,
                   RemarkHintOutOfRange
                   /*, ...*/]> {
  let cppNamespace = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}
//===--------------------------------------------------------------------===//
// Errors / Warnings
//===--------------------------------------------------------------------===//
//
// Always-on diagnostics live in separate files so they can be edited
// independently of the (filter-controlled) remark catalog above.

include "cuda_tile/Dialect/CudaTile/IR/Errors.td"
include "cuda_tile/Dialect/CudaTile/IR/Warnings.td"

//===--------------------------------------------------------------------===//
// Attribute Definitions
//===--------------------------------------------------------------------===//

def DiagnosticKind : I32EnumAttr<"DiagnosticKind",
      "Kinds of optimization diagnostics",
      [DK_Pass, DK_Miss, DK_Fail, DK_Analysis, DK_Error, DK_Warning]> {
  let cppNamespace       = "::mlir::cuda_tile::remark";
  let genSpecializedAttr = 0;
}

def CategoryID : I32EnumAttr<"CategoryID",
                             "TileIR remark categories",
                             [CategoryTensorCore,
                              CategoryMemory,
                              CategoryHints,
                              CategoryError,
                              CategoryWarning
                             ]> {
  let cppNamespace        = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}

#endif
