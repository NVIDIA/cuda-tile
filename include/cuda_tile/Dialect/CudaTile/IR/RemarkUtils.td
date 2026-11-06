//===- RemarkUtils.td - CUDA Tile Remark Utilities ---------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_REMARKUTILS_TD
#define CUDATILE_DIALECT_REMARKUTILS_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Properties.td"

/// Remark Category Definitions
class TileIRCategoryCase<string Name, int Val, string Doc> : I32EnumAttrCase<Name, Val, Doc>{}

/// Remark Category Definition
class TileIRCategory<string Name, int Id, string Doc> : I32EnumAttrCase<Name, Id, Doc>;

/// Diagnostic Kind Definition
class TileIRDiagnosticKind<string Name, int Val, string Doc> : I32EnumAttrCase<Name, Val, Doc>;

/// Remark Definition
class TileIRRemark<
        string                Name,
        TileIRDiagnosticKind  Kind,
        TileIRCategory        Cat,
        int                   Num,
        string                Doc>
  : I32EnumAttrCase<
        Name # "_" # Kind.str,
        !or(!shl(Num, 7), !or(!shl(Cat.value, 3), Kind.value)),
        Name # "@" # Doc> {
  assert !and(!ne(Kind.value, 4), !ne(Kind.value, 5)),
         "TileIRRemark<> must not use DK_Error or DK_Warning; use "
         "TileIRError<> / TileIRWarning<> so the call site goes through "
         "reportError / reportWarning.";
}

/// Always-on-diagnostic categories, owned by this file so that the
/// TileIRError / TileIRWarning classes below can bake their values into the
/// packed ID without depending on Remarks.td.
def CategoryError   : TileIRCategory<"Error",   6, "Error">;
def CategoryWarning : TileIRCategory<"Warning", 7, "Warning">;

/// All errors carry CategoryError; all warnings carry CategoryWarning. The
/// classes take no `Cat` parameter — if a finer-grained domain category is
/// ever needed, reintroduce the parameter without changing the bit layout.
class TileIRError<string Name, int Num, string Doc>
  : I32EnumAttrCase<
        Name,
        !or(!shl(Num, 7), !or(!shl(CategoryError.value, 3), 4)), // kind = DK_Error(4)
        Name # "@" # Doc>;


class TileIRWarning<string Name, int Num, string Doc>
  : I32EnumAttrCase<
        Name,
        !or(!shl(Num, 7), !or(!shl(CategoryWarning.value, 3), 5)), // kind = DK_Warning(5)
        Name # "@" # Doc>;

#endif // CUDATILE_DIALECT_REMARKUTILS_TD
