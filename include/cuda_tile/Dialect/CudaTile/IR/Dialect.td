//===- Dialect.td - CUDA Tile Dialect Definitions ----------*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDATILE_DIALECT_CUDATILE_IR_DIALECT_TD
#define CUDATILE_DIALECT_CUDATILE_IR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

class TableHeader<string labelArg, string contentTypeArg = "", int widthArg = -1> {
  string label = labelArg;
  string contentType = contentTypeArg;
  int width = widthArg;
}

class TableRow<list<string> columsArg> {
  list<string> columns = columsArg;
}

class Table<string labelArg, string descriptionArg, list<TableHeader> headersArg, list<TableRow> rowsArg> {
  string label = labelArg;
  string description = descriptionArg;
  list<TableHeader> headers = headersArg;
  list<TableRow> rows = rowsArg;
}

def CudaTile_Dialect : Dialect {
  let name = "cuda_tile";
  let cppNamespace = "::mlir::cuda_tile";
  let dependentDialects = [];
  let description = [{
    This dialect contains public CudaTile instruction set. It is entirely
    self-contained and independent of any other dialects.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;

  let extraClassDeclaration = [{
    template <typename... OpTys>
    void addExternalOperations() {
      (addOperations<OpTys>(), ...);
    }

  private:
    void registerAttributes();
    void registerTypes();
  }];
}

/// The metadata for the operation used during specification generation.
class CudaTileOpMetadata<string version, string group, string subGroup> {
  string sinceVersion = version;
  string cudaTileSpecGroup = group;
  string cudaTileSpecSubGroup = subGroup;
}

/// The base class for all CudaTile operations.
class CudaTileOpDef<string mnemonic, string version, string group, string subGroup = "", list<Trait> traits = []> :
    Op<CudaTile_Dialect, mnemonic, traits> {
  /// Store version for bytecode generation.
  string operationVersion = version;
  /// Examples of how to use the operation written in the MLIR dialect.
  ///
  /// Note: we choose this name to enable other examples to be written in the
  /// future.
  list<string> mlirExamples = [];

  list<Table> descriptionTables = [];

  CudaTileOpMetadata metadata = CudaTileOpMetadata<version, group, subGroup>;
}


//===----------------------------------------------------------------------===//
// Integer 32-bit Enum Attribute
//===----------------------------------------------------------------------===//

class CudaTileI32EnumAttrCase<string desc, string sym, int val, string str = sym> : I32EnumAttrCase<sym, val, str> {
  string description = desc;
}

class CudaTileI32EnumAttr<string name, string desc, list<CudaTileI32EnumAttrCase> cases> : I32EnumAttr<name, desc, cases> {
  string specPrefixDescription;
  string specSuffixDescription;
}

//===----------------------------------------------------------------------===//
// Integer 64-bit Enum Attribute
//===----------------------------------------------------------------------===//

class CudaTileI64EnumAttrCase<string desc, string sym, int val, string str = sym> : I64EnumAttrCase<sym, val, str> {
  string description = desc;
}

class CudaTileI64EnumAttr<string name, string desc, list<CudaTileI64EnumAttrCase> cases> : I64EnumAttr<name, desc, cases> {
  string specPrefixDescription;
  string specSuffixDescription;
}

class CudaTileEnumAttr<EnumAttrInfo enumInfo, string name = "",
               list <Trait> traits = []> : EnumAttr<CudaTile_Dialect, enumInfo, name, traits>;

// Bitwise Arithmetic Operations
class CudaTileBitwiseOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Bitwise", "", traits>;

// Integer Arithmetic Operations
class CudaTileIntegerOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Integer", "", traits>;

// Floating Point Arithmetic Operations
class CudaTileFloatingPointOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Floating Point", "", traits>;

// Atomic Operations
class CudaTileAtomicsOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Atomics", "", traits>;

// Conversion Operations
class CudaTileConversionOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Conversions", "", traits>;

// Core Operations
class CudaTileCoreOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Core", "", traits>;

// Control Flow Operations
class CudaTileControlFlowOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Control Flow", "", traits>;

// Memory Operations
class CudaTileMemOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Memory", "", traits>;

// TensorView Operations
class CudaTileViewOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Views", "", traits>;

// Miscellaneous Operations
class CudaTileMiscOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<mnemonic, version, "Miscellaneous", "", traits>;

#ifdef TILE_IR_INCLUDE_TESTS
// Testing Operations
class CudaTileTestingOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<"testing$" # mnemonic, version, "Testing", "", traits>;
#endif // TILE_IR_INCLUDE_TESTS

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

class CudaTileTypeDef<string name, string _mnemonic, string _specName,
                      list<Trait> traits = []>
  : TypeDef<CudaTile_Dialect, name, traits> {

  // The name used in the CUDA Tile IR spec to reference this type.
  string specName = _specName;

  let mnemonic = _mnemonic;
}

// The metadata for the argument used during specification generation.
class CudaTileArgMetadata<string version, string desc> : OpVariableDecorator {
  string sinceVersion = version;
  string specDesc = desc;
}

// Used to filter the set of variants documented for an argument.
class OnlyVariants<list<string> selectedVariants> : OpVariableDecorator {
  list<string> variants = selectedVariants;
}

// The wrapper class for declaring arguments for CudaTile operations.
class CudaTileArg<Constraint constraint, string desc, string version, list<OpVariableDecorator> decorators = []>
  : Arg<constraint, desc, decorators # [CudaTileArgMetadata<version, desc>]>;

// The wrapper class for declaring unused arguments for CudaTile operations. The
// arguments are defined but not currently processed by CUDA Tile IR's specific logic.
class CudaTileUnusedArg<Constraint constraint, string desc, string version, list<OpVariableDecorator> decorators = []>
  : Arg<constraint, desc, decorators # [CudaTileArgMetadata<version, desc>]> {
  let summary = "Defines an argument for a CudaTile operation that is syntactically "
                "present but not currently processed by CUDA Tile IR's specific logic.";
}

// The wrapper class for declaring attributes for CudaTile attributes.
class CudaTileAttrDef<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CudaTile_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;

  list<string> mlirExamples = [];

  list<Table> descriptionTables = [];
}

def CudaTile_DefaultDialect {
  // Helper record to store overrides for the OpAsmOpInterface. Used in block
  // Ops to remove the need for `cuda_tile.` prefix.
  string classDecl = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    // This will filter the `cuda_tile.` prefix in front of operations inside the
    // the block.
    static StringRef getDefaultDialect() {
      return CudaTileDialect::getDialectNamespace();
    }
  }];
}


#endif  // CUDATILE_DIALECT_CUDATILE_IR_DIALECT_TD
