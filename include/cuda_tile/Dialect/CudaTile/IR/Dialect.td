//===- Dialect.td - CUDA Tile Dialect Definitions ----------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_DIALECT_TD
#define CUDATILE_DIALECT_CUDATILE_IR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
// Preview feature records are required by the public entity base classes below.
include "cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.td"

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

  // Temporary opt-out of strict property assembly formats, deprecated upstream
  // since LLVM c495a9b796c4. Some formats still carry inherent attributes in
  // `attr-dict`. This only affects the textual IR, not bytecode. Remove once
  // all formats bind their attributes or use `prop-dict`.
  let useStrictPropertiesInAssemblyFormat = 0;

  let extraClassDeclaration = [{
    template <typename... OpTys>
    void addExternalOperations() {
      (addOperations<OpTys>(), ...);
    }

    /// Get whether to emit diagnostics for unsupported/invalid optimization hints.
    bool getWarnUnsupportedHints() const { return warnUnsupportedHints_; }

    /// Set whether to emit diagnostics for unsupported/invalid optimization hints.
    void setWarnUnsupportedHints(bool value) { warnUnsupportedHints_ = value; }

    /// Get whether to treat unsupported/invalid hints as errors (vs warnings).
    bool getErrorOnHints() const { return errorOnHints_; }

    /// Set whether to treat unsupported/invalid hints as errors (vs warnings).
    void setErrorOnHints(bool value) { errorOnHints_ = value; }

    /// Set the remark finalizer.
    void setRemarkFinalizer(remark::InFlightRemark::Finalizer finalizer) { finalizer_ = finalizer; }

    /// Get the remark finalizer.
    remark::InFlightRemark::Finalizer getRemarkFinalizer() { return finalizer_; }

    /// Configure the public-preview features enabled in this context.
    void setEnabledPreviewFeatures(PreviewFeatures features) {
      enabledPreviewFeatures_ = std::move(features);
    }

    /// Return the public-preview features enabled in this context.
    const PreviewFeatures &getEnabledPreviewFeatures() const {
      return enabledPreviewFeatures_;
    }

    /// Return whether a public-preview feature is enabled in this context.
    bool isPreviewFeatureEnabled(PreviewFeature feature) const {
      return enabledPreviewFeatures_.isEnabled(feature);
    }

  private:
    void registerAttributes();
    void registerTypes();

    /// Whether to emit diagnostics for unsupported/invalid optimization hints.
    /// Default: false (silent).
    bool warnUnsupportedHints_ = false;

    /// Whether to treat unsupported/invalid hints as errors (requires warnUnsupportedHints_).
    /// Default: false (warnings only).
    bool errorOnHints_ = false;

    /// The finalizer to use for remarks emitted during compilation. This is a backup
    /// for any finalizer that's provided by the user directly in the API, and is only
    /// used if the user provides nothing directly.
    remark::InFlightRemark::Finalizer finalizer_;

    /// Independently enabled public-preview features. Empty by default.
    PreviewFeatures enabledPreviewFeatures_;
  }]
  ;
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

// Helper: a typed array attribute constraint whose elements must satisfy the
// given element constraint. Use this in op arguments for ArrayAttr-backed
// fields that need element verification without introducing a dedicated attr
// mnemonic.
class CudaTileArrayOf<Attr element> : TypedArrayAttrBase<element,
    "array of " # element.summary # " values">;


//===----------------------------------------------------------------------===//
// Integer 32-bit Enum Attribute
//===----------------------------------------------------------------------===//

class CudaTileI32EnumAttrCase<string desc, string sym, int val, string version,
                              string str = sym>
    : I32EnumAttrCase<sym, val, str> {
  string description = desc;
  string sinceVersion = version;
}

// GPU architecture enum case with an architecture family name.
// Used by -gen-gpu-arch-constants to generate per-family constants and ranges.
class CudaTileGpuArchCase<string desc, string sym, int val,
                          string version, string str, string family>
    : CudaTileI32EnumAttrCase<desc, sym, val, version, str> {
  string archFamily = family;
}

// Per-architecture range override for optimization hint validation.
// Narrows the valid range for specific GPU architecture families.
class HintArchOverride<list<string> families, int minVal, int maxVal> {
  list<string> archFamilies = families;
  int overrideMin = minVal;
  int overrideMax = maxVal;
}

// Common versioned base for CudaTile integer enum attributes. Parallels MLIR's
// IntEnumAttr but carries `sinceVersion`, so a single type can constrain the
// enum-info arg of CudaTileEnumParameter/CudaTileEnumAttr -- for both 32- and
// 64-bit enums -- while exposing `sinceVersion` at the TableGen level (the base
// EnumInfo/EnumAttrInfo classes do not declare it).
class CudaTileIntEnumAttr<I intType, string name, string desc, string version,
                          list<IntEnumAttrCaseBase> cases>
    : IntEnumAttr<intType, name, desc, cases> {
  string sinceVersion = version;
  string specPrefixDescription;
  string specSuffixDescription;
}

class CudaTileI32EnumAttr<string name, string desc, string version,
                         list<CudaTileI32EnumAttrCase> cases>
    : CudaTileIntEnumAttr<I32, name, desc, version, cases>;

// Versionless backing enum cases for public-preview attributes and types.
class CudaTilePreviewI32EnumAttrCase<string desc, string sym, int val,
                                     string str = sym>
    : CudaTileI32EnumAttrCase<desc, sym, val, "", str>;

// Backing enum for public-preview containers. genSpecializedAttr = 0 so MLIR
// does not auto-generate a specialized attribute (which would escape the
// preview gate -- enforced in BytecodeAttrAnalysis). Expose it as a standalone
// attribute with CudaTilePreviewEnumAttr, or use it inline via
// CudaTilePreviewEnumParam.
class CudaTilePreviewI32EnumInfo<string name, string desc,
                                 list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, desc, cases> {
  let genSpecializedAttr = 0;
  string sinceVersion = "";
  string specPrefixDescription;
  string specSuffixDescription;
}

//===----------------------------------------------------------------------===//
// Integer 64-bit Enum Attribute
//===----------------------------------------------------------------------===//

class CudaTileI64EnumAttrCase<string desc, string sym, int val, string version,
                              string str = sym>
    : I64EnumAttrCase<sym, val, str> {
  string description = desc;
  string sinceVersion = version;
}

class CudaTileI64EnumAttr<string name, string desc, string version,
                         list<CudaTileI64EnumAttrCase> cases>
    : CudaTileIntEnumAttr<I64, name, desc, version, cases>;

// Backs an EnumAttr's single `$value` parameter, deriving its `sinceVersion`
// from the backing enum. MLIR's plain EnumParameter carries no version, so the
// bytecode analysis would otherwise reject the auto-generated `$value` of a
// versioned enum attr; this makes it a versioned parameter with no C++ special
// case. The arg is typed `CudaTileIntEnumAttr` (not the base `EnumInfo`) so
// that `enumInfo.sinceVersion` resolves.
class CudaTileEnumParameter<CudaTileIntEnumAttr enumInfo>
    : EnumParameter<enumInfo> {
  string sinceVersion = enumInfo.sinceVersion;
}

class CudaTileEnumAttr<CudaTileIntEnumAttr enumInfo, string name = "",
               list <Trait> traits = []> : EnumAttr<CudaTile_Dialect, enumInfo, name, traits> {
  string sinceVersion = enumInfo.sinceVersion;
  let parameters = (ins CudaTileEnumParameter<enumInfo>:$value);
}

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

// Public-preview Operations. Every preview entity names its owning feature so
// enabling one feature never enables unrelated preview IR.
class CudaTilePreviewOpDef<string mnemonic, CudaTilePreviewFeature feature,
                           list<Trait> traits = []> :
    CudaTileOpDef<"preview$" # mnemonic, "", "Preview", "", traits> {
  CudaTilePreviewFeature previewFeature = feature;
}

#ifdef TILE_IR_INCLUDE_TESTS
// Testing Operations
class CudaTileTestingOpDef<string mnemonic, string version, list<Trait> traits = []> :
    CudaTileOpDef<"testing$" # mnemonic, version, "Testing", "", traits>;
#endif // TILE_IR_INCLUDE_TESTS

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

class CudaTileTypeDef<string name, string _mnemonic, string _specName,
                      string version, list<Trait> traits = []>
  : TypeDef<CudaTile_Dialect, name, traits> {

  // The name used in the CUDA Tile IR spec to reference this type.
  string specName = _specName;

  // Version when this type was introduced.
  string sinceVersion = version;

  let mnemonic = _mnemonic;
}

// Public-preview types ship in all builds but are not part of the stable
// bytecode format. They carry no version metadata because they are only
// serialized at the exact current bytecode version.
class CudaTilePreviewTypeDef<string name, string mnemonic, string specName,
                             CudaTilePreviewFeature feature,
                             list<Trait> traits = []>
    : CudaTileTypeDef<name, "preview$" # mnemonic, specName, "", traits> {
  CudaTilePreviewFeature previewFeature = feature;
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

defvar CudaTileIEEERoundingModeVariants =
    ["NEAREST_EVEN", "ZERO", "NEGATIVE_INF", "POSITIVE_INF"];
defvar CudaTileFloatConversionRoundingModeVariants =
    ["NEAREST_EVEN", "ZERO", "NEGATIVE_INF", "POSITIVE_INF",
     "NEAREST_AWAY"];
defvar CudaTileDivFRoundingModeVariants =
    ["NEAREST_EVEN", "ZERO", "NEGATIVE_INF", "POSITIVE_INF", "APPROX",
     "FULL"];
defvar CudaTileSqrtRoundingModeVariants =
    ["NEAREST_EVEN", "ZERO", "NEGATIVE_INF", "POSITIVE_INF", "APPROX"];
defvar CudaTileApproxFullRoundingModeVariants = ["APPROX", "FULL"];
defvar CudaTileIntegerDivisionRoundingModeVariants =
    ["ZERO", "NEGATIVE_INF", "POSITIVE_INF"];
defvar CudaTileIntegerRoundingModeVariants = ["NEAREST_INT_TO_ZERO"];
defvar CudaTileNearestEvenRoundingModeVariants = ["NEAREST_EVEN"];

def OnlyCudaTileIEEERoundingModes :
    OnlyVariants<CudaTileIEEERoundingModeVariants>;
def OnlyCudaTileFloatConversionRoundingModes :
    OnlyVariants<CudaTileFloatConversionRoundingModeVariants>;
def OnlyCudaTileDivFRoundingModes :
    OnlyVariants<CudaTileDivFRoundingModeVariants>;
def OnlyCudaTileSqrtRoundingModes :
    OnlyVariants<CudaTileSqrtRoundingModeVariants>;
def OnlyCudaTileApproxFullRoundingModes :
    OnlyVariants<CudaTileApproxFullRoundingModeVariants>;
def OnlyCudaTileIntegerDivisionRoundingModes :
    OnlyVariants<CudaTileIntegerDivisionRoundingModeVariants>;
def OnlyCudaTileIntegerRoundingModes :
    OnlyVariants<CudaTileIntegerRoundingModeVariants>;
def OnlyCudaTileNearestEvenRoundingMode :
    OnlyVariants<CudaTileNearestEvenRoundingModeVariants>;

// Marks a required attribute whose rank must match the rank of a sibling
// operand on the same operation. The `operandName` must refer to a variadic operand defined on the same op.
class RequireSameOperandRank<string operandName> : OpVariableDecorator {
  string rankOperand = operandName;
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

// The wrapper for type-constrained parameters.
// Uses multiple inheritance to be both a parameter and a constraint.
// Inherits from AttrOrTypeParameter (so it can be used in parameters list)
// and from TypeConstraint (to preserve validation and error messages).
class CudaTileConstrainedTypeParam<Type typeConstraint, string version,
                                   string bcFallback = "">
  : AttrOrTypeParameter<typeConstraint.cppType, typeConstraint.summary>,
    TypeConstraint<typeConstraint.predicate, typeConstraint.summary, typeConstraint.cppType> {
  // Version when this parameter was introduced.
  string sinceVersion = version;

  // Bytecode-only fallback value for evolved parameters. When reading bytecode
  // older than `sinceVersion`, this expression initializes the parameter.
  // When writing to an older version, the parameter must equal this value.
  // Supports $_ctxt and $_param_<name> substitutions (see CudaTileTypeParam
  // for details). The fallback expression must produce a value convertible
  // to the constrained type.
  string bytecodeFallbackValue = bcFallback;
}

// The wrapper for simple type parameters with version information.
class CudaTileTypeParam<AttrOrTypeParameter param, string version,
                        string bcFallback = "">
  : AttrOrTypeParameter<param.cppType, param.summary, param.cppAccessorType> {
  // Version when this parameter was introduced.
  string sinceVersion = version;

  // Bytecode-only fallback value for evolved parameters. When reading bytecode
  // older than `sinceVersion`, this expression initializes the parameter.
  // When writing to an older version, the parameter must equal this value.
  // Supports $_ctxt and $_param_<name> substitutions, the latter being replaced
  // by the value of the corresponding parameter. When empty, falls back
  // to ODS `defaultValue`.
  string bytecodeFallbackValue = bcFallback;

  // Forward all fields from the base parameter.
  let allocator = param.allocator;
  let comparator = param.comparator;
  let cppStorageType = param.cppStorageType;
  let convertFromStorage = param.convertFromStorage;
  let syntax = param.syntax;
  let parser = param.parser;
  let printer = param.printer;
  let defaultValue = param.defaultValue;
}

// Versionless parameter wrappers for public-preview types.
class CudaTilePreviewTypeParam<AttrOrTypeParameter param>
    : CudaTileTypeParam<param, "">;

class CudaTilePreviewConstrainedTypeParam<Type typeConstraint>
    : CudaTileConstrainedTypeParam<typeConstraint, "">;

// A versionless public-preview extension to an otherwise stable type. The
// parameter is omitted from bytecode when it equals `bcFallback`; a differing
// value is serialized only as preview content at the current bytecode version.
// Preview extensions must be last in bytecode serialization order so their
// presence can be detected from the remaining type-entry payload.
class CudaTilePreviewTypeExtensionParam<string cppType, string desc,
                                        CudaTilePreviewFeature feature,
                                        string bcFallback>
    : TypeParameter<cppType, desc> {
  bit bytecodePreviewExtension = 1;
  CudaTilePreviewFeature previewFeature = feature;
  string bytecodeFallbackValue = bcFallback;
}

// Convenience wrapper for string-based type parameters (avoids TypeParameter wrapper).
class CudaTileStrTypeParam<string cppTypeStr, string desc, string version,
                           string bcFallback = "">
  : CudaTileTypeParam<TypeParameter<cppTypeStr, desc>, version, bcFallback>;

// Versionless enum parameter for public-preview attributes and types (an enum
// parameter is a plain EnumParameter in either container). Serializes as one
// varint and keeps the enum classification for bytecode analysis.
class CudaTilePreviewEnumParam<EnumInfo enumInfo>
  : EnumParameter<enumInfo> {
  string sinceVersion = "";
}

//===----------------------------------------------------------------------===//
// Attribute Definitions
//===----------------------------------------------------------------------===//

class CudaTileAttrDef<string attrName, string attrMnemonic, string version,
                      list<Trait> traits = []>
    : AttrDef<CudaTile_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;

  string sinceVersion = version;

  list<string> mlirExamples = [];

  list<Table> descriptionTables = [];
}

// Public-preview attributes ship in all builds but are not part of the stable
// bytecode format. They carry no version metadata because they are only
// serialized at the exact current bytecode version.
class CudaTilePreviewAttrDef<string attrName, string attrMnemonic,
                             CudaTilePreviewFeature feature,
                             list<Trait> traits = []>
    : CudaTileAttrDef<attrName, "preview$" # attrMnemonic, "", traits> {
  CudaTilePreviewFeature previewFeature = feature;
}

// The wrapper for attribute parameters with version information.
// Mirrors CudaTileTypeParam for types.
class CudaTileAttrParam<AttrOrTypeParameter param, string version,
                        string bcFallback = "">
  : AttrOrTypeParameter<param.cppType, param.summary, param.cppAccessorType> {
  // Version when this parameter was introduced.
  string sinceVersion = version;

  // Bytecode-only fallback value for evolved parameters. When reading bytecode
  // older than `sinceVersion`, this expression initializes the parameter.
  // When writing to an older version, the parameter must equal this value.
  // Supports $_ctxt and $_param_<name> substitutions (resolved by
  // cuda-tile-tblgen). When empty, falls back to ODS `defaultValue`.
  string bytecodeFallbackValue = bcFallback;

  // Forward all fields from the base parameter.
  let allocator = param.allocator;
  let comparator = param.comparator;
  let cppStorageType = param.cppStorageType;
  let convertFromStorage = param.convertFromStorage;
  let syntax = param.syntax;
  let parser = param.parser;
  let printer = param.printer;
  let defaultValue = param.defaultValue;
}

// Versionless parameter wrapper for public-preview attributes.
class CudaTilePreviewAttrParam<AttrOrTypeParameter param>
    : CudaTileAttrParam<param, "">;

// One-line public-preview enum attribute, the preview twin of
// CudaTileEnumAttr. Exposes a versionless preview enum (genSpecializedAttr = 0)
// as an attribute with a `<value>` spelling; it is classified as preview
// automatically. Still register a PreviewBytecodeAttrTag for `<Name>Attr`.
class CudaTilePreviewEnumAttr<string attrName, string attrMnemonic,
                              EnumInfo enumInfo,
                              CudaTilePreviewFeature feature,
                              list<Trait> traits = []>
    : CudaTilePreviewAttrDef<attrName, attrMnemonic, feature, traits> {
  let parameters = (ins CudaTilePreviewEnumParam<enumInfo>:$value);
  let assemblyFormat = "`<` $value `>`";
}

// Convenience wrapper for string-based attribute parameters.
class CudaTileStrAttrParam<string cppTypeStr, string desc, string version,
                           string bcFallback = "">
  : CudaTileAttrParam<TypeParameter<cppTypeStr, desc>, version, bcFallback>;

// The wrapper for an attribute's self type with version information.
// Subclasses AttributeSelfTypeParameter directly (not CudaTileAttrParam) so
// mlir-tblgen still recognizes it as the self type and generates
// TypedAttrInterface. Serialized through the type table (Kind::Type).
class CudaTileAttrSelfTypeParam<string version, string derivedType = "::mlir::Type">
  : AttributeSelfTypeParameter<"", derivedType> {
  // Version when this parameter was introduced.
  string sinceVersion = version;

  // A self type is always present, so it carries no default value. This
  // overrides the NoneType default inherited from AttributeSelfTypeParameter,
  // keeping isOptional() false (it keys off a non-empty defaultValue) so the
  // self type serializes as a required parameter.
  let defaultValue = "";
}

// Wrapper marking an attribute parameter whose declared C++ type is an
// AttrInterface.
class CudaTileAttrInterfaceParam<string interfaceCppType, string version,
                                 string desc = "", string bcFallback = "">
  : TypeParameter<interfaceCppType, desc> {
  string sinceVersion = version;
  string bytecodeFallbackValue = bcFallback;
}

// Versionless AttrInterface parameter wrapper for public-preview attributes.
class CudaTilePreviewAttrInterfaceParam<string interfaceCppType,
                                        string desc = "">
    : CudaTileAttrInterfaceParam<interfaceCppType, "", desc>;

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
