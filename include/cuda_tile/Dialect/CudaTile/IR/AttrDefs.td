//===- AttrDefs.td - CUDA Tile Attribute Definitions -------*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDATILE_DIALECT_CUDATILE_IR_ATTRDEFS_TD
#define CUDATILE_DIALECT_CUDATILE_IR_ATTRDEFS_TD

include "mlir/IR/EnumAttr.td"

include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// Integer Signedness Attribute
//===----------------------------------------------------------------------===//

def CudaTile_Signedness : CudaTileI32EnumAttr<"Signedness", "signedness",
    [CudaTileI32EnumAttrCase<"Treat the operands as unsigned integers.", "Unsigned", 0, "unsigned">,
     CudaTileI32EnumAttrCase<"Treat the operands as signed integers.", "Signed", 1, "signed">]> {
  let specPrefixDescription = "The :code:`signedness` attribute specifies the signedness of operand(s).";
  let specSuffixDescription = "";
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_SignednessAttr : CudaTileEnumAttr<CudaTile_Signedness, "signedness"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}

//===----------------------------------------------------------------------===//
// Integer Overflow Attributes
//===----------------------------------------------------------------------===//

def CudaTile_IntegerOverflow : CudaTileI32EnumAttr<
  "IntegerOverflow", "integer overflow",
  [
      CudaTileI32EnumAttrCase<"The compiler makes no assumptions regarding overflow behavior.", "NONE", 0, "none">,
      CudaTileI32EnumAttrCase<"The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.", "NSW", 1, "no_signed_wrap">,
      CudaTileI32EnumAttrCase<"The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.", "NUW", 2, "no_unsigned_wrap">,
      CudaTileI32EnumAttrCase<"The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.", "NW", 3, "no_wrap">,
  ]> {
  let specPrefixDescription = [{
    The :code:`overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

    These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation
    respects these assumptions dynamically during execution.
  }];
  let specSuffixDescription = "If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.";
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}


def CudaTile_IntegerOverflowAttr :
    CudaTileEnumAttr<CudaTile_IntegerOverflow, "overflow"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}

//===----------------------------------------------------------------------===//
// Optimization hints Attributes
//===----------------------------------------------------------------------===//

def CudaTile_OptimizationHintsAttr : CudaTileAttrDef<"OptimizationHints", "optimization_hints"> {
  let parameters = (ins "DictionaryAttr":$value);
  let description = [{
    The :code:`optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries.

    The hints are specified for each architecture (e.g., :code:`sm_100`, :code:`sm_120`) and for each architecture the user can specify
    specific hints for each operation.

    - :code:`num_cta_in_cga` - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for :ref:`op-cuda_tile.entry`.
    - :code:`allow_tma` - suggest whether to use TMA for :ref:`op-cuda_tile.load_view_tko` and :ref:`op-cuda_tile.store_view_tko`.
    - :code:`latency` - latency hint for :ref:`op-cuda_tile.load_view_tko` and :ref:`op-cuda_tile.store_view_tko`.

    For example they can be annotated as:

    .. code-block:: mlir

      optimization_hints=<
        sm_100 = {num_cta_in_cga = 8},
        sm_120 = {num_cta_in_cga = 16}
      >
  }];

  let descriptionTables = [
    Table<":code: Optimization Hints", "The below table shows the supported optimization hints for each operation type.",
      [TableHeader<"Optimization Hint", "code">, TableHeader<"EntryOp">,
       TableHeader<"LoadViewTkoOp, StoreViewTkoOp">,
       TableHeader<"LoadPtrTkoOp, StorePtrTkoOp">],
      [TableRow<["num_cta_in_cga", "yes", "no", "no"]>,
       TableRow<["allow_tma", "no", "yes", "no"]>,
       TableRow<["latency", "no", "yes", "yes"]>]
    >
  ];
  let hasCustomAssemblyFormat = 1;
  let cppNamespace = "::mlir::cuda_tile";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{

  private:
    static constexpr llvm::StringLiteral kNumCTAInCGA = "num_cta_in_cga";
    static constexpr llvm::StringLiteral kAllowTMA = "allow_tma";
    static constexpr llvm::StringLiteral kLatency = "latency";
    static constexpr llvm::StringLiteral kOccupancy = "occupancy";
    static constexpr llvm::StringLiteral allowedKeysArr[] = {
        "sm_80", "sm_90", "sm_100", "sm_103", "sm_110", "sm_120", "sm_121"};

    static bool isAllowedKey(llvm::StringRef key) {
      return llvm::is_contained(allowedKeysArr, key);
    }

    static mlir::LogicalResult verifyParamWithContext(llvm::function_ref<InFlightDiagnostic()> emitError,
                                               llvm::StringRef context,
                                               ArrayRef<StringRef> allowedKeys,
                                               DictionaryAttr &attr);
  public:
    std::optional<int> getNumCTAInCGA(StringRef sm);
    std::optional<bool> getAllowTMA(StringRef sm);
    std::optional<int> getLatency(StringRef sm);
    std::optional<int> getOccupancy(StringRef sm);
    static mlir::LogicalResult verifyWithOp(Operation *op, llvm::function_ref<InFlightDiagnostic()> emitError, DictionaryAttr value);

  }];
}

//===----------------------------------------------------------------------===//
// Rounding Mode Attributes
//===----------------------------------------------------------------------===//

def CudaTile_RoundingMode : CudaTileI32EnumAttr<
  "RoundingMode", "rounding mode",
  [   CudaTileI32EnumAttrCase<"Round to nearest (ties to even).", "NEAREST_EVEN", 0, "nearest_even">,
      CudaTileI32EnumAttrCase<"Round towards zero (truncate).", "ZERO", 1, "zero">,
      CudaTileI32EnumAttrCase<"Round towards negative infinity.", "NEGATIVE_INF", 2, "negative_inf">,
      CudaTileI32EnumAttrCase<"Round towards positive infinity.", "POSITIVE_INF", 3, "positive_inf">,
      CudaTileI32EnumAttrCase<"Approximate rounding mode.", "APPROX", 4, "approx">,
      CudaTileI32EnumAttrCase<"Full precision rounding mode.", "FULL", 5, "full">,

      // Integer roundings
      CudaTileI32EnumAttrCase<"Round towards zero to the nearest integer.", "NEAREST_INT_TO_ZERO", 6, "nearest_int_to_zero">
  ]> {
  let specPrefixDescription = "The :code:`rounding` attribute specifies the rounding mode to use for the operation.";
  let specSuffixDescription = "";
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_RoundingModeAttr : CudaTileEnumAttr<CudaTile_RoundingMode, "rounding"> {
  let assemblyFormat = "`<` $value `>`";
}




//===----------------------------------------------------------------------===//
// Comparison Attributes
//===----------------------------------------------------------------------===//

def CudaTile_ComparisonOrdering : CudaTileI32EnumAttr<"ComparisonOrdering", "comparison_ordering",
    [CudaTileI32EnumAttrCase<"Unordered comparison.", "UNORDERED", 0, "unordered">,
     CudaTileI32EnumAttrCase<"Ordered comparison.", "ORDERED", 1, "ordered">]> {
  let cppNamespace = "::mlir::cuda_tile";
  let genSpecializedAttr = 0;
  let specPrefixDescription = "The :code:`comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.";
}

def CudaTile_ComparisonOrderingAttr : CudaTileEnumAttr<CudaTile_ComparisonOrdering, "comparison_ordering"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_ComparisonPredicate : CudaTileI32EnumAttr<
    "ComparisonPredicate", "cmp_predicate",
    [
      CudaTileI32EnumAttrCase<"Equal comparison.", "EQUAL", 0, "equal">,
      CudaTileI32EnumAttrCase<"Not equal comparison.", "NOT_EQUAL", 1, "not_equal">,
      CudaTileI32EnumAttrCase<"Less than comparison.", "LESS_THAN", 2, "less_than">,
      CudaTileI32EnumAttrCase<"Less than or equal comparison.", "LESS_THAN_OR_EQUAL", 3, "less_than_or_equal">,
      CudaTileI32EnumAttrCase<"Greater than comparison.", "GREATER_THAN", 4, "greater_than">,
      CudaTileI32EnumAttrCase<"Greater than or equal comparison.", "GREATER_THAN_OR_EQUAL", 5, "greater_than_or_equal">
    ]> {
  let cppNamespace = "::mlir::cuda_tile";
  let genSpecializedAttr = 0;
  let specPrefixDescription = "The :code:`comparison_predicate` attribute specifies the kind of comparison to be performed.";
}

def CudaTile_ComparisonPredicateAttr : CudaTileEnumAttr<CudaTile_ComparisonPredicate, "comparison_predicate"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}


//===----------------------------------------------------------------------===//
// Op-specific Attributes
//===----------------------------------------------------------------------===//

def CudaTile_AtomicRMWModeAttr : CudaTileI32EnumAttr<
    "AtomicRMWMode", "",
    [
      CudaTileI32EnumAttrCase<"Perform bitwise AND as the modification operation.", "AND", 0, "and">,
      CudaTileI32EnumAttrCase<"Perform bitwise OR as the modification operation.", "OR", 1, "or">,
      CudaTileI32EnumAttrCase<"Perform bitwise XOR as the modification operation.", "XOR", 2, "xor">,
      CudaTileI32EnumAttrCase<"Perform integer addition as the modification operation.", "ADD", 3, "add">,
      CudaTileI32EnumAttrCase<"Perform floating-point addition as the modification operation.", "ADDF", 4, "addf">,
      CudaTileI32EnumAttrCase<"Perform maximum as the modification operation.", "MAX", 5, "max">,
      CudaTileI32EnumAttrCase<"Perform minimum as the modification operation.", "MIN", 6, "min">,
      CudaTileI32EnumAttrCase<"Perform unsigned maximum as the modification operation.", "UMAX", 7, "umax">,
      CudaTileI32EnumAttrCase<"Perform unsigned minimum as the modification operation.", "UMIN", 8, "umin">,
      CudaTileI32EnumAttrCase<"Perform exchange as the modification operation.", "XCHG", 9, "xchg">
    ]> {
  let specPrefixDescription = "The :code:`mode` attribute specifies the mode of the atomic read-modify-write operation.";
  let specSuffixDescription = "The :code:`mode` attribute has a default value of :code:`add`.";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_DivByAttr : CudaTileAttrDef<"DivBy", "div_by",
    [DeclareAttrInterfaceMethods<CudaTile_AssumePredicateAttrInterface>]> {

  let description = [{
    .. code-block:: mlir

      div_by< $divisor (, every $every^ along $along)?>

    The :code:`div_by` attribute must be used as a predicate for :code:`cuda_tile.assume`
    ops. The predicated value must be a :code:`tile` of integers or pointers, or
    a :code:`tensor_view`.

    If the predicated value is a :code:`tile`, the attribute indicates that some
    elements of the :code:`tile` are divisible by :code:`divisor`. If the predicated value
    is a :code:`tensor_view` the attribute indicates that the base address of the :code:`tensor_view` is
    divisible by :code:`divisor`. :code:`divisor` must be a positive power of :code:`2`.

    The :code:`every` and :code:`along` attributes control which elements are assumed to
    satisfy the divisibility property. When splitting the tensor in groups of
    size :code:`every` along dimension :code:`along`, the first element of each group is
    assumed to satisfy the divisibility property. The other elements are
    assumed to be monotonically increasing by :code:`1` within the group. In case
    of a :code:`tile` of pointers, the elements are assumed to be monotonically
    increasing by the byte width of the pointee type. The size of the last
    group may be smaller than :code:`every`.

    The :code:`every` and :code:`along` attributes are optional. When missing, they are
    assumed to have a default value of :code:`1` and :code:`0` in case of a :code:`tile`.
    I.e., all elements of the :code:`tile` are assumed to satisfy the divisibility
    property. (The value of :code:`along` does not matter in that case.) If the
    predicated value is a :code:`tensor_view` or a 0D :code:`tile`, :code:`every` and :code:`along` cannot be
    used.

    :code:`every`, and :code:`along` must be used together. If one is specified,
    so must be the other.

    .. note::

      If the predicated value is a tile of integers, :code:`every` is a property of
      the signed interpretation of the integer values. Otherwise, it is a
      property of the unsigned integer interpretation. E.g., :code:`every = 4`
      is incorrect for the following sequence of "i8" values (written in binary
      form) because they wrap around when interpreted as signed integers:
      :code:`[01111110, 01111111, 10000000, 10000001]`. :code:`every = 2` would
      be correct.

    The examples below demonstrate tensors that satisfy the assumed properties.
  }];

  let mlirExamples = [
    [{
      // Example 1: Each pointer is divisible by 16.
      // [ 0x10, 0x20, 0x80, 0x10, 0x0, 0x120, ... ]
      %0 = cuda_tile.assume #cuda_tile.div_by<16>, %ptrs
          : !cuda_tile.tile<128x!cuda_tile.ptr<f32>>
      // Note: Equivalent to #cuda_tile.div_by<16, every 1 along 0>.
    }],
    [{
    // Example 2: Each integer is divisible by 4.
    // [ 16, 24, 8, 4, 12, 12, 0, 16, ... ]
    %0 = cuda_tile.assume #cuda_tile.div_by<4>, %t
        : !cuda_tile.tile<128xi32>
    }],
    [{
    // Example 3: Group size [4].
    // [7, 8, 9, 10, 23, 24, 25, 26, 0, 1, 2, 3, ...]
    %0 = cuda_tile.assume #cuda_tile.div_by<1, every 4 along 0>, %t
        : !cuda_tile.tile<128xi32>
    }],
    [{
    // Example 4: 2-d Group size [1, 4] with divisibility 4.
    // [ [  4,  5,  6,  7, 12, 13, 14, 15 ],
    //   [  8,  9, 10, 11, 24, 25, 26, 27 ],
    //   [ 24, 25, 26, 27, 64, 65, 66, 67 ],
    //   [  0,  1,  2,  3,  4,  5,  6,  7 ] ]
    %0 = cuda_tile.assume #cuda_tile.div_by<4, every 4 along 1>, %t
        : !cuda_tile.tile<4x8xi32>
    }],
    [{
    // Example 5: 2-d Group size [4, 1] with divisibility 32.
    // Note that the elements within each column are monotonically increasing
    // by the byte width of the pointee type f32, e.g., 0x20, 0x24, 0x28, 0x2c.
    // [ [  0x20, 0x100,  0x40,  0x60,  0x40, 0x200, 0x340,  0x40 ],
    //   [  0x24, 0x104,  0x44,  0x64,  0x44, 0x204, 0x344,  0x44 ],
    //   [  0x28, 0x108,  0x48,  0x68,  0x48, 0x208, 0x348,  0x48 ],
    //   [  0x2c, 0x10c,  0x4c,  0x6c,  0x4c, 0x20c, 0x34c,  0x4c ] ]
    %0 = cuda_tile.assume #cuda_tile.div_by<32, every 4 along 0>, %ptrs
        : !cuda_tile.tile<4x8x!cuda_tile.ptr<f32>>
    }]
  ];


  let parameters = (ins "uint64_t":$divisor,
                        "std::optional<int64_t>":$every,
                        "std::optional<int64_t>":$along);

  // TODO: Specify assembly format instead of hand-written parsers/printers.
  // This requires a fix in MLIR. Optional type parameters are not supported
  // at the moment.
  let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `<` $divisor (`,` `every` $every^ `along` $along)? `>`";
  // }];
}

def CudaTile_SameElementsAttr : CudaTileAttrDef<
    "SameElements", "same_elements",
    [DeclareAttrInterfaceMethods<CudaTile_AssumePredicateAttrInterface>]> {
  let description = [{
    .. code-block:: mlir

      #same_elements< $values >

    The :code:`same_elements` attribute must be used as a predicate for
    :code:`cuda_tile.assume`. The predicated value must be a tensor of integers or
    pointers.

    :code:`same_elements` is specified for each dimension. A value of C for a
    dimension of size N indicates that, after dividing the respective
    dimension into N/C groups of size C, each group consists of the same
    elements. As N/C may not divide evenly, the last group may have fewer
    than C elements.

    If the "same elements" property does not hold along a dimension, the
    respective value should be set to 1.
    :code:`#cuda_tile.same_elements<[1, 1, ..., 1]>` is a correct predicate for any
    tensor of integers or pointers, where the number of ones matches the rank
    of the tensor. (Size-1 groups always have the same elements.)
  }];

  let mlirExamples = [[{
    // Integer tensor with same elements.
    %0 = cuda_tile.constant <i16: [[0, 0, 0, 0, 10, 10, 10, 10],
                                   [0, 0, 0, 0, 10, 10, 10, 10],
                                   [5, 5, 5, 5, 93, 93, 93, 93],
                                   [5, 5, 5, 5, 93, 93, 93, 93]]>
        : tile<4x8xi16>
    %1 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %0
        : !cuda_tile.tile<4x8xi16>

    // Pointer tensor with same elements.
    %2 = cuda_tile.constant <i64: [[ 0,  0,  0,  0,  8,  8,  8,  8],
                                   [ 0,  0,  0,  0,  8,  8,  8,  8],
                                   [64, 64, 64, 64, 32, 32, 32, 32],
                                   [64, 64, 64, 64, 32, 32, 32, 32]]>
        : tile<4x8xi64>
    %3 = cuda_tile.bitcast %2
        : !cuda_tile.tile<4x8xi64>
          -> !cuda_tile.tile<!cuda_tile.ptr<f32>>
    %4 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %3
        : !cuda_tile.tile<!cuda_tile.ptr<f32>>
  }]];

  let parameters = (ins "DenseI64ArrayAttr":$values);
  let assemblyFormat =  "`<` $values `>`";
}

def CudaTile_BoundedAttr : CudaTileAttrDef<
    "Bounded", "bounded",
    [DeclareAttrInterfaceMethods<CudaTile_AssumePredicateAttrInterface>]> {
  let description = [{
    .. code-block:: mlir

      #bounded<(lb|?), (ub|?)>

    The :code:`bounded` attribute must be used as a predicate for
    :code:`cuda_tile.assume`. The predicated value must be a tile of integers.

    :code:`bounded` specifies a lower and upper bound for all elements of the
    predicated tile when interpreted as signed integers. Bounds are optional:
    it is possible to leave a bound unspecified, as indicated by "?" in the
    assembly format. E.g., :code:`#bounded<0, ?>`. Both lower bound and upper
    bound are inclusive.

    The lower bounds must be less than or equal to the upper bound. A lower/
    upper bound that exceeds the range of valid values of the predicated value
    is invalid.
  }];

  let mlirExamples = [[{
    %1 = cuda_tile.assume #cuda_tile.bounded<0, ?>, %0
        : !cuda_tile.tile<4x8xi16>
  }]];

  let parameters = (ins OptionalParameter<"std::optional<int64_t>">:$lb,
                        OptionalParameter<"std::optional<int64_t>">:$ub);
  let assemblyFormat = [{
    `<` ($lb^) : (`?`)? `,` ($ub^) : (`?`)? `>`
  }];
}

def CudaTile_MemoryScopeAttr
    : CudaTileI32EnumAttr<"MemoryScope", "memory scope",
                  [CudaTileI32EnumAttrCase<"There may be concurrent accesses from within the same tile block.", "TL_BLK", 0, "tl_blk">,
                   CudaTileI32EnumAttrCase<"There may be concurrent accesses from within the same device (i.e., GPU).", "DEVICE", 1, "device">,
                   CudaTileI32EnumAttrCase<"There may be concurrent accesses from anywhere within the system (i.e., all devices).", "SYS", 2, "sys">]> {
  let specPrefixDescription = [{
    The :code:`memory_scope` attribute specifies a communication scope for memory operations.
    When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other
    threads which are participating in the communication, or data races may occur.
  }];
  let specSuffixDescription = "";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_MemoryOrderingSemanticsAttr
    : CudaTileI32EnumAttr<"MemoryOrderingSemantics", "memory ordering semantics",
                  [CudaTileI32EnumAttrCase<"No concurrent accesses to the source/destination location.", "WEAK", 0, "weak">,
                   CudaTileI32EnumAttrCase<"There may be concurrent access to the location, but this access does not establish a happens-before relationship.", "RELAXED", 1, "relaxed">,
                   CudaTileI32EnumAttrCase<" There may be concurrent accesses to the location. If this acquire observes a release operation, then *happens before* is established.", "ACQUIRE", 2, "acquire">,
                   CudaTileI32EnumAttrCase<"There may be concurrent access to the location. If this release is observed with an acquire operation, then *happens before* is established.", "RELEASE", 3, "release">,
                   CudaTileI32EnumAttrCase<"There may be concurrent accesses to the location. This has the effect of both a release and acquire operation.", "ACQ_REL", 4, "acq_rel">]> {
  let specPrefixDescription = [{
    The :code:`memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required.
    For example, :code:`weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location.
    For more information, refer to the :ref:`memory model section <section-memory-model>` of the specification.
  }];
  let specSuffixDescription = "";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_PaddingValue : CudaTileI32EnumAttr<
    "PaddingValue", "load padding value for out of bound access",
    [
      CudaTileI32EnumAttrCase<"zero", "zero", 0, "zero">,
      CudaTileI32EnumAttrCase<"negative zero", "neg_zero", 1, "neg_zero">,
      CudaTileI32EnumAttrCase<"NaN", "nan", 2, "nan">,
      CudaTileI32EnumAttrCase<"positive infinity", "pos_inf", 3, "pos_inf">,
      CudaTileI32EnumAttrCase<"negative infinity", "neg_inf", 4, "neg_inf">
    ]> {
    let specPrefixDescription = [{
      The :code:`padding_value` attribute specifies the value to return for an out-of-bounds access.
    }];

    let specSuffixDescription = [{
      Note that special padding values (:code:`neg_zero`, :code:`nan`, :code:`pos_inf`, :code:`neg_inf`)
      can only be used with floating-point element types.
    }];
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_PaddingValueAttr :
    CudaTileEnumAttr<CudaTile_PaddingValue, "padding_value"> {
  let cppNamespace = "::mlir::cuda_tile";
}

//===----------------------------------------------------------------------===//
// DebugInfo
//===----------------------------------------------------------------------===//

/// Wrapper class for declaring CudaTile debug info attributes.
class CudaTile_DIAttr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<CudaTile_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

/// Base class for all debug info attributes.
class CudaTile_DINodeAttr<string name,
                          string attrMnemonic,
                          list<Trait> traits = []>
    : CudaTile_DIAttr<name, attrMnemonic, traits, "DINodeAttr"> {
}

/// Represents a debug info scope.
class CudaTile_DIScopeAttr<string name,
                           string attrMnemonic,
                           list<Trait> traits = []>
    : CudaTile_DIAttr<name, attrMnemonic, traits, "DIScopeAttr"> {
}

/// Represents a local debug info scope.
class CudaTile_DILocalScopeAttr<string name,
                                string attrMnemonic,
                                list<Trait> traits = []>
    : CudaTile_DIAttr<name, attrMnemonic, traits, "DILocalScopeAttr"> {
}

//===----------------------------------------------------------------------===//
// DILocAttr
//===----------------------------------------------------------------------===//

def CudaTile_DILocAttr : LocationAttrDef<CudaTile_Dialect, "DILoc"> {
  let summary = "a source location with a debug info scope";
  let description = [{
    Represents a location in the source code that carries a corresponding
    debug info scope. This location is used to connect an operation with a
    particular debug scope, such as a function to its subprogram.
  }];
  let mnemonic = "di_loc";

  let parameters = (ins
    "FileLineColLoc":$sourceLoc,
    "DILocalScopeAttr":$scope
  );
  let assemblyFormat = "`<` $sourceLoc `in` $scope `>`";
}

//===----------------------------------------------------------------------===//
// DICompileUnitAttr
//===----------------------------------------------------------------------===//

def CudaTile_DICompileUnitAttr : CudaTile_DIScopeAttr<"DICompileUnit",
                                                      "di_compile_unit",
                                                      /*traits=*/[]> {
  let description = [{
    Represents a compilation unit, the root scope of all objects declared
    in a specific compilation unit; specifies the associated source file
    for the compilation unit.
  }];
  let parameters = (ins
    "DIFileAttr":$file
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DIFileAttr
//===----------------------------------------------------------------------===//

def CudaTile_DIFileAttr : CudaTile_DIScopeAttr<"DIFile",
                                               "di_file",
                                               /*traits=*/[]> {
  let description = [{
    Represents a source file; specifies the file name and directory of the
    source file.
  }];
  let parameters = (ins "StringAttr":$name, "StringAttr":$directory);
  let assemblyFormat = "`<` $name `in` $directory `>`";
}

//===----------------------------------------------------------------------===//
// DILexicalBlockAttr
//===----------------------------------------------------------------------===//

def CudaTile_DILexicalBlockAttr : CudaTile_DILocalScopeAttr<"DILexicalBlock",
                                                            "di_lexical_block",
                                                            /*traits=*/[]> {
  let description = [{
    Represents a lexical block nested within a subprogram; specifies the
    scope, file, line number and optional column number of the block. A
    lexical block, for example, may be used to represent the nested scope
    of a conditional statement.
  }];
  let parameters = (ins
    "DILocalScopeAttr":$scope,
    "DIFileAttr":$file,
    "unsigned":$line,
    OptionalParameter<"unsigned">:$column
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

def CudaTile_DISubprogramAttr : CudaTile_DILocalScopeAttr<"DISubprogram",
                                                          "di_subprogram",
                                                          /*traits=*/[]> {
  let description = [{
    Represents a function within the source language; specifies the scope, file,
    line number, name, and linkage name of the subprogram. Optionally the line
    number within the scope can be included.
  }];
  let parameters = (ins
    "DIFileAttr":$file,
    "unsigned":$line,
    "StringAttr":$name,
    "StringAttr":$linkageName,
    "DICompileUnitAttr":$compileUnit,
    OptionalParameter<"unsigned">:$scopeLine
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // CUDATILE_DIALECT_CUDATILE_IR_ATTRDEFS_TD
