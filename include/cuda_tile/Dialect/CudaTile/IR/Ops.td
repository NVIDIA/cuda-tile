//===- Ops.td - CUDA Tile Operation Definitions ------------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_OPS_TD
#define CUDATILE_DIALECT_CUDATILE_IR_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Interfaces.td"
include "cuda_tile/Dialect/CudaTile/IR/Types.td"
include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td"

// Public-preview operations ship in all builds.
include "cuda_tile/Dialect/CudaTile/IR/PreviewOps.td"

#ifdef TILE_IR_INCLUDE_TESTS
include "cuda_tile/Dialect/CudaTile/IR/TestingOps.td"
#endif // TILE_IR_INCLUDE_TESTS

// Commonly used strings for documentation.
//===----------------------------------------------------------------------===//
// Flush to zero flag's description.
defvar flush_to_zero_desc = "If set, flushes subnormal inputs and results to sign-preserving zero. Only supported for :code:`f32` data type.";
defvar signed_attr_desc = "Interpret integer(s) as :code:`signed` or :code:`unsigned`";
defvar approx_desc = "If set, use the fast approximation.";
defvar token_desc = "The optional token for operation ordering.";
defvar rounding_mode_desc = "The rounding mode for the operation.";
defvar cannonical_nan_desc = "When set, :code:`maxf` (or :code:`minf`) returns a :code:`NaN` if either of the two compared elements is :code:`NaN`.";
defvar overflow_desc = "The overflow behavior of the operation.";
defvar inbounds_desc = !strconcat(
    "Per-dimension bound-check guarantee for the tile coordinates. The ",
    "attribute is required and must contain exactly one bool per index ",
    "dimension; element i = true means the i-th tile coordinate is ",
    "statically known to stay inside the parent tensor along that ",
    "dimension, whereas i = false makes only the i-th dimension be ",
    "conservatively treated as potentially out-of-bounds. In textual MLIR ",
    "the `inbounds = [...]` clause may be omitted, in which case the parser ",
    "synthesizes an all-false vector sized to the index rank.");
defvar fast_accumulation_desc = !strconcat("\n", "Enable faster but less precise FP8 MMA on Hopper GPUs. "
  "Native Hopper FP8 accumulation uses an intermediate precision that is larger than half-precision (F16), but less than full precision (F32). "
  "When the flag is true, rely on native accumulation for faster execution. When false, intermediate results will be periodically promoted "
  "to a higher precision, causing modest performance degradation but providing higher accuracy.");

// NB: any suffix text prefix with :suffix so the RST emitter can normalize
// the white space.
//
// Integer Arithmetic Suffixes
defvar integer_arith_suffix = !strconcat("\n",
  ":suffix: Element-wise integer arithmetic operations are performed by the target architecture's native ",
  "integer instructions. The default semantics are wrap-around semantics on overflow or underflow. ",
  "See :ref:`op-group-integer` for more details.");
  
defvar floating_point_exact_suffix = !strconcat("\n",
  ":suffix: Element-wise floating-point arithmetic operations are performed by the target architecture's native ",
  "floating-point instructions. See :ref:`op-group-floating-point` for more details.");

defvar floating_point_arith_suffix = !strconcat("\n",
  ":suffix: Element-wise floating-point arithmetic operations are performed by the target architecture's native ",
  "floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied "
  "to each element of the result. See :ref:`op-group-floating-point` for more details.");

// Math Suffixes
defvar floating_point_math_suffix = !strconcat("\n",
  ":suffix: This operation is emulated in :code:`f32` when executed on half-precision "
  "inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details."
);

// Rounding Mode Suffix
defvar rounding_mode_suffix = !strconcat("\n",
  ":suffix: If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each"
  "element of the result."
);

//===----------------------------------------------------------------------===//
// AbsFOp
//===----------------------------------------------------------------------===//

def CudaTile_AbsFOp : CudaTileFloatingPointOpDef<"absf", "13.1",
    [Pure, SameOperandsAndResultShape, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise floating-point absolute value";
  let description = !strconcat([{
    The :code:`absf` operation computes the element-wise absolute value of the input float tile.

    .. math::
      \text{absf}(x)_i = |x|_i
  }], floating_point_exact_suffix);

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input float tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The absolute value of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// AbsIOp
//===----------------------------------------------------------------------===//

def CudaTile_AbsIOp : CudaTileIntegerOpDef<"absi", "13.1",
    [Pure, SameOperandsAndResultShape, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise integer absolute value";
  let description = !strconcat([{
    The :code:`absi` operation computes the absolute value of the input integer tile.

    The input tile is always interpreted as a signed integer.
    The output tile is always interpreted as an unsigned integer.

    .. math::
      \text{absi}(x) = |x|
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The input integer tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The absolute value of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def CudaTile_AddIOp : CudaTileIntegerOpDef<"addi", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise integer addition";
  let description = !strconcat([{
    The :code:`addi` operation computes the element-wise addition of two tiles with integer element types.

    .. math::
      \text{addi}(x, y)_i = x_i + y_i
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<DefaultValuedAttr<CudaTile_IntegerOverflowAttr, "::mlir::cuda_tile::IntegerOverflow::NONE">, overflow_desc, "13.1">:$overflow);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The sum of the input tiles.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`overflow` `` $overflow^)? attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def CudaTile_AddFOp : CudaTileFloatingPointOpDef<"addf", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise floating-point addition";
  let description = !strconcat([{
    The :code:`addf` operation computes the element-wise addition of two tiles with floating-point element type.

    .. math::
      \text{addf}(x, y)_i = x_i + y_i

    The addition of individual elements is performed by the target architecture's native floating-point addition
    for the given element type unless otherwise specified.
  }], floating_point_arith_suffix);

  let descriptionTables = [
    Table<":code:`addf` Modifiers", "The below table shows the supported modifiers and rounding modes for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["flush_to_zero", "yes", "no", "no", "no"]>,
       TableRow<["rounding<nearest_even>", "yes", "yes", "yes", "yes"]>,
       TableRow<["rounding<zero>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<negative_inf>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<positive_inf>", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments =
    (ins CudaTileArg<CudaTile_BaseFloatTileType, "The left hand side operand.", "13.1">:$lhs,
         CudaTileArg<CudaTile_BaseFloatTileType, "The right hand side operand.", "13.1">:$rhs,
         CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                     [OnlyCudaTileIEEERoundingModes]>:$rounding_mode,
         CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);

  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The sum of the input tiles.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs
    custom<IEEERoundingMode>($rounding_mode)
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

def CudaTile_AndIOp : CudaTileBitwiseOpDef<"andi", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise bitwise logical AND";
  let description = !strconcat([{
    The :code:`andi` operation produces a value that is the result of an
    element-wise, bitwise "and" of two tiles with integer element
    type.

    .. math::
      \text{andi}(x, y)_i = x_i \land y_i
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The bitwise AND of the input tiles.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def CudaTile_AssertOp : CudaTileControlFlowOpDef<"assert", "13.1"
    > {
  let summary = "Terminate kernel execution with an error message if condition is false-y";
  let description = [{
    The :code:`assert` operation takes as :code:`condition` a tile of
    :code:`i1` values. For each value that is :code:`0`, it prints the given
    error message, along with the index of the value within the tile.

    If at least one value is :code:`0`, an error is signalled to the host
    side. The kernel, including the tile block that failed the assertion,
    may keep running.

    Assertions are for debugging purposes. They can affect performance and it
    is therefore recommended to remove them in production code.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example(%arg0: tile<i1>) {
          assert %arg0, "assertion failed" : tile<i1>
      # }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TileOf<[CudaTile_Int1]>, "The condition tile to check.", "13.1">:$condition,
                       CudaTileArg<StrAttr, "The error message to display if assertion fails.", "13.1">:$message);
  let assemblyFormat = [{
    $condition `,` $message attr-dict `:` custom<CudaTileType>(type($condition))
  }];
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

// Naming convention see: https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces
defvar alloca_is_global_address =
  "Optional tag: marks the returned address as shareable across tile threads; "
  "if omitted, the address is private to the current tile thread (within the "
  "lifetime of the alloca)";

def CudaTile_AllocaOp : CudaTileMemOpDef<"alloca", "13.3"
    > {
  let summary = "Automatic memory allocation";
  let description = [{
    Allocates memory sufficient to hold :code:`num_elem` elements and returns a
    pointer to the allocated memory, or trap if the request cannot be
    satisfied. The returned address is guaranteed to be aligned to :code:`alignment`
    bytes, which must be a non-zero power of two and at least the natural size
    of the pointee type in bytes (e.g. :code:`8` for :code:`f64`, :code:`4` for
    :code:`i32`). By default, the returned address is valid only within the context
    of the current tile thread. If the :code:`global` attribute is set, the address
    may also be accessed by other tile threads. The lifetime of the allocation
    is limited to the block in which the alloca resides.

    For :code:`f4E2M1FN`, consecutive elements are packed two per byte, so an
    allocation of :code:`num_elem` elements occupies
    :code:`ceil(num_elem / 2)` bytes and has a natural alignment of one byte.
  }];
  let arguments = (ins
    CudaTileArg<ConfinedAttr<I64Attr, [IntMinValue<0>]>, "The number of elements to allocate.", "13.3">:$num_elem,
    CudaTileArg<ConfinedAttr<I64Attr, [IntMinValue<0>]>, "The alignment of the allocated memory.", "13.3">:$alignment,
    CudaTileArg<UnitAttr, alloca_is_global_address, "13.3">:$global
  );
  let results = (
    outs CudaTileArg<CudaTile_ScalarTileOf<CudaTile_PointerType>,
                    "The result tile of pointer.", "13.3">:$result
  );

  let assemblyFormat = [{
    `num_elem` `=` $num_elem `,` `alignment` `=` $alignment
    (`global` $global^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @alloca_kernel() {
        %0 = alloca num_elem = 64, alignment = 128 : tile<ptr<f32>>

        %1 = alloca num_elem = 64, alignment = 128 global : tile<ptr<f32>>
      # }
    # }
  }]];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

def CudaTile_AssumeOp : CudaTileMiscOpDef<"assume", "13.1",
    [AllTypesMatch<["value", "result"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Attach static information to an SSA value";
  let description = [{
    The :code:`assume` operation passes through :code:`value` as the result and
    attaches a predicate to it. The assumed predicate is a property of
    :code:`result`.

    This operation can be used to inject static information into the compiler,
    potentially resulting in more efficient code generation.

    :code:`predicate` must implement the :code:`AssumePredicateAttrInterface`.

    .. note::

      :code:`assume` does not check the correctness of the predicate.
      Incorrect predicates may inject incorrect static information and cause
      miscompilation. If an incorrect predicate is attached to an SSA value,
      the behavior of the program is undefined.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example(%input: tile<ptr<f32>>) {
        // Assume that all integers are divisible by 32.
        %int_tile = constant <i16: [32, 64, 0, 0, 32, -32, 1024, 0]> : tile<8xi16>
        %div_by_1 = assume div_by<32>, %int_tile : tile<8xi16>

        // Assume that every 4th element (starting with element 0) along
        // dimension 0 is divisible by 32 that and all integers are
        // montonically increasing by 1 within each group of 4.
        %int_tile_2 = constant <i16: [96, 97, 98, 99, 64, 65, 66, 67]> : tile<8xi16>
        %div_by_2 = assume div_by<32, every 4 along 0>, %int_tile_2 : tile<8xi16>

        // Assume that every rectangular chunk of size [1, 4, 2] has the same
        // values.
        # %input_rank3 = reshape %input : tile<ptr<f32>> -> tile<1x1x1xptr<f32>>
        # %ptr_3d = broadcast %input_rank3 : tile<1x1x1xptr<f32>> -> tile<1x8x8xptr<f32>>
        %same_elem = assume same_elements<[1, 4, 2]>, %ptr_3d : tile<1x8x8xptr<f32>>

        // Assume that every value is greater or equal to 5.
        %int_tile_3 = constant <i16: [5, 9, 10, 11, 6, 5, 5, 7]> : tile<8xi16>
        %bounded = assume bounded<5, ?>, %int_tile_3 : tile<8xi16>
      # }
    # }
  }]];

  let arguments = (ins CudaTileArg<AnyType, "The value to attach the predicate to.", "13.1">:$value,
                       CudaTileArg<CudaTile_AssumePredicateAttrInterface, "The predicate to attach to the value.", "13.1">:$predicate);
  let results = (outs CudaTileArg<AnyType, "The value with the attached predicate.", "13.1">:$result);
  let assemblyFormat = "custom<AssumePredicate>($predicate) `,` $value  attr-dict `:` custom<CudaTileType>(type($value))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Atan2Op
//===----------------------------------------------------------------------===//

def CudaTile_Atan2Op : CudaTileFloatingPointOpDef<"atan2", "13.2", [
    Pure, AllTypesMatch<["x", "y", "result"]>
  ]> {
  let summary = "Element-wise atan2";
  let description = !strconcat([{
    The :code:`atan2` operation calculates the principal value
    of the arc tangent of the ratio of first and second input
    arguments x / y. The quadrant of the result is determined
    by the signs of inputs x and y.

    .. math::

      (\operatorname{atan2}(x, y))_i = \mathrm{atan2}(x_i, y_i)

  }], floating_point_math_suffix);

  let arguments = (
    ins CudaTileArg<CudaTile_BaseFloatTileType, "The input x float tile.", "13.2">:$x,
        CudaTileArg<CudaTile_BaseFloatTileType, "The input y float tile.", "13.2">:$y
  );
  let results = (
    outs CudaTileArg<CudaTile_BaseFloatTileType, "The element-wise result tile.", "13.2">:$result
  );

  let assemblyFormat = [{
    $x `,` $y attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_atan2() {
        %x = constant <f32: [1.0, -1.0, 0.0, 2.0]> : tile<4xf32>
        %y = constant <f32: [1.0,  1.0, 1.0, 0.0]> : tile<4xf32>
        %res = atan2 %x, %y : tile<4xf32>
      # }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// AtomicCASTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_AtomicCASTkoOp : CudaTileAtomicsOpDef<"atomic_cas_tko", "13.1", [
    AllShapesMatch<["pointers", "cmp", "val", "result"]>,
    AllTypesMatch<["cmp", "val", "result"]>,
    AttrSizedOperandSegments]> {
  let summary = "Atomic compare-and-swap on global memory";

  let description = [{
    The :code:`atomic_cas` operation performs element-wise, atomic
    compare-and-swaps at the specified global memory :code:`pointers`. The data in
    memory is compared to :code:`cmp` and the data written to memory is specified
    by :code:`val`. The operation returns the original value that was stored in memory
    before the atomic operation was performed.

    The shape (and the element type) of :code:`pointers`, :code:`cmp`,
    :code:`val` and :code:`result` must match. The :code:`atomic_cas` operation
    performs the following steps for every :code:`(pointer, cmp, val)` tuple in
    one atomic transaction. (One atomic transaction per tuple.)

    .. code-block:: mlir

        atomic() {
          x = *pointer
          if x == cmp {
          *pointer = val
        }
        return x
      }

    An optional parameter, :code:`mask`, allows specifying which elements participate
    in the atomic operation. A false value at position i masks out the
    corresponding element in :code:`pointers`, excluding it from the operation. The
    returned value for a masked element at position i is :code:`cmp[i]`. If no mask is
    provided, all elements are included in the computation by default. The shape of
    mask must match that of :code:`pointers`, :code:`cmp`, and :code:`val`.

    A token-ordered atomic compare-and-swap is not constrained by program order. The compiler
    may reorder it (i.e. place them earlier or later in program order) unless
    constrained by tokens.

    Supported data types:
      - i32, i64: integer values
      - f32, f64: floating-point values

    For floating-point types, the comparison uses bitwise equality rather than
    IEEE-754 semantics. This means different :code:`NaN` bit patterns are treated as
    distinct values, and :code:`+0.0` and :code:`-0.0` are considered different if their bit
    representations differ.

    .. note::

      Each pointer must be aligned to the element type byte size (rounded to
      the next full byte for sub-byte types). Even stronger alignment guarantees
      can be specified with the :code:`cuda_tile.assume` operation.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example(%ptr: tile<ptr<i32>>) {
        %ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
        %ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<8xptr<i32>>
        %offsets = iota : tile<8xi32>
        %ptrs = offset %ptr_vec, %offsets : tile<8xptr<i32>>, tile<8xi32> -> tile<8xptr<i32>>
        %cmp = constant <i32: [0, 1, 2, 3, 4, 5, 6, 7]> : tile<8xi32>
        %val = constant <i32: [7, 6, 5, 4, 3, 2, 1, 0]> : tile<8xi32>
        %mask = constant <i1: [0, 1, 0, 1, 0, 1, 0, 1]> : tile<8xi1>

        // Atomic CAS without input token.
        %0, %token = atomic_cas_tko relaxed device %ptrs, %cmp, %val :
          tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token

        // Atomic CAS without input token.
        %1, %token1 = atomic_cas_tko relaxed device %ptrs, %cmp, %val, %mask :
          tile<8xptr<i32>>, tile<8xi32>, tile<8xi1> -> tile<8xi32>, token

        // Atomic CAS with input token.
        %token2 = make_token : token
        %2, %token3 = atomic_cas_tko relaxed device %ptrs, %cmp, %val token=%token2 :
          tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token

        return
      # }
    # }
  }]];

  let arguments = (ins
    CudaTileArg<
      CudaTile_MemoryOrderingSemanticsAttr,
      "The memory ordering semantics for the atomic operation.",
      "13.1",
      [OnlyVariants<["RELAXED", "ACQUIRE", "RELEASE", "ACQ_REL"]>]>:$memory_ordering_semantics,
    CudaTileArg<CudaTile_MemoryScopeAttr, "The memory scope for the atomic operation.", "13.1">:$memory_scope,
    CudaTileArg<CudaTile_PointerTileType, "The pointers to the memory locations to perform the atomic compare-and-swap operation on.", "13.1">:$pointers,
    CudaTileArg<CudaTile_TileType, "The values to compare against.", "13.1">:$cmp,
    CudaTileArg<CudaTile_TileType, "The values to swap in.", "13.1">:$val,
    CudaTileArg<Optional<CudaTile_TileOf<[CudaTile_Int1]>>, "The mask for the atomic operation.", "13.1">:$mask,
    CudaTileArg<Optional<CudaTile_TokenType>, "The token for the atomic operation.", "13.1">:$token);

  let results = (outs CudaTileArg<CudaTile_TileType, "The result of the atomic operation.", "13.1">:$result,
    CudaTileArg<CudaTile_TokenType, "The result token of the atomic operation.", "13.1">:$result_token);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $memory_ordering_semantics $memory_scope
    $pointers `,` $cmp `,` $val
    (`,` $mask^)?
    (`token` `` `=` `` $token^)?
    attr-dict
    `:` custom<CudaTileType>(type($pointers))
    `,` custom<CudaTileType>(type($val))
    (`,` custom<CudaTileType>(type($mask))^)?
    `->` custom<CudaTileType>(type($result))
    `,` custom<CudaTileType>(type($result_token))
  }];
}

//===----------------------------------------------------------------------===//
// AtomicRMWTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_AtomicRMWTkoOp : CudaTileAtomicsOpDef<"atomic_rmw_tko", "13.1", [
    AllShapesMatch<["pointers", "arg", "result"]>,
    AllTypesMatch<["arg", "result"]>,
    AttrSizedOperandSegments]> {
  let summary = "Atomic read-modify-write on global memory";
  let description = [{
    The :code:`atomic_rmw_tko` operation performs element-wise, atomic
    read-modify-write operations at the global memory locations specified
    by :code:`pointers`. The values written to memory are determined by
    :code:`mode` and :code:`arg`. The operation returns the original value
    stored at each location before the atomic update.

    The shapes of :code:`pointers`, :code:`arg`, and :code:`result` must
    match. The element type of the pointer type must match the element types
    of both :code:`arg` and :code:`result`. Each (:code:`pointer`, :code:`arg`)
    pair is processed in a single atomic transaction.

    .. code-block:: mlir

      atomic {
        x = *pointer
        y = mode(x, arg)
        *pointer = y
        return x
      }

    An optional parameter, :code:`mask`, specifies which elements participate
    in the atomic operation. A `False` value at position :code:`i` excludes
    the corresponding element in :code:`pointers` from the operation.
    The value returned for a masked-out element is implementation-defined.
    The shape of :code:`mask` must match the shape of :code:`pointers`.

    The :code:`addf` operation is defined to round to the nearest even value.

    .. note::

      The current implementation of :code:`addf` compiler flushes denormals to zero. This behavior
      will be fixed in a future version of the compiler and users should not rely on it.


    Token-ordered atomic read-modify-write operations are not constrained by
    program order. The compiler may reorder them (i.e., move them earlier or
    later in the program) unless further constrained by tokens.

    Supported data types by :code:`mode`:

      - ADD, AND, MAX, MIN, OR, UMAX, UMIN, XOR: i32, i64
      - ADDF: f16, bf16, f32, f64
      - XCHG: i32, i64, f32, f64

    The :code:`U` prefix in UMAX and UMIN distinguishes these from their
    signed counterparts (MAX and MIN) by interpreting the comparison as
    unsigned.

    .. note::

      :code:`bf16` is supported from Hopper (sm90) onward. On earlier architectures
      (e.g. Ampere/sm80), atomicRMW ADDF with bf16 is not supported.

    .. note::

      Each pointer must be aligned to the element type byte size (rounded to
      the next full byte for sub-byte types). Even stronger alignment guarantees
      can be specified with the :code:`cuda_tile.assume` operation.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_rmw(%ptr: tile<ptr<f32>>) {
        // Reshape the input pointer tile to have a 1d shape
        %ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
        // Broadcast the reshaped tile to a tile with 8 rows, effectively replicating the pointer 8 times
        %ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
        // Create a tile of offsets [0, 1, 2, ..., 7] to index into memory
        %offsets = iota : tile<8xi32>
        // Add the offsets to each pointer in the vector to create 8 unique pointers
        %ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
        %vals = constant <f32: [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]> : tile<8xf32>

        // Perform atomic addf operations on the memory locations pointed by %ptrs
        // without requiring an input token. Returns the original values and a result token
        %0, %res_token0 = atomic_rmw_tko relaxed device %ptrs, addf, %vals :
            tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token

        // Perform atomic add operations again, this time using the explicit input token
        %token = make_token : token
        %1, %res_token1 = atomic_rmw_tko relaxed device %ptrs, addf, %vals, token = %token :
            tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token
      # }
    # }
  }]];

  let arguments = (ins
    CudaTileArg<
      CudaTile_MemoryOrderingSemanticsAttr,
      "The memory ordering semantics for the load operation.",
      "13.1",
      [OnlyVariants<["RELAXED", "ACQUIRE", "RELEASE", "ACQ_REL"]>]>:$memory_ordering_semantics,
    CudaTileArg<CudaTile_MemoryScopeAttr, "The memory scope for the atomic operation.", "13.1">:$memory_scope,
    CudaTileArg<CudaTile_PointerTileType, "The pointer tile to perform atomic operation on.", "13.1">:$pointers,
    CudaTileArg<CudaTile_AtomicRMWModeAttr, "The atomic operation mode (e.g., add, max, min, etc.).", "13.1">:$mode,
    CudaTileArg<CudaTile_TileType, "The value tile to use in the atomic operation.", "13.1">:$arg,
    CudaTileArg<Optional<CudaTile_TileOf<[CudaTile_Int1]>>, "The mask for the load operation.", "13.1">:$mask,
    CudaTileArg<Optional<CudaTile_TokenType>, "The token for the atomic operation.", "13.1">:$token
  );
  let results = (outs CudaTileArg<CudaTile_TileType, "The result of the atomic operation.", "13.1">:$result,
    CudaTileArg<CudaTile_TokenType, "The result token of the load operation.", "13.1">:$result_token);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $memory_ordering_semantics $memory_scope
    $pointers `,` $mode `,` $arg
    (`,` $mask^)?
    (`token` `` `=` `` $token^)?
    attr-dict
    `:` custom<CudaTileType>(type($pointers))
    `,` custom<CudaTileType>(type($arg))
    (`,` custom<CudaTileType>(type($mask))^)?
    `->` custom<CudaTileType>(type($result))
    `,` custom<CudaTileType>(type($result_token))
  }];
}

//===----------------------------------------------------------------------===//
// AtomicRedViewTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_AtomicRedViewTkoOp : CudaTileAtomicsOpDef<"atomic_red_view_tko", "13.3",
    [AttrSizedOperandSegments]> {
  let summary = "View-based atomic reduction on global memory";

  let description = [{
    The :code:`atomic_red_view_tko` operation performs element-wise atomic
    read-modify-write operations on global memory. The tile identified by
    :code:`index` within :code:`view` is the destination; :code:`value`
    provides the operand for each element. Unlike :code:`atomic_rmw_tko`,
    this operation does not return the original values from memory.

    A view is a mapping from view-space indices to tiles. For example, a
    :ref:`type-partition_view` partitions a :ref:`type-tensor_view` into a
    grid of equally sized tiles. The :code:`index` operands select which tile
    in that grid to reduce into, and must match the rank of the view's index
    space. The :code:`index` operands are interpreted as unsigned integers.

    The :code:`view` must be a :ref:`type-partition_view` or
    :ref:`type-strided_view`; :ref:`type-gather_scatter_view` is not
    supported.

    The element type of :code:`value` must match the element type of
    :code:`view`, and the tile shape of :code:`value` must match the tile
    shape of :code:`view`. Each :code:`index` operand must be a scalar
    (rank-0) integer tile. Each element is updated atomically:

    .. code-block:: text

      ∀ (i, j, …) ∈ shape(value):
          view[index][i,j,…] := mode(view[index][i,j,…], value[i,j,…])   (atomically)

    The original value at each location is not returned; use :code:`atomic_rmw_tko`
    if the original values are needed.

    Supported data types by :code:`mode`:

      - ADD, AND, MAX, MIN, OR, UMAX, UMIN, XOR: i32, i64
      - ADDF: f16, bf16, f32, f64

    The :code:`U` prefix in UMAX and UMIN distinguishes these from their
    signed counterparts (MAX and MIN) by interpreting the comparison as
    unsigned.

    The :code:`addf` operation is defined to round to the nearest even value.

    .. note::

      The current implementation of :code:`addf` flushes denormals to zero. This behavior
      will be fixed in a future version of the compiler and users should not rely on it.

    .. note::

      :code:`bf16` is supported from Hopper (sm90) onward. On earlier architectures
      (e.g. Ampere/sm80), atomicRMW ADDF with bf16 is not supported.


    .. note::

      The pointer of the underlying :code:`tensor_view` must be aligned to the
      element type byte size (rounded to the next full byte for
      sub-byte types). Even stronger alignment guarantees can be
      specified with the :code:`cuda_tile.assume` operation.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%ptr: tile<ptr<f32>>) {
          // Create a view over an 8192x128 f32 matrix in global memory.
          // %ptr points to the base of the matrix (row-major, 128 columns).
          %tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
            : tensor_view<8192x128xf32, strides=[128,1]>

          // Partition the matrix into a grid of 64x64 tiles.
          // This creates a (8192/64) x (128/64) = 128x2 grid of tiles,
          // each addressable by a 2D integer index (row_tile, col_tile).
          %view = make_partition_view %tensor_view :
            partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

          // Tile indices used to select which partition to reduce into.
          %c0 = constant <i32: 0> : tile<i32>
          %c1 = constant <i32: 1> : tile<i32>

          // The local contribution to accumulate (64x64 tile of f32 values)
          %local_contribution = constant <f32: 1.0> : tile<64x64xf32>

          // Atomically accumulate local_contribution into tile (0,0):
          // the 64x64 region at rows [0,64), cols [0,64) of the global matrix.
          %t0 = atomic_red_view_tko relaxed device %view[%c0, %c0], addf, %local_contribution
            : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

          // Atomically accumulate into tile (0,1):
          // the 64x64 region at rows [0,64), cols [64,128) of the global matrix.
          %t1 = atomic_red_view_tko relaxed device %view[%c0, %c1], addf, %local_contribution
            : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

          // Same reduction with an input token to enforce ordering with a
          // prior operation (e.g. ensure %t1 completes before this starts).
          %token = make_token : token
          %t2 = atomic_red_view_tko relaxed device %view[%c0, %c1], addf, %local_contribution token = %token
            : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token
    #   }
    # }
  }]];

  let arguments =
      (ins CudaTileArg<
           CudaTile_MemoryOrderingSemanticsAttr,
           "The memory ordering semantics for the atomic operation.",
           "13.3", [OnlyVariants<["RELAXED"]>]>:$memory_ordering_semantics,
          CudaTileArg<
              CudaTile_MemoryScopeAttr,
              "The memory scope for the atomic operation.",
              "13.3", [OnlyVariants<["TL_BLK", "DEVICE"]>]>:$memory_scope,
          CudaTileArg<CudaTile_TileView, "Destination view", "13.3">:$view,
          CudaTileArg<Variadic<CudaTile_IntTileType>,
                      "The n-dimensional index of the tile to reduce into.",
                      "13.3">:$index,
          CudaTileArg<CudaTile_AtomicRMWModeAttr, "RMW mode", "13.3",
                      [OnlyVariants<["AND", "OR", "XOR", "ADD", "ADDF", "MAX",
                                     "MIN", "UMAX", "UMIN"]>]>:$mode,
          CudaTileArg<CudaTile_TileType, "Value tile for atomic operation",
                      "13.3">:$value,
          CudaTileArg<Optional<CudaTile_TokenType>, "Input token",
                      "13.3">:$token,
          CudaTileArg<DenseBoolArrayAttr, inbounds_desc,
                      "13.5", [RequireSameOperandRank<"index">]>:$inbounds);

  let results = (outs CudaTileArg<CudaTile_TokenType, "Result token", "13.3">:$result_token);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

def CudaTile_BitcastOp : CudaTileConversionOpDef<"bitcast", "13.1", [
    Pure, AllShapesMatch<["source", "result"]>]> {

  let summary = "Bitcast a tile from one element type to another";

  let description = [{
    The :code:`bitcast` operation casts the input tile from one element type to
    another without modifying the underlying bits.

    Only non-pointer types of the same bit width are allowed (e.g., :code:`i32` to :code:`f32`).
    Pointer types must use :ref:`op-cuda_tile.ptr_to_int` or :ref:`op-cuda_tile.int_to_ptr` instead.
  }];

  let arguments = (ins CudaTileArg<CudaTile_NumberTileType, "The source tile to cast.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_NumberTileType, "The casted tile.", "13.1">:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source attr-dict
    `:` custom<CudaTileType>(type($source)) `->` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

def CudaTile_BroadcastOp : CudaTileCoreOpDef<"broadcast", "13.1",
    [Pure, SameOperandsAndResultElementType,
     AllRanksMatch<["source", "result"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Broadcast tile to new shape";
  let description = [{
    The :code:`broadcast` operation expands each unary (:code:`1`) dimension in the input tile
    by duplicating the data along that dimension.

    Expansion happens only for dimensions of size one that are stretched or "copied" to match
    the size of the dimension implied by the result type of the operation. The operation
    does not change the rank of the source tile.  Any change to the rank of the source tile
    must be made using reshape-like operations before broadcasting.

    .. .. math::
      .. broadcast(x, idim_n, odim_n) = x
  }];

  let arguments = (ins CudaTileArg<CudaTile_TileType, "The tile to broadcast.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_TileType, "The broadcasted tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($source))
    `->` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CatOp
//===----------------------------------------------------------------------===//

def CudaTile_CatOp : CudaTileCoreOpDef<"cat", "13.1",
    [Pure, AllRanksMatch<["lhs", "rhs", "result"]>,
     AllElementTypeMatch<"all of {lhs, rhs, result} have the same element type", ["lhs", "rhs", "result"]>]> {
  let summary = "Concatenate tiles along specified dimension";
  let description = [{
    The :code:`cat` operation concatenates the two input tiles. The input tiles must have the same shape
    in all but the concatenating dimension. Concatenation happens along the dimension specified by the
    the attribute :code:`dim` the resulting dimension is the sum of the the two input tiles concatenating
    dimension.

    .. math::

      \text{cat}(x, y, dim_{cat})[ \vec{i} ] =
        \begin{cases}
          x[..., i_{cat}, ..., i_n] & \text{if } i_{cat} < d_{cat} \\
          y[..., i_{cat} - d_{cat}, ..., i_n] & \text{if } i_{cat} \geq d_{cat}
        \end{cases}

    .. \text{where } X \text{ has type tile}<d_0 \times d_1 \times \cdots \times d_n>
    ..      \text{ and } Y \text{ has type tile}<d_0 \times d_1 \times \cdots \times d_n>

  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example() {
      # %arg0 = constant <f32: 0.0> : tile<2x4xf32>
      # %arg1 = constant <f32: 1.0> : tile<2x4xf32>

          // A valid invocation of cat.
          %0 = cat %arg0, %arg1 dim = 1
            : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

          // >>> %arg0 = tile([[ A, B, C ],
          //                   [ D, E, F ]])
          // >>> %arg1 = tile([[ 1, 2, 3 ],
          //                   [ 4, 5, 6 ]])
          // >>> %0 = tile([[ A, B, C, 1, 2, 3 ],
          //                [ D, E, F, 4, 5, 6 ]])

          // A valid invocation of cat.
          %1 = cat %arg0, %arg1 dim = 0
            : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>

          // >>> %arg0 = tile([[ A, B, C ],
          //                   [ D, E, F ]])
          //
          // >>> %arg1 = tile([[ 1, 2, 3 ],
          //                   [ 4, 5, 6 ]])
          //
          // >>> %1 = tile([[ A, B, C ],
          //                [ D, E, F ],
          //                [ 1, 2, 3 ],
          //                [ 4, 5, 6 ]])
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_TileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<I64Attr, "The dimension along which to concatenate.", "13.1">:$dim);
  let results = (outs CudaTileArg<CudaTile_TileType, "The concatenated result tile.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `dim` `=` $dim
    attr-dict `:` custom<CudaTileType>(type($lhs)) `,` custom<CudaTileType>(type($rhs))
    `->` custom<CudaTileType>(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

def CudaTile_CosOp : CudaTileFloatingPointOpDef<"cos", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise cosine";
  let description = !strconcat([{
  The :code:`cos` operation computes the element-wise cosine of the
  input floating-point tile.

  .. math::

    \text{cos}(x)_i = \cos(x_i)
}], floating_point_math_suffix);

  let arguments = (ins
    CudaTileArg<CudaTile_BaseFloatTileType, "The input float tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The cosine of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_cos() {
        %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
        %res = cos %in : tile<4xf32>
      # }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// CosHOp
//===----------------------------------------------------------------------===//

def CudaTile_CosHOp : CudaTileFloatingPointOpDef<"cosh", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise hyperbolic cosine";
  let description = !strconcat([{
    The :code:`cosh` operation computes the element-wise hyperbolic cosine of the
    input tile with floating-point element type.

    .. math::

      \text{cosh}(x)_i = {\cosh x}_i

  }], floating_point_math_suffix);

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input floating-point tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The hyperbolic cosine of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def CudaTile_BreakOp : CudaTileControlFlowOpDef<"break", "13.1", [
    ReturnLike, Terminator, ParentOneOf<["IfOp", "LoopOp"]>
  ]> {
  let summary = "Break from loop";
  let description = [{
    The :code:`break` operation is a terminator operation of a :ref:`op-cuda_tile.loop`.

    It may yield any number of :code:`$operands` to the parent loop upon termination; their
    types must match the result types of the enclosing :ref:`op-cuda_tile.loop`. The number
    of values yielded and the execution semantics of how they are yielded are determined by
    the parent loop.

    The :code:`break` operation always returns control to the innermost enclosing loop operation,
    even when it is nested within other control constructs such as :code:`if` or additional loops.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example() {
        // Break from the body of a loop.
        loop {
            break
        }

        // Break from an if nested within the loop.
        loop  {
            %condition = constant <i1: 1> : tile<i1>
            if %condition  {
                break
            }
            // ...
        }

        %initValue0 = constant <f32: 0.0> : tile<f32>
        // Break from an if nested within the loop, while yielding values.
        %results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
            %condition = constant <i1: 1> : tile<i1>
            if %condition  {
                // ...
                yield
            } else {
                // %if.loopValue0 = ...
                %loopValue0 = constant <f32: 1.0> : tile<f32>
                break %loopValue0 : tile<f32>
            }
            %loopValue1 = constant <f32: 1.0> : tile<f32>
            continue %loopValue1 : tile<f32>
        }
      # }
    # }
  }]];

  let arguments = (ins CudaTileArg<Variadic<AnyType>, "The operands to yield to the parent loop upon termination.", "13.1">:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict ($operands^ `:` custom<CudaTileType>(type($operands)))?
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CeilOp
//===----------------------------------------------------------------------===//

def CudaTile_CeilOp : CudaTileFloatingPointOpDef<"ceil", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise ceiling";
  let description = [{
    The :code:`ceil` operation computes the element-wise ceiling on the input
    floating-point tile. The ceiling operation rounds each element up to the
    largest integer value that is greater than or equal to the input value.


    .. math::

      \text{ceil}(x)_i = \min\{n \in \mathbb{Z} \mid n \geq x_i\}
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example() {
        # %source = constant <f32: 0.5> : tile<f32>
        %result = ceil %source : tile<f32>
      # }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input float tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The ceiling of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

def CudaTile_CmpFOp : CudaTileFloatingPointOpDef<"cmpf", "13.1", [Pure, AllTypesMatch<["lhs", "rhs"]>, TypesMatchWith<
    "Result type has i1 element type and same shape as operands",
    "lhs", "result", "::getI1SameShape($_self)">]> {
  let summary = "Element-wise floating-point comparison";
  let description = [{
    The :code:`cmpf` operation is a generic comparison for float-like types. The
    operands must have the same shape and type, and this type must be a float type.

    The result is :code:`1` if the comparison is true and :code:`0` otherwise. The comparison is
    performed element-wise and the element of the result indicates whether the
    comparison is true for the operand elements with the same indices as those of
    the result.

    .. math::
      \text{cmpf}(x, y, \text{pred})_i = \begin{cases}
        1 & \text{if } x_i \text{ pred } y_i \\
        0 & \text{otherwise}
      \end{cases}
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
    #   entry @example() {
          %lhs0 = constant <f16: 0.0> : tile<f16>
          %rhs0 = constant <f16: 0.0> : tile<f16>

          // Custom form of scalar "ordered equal" comparison.
          %x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16> -> tile<i1>

          %lhs1 = constant <f16: 0.0> : tile<2x2xf16>
          %rhs1 = constant <f16: 0.0> : tile<2x2xf16>

          // Custom form of scalar "unordered less than" comparison.
          %x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16> -> tile<2x2xi1>

          %lhs2 = constant <f64: 0.0> : tile<2x2xf64>
          %rhs2 = constant <f64: 0.0> : tile<2x2xf64>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_ComparisonPredicateAttr, "The comparison predicate.", "13.1">:$comparison_predicate,
                       CudaTileArg<CudaTile_ComparisonOrderingAttr, "The comparison ordering.", "13.1">:$comparison_ordering,
                       CudaTileArg<CudaTile_BaseFloatTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_BaseFloatTileType, "The right hand side operand.", "13.1">:$rhs);

  let assemblyFormat = [{
    custom<ComparisonPredicate>($comparison_predicate) custom<ComparisonOrdering>($comparison_ordering) $lhs `,`
    $rhs attr-dict `:` custom<CudaTileType>(type($lhs)) `->` custom<CudaTileType>(type($result))
  }];

  let results = (outs CudaTileArg<CudaTile_TileOf<[CudaTile_Int1]>, "The result of the comparison.", "13.1">:$result);

  let extraClassDeclaration = [{
    static cuda_tile::ComparisonPredicate getPredicateByName(StringRef name);
  }];
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def CudaTile_CmpIOp : CudaTileIntegerOpDef<"cmpi", "13.1", [Pure, AllTypesMatch<["lhs", "rhs"]>, TypesMatchWith<
    "Result type has i1 element type and same shape as operands",
    "lhs", "result", "::getI1SameShape($_self)">]> {
  let summary = "Element-wise integer comparison";
  let description = [{
    The :code:`cmpi` operation is a generic comparison for integer-like types. The
    operands must have the same shape and type, and this type must be an integer type.
    The result type has i1 element type and the same shape as the operands.

    The result is :code:`1` if the comparison is true and :code:`0` otherwise. The comparison is
    performed element-wise and the element of the result indicates whether the
    comparison is true for the operand elements with the same indices as those of
    the result.

    .. math::
      \text{cmpi}(x, y, \text{pred})_i = \begin{cases}
        1 & \text{if } x_i \text{ pred } y_i \\
        0 & \text{otherwise}
      \end{cases}
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %lhs0 = constant <i16: 0> : tile<i16>
          %rhs0 = constant <i16: 0> : tile<i16>

          // Scalar "signed less than" comparison.
          %x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16> -> tile<i1>

          %lhs1 = constant <i64: 0> : tile<2x2xi64>
          %rhs1 = constant <i64: 0> : tile<2x2xi64>

          // Tile equality comparison.
          // There is no difference between "signed" and "unsigned" when performing equality and inequality comparison.
          %x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64> -> tile<2x2xi1>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_ComparisonPredicateAttr, "The comparison predicate.", "13.1">:$comparison_predicate,
                       CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness);

  let assemblyFormat = [{
    custom<ComparisonPredicate>($comparison_predicate) $lhs `,` $rhs `,`
    custom<Signedness>($signedness) attr-dict `:` custom<CudaTileType>(type($lhs)) `->` custom<CudaTileType>(type($result))
  }];

  let results = (outs CudaTileArg<CudaTile_TileOf<[CudaTile_Int1]>, "The result of the comparison.", "13.1">:$result);

  let extraClassDeclaration = [{
    static cuda_tile::ComparisonPredicate getPredicateByName(StringRef name);
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def CudaTile_ConstantOp : CudaTileCoreOpDef<"constant", "13.1",
    [ConstantLike, Pure,  AllTypesMatch<["value", "result"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Construct a constant tile";
  let description = [{
    The :code:`constant` operation creates a tile initialized by :code:`$value`.

    There are two main forms of using the operation:

    - One where the value is a single constant specified by :code:`<D: c>`
      and the tile is filled with identical values for all elements with element type :code:`D`.

    - One where the value is a list of constants specified by :code:`dense<D: [c0, c1, c2, ...]>`
      and the constant value's shape must match the tile's shape with the element type :code:`D`.

    The annotated type of the tile constrains its rank, shape, and element type.
  }];

  let arguments = (ins CudaTileArg<Builtin_DenseTypedElementsAttr, "The constant value to create.", "13.1">:$value);
  let results = (outs CudaTileArg<CudaTile_NumberTileType, "The constant tile.", "13.1">:$result);
  let hasFolder = 1;
  let assemblyFormat = [{ custom<DenseTypedElementsAttr>($value, type($result)) attr-dict }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example() {
        %c0 = constant <i32: 0> : tile<i32>
        %c1 = constant <i64: 1> : tile<i64>
        %c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
        %c3 = constant <f32: 0.0> : tile<2x4xf32>
        %c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
    #  }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def CudaTile_ContinueOp : CudaTileControlFlowOpDef<"continue", "13.1", [
    Terminator, ParentOneOf<["ForOp", "IfOp", "LoopOp"]>
  ]> {
  let summary = "Continue to next loop iteration";
  let description = [{
    The :code:`continue` operation represents a block terminator that returns control to
    a loop operation, such as :ref:`op-cuda_tile.for` and :ref:`op-cuda_tile.loop`. The operation
    may yield any number of :code:`$operands` to the parent loop upon termination; their types
    must match the parent loop's loop-carried types — :code:`iter_values` types for
    :ref:`op-cuda_tile.loop`, result types for :ref:`op-cuda_tile.for`.

    The requirements and semantics of the :code:`continue` operation are defined by the parent loop
    operation, see the loop operation's description for particular semantics.

    The :code:`continue` operation always returns control to the innermost enclosing loop operation,
    even when it is nested within other control constructs such as :code:`if` or additional loops.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %lowerBound = constant <i32: 0> : tile<i32>
          %upperBound = constant <i32: 10> : tile<i32>
          %step = constant <i32: 1> : tile<i32>
          %condition = constant <i1: 1> : tile<i1>
          // Continue from the body of a loop.
          for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
              continue
          }

          // Continue from an if nested within the loop.
          for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
              if %condition  {
                  continue
              }
              // ...
          }

        // Continue from an if nested within the loop, while yielding values.
        %initVar0 = constant <f32: 0.0> : tile<f32>
        %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                  iter_values(%var0 = %initVar0) -> (tile<f32>)
          {
              if %condition {
                  // ...
                  yield
              } else {
                  %loopValue0 = constant <f32: 1.0> : tile<f32>
                  continue %loopValue0 : tile<f32>
              }
              %loopValue1 = constant <f32: 1.0> : tile<f32>
              continue %loopValue1 : tile<f32>
          }
      # }
    # }
  }]];

  let arguments = (ins CudaTileArg<Variadic<AnyType>, "The values to yield to the parent loop.", "13.1">:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict ($operands^ `:` custom<CudaTileType>(type($operands)))?
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetIndexSpaceShapeOp
//===----------------------------------------------------------------------===//

def CudaTile_GetIndexSpaceShapeOp :
    CudaTileViewOpDef<"get_index_space_shape", "13.1", [Pure]> {
  let summary = "Query the index space dimension size";
  let description = [{
    The :code:`get_index_space_shape` operation returns the shape of the index
    space of :code:`src`.

    The result tile has the same rank as the view's index space with the elements
    representing the size of the corresponding dimension.

    The result values should be interpreted as unsigned integers.

    .. warning::

      If the individual index space dimension do not fit in the result tile's element type
      the behavior is undefined.
  }];

  let arguments =
    (ins CudaTileArg<CudaTile_TileView, "The source view type.", "13.1">:$src);
  let results =
    (outs CudaTileArg<
        Variadic<CudaTile_ScalarTileOf<CudaTile_AnyInt>>,
        [{The shape of the index space, each value representing the size of the
          corresponding dimension.}],
        "13.1"
      >:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example(%base: tile<ptr<f32>>) {
        %tensor_view = make_tensor_view %base,
            shape = [2, 2, 4], strides = [2, 2, 1]
            : tensor_view<2x2x4xf32, strides=[2,2,1]>
        %partition_view = make_partition_view %tensor_view :
          partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>>
        %dim0, %dim1, %dim2 = get_index_space_shape %partition_view :
          partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>> -> tile<i64>
      # }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// GetTensorShapeOp
//===----------------------------------------------------------------------===//

def CudaTile_GetTensorShapeOp :
    CudaTileViewOpDef<"get_tensor_shape", "13.1", [Pure]> {
  let summary = "Query the shape of a tensor view";
  let description = [{
    The :code:`get_tensor_shape` operation returns the shape of the tensor
    backing the provided tensor view.

    The result values should be interpreted as unsigned integers.

    .. warning::

      If the tensor dimensions do not fit in the result tile's element type
      the behavior is undefined.
  }];

  let arguments = (ins
    CudaTileArg<
      CudaTile_TensorViewType,
      "The source tensor view.",
      "13.1"
    >:$src);
  let results = (outs
    CudaTileArg<
      Variadic<CudaTile_ScalarTileOf<CudaTile_AnyInt>>,
      // You can't line break here right now causes the docs to break.
      [{The shape of the tensor, each value representing the size of the corresponding dimension.}],
      "13.1"
    >:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let mlirExamples = [[{
    # cuda_tile.module @module {
      # entry @example(%base: tile<ptr<f32>>) {
        # %tensor_view = make_tensor_view %base,
        #     shape = [32, 32], strides = [32, 1]
        #     : tensor_view<32x32xf32, strides=[32,1]>
        %dim0, %dim1 = get_tensor_shape %tensor_view : tensor_view<32x32xf32, strides=[32,1]> -> tile<i64>
      # }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

def CudaTile_DivFOp : CudaTileFloatingPointOpDef<"divf", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise floating-point division";
  let description = !strconcat([{
    The :code:`divf` operation computes the element-wise division of two input tiles
    with floating-point element types.

    The :code:`approx` rounding mode implements a fast approximation of divide,
    computed as a multiplication by reciprocal. For :code:`|rhs|` in normalized range
    :code:`[2^(-126), 2^(126)]` the maximum ULP (Unit in the Last Place) error is :code:`2`.
    For :code:`2^(126) < |rhs| < 2^(128)`, if :code:`lhs` is infinity the operation returns :code:`NaN`,
    otherwise :code:`0`.

    The :code:`full` rounding mode implements a relatively fast, full-range
    approximation that scales operands to achieve better accuracy, but is not fully
    IEEE 754 compliant. The maximum ulp error is 2 across the full range of inputs.

    .. math::
      \text{div(lhs, rhs)}_i = \text{lhs}_i / \text{rhs}_i
  }], floating_point_arith_suffix);

  let descriptionTables = [
    Table<":code:`divf` Modifiers", "The below table shows the supported modifiers and rounding modes for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["flush_to_zero", "yes", "no", "no", "no"]>,
       TableRow<["approx", "yes", "no", "no", "no"]>,
       TableRow<["full", "yes", "no", "no", "no"]>,
       TableRow<["rounding<nearest_even>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<zero>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<negative_inf>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<positive_inf>", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The dividend input floating-point tile.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_BaseFloatTileType, "The divisor input floating-point tile.", "13.1">:$rhs,
                       CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                                   [OnlyCudaTileDivFRoundingModes]>:$rounding_mode,
                       CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);

  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the :code:`divf` operation.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs
    custom<DivFOpRoundingMode>($rounding_mode)
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
   let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DivIOp
//===----------------------------------------------------------------------===//

def CudaTile_DivIOp : CudaTileIntegerOpDef<"divi", "13.1",
    [NoMemoryEffect, ConditionallySpeculatable,
     AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise integer division";
  let description = !strconcat([{
    The :code:`divi` operation computes the element-wise division of two tile values with integer element type.

    The use of the rounding flag `negative_inf` with `unsigned` is not a valid combination.

    If the `unsigned` flag is provided, the operands are treated as unsigned integers, otherwise they are
    treated as signed integers.

    The behavior is undefined if the right hand side is zero. A signed division overflow (minimum value
    divided by -1) is undefined behavior.

    .. math::
      \text{div(lhs, rhs)}_i = \text{lhs}_i / \text{rhs}_i
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness,
                       CudaTileArg<DefaultValuedAttr<CudaTile_RoundingModeAttr, "RoundingMode::ZERO">, "Set the rounding direction (implementing :spelling:ignore:`floordiv`/:spelling:ignore:`ceildiv`).", "13.1",
                                   [OnlyCudaTileIntegerDivisionRoundingModes]>:$rounding);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The result of the division.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) (`rounding` `` $rounding^)? attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    ::mlir::Speculation::Speculatability getSpeculatability();
  }];
}

//===----------------------------------------------------------------------===//
// MmaFOp
//===----------------------------------------------------------------------===//

def MmaFOp_OperandTileType : CudaTile_TileOf<[CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32,
                                            CudaTile_Float64, CudaTile_TFloat32, CudaTile_Float8E4M3FN,
                                            CudaTile_Float8E5M2],
                                            summary = "mmaf operand tile type">;
def MmaFOp_ResultTileType : CudaTile_TileOf<[CudaTile_Float16, CudaTile_Float32, CudaTile_Float64],
                                           summary = "mmaf acc/result tile type">;

def CudaTile_MmaFOp : CudaTileFloatingPointOpDef<"mmaf", "13.1",
    [Pure, AllTypesMatch<["acc", "result"]>,
     AllElementTypeMatch<"all of {lhs, rhs} have the same element type", ["lhs", "rhs"]>,
     AllRanksMatch<["lhs", "rhs", "acc"]>]> {
  let summary = "Floating-point matrix-multiply-accumulate";

  let description = [{
    The :code:`mmaf` operation implements an MMA (matrix-multiply-accumulate) operation for floating-point tiles.
    It performs matrix multiplication on the floating-point tiles :code:`lhs` and :code:`rhs`, then adds the tile :code:`acc` to the result.
    :code:`lhs`, :code:`rhs`, and :code:`acc` must be 2D tiles or 3D tiles. The latter case
    indicates a batched matrix multiplication.

    .. math::
      \text{mmaf}(A, B, C)_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj} + C_{ij}

    The types of all operands must be a supported combination (see :ref:`table-cuda_tile.mmaf-0`).

    Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
    MMA expects the operands :code:`lhs`, :code:`rhs`, and :code:`acc` to have shapes :code:`M x K`,
    :code:`K x N`, and :code:`M x N` (respectively). Batched (3D) MMA expects the operands
    to have shapes :code:`B x M x K`, :code:`B x K x N`, and :code:`B x M x N` (respectively).

    To perform a matrix multiply without accumulation, pass a constant zero tile for the
    :code:`acc` parameter.
    
    .. warning::

      :code:`mmaf` results are not bit-identical across tile sizes, target architectures,
      compiler/driver versions, or ``fast_acc`` settings. See
      :ref:`MMA Numerical Stability <subsection-mma-numerical-stability>`
      for the reasons MMA results are not stable and what programs can and
      cannot assume.

  }];

  let descriptionTables = [
    Table<":code:`mmaf` Supported Data Types", "The table below shows the "
      "supported output types for each possible :code:`mmaf` input type. "
      "Input operands must be of the same element type.",
      [TableHeader<"Input Type", "code">, TableHeader<"Supported Output Types">],
      [TableRow<["f8E4M3FN", ":code:`f16` or :code:`f32`"]>,
      TableRow<["f8E5M2", ":code:`f16` or :code:`f32`"]>,
      TableRow<["f16", ":code:`f16` or :code:`f32`"]>,
      TableRow<["bf16", ":code:`f32`"]>,
      TableRow<["tf32", ":code:`f32`"]>,
      TableRow<["f32", ":code:`f32`"]>,
      TableRow<["f64", ":code:`f64`"]>,
      ]
    >
  ];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %lhs0 = constant <f16: 0.0> : tile<4x8xf16>
          %rhs0 = constant <f16: 0.0> : tile<8x2xf16>
          %acc0 = constant <f32: 0.0> : tile<4x2xf32>

          %0 = mmaf %lhs0, %rhs0, %acc0
              : tile<4x8xf16>, tile<8x2xf16>,
                tile<4x2xf32>

          %lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
          %rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
          %acc1 = constant <f32: 0.0> : tile<2x4x2xf32>

          %1 = mmaf %lhs1, %rhs1, %acc1
              : tile<2x4x8xf16>, tile<2x8x2xf16>,
                tile<2x4x2xf32>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<MmaFOp_OperandTileType, "The left hand side matrix operand.", "13.1">:$lhs,
                   CudaTileArg<MmaFOp_OperandTileType, "The right hand side matrix operand.", "13.1">:$rhs,
                   CudaTileArg<MmaFOp_ResultTileType, "The accumulator matrix operand.", "13.1">:$acc,
                   CudaTileArg<UnitAttr, fast_accumulation_desc, "13.3">:$fast_acc);
  let results = (outs CudaTileArg<MmaFOp_ResultTileType, "The result matrix after multiplication and accumulation.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc
    (`fast_acc` $fast_acc^)?
    attr-dict `:`
    custom<CudaTileType>(type($lhs)) `,` custom<CudaTileType>(type($rhs)) `,` custom<CudaTileType>(type($acc))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MmaFScaledOp
//===----------------------------------------------------------------------===//

def MmaFScaledOp_OperandTileType : CudaTile_TileOf<[CudaTile_Float8E4M3FN,
                                                    CudaTile_Float8E5M2,
                                                    CudaTile_Float4E2M1FN],
                                                    summary = "mmaf_scaled operand tile type">;
def MmaFScaledOp_ScaleTileType : CudaTile_TileOf<[CudaTile_Float8E4M3FN,
                                                  CudaTile_Float8E8M0FNU,
                                                  CudaTile_Float8E5M3FNU
                                                  ],
                                                  summary = "mmaf_scaled scale tile type">;
def MmaFScaledOp_ResultTileType : CudaTile_TileOf<[CudaTile_Float32],
                                                   summary = "mmaf_scaled result tile type">;

def CudaTile_MmaFScaledOp : CudaTileFloatingPointOpDef<"mmaf_scaled", "13.3",
    [Pure, AllTypesMatch<["acc", "result"]>,
     AllElementTypeMatch<"all of {lhs, rhs} have the same element type", ["lhs", "rhs"]>,
     AllElementTypeMatch<"all of {lhs_scale, rhs_scale} have the same element type", ["lhs_scale", "rhs_scale"]>,
     AllRanksMatch<["lhs", "rhs", "acc", "lhs_scale", "rhs_scale"]>]> {
  let summary = "Floating-point matrix-multiply-accumulate with scaled inputs";

  let description = [{
    The :code:`mmaf_scaled` operation is a variant of the floating-point MMA operation (see :ref:`op-cuda_tile.mmaf`)
    for low-precision input types that use block scaling. In addition to the normal MMA operands, it takes two additional
    operands: :code:`$lhs_scale` and :code:`$rhs_scale`, which are the scale factors for the left hand side and right hand side operands,
    respectively. Each element of the scale factor tiles are used to scale a block of continuous elements along the K dimension
    in the corresponding input tile.

    Conceptually, the operation is equivalent to broadcasting :code:`lhs_scale` and :code:`rhs_scale` along the K dimension to match the shape
    of the left hand side and right hand side operands, performing the elementwise multiplications :code:`lhs*sfa` and :code:`rhs*sfb`,
    and then performing the standard MMA on the scaled tiles.

    .. math::
      \text{mmaf_scaled}(lhs, rhs, acc, lhs\_scale, rhs\_scale)_{i,j} = \sum_{k=0}^{K-1} (lhs_{i,k}*lhs\_scale_{i,k/V}) \times (rhs_{k,j}*rhs\_scale_{k/V,j}) + acc_{i,j}

    where :math:`V` is the scale-factor vector (or block) size, derived as the K
    dimension of :code:`lhs`/:code:`rhs` divided by the K dimension of
    :code:`lhs_scale`/:code:`rhs_scale`. The K dimensions must therefore divide
    evenly, and :math:`V` is constrained by the input/scale element types:

    - :code:`f8E4M3FN` and :code:`f8E5M2` inputs: :math:`V` must be 32.
    - :code:`f4E2M1FN` inputs with :code:`f8E4M3FN` scales: :math:`V` must be 16.
    - :code:`f4E2M1FN` inputs with :code:`f8E8M0FNU` scales: :math:`V` must be 16 or 32.

    The same rank and shape constraints apply as for :ref:`op-cuda_tile.mmaf`. The operand types must be a supported combination
    (see :ref:`table-cuda_tile.mmaf_scaled-0`). :code:`lhs` and :code:`rhs` must be of the same element type.
    :code:`lhs_scale` and :code:`rhs_scale` must be of the same element type.

    .. note::
        Not every valid operand and vector size configuration is efficiently supported on all architectures.
        Only scale-vector sizes of 16 and 32 are accepted; for :code:`f8E4M3FN` scales, only vector
        size 16 (the NVFP4 configuration) is supported.

    .. note::
        When using :code:`f8E4M3FN` as the scale type, values must be non-negative. Negative scale values may produce undefined values.

    .. warning::

        :code:`mmaf_scaled` is not supported on Hopper (sm_90). It is available on Blackwell (sm_100, sm_120) and later.

    .. warning::

        :code:`mmaf_scaled` results are not bit-identical across tile sizes, target
        architectures, or compiler/driver versions. See
        :ref:`MMA Numerical Stability <subsection-mma-numerical-stability>`
        for the reasons MMA results are not stable and what programs can and
        cannot assume.

  }]
  ;

    let descriptionTables = [
    Table<":code:`mmaf_scaled` Supported Data Types", "The table below shows the "
      "supported type and vector size combinations for the :code:`mmaf_scaled` operation.",
      [TableHeader<":code:`lhs`/:code:`rhs` Element Type", "code">,
       TableHeader<":code:`lhs_scale`/:code:`rhs_scale` Element Type", "code">,
       TableHeader<"Output Type", "code">],
       [
        TableRow<["f8E4M3FN, f8E5M2, f4E2M1FN", "f8E8M0FNU", "f32"]>,
        TableRow<["f4E2M1FN", "f8E4M3FN", "f32"]>,
        TableRow<["f4E2M1FN", "f8E5M3FNU", "f32"]>,
       ]
    >
  ];

  let arguments = (ins CudaTileArg<MmaFScaledOp_OperandTileType, "The left hand side matrix operand.", "13.3">:$lhs,
                   CudaTileArg<MmaFScaledOp_OperandTileType, "The right hand side matrix operand.", "13.3">:$rhs,
                   CudaTileArg<MmaFScaledOp_ResultTileType, "The accumulator matrix operand.", "13.3">:$acc,
                   CudaTileArg<MmaFScaledOp_ScaleTileType, "The left hand side scale operand.", "13.3">:$lhs_scale,
                   CudaTileArg<MmaFScaledOp_ScaleTileType, "The right hand side scale operand.", "13.3">:$rhs_scale);
  let results = (outs CudaTileArg<MmaFScaledOp_ResultTileType, "The result matrix after multiplication and accumulation.", "13.3">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc `,` $lhs_scale `,` $rhs_scale attr-dict `:`
    custom<CudaTileType>(type($lhs)) `,` custom<CudaTileType>(type($rhs)) `,`
    custom<CudaTileType>(type($acc)) `,` custom<CudaTileType>(type($lhs_scale)) `,`
    custom<CudaTileType>(type($rhs_scale))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MmaIOp
//===----------------------------------------------------------------------===//

def MmaIOp_OperandTileType : CudaTile_TileOf<[CudaTile_Int8],
                                            summary = "mmai operand tile type">;

def CudaTile_MmaIOp : CudaTileIntegerOpDef<"mmai", "13.1",
    [Pure, AllTypesMatch<["acc", "result"]>,
     AllElementTypeMatch<"all of {lhs, rhs} have the same element type", ["lhs", "rhs"]>,
     AllRanksMatch<["lhs", "rhs", "acc"]>]> {
  let summary = "Integer matrix-multiply-accumulate";

  let description = [{
    The :code:`mmai` operation implements an MMA (matrix-multiply-accumulate) operation for integer tiles.
    It performs matrix multiplication on the integer tiles :code:`lhs` and :code:`rhs`, then adds the tile :code:`acc` to the result.
    :code:`lhs`, :code:`rhs`, and :code:`acc` must be 2D tiles or 3D tiles. The latter case indicates a batched matrix multiplication.

    .. math::
      \text{mmai}(A, B, C)_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj} + C_{ij}

    Input tiles :code:`lhs` and :code:`rhs` must be of integer type :code:`i8`. The signedness of
    :code:`lhs` and :code:`rhs` are specified separately by the :code:`signedness_lhs` and
    :code:`signedness_rhs` attributes, respectively. The accumulator tile :code:`acc` must be
    of type :code:`i32` and is always interpreted as signed. The output tile :code:`result`
    is of type :code:`i32` and is always interpreted as signed.

    Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
    MMA expects the operands :code:`lhs`, :code:`rhs`, and :code:`acc` to have shapes :code:`M x K`,
    :code:`K x N`, and :code:`M x N` (respectively). Batched (3D) MMA expects the operands
    to have shapes :code:`B x M x K`, :code:`B x K x N`, and :code:`B x M x N` (respectively).
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %lhs0 = cuda_tile.constant <i8: 0> : tile<4x8xi8>
          %rhs0 = cuda_tile.constant <i8: 0> : tile<8x2xi8>
          %acc0 = cuda_tile.constant <i32: 0> : tile<4x2xi32>

          %0 = mmai %lhs0, %rhs0, %acc0 signed signed
              : tile<4x8xi8>, tile<8x2xi8>,
                tile<4x2xi32>

          %lhs1 = cuda_tile.constant <i8: 0> : tile<2x4x8xi8>
          %rhs1 = cuda_tile.constant <i8: 0> : tile<2x8x2xi8>
          %acc1 = cuda_tile.constant <i32: 0> : tile<2x4x2xi32>

          %1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned
              : tile<2x4x8xi8>, tile<2x8x2xi8>,
                tile<2x4x2xi32>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<MmaIOp_OperandTileType, "The left hand side matrix operand.", "13.1">:$lhs,
                   CudaTileArg<MmaIOp_OperandTileType, "The right hand side matrix operand.", "13.1">:$rhs,
                   CudaTileArg<CudaTile_TileOf<[CudaTile_Int32], summary = "mmai acc tile type">, "The accumulator matrix operand.", "13.1">:$acc,
                   CudaTileArg<CudaTile_SignednessAttr, "The signedness of the :code:`lhs` operand.", "13.1">:$signedness_lhs,
                   CudaTileArg<CudaTile_SignednessAttr, "The signedness of the :code:`rhs` operand.", "13.1">:$signedness_rhs);
  let results = (outs CudaTileArg<CudaTile_TileOf<[CudaTile_Int32], summary = "mmai result tile type">, "The result matrix after multiplication and accumulation.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc custom<Signedness>($signedness_lhs) custom<Signedness>($signedness_rhs) attr-dict `:`
    custom<CudaTileType>(type($lhs)) `,` custom<CudaTileType>(type($rhs)) `,` custom<CudaTileType>(type($acc))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

def CudaTile_ExtractOp : CudaTileCoreOpDef<"extract", "13.1", [
    Pure, AllRanksMatch<["source", "result"]>
  ]> {
  let summary = "Extract a subtile from a tile";
  let description = [{
    The :code:`extract` operation extracts a subtile from the given source tile.

    The shape of the result tile must divide the shape of the source tile
    evenly e.g., :code:`tile<4xf32>` is a valid extraction from :code:`tile<8xf32>`, but
    :code:`tile<3xf32>` is not.

    The :code:`$indices` indicate the number of the slice to extract, but *importantly* not the offsets
    used to construct the subtile for extraction. The semantics of extract means that only
    full size slices can be extracted.

    Slices of a source tile with the same shape are non-overlapping by definition for
    unique indices.

    The :code:`indices` operands are interpreted as unsigned integers.

    .. warning::

      If the :code:`indices` specify a non-existent (i.e., out-of-bounds) slice, the
      behavior of the operation is undefined.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          // Extract a subtile from %t at dim_0 = [4;8) and dim_1 = [4;6).
          %c1 = constant <i32: 1> : tile<i32>
          %c2 = constant <i32: 2> : tile<i32>
          %t = constant <f32: 0.0> : tile<32x8xf32>
          // Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
          %0 = extract %t[%c1, %c2]
              : tile<32x8xf32> -> tile<4x2xf32>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TileType, "The source tile to extract from.", "13.1">:$source,
                       CudaTileArg<Variadic<CudaTile_ScalarTileOf<CudaTile_Int32>>, "The indices of the slice to extract.", "13.1">:$indices);
  let results = (outs CudaTileArg<CudaTile_TileType, "The extracted subtile.", "13.1">:$result);
  let assemblyFormat = [{
    $source `[` $indices `]` attr-dict
    `:` custom<CudaTileType>(type($source)) `->` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

def CudaTile_InsertOp : CudaTileCoreOpDef<"insert", "13.4",
    [Pure, AllTypesMatch<["destination", "result"]>]> {
  let summary = "Insert a subtile into a tile";
  let description = [{
    The :code:`insert` operation inserts a :code:`$source` subtile into a
    :code:`$destination` tile and returns the updated tile as a result.

    The shape of the source tile must divide the shape of the result/destination
    tile evenly. E.g., :code:`tile<4xf32>` is a valid source tile for a
    :code:`tile<8xf32>` destination, but :code:`tile<3xf32>` is not.

    The :code:`$indices` indicate the index of the subtile to insert into, but
    *importantly* not the offsets. (Same as :code:`cuda_tile.extract`.)

    The subtiles (with the same shape as the source tile) within the destination
    tile are non-overlapping.

    The :code:`indices` operands are interpreted as unsigned integers.

    .. warning::

      If the :code:`indices` specify a non-existent (i.e., out-of-bounds)
      subtile, the behavior of the operation is undefined.

  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          // Insert a subtile at dim_0 = [4;8) and dim_1 = [4;6).
          %c1 = constant <i32: 1> : tile<i32>
          %c2 = constant <i32: 2> : tile<i32>
          %src = constant <f32: 1.0> : tile<4x2xf32>
          %dst = constant <f32: 0.0> : tile<32x8xf32>
          // Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
          %0 = insert %src, %dst[%c1, %c2] : tile<4x2xf32>, tile<32x8xf32>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TileType, "The source tile.", "13.4">:$source,
                       CudaTileArg<CudaTile_TileType, "The destination tile.", "13.4">:$destination,
                       CudaTileArg<Variadic<CudaTile_ScalarTileOf<CudaTile_Int32>>, "The indices of the slice to insert into.", "13.4">:$indices);
  let results = (outs CudaTileArg<CudaTile_TileType, "The updated destination subtile.", "13.4">:$result);
  let assemblyFormat = [{
    $source `,` $destination `[` $indices `]` attr-dict
    `:` custom<CudaTileType>(type($source)) `,` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

def CudaTile_ExpOp : CudaTileFloatingPointOpDef<"exp", "13.1", [
    Pure, AllTypesMatch<["source", "result"]>
  ]> {
  let summary = "Element-wise exponential";
  let description = !strconcat([{
    The :code:`exp` operation computes the element-wise exponential of the input
    floating-point tile. Default rounding mode is `full`.

    The :code:`full` rounding mode relies on the CUDA Math API. For more details please refer to 
    the CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#cuda-c-mathematical-standard-library-functions.
    Exponential functions provided by the CUDA Math API are supported for the float and double data types only,
    support for other data types is not guaranteed.

    The :code:`approx` rounding mode enables fast hardware-level approximation
    using exp2 with log2(e) scaling.

    .. math::

      \text{exp}(x)_i = e^{x_i}

  }], floating_point_math_suffix);

  let descriptionTables = [
    Table<":code:`exp` Modifiers", "The below table shows the supported modifiers for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["approx", "yes", "no", "no", "no"]>,
       TableRow<["full", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input float tile.", "13.1">:$source,
                       CudaTileArg<DefaultValuedAttr<CudaTile_RoundingModeAttr, "RoundingMode::FULL">, rounding_mode_desc, "13.3",
                                   [OnlyCudaTileApproxFullRoundingModes]>:$rounding_mode);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The exponential of the input tile.", "13.1">:$result);

  let assemblyFormat = [{
    $source
    custom<ExpOpRoundingMode>($rounding_mode)
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_exp() {
        %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
        %res = exp %in : tile<4xf32>
        %res_approx = exp %in rounding<approx> : tile<4xf32>
      # }
    # }
  }]];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Exp2Op
//===----------------------------------------------------------------------===//

def CudaTile_Exp2Op : CudaTileFloatingPointOpDef<"exp2", "13.1", [
    Pure, AllTypesMatch<["source", "result"]>
  ]> {
  let summary = "Element-wise power of two";
  let description = !strconcat([{
    The :code:`exp2` operation computes the element-wise power of two of the input
    floating-point tile.

    .. math::

      \text{exp2}(x)_i = 2^{x_i}
  }], floating_point_math_suffix);

  let descriptionTables = [
    Table<":code:`exp2` Modifiers", "The below table shows the supported modifiers for each data type.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["flush_to_zero", "yes", "no", "no", "no"]>]
    >
  ];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input floating-point tile.", "13.1">:$source,
                       CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of raising 2 to the power of the input tile.", "13.1">:$result);

  let assemblyFormat = [{
    $source
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_exp2() {
        %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
        %res = exp2 %in : tile<4xf32>
      # }
    # }
  }]];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtIOp
//===----------------------------------------------------------------------===//

// IntTileType + Int4
def ExtIOp_IntegerType : CudaTile_TileOf<
  !listconcat(CudaTile_IntTileType.allowedElementTypes, [CudaTile_Int4])>;

def CudaTile_ExtIOp : CudaTileConversionOpDef<"exti", "13.1", [
    Pure, AllShapesMatch<["from", "to"]>]> {
  let summary = "Extend the width of an integer tile";

  let description = [{
    The :code:`exti` operation converts a tile of integers of a given width to a
    strictly larger width. Zero-extension is used
    for :code:`unsigned` integers and sign-extension is used for :code:`signed`
    integers.
  }];

  let arguments = (ins CudaTileArg<ExtIOp_IntegerType, "The input integer tile to extend.", "13.1">:$from,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness);
  let results = (outs CudaTileArg<ExtIOp_IntegerType, "The extended integer tile.", "13.1">:$to);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $from custom<Signedness>($signedness) attr-dict
    `:` custom<CudaTileType>(type($from)) `->` custom<CudaTileType>(type($to))
  }];

  let builders = [
    OpBuilder<(ins "Type":$resTy,
                   "ValueRange":$operands, "mlir::cuda_tile::Signedness":$signedness), [{
      assert(operands.size() == 1 && "expected a single operand");
      return build($_builder, $_state, resTy, operands[0], signedness);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

def CudaTile_ForOp : CudaTileControlFlowOpDef<"for", "13.1", [
    AutomaticAllocationScope,
    AllTypesMatch<["lowerBound", "upperBound", "step"]>,
    AllTypesMatch<["initValues", "resultValues"]>,
    OpAsmOpInterface,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ContinueOp">,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames", "getAsmBlockArgumentNames"]>
  ]> {
  let summary = "For loop over integer range";

  let description = [{
    The :code:`for` operation is a structured range-based sequential loop.

    The loop operation consists of (1) a range formed by :code:`lowerBound`, :code:`upperBound`, and :code:`step`,
    (2) a set of loop-carried values which are initialized by :code:`initValues` and updated by each iteration of the loop, and
    (3) a region which represents the loop body.

    The iteration space is defined by the interval :math:`[lowerBound, upperBound)` with each value
    separated by :code:`step`.

    .. math::

      range(L_b, U_b, S) = \{ L_b + i \cdot S \mid i \in \mathbb{Z}, L_b + i \cdot S < U_b \}

    :code:`lowerBound`, :code:`upperBound`, and :code:`step` must be of the same type.
    :code:`lowerBound` and :code:`upperBound` specify a half-open (or exclusive) range: the range
    includes the :code:`lowerBound` but does not include the :code:`upperBound`.
    :code:`step` must be positive but the bounds may be negative or zero.

    The :code:`lowerBound`, :code:`upperBound`, and :code:`step` operands are interpreted as signed integers.

    The first iteration of the loop receives the induction variable initialized to the value of :code:`lowerBound`
    and the loop-carried values initialized to the values of :code:`initValues`.

    The loop body is executed for each value in the range, receiving an integer induction variable
    incremented by :code:`step` on each iteration and the loop-carried values which correspond to the
    loop-carried values yielded by the previous loop iteration.

    The loop terminates when the induction variable is greater than or equal to
    :code:`upperBound`. By default, signed comparison is used between the
    upperBound and the induction variable. To use unsigned comparison instead,
    specify the optional :code:`unsigned` unit attribute.

    The body of the loop must be terminated by a :ref:`op-cuda_tile.continue` that yields
    the next iteration's value for each loop carried variable.

    The for operation produces one return value for each loop carried variable. The type of the :math:`i`-th return
    value is that of the :math:`i`-th loop carried variable and its value is the final value of the
    :math:`i`-th loop carried variable.

    .. warning::

      - Loop carried variables can not be a :tileirty:`tensor_view` or view type.
      - :code:`for` operations cannot terminate early and must end in a :ref:`op-cuda_tile.continue`.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %lowerBound = constant <i32: 0> : tile<i32>
          %upperBound = constant <i32: 10> : tile<i32>
          %step = constant <i32: 1> : tile<i32>

          // A simple loop iterating over an i32 range.
          for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
              continue
          }

          %initVal0 = constant <f32: 0.0> : tile<f32>
          // A similar loop to the above, but with a loop carried value, val0.
          %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                              iter_values(%val00 = %initVal0) -> (tile<f32>) {
            %loopVal0 = constant <f32: 1.0> : tile<f32>
            continue %loopVal0 : tile<f32>
          }
    #   }
    # }
  }]];

  let arguments = (ins
    CudaTileArg<CudaTile_ScalarTileOf<CudaTile_AnyInt>, "The lower bound of the loop.", "13.1">:$lowerBound,
    CudaTileArg<CudaTile_ScalarTileOf<CudaTile_AnyInt>, "The upper bound of the loop.", "13.1">:$upperBound,
    CudaTileArg<CudaTile_ScalarTileOf<CudaTile_AnyInt>, "The step of the loop.", "13.1">:$step,
    CudaTileArg<Variadic<AnyType>, "The initial values of the loop-carried values.", "13.1">:$initValues,
    CudaTileArg<UnitAttr, "If present, use unsigned integer comparison for loop termination.", "13.2">:$unsignedCmp
  );
  let results = (outs CudaTileArg<Variadic<AnyType>, "The values of the loop-carried variables after loop termination.", "13.1">:$resultValues);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound, "Value":$step,
      CArg<"ValueRange", "{}">:$initArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">,
      CArg<"bool", "false">:$unsignedCmp)>
  ];

  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl # [{
    Value getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterValues() {
      return getBody()->getArguments().drop_front(getNumInductionVars());
    }

    /// Return the `index`-th region iteration argument.
    BlockArgument getRegionIterVar(unsigned index) {
      assert(index < getNumRegionIterVars() &&
        "expected an index less than the number of region iter vars");
      return getBody()->getArguments().drop_front(getNumInductionVars())[index];
    }

    /// Returns the number of induction variables, always 1 for ForOp.
    unsigned getNumInductionVars() { return 1; }
    /// Returns the number of region arguments for loop-carried values.
    unsigned getNumRegionIterVars() {
      return getBody()->getNumArguments() - getNumInductionVars();
    }

    /// Return the total number of region arguments (iteration variable + loop-carried values)
    unsigned getNumRegionArgs() { return getBody()->getNumArguments(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FloorOp
//===----------------------------------------------------------------------===//

def CudaTile_FloorOp : CudaTileFloatingPointOpDef<"floor", "13.1", [
    Pure, AllTypesMatch<["source", "result"]>
  ]> {
  let summary = "Element-wise floor rounding";
  let description = !strconcat([{
    The :code:`floor` operation computes the element-wise floor on the input floating-point tile
    rounding each element down to the largest integer that is less than or equal to the element.

    .. math::
      \text{floor}_i(x_i) = \max\{n \in \mathbb{Z} \mid n \leq x_i\}
  }], floating_point_arith_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %source = constant <f32: 1.5> : tile<f32>
          %result = floor %source : tile<f32>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input tile to the floor operation.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the floor operation.", "13.1">:$result);

  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

def CudaTile_FmaTile : CudaTile_TileOf<[CudaTile_Float16,
                                        CudaTile_BFloat16,
                                        CudaTile_Float32,
                                        CudaTile_Float64]>;

def CudaTile_FmaOp : CudaTileFloatingPointOpDef<"fma", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "acc", "result"]>]> {
  let summary = "Floating point fused multipy-add";
  let description = [{
    Takes three operands :code:`lhs`, :code:`rhs` and :code:`acc`, returns :code:`result = lhs * rhs + acc`.

    .. math::
      \text{fma}(x, y, z)_i = x_i \times y_i + z_i
  }];

  let descriptionTables = [
    Table<":code:`fma` Modifier", "The below table shows the supported modifiers and rounding modes for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["flush_to_zero", "yes", "no", "no", "no"]>,
       TableRow<["rounding<nearest_even>", "yes", "yes", "yes", "yes"]>,
       TableRow<["rounding<zero>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<negative_inf>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<positive_inf>", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments = (ins
      CudaTileArg<CudaTile_FmaTile, "The left hand side operand.", "13.1">:$lhs,
      CudaTileArg<CudaTile_FmaTile, "The right hand side operand.", "13.1">:$rhs,
      CudaTileArg<CudaTile_FmaTile, "The accumulator operand.", "13.1">:$acc,
      CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                  [OnlyCudaTileIEEERoundingModes]>:$rounding_mode,
      CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);

  let results = (outs CudaTileArg<CudaTile_FmaTile, "The fused multiply-add of the input tiles.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc
    custom<IEEERoundingMode>($rounding_mode)
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FToFOp
//===----------------------------------------------------------------------===//

def CudaTile_FToFOp : CudaTileConversionOpDef<"ftof", "13.1", [
    Pure, AllShapesMatch<["from", "to"]>]> {
  let summary = "Convert between floating-point types";
  let description = [{
    The :code:`ftof` operation converts a tile of a given floating-point element type into one
    of a different floating-point element type (for example, from :code:`f32` to :code:`f64`).

    The source type and the result type must be different.

    The :code:`rounding_mode` attribute specifies the rounding behavior for the operation.
    The supported rounding modes depend on the source and destination types. Rules
    are evaluated top-to-bottom; the first matching row applies:

    .. csv-table::
       :header: "Conversion", ":code:`nearest_even`", ":code:`zero`", ":code:`negative_inf`", ":code:`positive_inf`", ":code:`nearest_away`"
       :widths: auto

       "To :code:`f8E8M0FNU`",                                          "",  "✓", "",  "✓", ""
       "To :code:`f8E4M3FN`, :code:`f8E5M2`, or :code:`f4E2M1FN`",       "✓", "",  "",  "",  ""
       "Strict widening (source exactly representable in destination)", "✓", "✓", "✓", "✓", "✓"
       ":code:`f64` → :code:`f32`",                                     "✓", "✓", "✓", "✓", ""
       ":code:`f32` → :code:`tf32`",                                    "✓", "✓", "",  "",  "✓"
       "All other narrowing conversions",                               "✓", "✓", "",  "",  ""

    A conversion is a **strict widening** when every value representable in
    the source type is exactly representable in the destination type, so the
    conversion is lossless regardless of rounding mode. For example,
    :code:`f16` → :code:`f32` is a strict widening: every :code:`f16` value
    has an exact :code:`f32` representation. By contrast,
    :code:`f16` → :code:`bf16` is **not** a widening because :code:`bf16`
    has fewer mantissa bits than :code:`f16` (7 vs. 10), so some
    :code:`f16` values cannot be represented exactly. Conversely,
    :code:`bf16` → :code:`f16` is **not** a widening because :code:`f16`
    has a smaller exponent range than :code:`bf16` (5-bit vs. 8-bit
    exponent), so some :code:`bf16` values overflow the :code:`f16`
    range. Such conversions appear in the "All other narrowing
    conversions" row.

    .. warning::

      Different floating-point types have different conversion behaviors for out-of-finite-range
      values and special values. See :ref:`table-float-conversion-semantics` for more details.
  }];

  let arguments = (ins
    CudaTileArg<CudaTile_FloatTileType, "The input floating-point tile.", "13.1">:$from,
    CudaTileArg<DefaultValuedAttr<CudaTile_RoundingModeAttr, "::mlir::cuda_tile::RoundingMode::NEAREST_EVEN">, rounding_mode_desc, "13.1",
      [OnlyCudaTileFloatConversionRoundingModes]>:$rounding_mode);
  let results = (outs
    CudaTileArg<CudaTile_FloatTileType, "The result floating-point tile.", "13.1">:$to);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $from custom<IEEERoundingMode>($rounding_mode)
    attr-dict `:` custom<CudaTileType>(type($from))
    `->` custom<CudaTileType>(type($to))
  }];
}

//===----------------------------------------------------------------------===//
// FToIOp
//===----------------------------------------------------------------------===//

def CudaTile_FToIOp : CudaTileConversionOpDef<"ftoi", "13.1", [
    Pure, AllShapesMatch<["from", "to"]>]> {
  let summary = "Convert a tile from floating-point values to integer values";
  let description = [{
    The :code:`ftoi` operation converts a floating-point tile into an integer tile.

    In contrast to a :ref:`op-cuda_tile.bitcast` which is bits preserving, this preserves the numerical
    value of the tile, rounded towards zero to the nearest integer of the provided type.

    .. warning::

      If the input floating-point value is :code:`Inf` or :code:`NaN`, or,
      after rounding, is outside the representable range of the target integer
      type, using the returned value is undefined behavior. Users must ensure
      that input values are finite and within the representable range of the
      target type.

    If the :code:`saturating` modifier is present, out-of-range values are
    clamped to the closest representable value of the target type instead:
    values exceeding the maximum are clamped to the maximum, values below the
    minimum are clamped to the minimum, and :code:`NaN` is converted to 0.
  }];

  let arguments = (ins CudaTileArg<CudaTile_FloatTileType, "The input floating-point tile.", "13.1">:$from,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness,
                       CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                         [OnlyCudaTileIntegerRoundingModes]>:$rounding_mode,
                       CudaTileArg<UnitAttr, "If present, use saturating conversion (clamp out-of-range values).", "13.4">:$saturating);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The result integer tile.", "13.1">:$to);

  let assemblyFormat = [{
    $from custom<Signedness>($signedness)
     custom<IntegerRoundingMode>($rounding_mode)
     (`saturating` $saturating^)?
     attr-dict
    `:` custom<CudaTileType>(type($from)) `->` custom<CudaTileType>(type($to))
  }];
  let builders = [
    OpBuilder<(ins "Type":$resTy,
                   "ValueRange":$operands, "mlir::cuda_tile::Signedness":$signedness), [{
      assert(operands.size() == 1 && "expected a single operand");
      return build($_builder, $_state, resTy, operands[0], signedness, RoundingMode::NEAREST_INT_TO_ZERO, /*saturating=*/false);
    }]>,
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// EntryOp
//===----------------------------------------------------------------------===//

def CudaTile_EntryOp : CudaTileCoreOpDef<"entry", "13.1", [
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface, SingleBlock,
  SingleBlockImplicitTerminator<"ReturnOp">, SymbolName, SymbolVisibility
]> {
  let summary = "Define a tile kernel";
  let description = [{
    The :code:`entry` operation defines a tile kernel; a kernel is a function that can
    serve as the program entry point. It has a unique name per-module. A kernel can
    not return any value, and all its arguments must be scalar (rank-0) tile types.
    It must be launched from the host side using :code:`cuLaunchKernel` or similar
    CUDA runtime API functions.

    Tile kernels require that the user specifies the 3-d grid dimensions at launch which
    defines the number of tile blocks (or kernel instances) that will execute the kernel
    in parallel.

    For detailed semantics of tile kernels see :ref:`sub_sec_tile_kernel`.
  }];

  let arguments = (ins CudaTileArg<SymbolNameAttr, "The name of the function.", "13.1">:$sym_name,
                       CudaTileArg<TypeAttrOf<FunctionType>, "The type of the function.", "13.1">:$function_type,
                       CudaTileArg<OptionalAttr<StrAttr>, "The visibility of the function.", "13.1">:$sym_visibility,
                       CudaTileArg<OptionalAttr<DictArrayAttr>, "The argument attributes of the function. CUDA Tile IR currently defines none for entry functions.", "13.5">:$arg_attrs,
                       CudaTileArg<OptionalAttr<DictArrayAttr>, "The result attributes of the function: none of these are supported by CUDA Tile IR at the moment.", "13.1">:$res_attrs,
                       CudaTileArg<OptionalAttr<CudaTile_OptimizationHintsAttr>, "Compiler architecture-specific optimization hints", "13.1">:$optimization_hints);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl # [{
    // FunctionOpInterface Methods

    /// Returns the region on the current operation
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    static void build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      ::mlir::StringAttr sym_name,
                      ::mlir::TypeAttr function_type,
                      ::mlir::ArrayAttr arg_attrs,
                      ::mlir::ArrayAttr res_attrs) {
        build(odsBuilder, odsState, sym_name, function_type,
              /*sym_visibility=*/{}, arg_attrs, res_attrs,
              OptimizationHintsAttr::get(odsBuilder.getContext(),
                  DictionaryAttr::get(odsBuilder.getContext())));
    }
    static void build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      ::llvm::StringRef sym_name,
                      ::mlir::FunctionType function_type,
                      ::mlir::ArrayAttr arg_attrs,
                      ::mlir::ArrayAttr res_attrs) {
        build(odsBuilder, odsState, sym_name, function_type,
              /*sym_visibility=*/{}, arg_attrs, res_attrs,
              OptimizationHintsAttr::get(odsBuilder.getContext(),
                  DictionaryAttr::get(odsBuilder.getContext())));
    }

  }];
}

//===----------------------------------------------------------------------===//
// GetTileBlockIdOp
//===----------------------------------------------------------------------===//

def CudaTile_GetTileBlockIdOp : CudaTileCoreOpDef<"get_tile_block_id", "13.1", [Pure]> {
    let summary = "Get the currently executing tile block coordinates";

    let description = [{
      :code:`get_tile_block_id` returns a 3-d tile block coordinates (or ID) of the currently
      executing tile block.

      A tile ID has three dimensions: :code:`x`, :code:`y`, and :code:`z`. This operation returns all
      three of them simultaneously. The value of each dimension returned by this
      operation is between :code:`0` (including) and the value returned by :code:`get_num_tile_blocks`
      for the respective axis (excluding), represented by the inclusive interval
      :code:`[0, get_num_tile_blocks(dim) - 1]` . Grid dimensions unspecified at kernel
      launch (i.e., a 1-d or 2-d grid) will always be :code:`0` for all tile blocks.

      .. note::
        **Grid Dimension Limitation**: Grid dimensions are limited to 2^24-1 (16,777,215)
        per axis. Larger dimensions may result in incorrect tile block ID calculations. Use multiple
        kernel launches for larger workloads.
    }];

    let results = (outs CudaTileArg<CudaTile_ScalarTileOf<CudaTile_Int32>, "The tile block ID for dimension :code:`x`.", "13.1">:$blockId_x,
                        CudaTileArg<CudaTile_ScalarTileOf<CudaTile_Int32>, "The tile block ID for dimension :code:`y`.", "13.1">:$blockId_y,
                        CudaTileArg<CudaTile_ScalarTileOf<CudaTile_Int32>, "The tile block ID for dimension :code:`z`.", "13.1">:$blockId_z);
    let assemblyFormat = "attr-dict `:` custom<CudaTileType>(type($blockId_x))";
}

//===----------------------------------------------------------------------===//
// GetNumTileBlocksOp
//===----------------------------------------------------------------------===//

def CudaTile_GetNumTileBlocksOp : CudaTileCoreOpDef<"get_num_tile_blocks", "13.1", [Pure]> {
    let summary = "Get total number of tile blocks";

    let description = [{
      The :code:`get_num_tile_blocks` operation queries the total number of tile blocks
      in the form of a 3-tuple specifying the extent of each grid dimension.

      A tile :code:`id` is a coordinate in 3-space and therefore the must also be a 3-tuple containing
      the extent of each dimension: :code:`x`, :code:`y` and :code:`z`.

      When launching 1- or 2-dimensional grids, the unspecified dimensions will have a cardinality of 1.

      For example if the grid used to launch the kernel is :code:`(1024, 1024)` then the
      result of this operation will be :code:`(1024, 1024, 1)`.

      .. note::
        **Grid Dimension Limitation**: Grid dimensions are limited to 2^24-1 (16,777,215)
        per axis. Larger dimensions may result in incorrect tile block ID calculations. Use multiple
        kernel launches for larger workloads.
    }];

    let results = (outs CudaTileArg<CudaTile_ScalarTileOf<CudaTile_Int32>, "The number of tile blocks in dimension :code:`x`.", "13.1">:$gridSize_x,
                        CudaTileArg<CudaTile_ScalarTileOf<CudaTile_Int32>, "The number of tile blocks in dimension :code:`y`.", "13.1">:$gridSize_y,
                        CudaTileArg<CudaTile_ScalarTileOf<CudaTile_Int32>, "The number of tile blocks in dimension :code:`z`.", "13.1">:$gridSize_z);
    let assemblyFormat = "attr-dict `:` custom<CudaTileType>(type($gridSize_x))";

    let mlirExamples = [[{
      # cuda_tile.module @module {
        entry @example() {
          %x, %y, %z = get_num_tile_blocks : tile<i32>
        }
      # }
    }]];
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

def CudaTile_GetGlobalOp  : CudaTileCoreOpDef<"get_global", "13.1", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get a pointer to a global variable";

  let description = [{
    The :code:`get_global` operation returns a pointer to the specified :code:`global`
    variable. A global variable is a form of static global memory allocation that can
    be declared using the :ref:`op-cuda_tile.global` operation.

    The element type of the returned pointer will be of the same type as the
    element type of the declared global variable.

    For detailed semantics of global variables see :ref:`sub_sec_tile_global`.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
        global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

        entry @example() {
          %ptr = get_global @val : tile<ptr<f32>>
          return
        }
    # }
  }]];

  let arguments = (ins CudaTileArg<FlatSymbolRefAttr, "The name of the global variable.", "13.1">:$name);
  let results = (outs CudaTileArg<CudaTile_ScalarTileOf<CudaTile_PointerType>, "The result of the get_global operation.", "13.1">:$result);
  let assemblyFormat = "$name attr-dict `:` custom<CudaTileType>(type($result))";
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

def CudaTile_GlobalOp : CudaTileCoreOpDef<"global", "13.1", [Symbol, SymbolName]> {
  let summary = "Allocate static global memory";

  let description = [{
    The :code:`global` operation statically allocates a mutable 1-dimensional location in global
    memory and initializes it using :code:`value`. The initialization of the allocation is performed
    at `CUDA module <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>`_
    load time. The lifetime of the allocation is the same as the lifetime of the module.

    The allocation may be read or written to by first using :ref:`op-cuda_tile.get_global` to obtain a pointer to the
    the memory and then read using :ref:`op-cuda_tile.load_ptr_tko` or written to using :ref:`op-cuda_tile.store_ptr_tko`.

    The initial values are stored in memory in linear order, so the pointer returned by :ref:`op-cuda_tile.get_global`
    points to the first element, and offsetting the pointer by `x` would allow to load element at position `x`.

    :code:`global` operations must be directly nested within the |cuda_tile| module. They cannot be defined inside functions.
    As globals are defined at the module scope their names are globally unique symbols and must not collide with any other
    symbol in the module.

    The :code:`symbol_visibility` attribute controls the visibility and accessibility of the global variable.
    When not explicitly specified, it defaults to :code:`public`:
    - :code:`public` (default): Accessible from host code, never eliminated
    - :code:`private`: Device-only access, enables aggressive optimization

    The optional :code:`constant` attribute marks the global as immutable during the execution of a kernel, allowing the compiler
    to place it in read-only memory and enable additional optimizations. Defaults to false (mutable).

    For more detailed semantics of global variables see :ref:`sub_sec_tile_global`.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
        // Default public visibility (implicit)
        global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
        // Explicit public visibility (same as default)
        global public @host_accessible <f32: [1.0, 2.0]> : tile<2xf32>
        // Device-only optimization
        global private @device_only <i32: [42]> : tile<i32>
        // Read-only/constant global allocation
        global constant @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
        // Device-only, read-only/constant global allocation
        global private constant @device_only <i32: [42]> : tile<i32>        
        entry @example() {}
    # }
  }]];

  let arguments = (ins CudaTileArg<SymbolNameAttr, "The name of the global variable.", "13.1">:$sym_name,
                       CudaTileArg<Builtin_DenseTypedElementsAttr, "The value to initialize the allocation with.", "13.1">:$value,
                       CudaTileArg<DefaultValuedAttr<I64Attr, "0">, "The alignment of the buffer.", "13.1">:$alignment,
                       CudaTileArg<UnitAttr, "Whether the global is constant / read-only.", "13.3">:$constant,
                       CudaTileArg<DefaultValuedAttr<CudaTile_SymbolVisibilityAttr, "::mlir::cuda_tile::SymbolVisibility::Public">, "The symbol visibility of the global.", "13.3">:$symbol_visibility);

  let assemblyFormat = "custom<SymbolVisibility>($symbol_visibility) (`constant` $constant^)? $sym_name (`alignment` `=` $alignment^)? attr-dict custom<DenseTypedElementsAttrNoResult>($value)";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Discardability follows `symbol_visibility`, not the builtin
    /// `sym_visibility` attribute this op does not model.
    bool canDiscardOnUseEmpty() {
      return getSymbolVisibility() == SymbolVisibility::Private;
    }

    SymbolTable::Visibility getVisibility() {
      return getSymbolVisibility() == SymbolVisibility::Private
                 ? SymbolTable::Visibility::Private
                 : SymbolTable::Visibility::Public;
    }

    void setVisibility(SymbolTable::Visibility visibility) {
      assert(visibility != SymbolTable::Visibility::Nested &&
             "global symbols do not support nested visibility");
      setSymbolVisibility(visibility == SymbolTable::Visibility::Private
                              ? SymbolVisibility::Private
                              : SymbolVisibility::Public);
    }
  }];
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def CudaTile_IfOp : CudaTileControlFlowOpDef<"if", "13.1", [
    NoRegionArguments, OpAsmOpInterface,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"impl::IfOpImplicitTerminatorType">]> {
  let summary = "Conditional execution";
  let description = [{
    The :code:`if` operation represents an if-then-else construct.

    The `if` operation consists of (1) a control operand which is a :code:`tile<i1>` value, (2) a true branch :code:`thenRegion`
    and (3) an optional false branch :code:`elseRegion`.

    The :code:`if` operation may produce results by yielding values in each branch using :ref:`op-cuda_tile.yield`.

    If the :code:`if` operation has results, both branches must yield values whose types match
    the result types. If the :code:`if` operation has no results, both branches must yield
    nothing. When results are produced the :code:`else` branch is required.

    The values returned will be dependent on which branch is taken.

    .. warning::

      The :code:`if` operation has a set of additional restrictions today:

      - Results of :code:`if` must not be a :tileirty:`tensor_view` or view type.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %condition = constant <i1: 1> : tile<i1>

          // A simple if operation that conditionally executes a region.
          if %condition  {
            // ...
          }

          // An if operation with an "else" branch.
          if %condition  {
            // ...
          } else {
            // ...
          }

          // An if operation that produces results (mixed types f32, i32).
          // Because results are produced, the else branch is required and
          // both branches must yield values whose types match the result types.
          %x, %y = if %condition -> (tile<f32>, tile<i32>) {
            %x_then = constant <f32: 1.0> : tile<f32>
            %y_then = constant <i32: 2> : tile<i32>
            yield %x_then, %y_then : tile<f32>, tile<i32>
          } else {
            %x_then = constant <f32: 1.0> : tile<f32>
            %y_then = constant <i32: 42> : tile<i32>
            yield %x_then, %y_then : tile<f32>, tile<i32>
          }
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_ScalarTileOf<CudaTile_Int1>, "The condition of the if operation.", "13.1">:$condition);
  let results = (outs CudaTileArg<Variadic<AnyType>, "The results of the if operation.", "13.1">:$results);

  let regions = (region
    SizedRegion<1>:$thenRegion, MaxSizedRegion<1>:$elseRegion
  );

  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl # [{
    /// Return the single block of the `thenRegion`.
    Block *getThenBlock();
    Operation *getThenTerminator();

    /// Return the single block of the `elseRegion`.
    Block *getElseBlock();
    Operation *getElseTerminator();
  }];

  let assemblyFormat = [{
    $condition (`->` `(` custom<CudaTileType>(type($results))^ `)`)?
    custom<IfOpRegion>($thenRegion)
    (`else` custom<IfOpRegion>($elseRegion)^)? attr-dict
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IntToPtrOp
//===----------------------------------------------------------------------===//

def CudaTile_IntToPtrOp : CudaTileConversionOpDef<"int_to_ptr", "13.1", [
    Pure, AllShapesMatch<["source", "result"]>]> {

  let summary = "Convert a tile of integers to a tile of pointers";

  let description = [{
    The :code:`int_to_ptr` operation converts a tile of integers to a tile of pointers.

    The :code:`source` operand is interpreted as an unsigned integer.

    The inverse of this operation is :ref:`op-cuda_tile.ptr_to_int`.
  }];

  let arguments = (ins
    CudaTileArg<CudaTile_IntTileInt64Type, "The input tile of integers.", "13.1">:$source
  );
  let results = (outs
    CudaTileArg<CudaTile_PointerTileType, "The output tile of pointers.", "13.1">:$result
  );
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($source)) `->` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

def CudaTile_IotaOp : CudaTileCoreOpDef<"iota", "13.1", [Pure]> {
  let summary = "Generate a 1-d tile range from 0 to n-1";
  let description = [{
    The :code:`iota` operation generates a 1-d tile with a sequence of integer
    values. The starting value is :code:`0` and the stride is :code:`1`. If the shape of
    the result tile is :code:`(n)`, then the generated values are :code:`[0, n - 1]`.

    .. math::
      \text{iota}(n)_i = i \quad \text{for } i \in [0, n-1]

    The result values should be interpreted as unsigned integers.

    .. note::

      The number of elements in the result tile must not exceed
      the maximum value that the element type can express.
  }];
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The result of the iota operation.", "13.1">:$result);
  let assemblyFormat = "attr-dict `:` custom<CudaTileType>(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// JoinTokensOp
//===----------------------------------------------------------------------===//

def CudaTile_JoinTokensOp
    : CudaTileMemOpDef<"join_tokens", "13.1", [Pure]> {
  let summary = "Product a new token which depends on the input tokens";
  let description = [{
    The :code:`join_tokens` operation produces a fresh token which depends on all input tokens
    (at least two are required). Token-ordered operations which consume the new token will then
    be ordered with respect to all joined tokens.
  }];

  let arguments = (ins CudaTileArg<Variadic<CudaTile_TokenType>, "The input tokens to join.", "13.1">:$tokens);
  let results = (outs CudaTileArg<CudaTile_TokenType, "The joined token.", "13.1">:$result);
  let assemblyFormat = [{
    $tokens attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

// IntTileType + Int4
def TruncIOp_IntegerType : CudaTile_TileOf<
  !listconcat(CudaTile_IntTileType.allowedElementTypes, [CudaTile_Int4])>;

def CudaTile_TruncIOp : CudaTileConversionOpDef<"trunci", "13.1", [
    Pure, AllShapesMatch<["from", "to"]>]> {
  let summary = "Truncates the width of an integer tile";
  let description = [{
    The :code:`trunci` operation converts a tile of integers of a given element type to
    one with a strictly smaller width.

    The optional `overflow` attribute specifies whether an overflow can occur
    when interpreting the operand as a signed and/or unsigned integer. In case
    of "no signed wrap", all truncated bits must have the same value as the
    most significant bit of the truncated result. In case of "no unsigned
    wrap", the truncated bits must be zero.
  }];

  let arguments = (ins CudaTileArg<TruncIOp_IntegerType, "The input integer tile to truncate.", "13.1">:$from,
                       CudaTileArg<DefaultValuedAttr<CudaTile_IntegerOverflowAttr, "::mlir::cuda_tile::IntegerOverflow::NONE">, overflow_desc, "13.1">:$overflow);
  let results = (outs CudaTileArg<TruncIOp_IntegerType, "The truncated integer tile.", "13.1">:$to);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $from (`overflow` `` $overflow^)? attr-dict
    `:` custom<CudaTileType>(type($from))
    `->` custom<CudaTileType>(type($to))
  }];
}

//===----------------------------------------------------------------------===//
// IToFOp
//===----------------------------------------------------------------------===//

def CudaTile_IToFOp : CudaTileConversionOpDef<"itof", "13.1",
    [Pure, AllShapesMatch<["from", "to"]>]> {
  let summary = "Convert integer to floating-point";
  let description = [{
    The :code:`itof` operation converts an integer tile into a float tile.
    In contrast to :ref:`op-cuda_tile.bitcast`, this preserves the numerical value of the tile,
    rounded to the nearest floating-point number of the provided type. Direct conversion to
    :code:`f8E8M0FNU` is not allowed; first convert to a wider floating-point type and then
    to :code:`f8E8M0FNU` via :ref:`op-cuda_tile.ftof`.

    .. warning::

      Different floating-point types have different conversion behaviors for out-of-finite-range
      values and special values. See :ref:`table-float-conversion-semantics` for more details.
  }];

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The input integer tile.", "13.1">:$from,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness,
                       CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                                   [OnlyCudaTileNearestEvenRoundingMode]>:$rounding_mode);
  let results = (outs CudaTileArg<CudaTile_FloatTileType, "The converted floating-point tile.", "13.1">:$to);
  let assemblyFormat = [{
    $from custom<Signedness>($signedness)
    custom<IEEERoundingMode>($rounding_mode)
    attr-dict
    `:` custom<CudaTileType>(type($from)) `->` custom<CudaTileType>(type($to))
  }];
  let builders = [
    OpBuilder<(ins "Type":$resTy,
                   "ValueRange":$operands, "mlir::cuda_tile::Signedness":$signedness), [{
      assert(operands.size() == 1 && "expected a single operand");
      return build($_builder, $_state, resTy, operands[0], signedness, RoundingMode::NEAREST_EVEN);
    }]>,
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoadViewTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_LoadViewTkoOp : CudaTileViewOpDef<"load_view_tko", "13.1",
    [AttrSizedOperandSegments]> {
  let summary = "Load a tile from a tile view";
  let description = [{
    The :code:`load_view_tko` operation loads a tile from a tile view.

    A view is mapping from view-space indices to a particular element in the view, each
    view type has a defined mapping from view-space indices to tiles produced from elements
    of the view.

    For example, the :ref:`type-partition_view` partitions a :ref:`type-tensor_view` into
    a grid of equally sized tiles. The view indexes one of the partitioned tiles in the grid.

    For a given view the rank of the indices must match the rank of the view's index
    space. The space of valid indices depends on which view is passed to the operation.
    For example the index space of a :ref:`type-partition_view` is equal to the
    rank of the partitioned tiles.

    The :code:`index` operands are interpreted as unsigned integers.

    The :code:`inbounds` attr lets the user (or a preceding analysis) declare
    that selected tile-coordinate dimensions are statically known to address
    only valid memory. It is required and must contain exactly one bool per
    index dimension; it is interpreted element-wise:

    - element :code:`i = true` is a guarantee that the i-th tile coordinate
      stays inside the parent tensor along dimension :code:`i`. Violating the
      guarantee is undefined behavior.
    - element :code:`i = false` carries no such guarantee: the i-th tile coordinate
      is conservatively treated as potentially out-of-bounds and handled
      according to the semantics of the concrete view type
      (see for example :ref:`type-partition_view`).

    In textual MLIR the :code:`inbounds = [...]` clause may be omitted, in
    which case the parser synthesizes an all-:code:`false` vector sized to
    the index rank (every dimension conservatively treated as potentially
    out-of-bounds).

    When :code:`memory_ordering_semantics` is :code:`weak`, :code:`memory_scope` must be
    omitted; when it is :code:`relaxed` or :code:`acquire`, :code:`memory_scope` is required.

    .. note::

      The pointer of the underlying :code:`tensor_view` must be aligned to the
      element type byte size (rounded to the next full byte for
      sub-byte types). Even stronger alignment guarantees can be
      specified with the :code:`cuda_tile.assume` operation.

      An :code:`f4E2M1FN` access additionally requires the underlying tensor
      view to identify exactly one statically encoded unit-stride packing
      dimension. The tensor view and concrete tile-view types specify
      additional evenness preconditions on extents, strides, and origins. Some
      may be runtime values, while tile extents and traversal strides are
      static type attributes. If an access executes with any applicable value
      odd, behavior is undefined.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%ptr: tile<ptr<f32>>, %index: tile<i32>) {
          %tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
            : tensor_view<8192x128xf32, strides=[128,1]>

          // This example uses the PartitionView on a 8192x128xf32 tensor_view,
          // dividing the tensor_view in tiles of 64x64.

          %view = make_partition_view %tensor_view : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

          %c0 = constant <i32: 0> : tile<i32>
          %c1 = constant <i32: 1> : tile<i32>

          // Load a tile at index (0, 0) in the view's index space.
          // For this PartitionView, this is the rectangular tile such that
          // X=[0,64) and Y=[0,64), in the coordinates of tiles.
          %tile0, %res_token0 = load_view_tko weak %view[%c0, %c0]
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

          // Load a tile at index (0, 1) in the view's index space.
          // For this PartitionView, this is the rectangular tile such that
          // X=[0,64) and Y=[64,128), in the coordinates of tiles.
          %tile1, %res_token1 = load_view_tko weak %view[%c0, %c1]
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

          // Same example as above but with memory token as input.
          %token = make_token : token
          %tile2, %res_token2 = load_view_tko weak %view[%c0, %c1] token = %token
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

          // Loads a tile at the dynamic index (%index, %index) in the view's index space.
          %tile3, %res_token3 = load_view_tko weak %view[%index, %index]
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token
    #   }
    # }
  }],
  [{
    # cuda_tile.module @module {
    #   entry @example_gather(%ptr: tile<ptr<f32>>) {
          // Create a tensor view
          %tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
            : tensor_view<8192x128xf32, strides=[128,1]>

          // Create a gather/scatter view with tile size (64x64) along dimension 0
          %gsview = make_gather_scatter_view %tensor_view :
            gather_scatter_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>, sparse_dim=0>

          // The 1D tensor index MUST be at position `sparse_dim` (here sparse_dim=0)
          // Index at sparse_dim=0: 1D tensor containing gather indices
          %gather_indices = constant <i32: [0, 10, 20, 30, 40, 50, 60, 70]> : tile<8xi32>
          // Index at sparse_dim=1: scalar for the other dimension
          %col_idx = constant <i32: 0> : tile<i32>

          // Load tiles using gather semantics
          // Index order: [%gather_indices (at sparse_dim=0), %col_idx (at sparse_dim=1)]
          %gathered_tile, %res_token = load_view_tko weak %gsview[%gather_indices, %col_idx]
            : gather_scatter_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>, sparse_dim=0>, tile<8xi32>, tile<i32>
            -> tile<64x64xf32>, token
    #   }
    # }
  }]];

  let arguments = (ins
    CudaTileArg<
      CudaTile_MemoryOrderingSemanticsAttr,
      "The memory ordering semantics for the load operation.",
      "13.1",
      [OnlyVariants<["WEAK", "RELAXED", "ACQUIRE"]>]>:$memory_ordering_semantics,
    CudaTileArg<OptionalAttr<CudaTile_MemoryScopeAttr>, "The memory scope for the atomic operation.", "13.1">:$memory_scope,
    CudaTileArg<CudaTile_TileView, "The view from which the tile will be loaded.", "13.1">:$view,
    CudaTileArg<Variadic<CudaTile_IntTileType>, "The n-dimensional index of the desired element to load from the view.", "13.1">:$index,
    CudaTileArg<Optional<CudaTile_TokenType>, "The optional token for the load operation.", "13.1">:$token,
    CudaTileArg<OptionalAttr<CudaTile_OptimizationHintsAttr>, "Optimization hints for operation", "13.1">:$optimization_hints,
    CudaTileArg<DenseBoolArrayAttr, inbounds_desc, "13.4",
                [RequireSameOperandRank<"index">]>:$inbounds);
  let results = (outs CudaTileArg<CudaTile_TileType, "The loaded tile.", "13.1">:$tile,
    CudaTileArg<CudaTile_TokenType, "The result token.", "13.1">:$result_token);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let builders = [
    // Convenience builder (attr form of MemoryOrderingSemantics) that
    // synthesizes the required `inbounds` attribute as an all-`false` vector
    // sized to the index rank, so callers that do not have per-dimension
    // inbounds information do not have to construct the attribute themselves.
    OpBuilder<(ins "::mlir::Type":$tile,
                   "::mlir::Type":$result_token,
                   "::mlir::cuda_tile::MemoryOrderingSemanticsAttr":$memory_ordering_semantics,
                   "::mlir::cuda_tile::MemoryScopeAttr":$memory_scope,
                   "::mlir::Value":$view,
                   "::mlir::ValueRange":$index,
                   "::mlir::Value":$token,
                   "::mlir::cuda_tile::OptimizationHintsAttr":$optimization_hints), [{
      auto inbounds = ::mlir::DenseBoolArrayAttr::get(
          $_builder.getContext(),
          ::llvm::SmallVector<bool>(index.size(), /*value=*/false));
      build($_builder, $_state, tile, result_token, memory_ordering_semantics,
            memory_scope, view, index, token, optimization_hints, inbounds);
    }]>,
    // Same convenience builder using the enum form of MemoryOrderingSemantics.
    OpBuilder<(ins "::mlir::Type":$tile,
                   "::mlir::Type":$result_token,
                   "::mlir::cuda_tile::MemoryOrderingSemantics":$memory_ordering_semantics,
                   "::mlir::cuda_tile::MemoryScopeAttr":$memory_scope,
                   "::mlir::Value":$view,
                   "::mlir::ValueRange":$index,
                   "::mlir::Value":$token,
                   "::mlir::cuda_tile::OptimizationHintsAttr":$optimization_hints), [{
      auto semAttr = ::mlir::cuda_tile::MemoryOrderingSemanticsAttr::get(
          $_builder.getContext(), memory_ordering_semantics);
      auto inbounds = ::mlir::DenseBoolArrayAttr::get(
          $_builder.getContext(),
          ::llvm::SmallVector<bool>(index.size(), /*value=*/false));
      build($_builder, $_state, tile, result_token, semAttr, memory_scope,
            view, index, token, optimization_hints, inbounds);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// LoadOpBase (abstract)
//===----------------------------------------------------------------------===//

def LoadOpBaseDoc {
  string summary =
      "Load and gather data from global memory using a pointer tile";
  string description = [{
    This :code:`load` OP performs a gather operation by loading
    a tile of data from global memory into a result tile based on a
    tile of pointers provided by the :code:`source` operand.

    The :code:`source` operand is a tile of pointers, which specifies the memory
    locations from which the data is gathered. Each pointer must be aligned to
    the element type byte size. Even stronger alignment guarantees can be
    specified with the :code:`cuda_tile.assume` operation.

    The operation loads this data and returns it as the :code:`result` tile.
    When loading i1 values, each value is loaded from a full byte in memory.
    Any nonzero byte is canonicalized to 0x01, and zero bytes become 0x00.
    Direct pointer-tile loads do not support :code:`f4E2M1FN`. Use a tile view
    to load f4 memory. Alternatively, load its packed byte representation as
    :code:`i8` and convert it with :code:`cuda_tile.unpack`.

    Optionally, a :code:`mask` operand can be provided to control the gathering of
    elements. If present, only the elements specified by the :code:`mask` are loaded.
    The shape of the :code:`mask` must match the shape of the :code:`result`.

    When :code:`mask` is present one :code:`paddingValue` can be optionally present as well.
    The :code:`paddingValue` must have the same shape of the :code:`source` tile. If
    it is not present, the value of masked elements are undefined.
  }];
}

class CudaTile_LoadOpBase<string mnemonic, string version>
    : CudaTileMemOpDef<
          mnemonic, version,
          [AttrSizedOperandSegments,
           TypesMatchWith<
               "`source` type is expected a pointer type of `result` type",
               "result", "source", "$_self",
               "mlir::OpTrait::cuda_tile::impl::verifyLoadStoreType">,
           OptionalTypesMatchWith<
               "shape of 'mask' must match the shape of 'source'", "source",
               "mask", "$_self",
               "mlir::OpTrait::cuda_tile::impl::verifyLoadStoreMask">,
           OptionalTypesMatchWith<
               "type of 'paddingValue' must match the type of 'result'",
               "result", "paddingValue", "$_self",
               "mlir::OpTrait::cuda_tile::impl::verifyLoadPadding">]> {}

//===----------------------------------------------------------------------===//
// LoadPtrTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_LoadPtrTkoOp : CudaTile_LoadOpBase<"load_ptr_tko", "13.1"> {
  let summary =
      !strconcat(LoadOpBaseDoc.summary, " without ordering guarantees");

  let description = !strconcat(LoadOpBaseDoc.description, [{
    When :code:`memory_ordering_semantics` is :code:`weak`, :code:`memory_scope`
    must be omitted; when it is :code:`relaxed` or :code:`acquire`, :code:`memory_scope`
    is required.

    Token-ordered operations are not constrained by program order. The compiler may
    reorder them (i.e. place them earlier or later in program order) unless further
    constrained by tokens.
  }]);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%ptr: tile<ptr<f32>>) {
          %mask = constant <i1: 1> : tile<i1>
          %padding = constant <f32: 0.0> : tile<f32>

            // Load without token.
            %result0, %res_token0 = load_ptr_tko weak %ptr, %mask, %padding
                : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

            // Load with token.
            %token0 = make_token : token
            %result1, %res_token1 = load_ptr_tko weak %ptr, %mask, %padding token=%token0
                : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

            return
      # }
    # }
  }]];

  let arguments = (ins
      CudaTileArg<
        CudaTile_MemoryOrderingSemanticsAttr,
        "The memory ordering semantics for the load operation.",
        "13.1",
        [OnlyVariants<["WEAK", "RELAXED", "ACQUIRE"]>]>:$memory_ordering_semantics,
      CudaTileArg<OptionalAttr<CudaTile_MemoryScopeAttr>, "The memory scope for the atomic operation.", "13.1">:$memory_scope,
      CudaTileArg<CudaTile_PointerTileType, "The source tile of pointers.", "13.1">:$source,
      CudaTileArg<Optional<CudaTile_TileOf<[CudaTile_Int1]>>, "The mask for the load operation.", "13.1">:$mask,
      CudaTileArg<Optional<CudaTile_NumberTileType>, "The padding value for the load operation.", "13.1">:$paddingValue,
      CudaTileArg<Optional<CudaTile_TokenType>, "The token for the load operation.", "13.1">:$token,
      CudaTileArg<OptionalAttr<CudaTile_OptimizationHintsAttr>, "Optimization hints for operation", "13.1">:$optimization_hints);

  let results = (outs CudaTileArg<CudaTile_TileType, "The result of the load operation.", "13.1">:$result,
      CudaTileArg<CudaTile_TokenType, "The result token of the load operation.", "13.1">:$result_token);

  let assemblyFormat = [{
    $memory_ordering_semantics
    ($memory_scope^)?
    $source
    (`,` $mask^)? (`,` $paddingValue^)?
    (`token` `` `=` `` $token^)?
    (`optimization_hints` `=` $optimization_hints^)?
    attr-dict `:`
    custom<CudaTileType>(type($source))
    (`,` custom<CudaTileType>(type($mask))^)?
    (`,` custom<CudaTileType>(type($paddingValue))^)?
    `->` custom<CudaTileType>(type($result))
    `,` custom<CudaTileType>(type($result_token))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

def CudaTile_LogOp : CudaTileFloatingPointOpDef<"log", "13.1", [
    Pure, AllTypesMatch<["source", "result"]>
  ]> {
  let summary = "Element-wise natural logarithm";
  let description = !strconcat([{
    The :code:`log` operation computes the element-wise natural logarithm of a
    floating-point tile.

    .. math::

      \text{log}(x)_i = \ln(x_i)
  }], floating_point_math_suffix);

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input floating-point tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the log operation.", "13.1">:$result);
  let assemblyFormat = [{
    $source
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Log2Op
//===----------------------------------------------------------------------===//

def CudaTile_Log2Op : CudaTileFloatingPointOpDef<"log2", "13.1", [
    Pure, AllTypesMatch<["source", "result"]>
  ]> {
  let summary = "Element-wise base-2 logarithm";
  let description = !strconcat([{
    The :code:`log2` operation computes the element-wise base-2 logarithm
    of a floating-point tile.

    .. math::

      \text{log2}(x)_i = \log_2(x_i)
  }], floating_point_math_suffix);

  let arguments = (ins
    CudaTileArg<CudaTile_BaseFloatTileType, "The input floating-point tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the log2 operation.", "13.1">:$result);

  let assemblyFormat = [{
    $source
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_log2() {
        %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
        %res = log2 %in : tile<4xf32>
      # }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

def CudaTile_LoopOp : CudaTileControlFlowOpDef<"loop", "13.1", [
    AutomaticAllocationScope,
    OpAsmOpInterface,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"impl::LoopOpImplicitTerminatorType">
  ]> {
  let summary = "Loop until a break operation";
  let description = [{
    The :code:`loop` operation represents an, unstructured, infinite loop that executes
    until a :ref:`op-cuda_tile.break` is reached.

    The loop consists of a (1) a set of loop-carried values which are initialized by :code:`initValues` and updated by each iteration of the loop, and
    (2) a region which represents the loop body.

    The loop will execute the body of the loop until a :ref:`op-cuda_tile.break` is dynamically executed.

    Each control path of the loop must be terminated by:

    - a :ref:`op-cuda_tile.continue` that yields the next iteration's value for each loop carried variable.
    - a :ref:`op-cuda_tile.break` that terminates the loop and yields the final loop carried values.
    - a :ref:`op-cuda_tile.return` that terminates the loop and returns from the enclosing function.

    As long as each loop iteration is terminated by one of these operations they may be combined with other control
    flow operations to express different control flow patterns.

    The loop operation produces one return value for each loop carried variable. The type of the :math:`i`:spelling:ignore:`th` return
    value is that of the :math:`i`:spelling:ignore:`th` loop carried variable and its value is the final value of the
    :math:`i`:spelling:ignore:`th` loop carried variable.

    .. warning::

      Loop operations have a set of additional restrictions today:

      - Loop carried variables can not be a :tileirty:`tensor_view` or view type.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          // A simple "while-do" loop.
          loop {
              %cond = constant <i1: 1> : tile<i1>
              if %cond {
                  continue
              }
              break
          }
    #   }
    # }
    }],
    [{
    # cuda_tile.module @module {
    #   entry @example() {
          // A simple "do-while" loop.
          loop {
              //... body of the loop.

              %cond = constant <i1: 1> : tile<i1>
              if %cond {
                  continue
              }
              break
          }
    #   }
    # }
    }],
    [{
    # cuda_tile.module @module {
    #   entry @example() {
          %initValue0 = constant <f32: 0.0> : tile<f32>
          // A loop that yields carried-iteration values, returning the final values.
          %results = loop iter_values(%value0 = %initValue0) : tile<f32> -> tile<f32> {
              %cond = constant <i1: 1> : tile<i1>
              if %cond {
                  %loopValue0 = constant <f32: 0.0> : tile<f32>
                  continue %loopValue0 : tile<f32>
              }
              break %value0 : tile<f32>
          }
    #   }
    # }
    }],
    [{
    # cuda_tile.module @module {
    #   entry @example() {
          // A loop with early return from within if statement.
          loop {
              %condition = constant <i1: 1> : tile<i1>
              if %condition {
                  // Early return from loop - returns from the enclosing function
                  return
              }
              continue
          }
    #   }
    # }
    }],
    [{
    # cuda_tile.module @module {
    #   entry @example() {
          // A loop with direct early return (not nested in if).
          loop {
              // Direct early return from loop
              return
          }
    #   }
    # }
    }],
    [{
    # cuda_tile.module @module {
    #   entry @example() {
          %initValue0 = constant <i32: 0> : tile<i32>
          // A loop that uses loop-carried values and returns a different type.
          %results = loop iter_values(%value0 = %initValue0) : tile<i32> -> tile<f32> {
              %cond = constant <i1: 1> : tile<i1>

              if %cond {
                  %newLoopValue = constant <i32: 0> : tile<i32>
                  continue %newLoopValue : tile<i32>
              }

              %finalReturnValue = constant <f32: 0.0> : tile<f32>
              break %finalReturnValue : tile<f32>
          }
    #   }
    # }
    }]];


  let arguments = (ins CudaTileArg<Variadic<AnyType>, "The initial values of the loop.", "13.1">:$initValues);
  let results = (outs CudaTileArg<Variadic<AnyType>, "The result values of the loop.", "13.1">:$resultValues);
  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl # [{
    /// Return the iteration values of the loop region.
    Block::BlockArgListType getRegionIterValues() {
      return getRegion().getArguments();
    }

    /// Return the `index`-th region iteration value.
    BlockArgument getRegionIterValue(unsigned index) {
      return getRegionIterValues()[index];
    }

    /// Returns the number of region arguments for loop-carried values.
    unsigned getNumRegionIterValues() { return getRegion().getNumArguments(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MakeTensorView
//===----------------------------------------------------------------------===//

def CudaTile_MakeTensorViewOp : CudaTileViewOpDef<"make_tensor_view", "13.1",
    [AttrSizedOperandSegments, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create :code:`tensor_view` from a pointer to global memory";
  let description = [{
    The :code:`make_tensor_view` operation constructs a :code:`tensor_view` from a global
    memory pointer, a dynamic shape and dynamic strides. See :ref:`type-tensor_view` for more
    details. The :code:`base` pointer's pointee type must equal the element type of the
    resulting :code:`tensor_view`. If :code:`base` carries a :code:`ptr_attr`, the
    :code:`tensor_view` must carry the same :code:`ptr_attr`; if :code:`base` does not
    carry one, the :code:`tensor_view` must not carry one either.

    The constructor supports taking dynamic arrays for shapes and strides, enabling workloads
    to operate on global memory tensors of dynamic shape and strides. Static values are
    reflected directly in the resulting :code:`tensor_view` type; dynamic values appear as
    :code:`?` and the corresponding :code:`dynamicShape` / :code:`dynamicStrides` operands
    must be supplied (one per :code:`?`). All dynamic shape and stride operands must be
    scalar tiles of the same integer element type, interpreted as unsigned integers. See
    below for concrete examples.

  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%base: tile<ptr<f32>>) {
          // tensor_view to a scalar tile of f32
          %a0 = make_tensor_view %base,
              shape = [], strides = [] : tensor_view<f32>

          // tensor_view to a tile of static shape and strides
          %a1 = make_tensor_view %base,
              shape = [32, 32], strides = [32, 1]
              : tensor_view<32x32xf32, strides=[32,1]>

        %sh0 = constant <i32: 32> : tile<i32>
        %sh1 = constant <i32: 32> : tile<i32>
        %st0 = constant <i32: 32> : tile<i32>
        %st1 = constant <i32: 1> : tile<i32>

          // tensor_view to a tile with partially dynamic shape and strides
          // all dynamic values must be of the same type, here tile<i32>
          %a2 = make_tensor_view %base,
                  shape = [%sh0, %sh1], strides = [%st0, %st1]
                  : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>
      # }
    # }
    }]];

  let arguments = (ins CudaTileArg<CudaTile_ScalarTileOf<CudaTile_PointerType>,
                                  "The scalar base pointer to a portion of global memory.", "13.1">:$base,
                       CudaTileArg<Variadic<CudaTile_ScalarTileOf<CudaTile_AnyInt>>, "The array of values representing the shape of the view, may be fully dynamic.", "13.1">:$dynamicShape,
                       CudaTileArg<Variadic<CudaTile_ScalarTileOf<CudaTile_AnyInt>>, "The array of values representing the strides of the view, may be fully dynamic.", "13.1">:$dynamicStrides);

  let results = (outs CudaTileArg<CudaTile_TensorViewType, "The constructed tensor_view.", "13.1">:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MaxFOp
//===----------------------------------------------------------------------===//

def CudaTile_MaxFOp : CudaTileFloatingPointOpDef<"maxf", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise floating-point maximum";
  let description = [{
    The :code:`maxf` operation computes the element-wise maximum of two input
    tiles with floating-point element types.

    The :code:`propagate_nan` controls how :code:`maxf` will interpret :code:`NaN`. If
    the :code:`propagate_nan` modifier is set, :code:`maxf` returns a canonical :code:`NaN`
    if either of the compared elements is :code:`NaN` (IEEE 754-2019's maximum). While if
    the :code:`propagate_nan` modifier is not set, :code:`maxf` returns a canonical :code:`NaN`
    only if both elements are :code:`NaN`; otherwise, it returns the non-:code:`NaN` element (IEEE
    754-2019's :spelling:ignore:`maximumNumber`).

    If neither element is :code:`NaN`, :code:`maxf` will return the greater of the
    inputs. :code:`+0.0` is considered greater than :code:`-0.0`.

    .. math::
      \text{maxi}(x, y)_i = \begin{cases}
        x_i & \text{if } x_i \geq y_i \\
        y_i & \text{if } x_i < y_i
      \end{cases}
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      #   entry @example_maxf(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) {
            // Create tensor view from a pointer to global memory
            %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
            %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
            // Convert tensor views to partition views and load tiles from partition views.
            %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
            %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
            %c0 = constant <i32: 0> : tile<i32>
            %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
            %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
            // IEEE 754-2019's maximum
            %4 = maxf %2, %3 propagate_nan : tile<2x4xf32>
            // IEEE 754-2019's maximumNumber
            %5 = maxf %2, %3 : tile<2x4xf32>
            // flush denormals to sign-preserving zero
            %6 = maxf %2, %3 flush_to_zero : tile<2x4xf32>
      # }
    # }
  }]];

  let arguments =
    (ins CudaTileArg<CudaTile_BaseFloatTileType, "The left hand side operand.", "13.1">:$lhs,
         CudaTileArg<CudaTile_BaseFloatTileType, "The right hand side operand.", "13.1">:$rhs,
         CudaTileArg<UnitAttr, cannonical_nan_desc, "13.1">:$propagate_nan,
         CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);

  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the :code:`maxf` operation.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs
    oilist(`flush_to_zero` $flush_to_zero |
           `propagate_nan` $propagate_nan)
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MaxIOp
//===----------------------------------------------------------------------===//

def CudaTile_MaxIOp : CudaTileIntegerOpDef<"maxi", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise integer maximum";
  let description = !strconcat([{
    The :code:`maxi` operation computes the element-wise maximum between two input integer tiles.

    .. math::
      \text{maxi}(x, y)_i = \begin{cases}
        x_i & \text{if } x_i \geq y_i \\
        y_i & \text{if } x_i < y_i
      \end{cases}
  }], integer_arith_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
      #   entry @example_maxi(%arg0: tile<ptr<i32>>, %arg1: tile<ptr<i32>>) {
            // Create tensor view from a pointer to global memory
            %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
            %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
            // Convert tensor views to partition views and load tiles from them.
            %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
            %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
            %c0 = constant <i32: 0> : tile<i32>
            %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
            %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
            // Signless i32 treated as unsigned
            %4 = maxi %2, %3 unsigned : tile<2x4xi32>
            // Signless i32 treated as signed
            %5 = maxi %2, %3 signed : tile<2x4xi32>
      # }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The result of the maxi operation.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict
    `:` custom<CudaTileType>(type($result))
  }];

  let builders = [
    OpBuilder<(ins "Type":$resTy,
                   "ValueRange":$operands, "mlir::cuda_tile::Signedness":$signedness), [{
      assert(operands.size() == 2 && "expected two operands");
      return build($_builder, $_state, resTy, operands[0],
                   operands[1], signedness);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// MinFOp
//===----------------------------------------------------------------------===//

def CudaTile_MinFOp : CudaTileFloatingPointOpDef<"minf", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise floating-point minimum";
  let description = [{
    The :code:`minf` operation computes the element-wise minimum of two input
    tiles with floating-point element types.

    The :code:`propagate_nan` controls how :code:`minf` will interpret :code:`NaN`. If
    the :code:`propagate_nan` modifier is set, :code:`minf` returns a canonical :code:`NaN`
    if either of the compared elements is :code:`NaN` (IEEE 754-2019's minimum). While if
    the :code:`propagate_nan` modifier is not set, :code:`minf` returns a canonical :code:`NaN`
    only if both elements are :code:`NaN`; otherwise, it returns the non-:code:`NaN` element (IEEE
    754-2019's :spelling:ignore:`minimumNumber`).

    If neither element is :code:`NaN`, :code:`minf` will return the lowest of the
    inputs. :code:`-0.0` is considered less than :code:`+0.0`.

    .. math::
      \text{minf}(x, y)_i = \begin{cases}
        x_i & \text{if } x_i \leq y_i \\
        y_i & \text{if } x_i > y_i
      \end{cases}
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
      #   entry @example_minf(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) {
            // Create tensor view from a pointer to global memory
            %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
            %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
            // Convert tensor views to partition views and load tiles from partition views.
            %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
            %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
            %c0 = constant <i32: 0> : tile<i32>
            %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
            %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
            // IEEE 754-2019's minimum
            %4 = minf %2, %3 propagate_nan : tile<2x4xf32>
            // IEEE 754-2019's minimumNumber
            %5 = minf %2, %3 : tile<2x4xf32>
            // flush denormals to sign-preserving zero
            %6 = minf %2, %3 flush_to_zero : tile<2x4xf32>
      # }
    # }
  }]];

  let arguments =
    (ins CudaTileArg<CudaTile_BaseFloatTileType, "The left hand side operand.", "13.1">:$lhs,
      CudaTileArg<CudaTile_BaseFloatTileType, "The right hand side operand.", "13.1">:$rhs,
      CudaTileArg<UnitAttr, cannonical_nan_desc, "13.1">:$propagate_nan,
      CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The minimum of the input tiles.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs
    oilist(`flush_to_zero` $flush_to_zero |
           `propagate_nan` $propagate_nan)
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MinIOp
//===----------------------------------------------------------------------===//

def CudaTile_MinIOp : CudaTileIntegerOpDef<"mini", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise integer minimum";
  let description = !strconcat([{
    The :code:`mini` operation computes the element-wise minimum between the two input tiles with
    integer element types.

    .. math::
      \text{mini}(x, y)_i = \begin{cases}
        x_i & \text{if } x_i \leq y_i \\
        y_i & \text{if } x_i > y_i
      \end{cases}
  }], integer_arith_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
      #   entry @example_mini(%arg0: tile<ptr<i32>>, %arg1: tile<ptr<i32>>) {
            // Create tensor view from a pointer to global memory
            %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
            %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
            // Convert tensor views to partition views and load tiles from partition views.
            %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
            %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
            %c0 = constant <i32: 0> : tile<i32>
            %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
            %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
            // Signless i32 treated as unsigned
            %4 = mini %2, %3 unsigned : tile<2x4xi32>
            // Signless i32 treated as signed
            %5 = mini %2, %3 signed : tile<2x4xi32>
      # }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The minimum of the input tiles.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict
    `:` custom<CudaTileType>(type($result))
  }];

  let builders = [
    OpBuilder<(ins "Type":$resTy,
                   "ValueRange":$operands, "mlir::cuda_tile::Signedness":$signedness), [{
      assert(operands.size() == 2 && "expected two operands");
      return build($_builder, $_state, resTy, operands[0],
                   operands[1], signedness);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def CudaTile_ModuleOp : CudaTileCoreOpDef<"module", "13.1", [
    IsolatedFromAbove, OpAsmOpInterface, NoRegionArguments, SingleBlock,
    Symbol, SymbolName, SymbolTable]
        # GraphRegionNoTerminator.traits> {
  let summary = "Top-level module containing a series of defined items.";
  let description = [{
    A :code:`module` operation represents a single compilation unit and contains
    zero or more items (global variables, functions, or kernels).

    For detailed description of the semantics of modules, and the full definition of each item type see
    :ref:`sub_sec_modules`.

    The :code:`module` operation is the top-level operation in a |cuda_tile| module and must
    contain only |cuda_tile| operations and no other dialects.

    The optional :code:`producer` attribute contains free-form text identifying what tool
    generated this module (e.g., compiler version, build options).
  }];
  let arguments = (ins CudaTileArg<SymbolNameAttr, "The name of the module.", "13.1">:$sym_name,
                       CudaTileArg<OptionalAttr<StrAttr>, "Optional producer identifier.", "13.3">:$producer);
  let regions = (region MaxSizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;

  // We need to ensure that the region has a block; the auto-generated
  // builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, CArg<"StringRef", "\"\"">:$producer)>
  ];

  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl # [{
    ::mlir::SymbolTable::Visibility getVisibility() {
      return ::mlir::SymbolTable::Visibility::Public;
    }

    void setVisibility(::mlir::SymbolTable::Visibility visibility) {
      assert(visibility == ::mlir::SymbolTable::Visibility::Public &&
             "cuda_tile modules only support public visibility");
    }
  }];
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

def CudaTile_MulFOp : CudaTileFloatingPointOpDef<"mulf", "13.1", [
    Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise floating-point multiplication";
  let description = !strconcat([{
    The :code:`mulf` operation computes the element-wise product between the two input tiles with
    with floating-point element types.

    .. math::
      \text{mulf}(x, y)_i = x_i \times y_i
  }], floating_point_arith_suffix);

  let descriptionTables = [
    Table<":code:`mulf` Modifiers", "The below table shows the supported modifiers and rounding modes for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["flush_to_zero", "yes", "no", "no", "no"]>,
       TableRow<["rounding<nearest_even>", "yes", "yes", "yes", "yes"]>,
       TableRow<["rounding<zero>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<negative_inf>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<positive_inf>", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments =
    (ins CudaTileArg<CudaTile_BaseFloatTileType, "The left hand side operand.", "13.1">:$lhs,
      CudaTileArg<CudaTile_BaseFloatTileType, "The right hand side operand.", "13.1">:$rhs,
      CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                  [OnlyCudaTileIEEERoundingModes]>:$rounding_mode,
      CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);

  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The product of the input tiles.", "13.1">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs
    custom<IEEERoundingMode>($rounding_mode)
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

// Supported types for MulIOp.
def CudaTile_MulIOp : CudaTileIntegerOpDef<"muli", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise integer multiplication";
  let description = !strconcat([{
    The :code:`muli` operation computes the element-wise product between the two input tiles with
    integer element types.

    .. math::
      \text{muli}(x, y)_i = x_i \times y_i
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side input integer tile.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side input integer tile.", "13.1">:$rhs,
                       CudaTileArg<DefaultValuedAttr<CudaTile_IntegerOverflowAttr, "::mlir::cuda_tile::IntegerOverflow::NONE">, overflow_desc, "13.1">:$overflow);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The product of the input tiles.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`overflow` `` $overflow^)? attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// MulhiIOp
//===----------------------------------------------------------------------===//

def CudaTile_MulhiIOp : CudaTileIntegerOpDef<"mulhii", "13.1",
    [Pure, AllTypesMatch<["x", "y", "result"]>]> {
  let summary = "Element-wise high bits of integer multiplication";
  let description = !strconcat([{
    The :code:`mulhii` operation produces the most significant N bits of the 2N-bit
    product of two N-bit integer tiles. For :code:`i64`, this is the most significant 64
    bits of the full 128-bit product; for :code:`i8`, it is the most significant 8
    bits of the full 16-bit product; etc.

    This is in contrast to :code:`muli`, which produces the lower N bits of the 2N-bit
    product.

    The :code:`mulhii` operation is only defined for unsigned integers.

    .. math::
      \text{mulhii}(x_i, y_i) = x_i \times y_i >> \text{bitwidth}(\text{type}(x_i))
  }], integer_arith_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          // 2^31 * 2 = 2^32, or 0x100000000.
          // The most significant 32 bits of the product are 0x00000001.
          // The lower 32 bits of the product are 0x00000000.
          %a = constant <i32: 2147483648> : tile<i32>  // %a = 2^31
          %b = constant <i32: 2> : tile<i32>           // %b = 2
          %res_hi = mulhii %a, %b : tile<i32>          // %res_hi = 1
          %res_lo = muli %a, %b : tile<i32>            // %res_lo = 0
    #   }
    # }
    }]];

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side input integer tile.", "13.1">:$x,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side input integer tile.", "13.1">:$y);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The most significant bits of the product of the input tiles.", "13.1">:$result);

  let assemblyFormat = [{
    $x `,` $y attr-dict
    `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// NegIOp
//===----------------------------------------------------------------------===//

def CudaTile_NegIOp : CudaTileIntegerOpDef<"negi", "13.1", [
    Pure, AllTypesMatch<["source", "result"]>
  ]> {
  let summary = "Element-wise integer negation";
  let description = !strconcat([{
    The :code:`negi` operation computes the element-wise negation of the input integer tile.
    The input and output tiles are always interpreted as signed integers; the :code:`overflow`
    flag :code:`no_unsigned_wrap` is therefore rejected.

    .. math::
      \text{negi}(x_i) = -x_i
  }], integer_arith_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %source = constant <i16: [0, 1, 2, 3]> : tile<4xi16>
          %result = negi %source : tile<4xi16>
          // %result = [0, -1, -2, -3]
    #   }
    # }
  }]];

  let hasVerifier = 1;
  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The input integer tile.", "13.1">:$source,
                       CudaTileArg<DefaultValuedAttr<CudaTile_IntegerOverflowAttr, "::mlir::cuda_tile::IntegerOverflow::NONE">, overflow_desc, "13.2">:$overflow);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The negated integer tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source (`overflow` `` $overflow^)? attr-dict `:` custom<CudaTileType>(type($result))
  }];

}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def CudaTile_NegFOp : CudaTileFloatingPointOpDef<"negf", "13.1", [
    Pure, AllTypesMatch<["source", "result"]>
  ]> {
  let summary = "Element-wise floating-point negation";
  let description = !strconcat([{
    :code:`negf` is an element-wise operation that negates the sign of :code:`source`.

    .. math::
      \text{negf}(x)_i = -x_i
  }], floating_point_exact_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %source = constant <f32: 0.0> : tile<4xf32>
          %result = negf %source : tile<4xf32>
    #   }
    # }
  }]];

  let arguments =
    (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The negated floating-point tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// MakeTokenOp
//===----------------------------------------------------------------------===//

def CudaTile_MakeTokenOp
    : CudaTileMemOpDef<"make_token", "13.1", [Pure]> {
  let summary = "Create a fresh token with no prior dependencies";
  let description = [{
    The :code:`make_token` operation creates a fresh token with no prior dependencies.
  }];
  let arguments = (ins);
  let results = (outs CudaTileArg<CudaTile_TokenType, "A fresh token with no prior dependencies.", "13.1">:$result);
  let assemblyFormat = "attr-dict `:` custom<CudaTileType>(type($result))";
}

//===----------------------------------------------------------------------===//
// MemoryFenceAliasTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_MemoryFenceAliasTkoOp
    : CudaTileMemOpDef<"memory_fence_alias_tko", "13.4"> {
  let summary = "Token-ordered proxy alias fence";
  let description = [{
    The :code:`memory_fence_alias_tko` operation is a token-ordered proxy fence
    for operations that may access the same physical memory through different
    virtual aliases. It models alias proxy synchronization separately from
    ordinary memory-ordering fences.

    Along the token chain, this operation enforces that all earlier targeted memory
    operations that access virtually aliased memory have completed after the fence. 
  }];
  let mlirExamples = [[{
    cuda_tile.module @kernels {
      entry @example(%tok: !cuda_tile.token) {
        // Synchronize memory accesses through virtual aliases.
        %f0 = memory_fence_alias_tko %tok
        return
      }
    }
  }]];
  let arguments = (ins
      CudaTileArg<CudaTile_TokenType,
          "Input token from a prior token-ordered operation.", "13.4">:$token);
  let results = (outs
      CudaTileArg<CudaTile_TokenType,
          "Result token for subsequent token-ordered operations.", "13.4">:$result_token);
  let assemblyFormat = "$token attr-dict";
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

def CudaTile_OffsetOp : CudaTileCoreOpDef<"offset", "13.1", [
    Pure, Elementwise, SameOperandsAndResultShape,
    AllTypesMatch<["result", "ptr"]>]> {
  let summary = "Offsets a tile of pointers";

  let description = [{
    :code:`offset` advances a tile of pointers. It takes :code:`ptr` as base
    and :code:`offset` as increment, and performs element-wise addition of
    :code:`ptr` by :code:`offset`, measured in pointee elements:

    .. math::
      \text{offset}(\text{ptr}, \text{offset})_i = \text{ptr}_i + \text{offset}_i

    .. code-block:: mlir

        result[i,j] = ptr[i,j] + offset[i,j] elements

    :code:`ptr` is interpreted as an unsigned integer. :code:`offset` is
    interpreted as a signed integer. For byte-addressable pointee types, the
    physical byte displacement is :code:`offset` multiplied by the pointee's
    storage size in bytes. For :code:`f4E2M1FN`, :code:`offset` must be even and
    the byte displacement is :code:`offset / 2`; executing this operation with
    an odd f4 element offset is undefined behavior.

    The displacement calculation must not overflow (wrap around) in a signed
    sense. The pointer addition must not overflow (wrap around) in an unsigned
    sense. In case of an overflow, the result is undefined.
  }];

  let arguments = (ins CudaTileArg<CudaTile_PointerTileType, "The base pointer tile to advance.", "13.1">:$ptr,
    CudaTileArg<CudaTile_IntTileType, "The offset tile to add to the pointer.", "13.1">:$offset);
  let results = (outs CudaTileArg<CudaTile_PointerTileType, "The resulting pointer tile after advancement.", "13.1">:$result);
  let assemblyFormat = [{
    $ptr `,` $offset attr-dict `:` custom<CudaTileType>(type($ptr)) `,`
    custom<CudaTileType>(type($offset)) `->` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

// NumberTileType + Int4
def PackOp_UnpackedType : CudaTile_TileOf<
  !listconcat(CudaTile_NumberTileType.allowedElementTypes, [CudaTile_Int4])>;

def CudaTile_PackOp : CudaTileCoreOpDef<"pack", "13.3", [
    Pure, AllRanksMatch<["source", "result"]>]> {
  let summary = "Pack a tile into a byte array";
  let description = [{
    The :code:`pack` operation takes a rank-1 numeric tile and produces a rank-1 :code:`tile<i8>`.

    Similar to :code:`bitcast`, underlying bit-values are not changed. However, :code:`pack` does not
    operate elementwise, instead reinterpreting the entire tile as a byte array.

    Input and output tiles must be rank-1 to eliminate packing ambiguity. The size of the output
    tile must match the number of bytes in the input tile. The input tile cannot be an 8-bit type
    (bitcast should be used instead).
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %arg0 = constant <f16: 0.0> : tile<64xf16>
          %0 = pack %arg0 : tile<64xf16> -> tile<128xi8>
    #   }
    # }
  }], [{
    # cuda_tile.module @module {
    #   entry @example() {
          %arg0 = constant <f4E2M1FN: 0.0> : tile<64xf4E2M1FN>
          %0 = pack %arg0 : tile<64xf4E2M1FN> -> tile<32xi8>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<PackOp_UnpackedType, "The input tile.", "13.3">:$source);
  let results = (outs CudaTileArg<CudaTile_TileOf<[CudaTile_Int8]>, "The packed tile.", "13.3">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($source)) `->` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

def CudaTile_PermuteOp : CudaTileCoreOpDef<"permute", "13.1", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
    AllElementTypeMatch<"all of {source, result} have the same element type", ["source", "result"]>,
    AllRanksMatch<["source", "result"]>]> {
  let summary = "Permute tile dimensions";
  let description = [{
    Permute the dimensions of the input tile :code:`source` according to the
    :code:`permutation` array. The :code:`source` must have rank at least 2, and
    :code:`permutation` must list each dimension index in :math:`[0, \text{rank})`
    exactly once.

    For example, if the input tile has shape :code:`[2, 4, 8]`, and the permutation is :code:`[2, 0, 1]`,
    the output tile will have shape :code:`[8, 2, 4]`.

    This operation logically is a change in the indexing of the tile.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %arg0 = constant <f16: 0.0> : tile<2x4x8xf16>
          %0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
    #   }
    # }
  }]];

  let arguments =
    (ins CudaTileArg<CudaTile_TileType, "The input tile.", "13.1">:$source,
         CudaTileArg<DenseI32ArrayAttr, "The permutation of the dimensions.", "13.1">:$permutation);
  let results = (outs CudaTileArg<CudaTile_TileType, "The permuted tile.", "13.1">:$result);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $source $permutation  attr-dict
    `:` custom<CudaTileType>(type($source))
    `->` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// FPowFOp
//===----------------------------------------------------------------------===//

def CudaTile_FPowFOp : CudaTileFloatingPointOpDef<"fpowf", "13.1",
    [Pure,
     AllTypesMatch<["result", "source", "exponent"]>]> {
  let summary = "Element-wise floating-point exponentiation with float exponent";

  let description = !strconcat([{
    The :code:`fpowf` operation computes the element-wise exponentiation of the source
    floating-point tile raised to the power of the exponent floating-point tile. Both
    the source and exponent must have the same floating-point tile type.

    .. math::
      \text{fpowf}(x, y)_i = x_i^{y_i}
  }], floating_point_math_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %source = constant <f32: 0.0> : tile<4xf32>
          %exponent = constant <f32: 2.0> : tile<4xf32>
          %result = fpowf %source, %exponent : tile<4xf32>
    #   }
    # }
  }]];

  let arguments =
    (ins CudaTileArg<CudaTile_BaseFloatTileType, "The base tile.", "13.1">:$source,
         CudaTileArg<CudaTile_BaseFloatTileType, "The exponent tile.", "13.1">:$exponent);

  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the fpowf operation.", "13.1">:$result);
  let assemblyFormat = [{
    $source `,` $exponent attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// FPowIOp
//===----------------------------------------------------------------------===//

def CudaTile_PowIExponentTileType : CudaTile_TileOf<[
  CudaTile_Int1, CudaTile_Int8, CudaTile_Int16, CudaTile_Int32
]>;

def CudaTile_FPowIOp : CudaTileFloatingPointOpDef<"fpowi", "13.4",
    [Pure,
     AllTypesMatch<["result", "source"]>,
     AllShapesMatch<["source", "exponent"]>]> {
  let summary = "Element-wise floating-point exponentiation with integer exponent";

  let description = !strconcat([{
    The :code:`fpowi` operation computes the element-wise exponentiation of the source
    floating-point tile raised to the power of the integer exponent tile.

    The integer exponent is always treated as signed.

    .. math::
      \text{fpowi}(x, n)_i = x_i^{n_i}
  }], floating_point_math_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %source = constant <f32: 2.0> : tile<4xf32>
          %exponent = constant <i32: 3> : tile<4xi32>
          %result = fpowi %source, %exponent : tile<4xf32>, tile<4xi32>
    #   }
    # }
  }]];

  let arguments =
    (ins CudaTileArg<CudaTile_BaseFloatTileType, "The base tile.", "13.4">:$source,
         CudaTileArg<CudaTile_PowIExponentTileType, "The exponent tile.", "13.4">:$exponent);

  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the fpowi operation.", "13.4">:$result);
  let assemblyFormat = [{
    $source `,` $exponent attr-dict
    `:` custom<CudaTileType>(type($source)) `,` custom<CudaTileType>(type($exponent))
  }];
}

//===----------------------------------------------------------------------===//
// PrintTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_PrintTkoOp : CudaTileMiscOpDef<"print_tko", "13.1",
    [AttrSizedOperandSegments
    ]> {
  let summary = "Print a formatted string (token-ordered)";
  let description = [{
    The :code:`print_tko` operation prints a C-printf-style format string,
    interleaved with the given operands. The number of format expressions
    (starting with the :code:`%` character) must match the number of operands.
    If a format expression is not applicable to its respective operand, then
    the output is undefined.

    Token-ordered print operations are not constrained by program order. The
    compiler may reorder them (i.e., move them earlier or later in the program)
    unless further constrained by tokens.

    This operation is meant for debugging. Its implementation is not optimized
    for performance, so it should not be used in production mode. Prints are
    not guaranteed to be atomic. I.e., the output of prints that execute
    simultaneously may be interleaved.

    .. note::

      This op was renamed from :code:`print` to :code:`print_tko` in 13.2. The
      op code did not change.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          # %arg = constant <f32: 0.0> : tile<4xf32>
          print_tko "Hello world: %f\n", %arg : tile<4xf32> -> token
          print_tko "%+08.3f", %arg : tile<4xf32> -> token
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<StrAttr, "The format string.", "13.1">:$str,
                       CudaTileArg<Variadic<CudaTile_TileType>, "The arguments to format and print.", "13.1">:$args,
                       CudaTileArg<Optional<CudaTile_TokenType>, token_desc, "13.2">:$token);
  let results = (outs CudaTileArg<CudaTile_TokenType, "The result token for synchronization.", "13.2">:$result_token);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $str (`,` $args^)? (`token` `` `=` `` $token^)?
    attr-dict
    (`:` custom<CudaTileType>(type($args))^)? `->` custom<CudaTileType>(type($result_token))
  }];
}

//===----------------------------------------------------------------------===//
// PtrToIntOp
//===----------------------------------------------------------------------===//

def CudaTile_PtrToIntOp : CudaTileConversionOpDef<"ptr_to_int", "13.1", [
    Pure, AllShapesMatch<["source", "result"]>]> {

  let summary = "Convert a tile of pointers to a tile of integers";

  let description = [{
    The :code:`ptr_to_int` operation converts a tile of pointer-type elements to a tile of :code:`i64` elements.

    The result values should be interpreted as unsigned integers.

    The inverse of this operation is :ref:`op-cuda_tile.int_to_ptr`.
  }];

  let arguments = (ins
    CudaTileArg<CudaTile_PointerTileType, "The input tile of pointers.", "13.1">:$source
  );
  let results = (outs
    CudaTileArg<CudaTile_IntTileInt64Type, "The output tile of integers.", "13.1">:$result
  );
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($source)) `->` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// PtrToPtrOp
//===----------------------------------------------------------------------===//

def CudaTile_PtrToPtrOp : CudaTileConversionOpDef<"ptr_to_ptr", "13.1", [
    Pure, AllShapesMatch<["source", "result"]>]> {

  let summary = "Reinterpret a tile of one pointer type as another";

  let description = [{
    The :code:`ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another
    element. Casts between pointer and non-pointer types are disallowed.

    In order to perform those conversions, use :ref:`op-cuda_tile.ptr_to_int` or :ref:`op-cuda_tile.int_to_ptr`.
    These operations are distinct to enable future compiler reasoning about pointer provenance.
  }];

  let arguments = (ins
    CudaTileArg<CudaTile_PointerTileType, "Tile with source pointer element type.", "13.1">:$source
  );
  let results = (outs
    CudaTileArg<CudaTile_PointerTileType, "Tile with target pointer element type.", "13.1">:$result
  );
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($source)) `->` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

def CudaTile_ReduceOp : CudaTileCoreOpDef<"reduce", "13.1", [
    InferTypeOpAdaptor, OpAsmOpInterface, RecursiveMemoryEffects,
    SameOperandsShape, SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames", "getAsmBlockArgumentNames"]>
  ]> {

  let summary = "Variadic tile reduction across dimensions";

  let description = [{
    The :code:`reduce` operation applies a custom reduction function along a specified dimension of
    one or more input tiles, producing the same number of output tiles, using an associative
    and commutative function together with an identity value.

    The reduction function must be an associative and commutative operation defined within
    the :code:`reduce` operation's region. A single reduction operation can reduce over any number
    of input tiles in parallel, producing a reduced output tile for each. The operation assumes an
    associative and commutative combine function, allowing implementations to choose any valid
    evaluation strategy, including parallel implementations. As a result, the exact evaluation
    order is implementation-defined, but the result remains deterministic across runs of the same
    kernel on the same device.

    .. note::

      Only memory-effect-free operations are allowed in the body of :code:`reduce`.

    All input tiles must have the same shape. The output tiles will have a matching shape in every
    dimension except the one being reduced, which is removed.

    For each input tile, a constant identity value must be provided that matches the element type of
    the input tile. Identity :code:`i` of :code:`identities` corresponds to input tile
    :code:`i` of :code:`operands`. The correct identity value is a property of the reduction
    function in the :code:`body`. (For example, if the reduction function performs :code:`min`,
    the identity is :code:`+inf`, while if the reduction function performs a :code:`sum`,
    the identity is :code:`0`.)

    The :code:`body` region represents the associative and commutative operation. The
    region's block arguments must be scalar (rank-0) tiles whose element types match the
    corresponding input tiles. The body must accept :code:`2N` arguments, where :code:`N`
    is the number of input tiles. Region arguments are bound in operand order as
    :code:`[op_0_lhs, op_0_rhs, op_1_lhs, op_1_rhs, ...]`, so each pair
    :code:`(op_i_lhs, op_i_rhs)` corresponds to the :code:`i`-th input tile. Because the
    operation is associative and commutative, either argument may be an input element of
    the corresponding tile, the identity value for that tile, or a partial reduction
    result over a subset of indices along the specified :code:`dim`. The body
    should yield the combined value for each input pair.

    .. note::

      Associativity and commutativity of the operation permits the compiler to reorganize the
      applications of the operation to achieve efficient parallel reductions on the GPU.

  }];


  let mlirExamples = [[{
      # cuda_tile.module @module {
      #   entry @example() {
            %input = constant <f32: 0.0> : tile<8xf32>
            %0 = reduce %input dim=0 identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<f32>
              (%lhs: tile<f32>, %rhs: tile<f32>) {
                %add_result = addf %lhs, %rhs : tile<f32>
                yield %add_result : tile<f32>
              }
      #   }
      # }
    }],
    [{
      # cuda_tile.module @module {
      #   entry @example() {
            %input = constant <f32: 0.0> : tile<8x64xf32>
            %0 = reduce %input dim=1 identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8xf32>
              (%lhs: tile<f32>, %rhs: tile<f32>) {
                %add_result = addf %lhs, %rhs : tile<f32>
                yield %add_result : tile<f32>
              }
      #   }
      # }
    }]];

  let arguments = (ins CudaTileArg<Variadic<CudaTile_TileType>, "The set of tiles to reduce.", "13.1">:$operands,
                       CudaTileArg<ConfinedAttr<I32Attr, [IntNonNegative]>, "The index of the dimension to perform reduction on.", "13.1">:$dim,
                       CudaTileArg<ArrayAttr, "The reduction identities for each operand.", "13.1">:$identities);
  let results = (outs CudaTileArg<Variadic<CudaTile_TileType>, "The set of reduced tiles.", "13.1">:$results);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $operands attr-dict ` `
    `dim` `` `=` `` $dim `identities` `` `=` `` $identities
    `:` custom<CudaTileType>(type($operands)) `->`
    custom<CudaTileType>(type($results))
    custom<ArgumentRegion>($body)
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl;
}

//===----------------------------------------------------------------------===//
// RemIOp
//===----------------------------------------------------------------------===//

def CudaTile_RemIOp : CudaTileIntegerOpDef<"remi", "13.1", [
    NoMemoryEffect, ConditionallySpeculatable,
    AllTypesMatch<["result", "lhs", "rhs"]>,
    AllShapesMatch<["result", "lhs", "rhs"]>]> {
  let summary = "Element-wise integer remainder";
  let description = !strconcat([{
    The :code:`remi` operation computes the element-wise remainder of the input tiles
    with integer element types using truncated division (rounding towards zero).
    Division by zero is undefined behavior.

    .. math::
      \text{remi}(x, y)_i = x_i - \text{trunc}(x_i / y_i) \times y_i

    If the operation is signed, the sign of the result matches the sign
    of the dividend (:code:`lhs`). For example:

    - :code:`remi(7, 3) = 1`
    - :code:`remi(7, -3) = 1`
    - :code:`remi(-7, 3) = -1`
    - :code:`remi(-7, -3) = -1`

  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The remainder after division.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let builders = [
    OpBuilder<(ins "Type":$resTy,
                   "ValueRange":$operands, "mlir::cuda_tile::Signedness":$signedness), [{
      assert(operands.size() == 2 && "expected two operands");
      return build($_builder, $_state, resTy, operands[0],
                   operands[1], signedness);
    }]>,
  ];
  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    ::mlir::Speculation::Speculatability getSpeculatability();
  }];
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def CudaTile_ReshapeOp : CudaTileCoreOpDef<"reshape", "13.1", [
    Pure, SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Reshape tile dimensions";
  let description = [{
    The :code:`reshape` operation changes the shape of the :code:`source` operand. :code:`reshape` is
    only a change in the indexing of the tile. The number of elements and element type
    must remain unchanged.

    0-d tiles (i.e., scalars) contain precisely one element and thus are the one exception
    where a 0-d tile can be reshaped to shape where the :code:`size(shape) == 1`.

    Conceptually reshaping a tile is equivalent to first creating a 1-d tile from the data of the source assuming
    a row-major layout and then converting the 1-d tile into the new shape in a row-major layout.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %cst = constant <i8: 0> : tile<i8>
          %0 = reshape %cst
              : tile<i8> -> tile<1x1x1xi8>

          %t = constant <f32: 0.0> : tile<8x2xf32>
          %1 = reshape %t
              : tile<8x2xf32> -> tile<2x2x4x1xf32>
    #   }
    # }
  }],
  [{
    # cuda_tile.module @module {
    #   entry @example() {
          %cst = constant <i32: [[0, 1, 2, 3], [4, 5, 6, 7]]>
              : tile<2x4xi32>
          %r0 = reshape %cst
        : tile<2x4xi32> -> tile<2x2x2xi32>

        // Step 1: Turn source into 1D tile. Use row-major by convention.
        // %tmp: [0, 1, 2, 3, 4, 5, 6, 7]
        %tmp = reshape %cst
            : tile<2x4xi32> -> tile<8xi32>

        // Step 2: Turn 1D tile into result tile. Use row-major by convention.
        // %r: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        %r1 =  reshape %tmp
                : tile<8xi32> -> tile<2x2x2xi32>

    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TileType, "The source tile to reshape.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_TileType, "The reshaped tile.", "13.1">:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source attr-dict
    `:` custom<CudaTileType>(type($source))
    `->` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def CudaTile_ReturnOp : CudaTileControlFlowOpDef<"return", "13.1", [
    ParentOneOf<["EntryOp", "FuncOp", "IfOp", "LoopOp"
#ifdef TILE_IR_INCLUDE_TESTS
      , "Test_FuncOp"
#endif // TILE_IR_INCLUDE_TESTS
]>, ReturnLike, Terminator]> {
  let summary = "Return value(s) from a function";
  let description = [{
    The :code:`return` operation returns control to the caller of a function.

    .. warning::
      Currently :code:`return` implements restricted return semantics, notably:

      * :ref:`op-cuda_tile.entry` operations do not produce return value(s) and thus
        :code:`return` may be used to terminate the execution of the kernel by invoking
        the operation with no operands
      * :code:`return` is not supported in :ref:`op-cuda_tile.for` operations
        (use :ref:`op-cuda_tile.loop` for early-exit semantics)

  }]
  ;

  let mlirExamples = [
  [{
    # cuda_tile.module @module {
        preview$func @foo() -> (tile<i32>, tile<f16>) {
          %0 = constant <i32: 0> : tile<i32>
          %1 = constant <f16: 0.0> : tile<f16>
          // ...
          return %0, %1 : tile<i32>, tile<f16>
        }
    # }
  }],
  [{
    # cuda_tile.module @module {
        entry @foo() {
          %0 = constant <i32: 0> : tile<i32>
          %1 = constant <f16: 0.0> : tile<f16>
          // ...
          return
        }
    # }
  }]];

  let arguments = (ins CudaTileArg<Variadic<AnyType>, "The values to return.", "13.1">:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{});
  }]>];

  let assemblyFormat = [{
    attr-dict ($operands^ `:` custom<CudaTileType>(type($operands)))?
  }];
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

def CudaTile_ScanOp : CudaTileCoreOpDef<"scan", "13.1", [
    InferTypeOpAdaptor, OpAsmOpInterface, RecursiveMemoryEffects,
    SameOperandsShape, SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "A parallel prefix sum operation";

  let description = [{
    The :code:`scan` operation computes an inclusive parallel prefix along a given
    dimension of the input tiles using an associative and commutative function
    together with an identity value.

    The :code:`scan` operation applies a scan function over tile elements of a given
    type. It operates on :code:`operands` and :code:`identities` across the specified
    :code:`dim`, producing new :code:`results` tile values. The operation assumes an
    associative and commutative combine function, allowing implementations to choose
    any valid evaluation strategy, including parallel implementations. As a result,
    the exact evaluation order is implementation-defined, but the result remains
    deterministic across runs of the same kernel on the same device.

    .. note::

      Only memory-effect-free operations are allowed in the body of :code:`scan`.

    .. math::
      \text{scan}(X, \text{dim}, \text{identity}, f)_{i_1,\ldots,i_d}[j] \;=\;
      \text{fold}\!\left(f, \text{identity},
        \left(X_{i_1,\ldots,i_{\text{dim}-1}, 0, i_{\text{dim}+1},\ldots,i_d}, \ldots,
              X_{i_1,\ldots,i_{\text{dim}-1}, j, i_{\text{dim}+1},\ldots,i_d}\right)\right)

    The scan preserves all intermediate accumulator values:

    .. math::
      \text{result}[0] \;=\; f(\text{identity}, X[\ldots, 0, \ldots]) \\
      \text{result}[1] \;=\; f(\text{result}[0], X[\ldots, 1, \ldots]) \\
      \vdots \\
      \text{result}[j] \;=\; f(\text{result}[j-1], X[\ldots, j, \ldots])

    When :code:`reverse` is :code:`true`, the prefix is taken in decreasing index order.
    Let :math:`N` be the size of the scanned dimension; then:

    .. math::
      \text{scan}_{\text{rev}}(X)[j] \;=\;\
      \text{fold}\!\left(f, \text{identity},
        \left(X[\ldots, N\!-\!1,\ldots], \ldots, X[\ldots, j,\ldots]\right)\right)

    The :code:`identities` attribute is a list of identity elements for each input
    tile; the identity at position :code:`i` binds with the operand tile at the same
    position. The correct identity is a property of the scan function in the :code:`body`
    (e.g., :code:`sum` uses 0, :code:`prod` uses 1, :code:`min` uses +inf, :code:`max` uses -inf).

    The :code:`body` region represents the associative and commutative operation. The
    region's block arguments must be scalar (rank-0) tiles whose element types match the
    corresponding input tiles. The body must accept :code:`2N` arguments, where :code:`N`
    is the number of input tiles. Region arguments are bound in operand order as
    :code:`[op_0_lhs, op_0_rhs, op_1_lhs, op_1_rhs, ...]`, so each pair
    :code:`(op_i_lhs, op_i_rhs)` corresponds to the :code:`i`-th input tile. Because the
    operation is associative and commutative, either argument may be an input element of
    the corresponding tile, the identity value for that tile, or a partial scan result
    over a contiguous range of indices along :code:`dim`. The body should yield the
    combined value for each input tile.

    .. note::

      Associativity and commutativity of the operation permits the compiler to reorganize the
      applications of the operation to achieve efficient parallel prefix scans on the GPU.

    .. warning::

      The `scan` operation is restricted to only support single tile input.
  }];

  let mlirExamples = [[{
   # cuda_tile.module @module {
     # entry @example() {
        %input = constant <f32: 0.0> : tile<8x16xf32>
        %result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
        (%lhs: tile<f32>, %rhs: tile<f32>) {
          %prod = mulf %lhs, %rhs rounding<nearest_even>: tile<f32>
          yield %prod : tile<f32>
        }
      # }
     # }
  }]];

  let arguments = (ins CudaTileArg<Variadic<CudaTile_TileType>, "The a set of tiles to scan.", "13.1">:$operands,
                       CudaTileArg<ConfinedAttr<I32Attr, [IntNonNegative]>, "The index of the dimension along which to scan.", "13.1">:$dim,
                       CudaTileArg<BoolAttr, "Whether to scan in reverse order.", "13.1">:$reverse,
                       CudaTileArg<ArrayAttr, "The identities of the scan operation.", "13.1">:$identities);
  let results = (outs CudaTileArg<Variadic<CudaTile_TileType>, "The resulting tiles from the scan operation.", "13.1">:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $operands attr-dict ` `
    `dim` `` `=` `` $dim `reverse` `` `=` `` $reverse `identities` `` `=` `` $identities
    `:` custom<CudaTileType>(type($operands))
    `->` custom<CudaTileType>(type($results))
    custom<ArgumentRegion>($body)
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl;
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def CudaTile_SelectOp : CudaTileCoreOpDef<"select", "13.1",
    [Pure,
    AllTypesMatch<["val_if_true", "val_if_false", "result"]>,
    AllShapesMatch<["cond", "val_if_true", "val_if_false", "result"]>]> {
  let summary = "Select values based on condition";
  let description = [{
    The :code:`select` op chooses values based on the binary conditions supplied as
    the :code:`cond` operand. The :code:`val_if_true` operand contains the value(s) to use
    if the condition is 1. The :code:`val_if_false` operand contains the value(s) to
    use if the condition is 0. The choice is made element-wise according to the
    values in the condition tile.

    .. math::
      \text{select}(\text{cond}, x, y)_i = \begin{cases}
        x_i & \text{if } \text{cond}_i = 1 \\
        y_i & \text{if } \text{cond}_i = 0
      \end{cases}

    All tiles must have the same shape. The tiles :code:`val_if_true`,
    :code:`val_if_false`, and the result must have the same element type;
    sub-byte element types are not supported. The :code:`cond` tile must be a
    tile of :code:`i1` values.
  }];

  let arguments = (ins
    CudaTileArg<CudaTile_TileOf<[CudaTile_Int1]>, "The condition tile.", "13.1">:$cond,
    CudaTileArg<CudaTile_TileType, "The value if true tile.", "13.1">:$val_if_true,
    CudaTileArg<CudaTile_TileType, "The value if false tile.", "13.1">:$val_if_false);
  let results = (outs CudaTileArg<CudaTile_TileType, "The tile of selected values.", "13.1">:$result);
  let assemblyFormat = [{
    $cond `,` $val_if_true `,` $val_if_false attr-dict `:`
    custom<CudaTileType>(type($cond)) `,` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

// Supported types for ShLIOp and ShRIOp.
def CudaTile_ShLIOp : CudaTileIntegerOpDef<"shli", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise shift-left";
  let description = !strconcat([{
    The :code:`shli` operation computes the element-wise left shift of the :code:`lhs` integer operand by
    the :code:`rhs` operand. The lower-order bits on the right are filled with zeros.

    .. math::
      \text{shli}(x, y)_i = x_i \ll y_i

    The :code:`rhs` operand is interpreted as an unsigned integer.
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand (shift amount).", "13.1">:$rhs,
                       CudaTileArg<DefaultValuedAttr<CudaTile_IntegerOverflowAttr, "::mlir::cuda_tile::IntegerOverflow::NONE">, overflow_desc, "13.1">:$overflow);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The result of the left shift operation.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`overflow` `` $overflow^)? attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// ShRIOp
//===----------------------------------------------------------------------===//

def CudaTile_ShRIOp : CudaTileIntegerOpDef<"shri", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise shift-right";
  let description = !strconcat([{
    The :code:`shri` operation computes the element-wise right shift of the :code:`lhs` integer operand by
    the value of the :code:`rhs` operand for tiles with integer element types.

    .. math::
      \text{shri}(x, y)_i = x_i \gg y_i

    When :code:`unsigned`, higher-order bits
    are zero-filled; when :code:`signed`, the higher-order bits are filled with
    the sign bit.

    The :code:`rhs` operand is always interpreted as an unsigned integer.
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand (shift amount).", "13.1">:$rhs,
                       CudaTileArg<CudaTile_SignednessAttr, signed_attr_desc, "13.1">:$signedness);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The result of the right shift operation.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let builders = [
    OpBuilder<(ins "Type":$resTy,
                   "ValueRange":$operands, "mlir::cuda_tile::Signedness":$signedness), [{
      assert(operands.size() == 2 && "expected two operands");
      return build($_builder, $_state, resTy, operands[0],
                   operands[1], signedness);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

def CudaTile_SinOp : CudaTileFloatingPointOpDef<"sin", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise sine";
  let description = !strconcat([{
    The :code:`sin` operation computes the element-wise sine of the input floating-point tile.

    .. math::

      \text{sin}(x)_i = \sin(x_i)
  }], floating_point_math_suffix);

  let arguments = (ins
    CudaTileArg<CudaTile_BaseFloatTileType, "The input float tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The sine of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_sin() {
        %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
        %res = sin %in : tile<4xf32>
      # }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// SinHOp
//===----------------------------------------------------------------------===//

def CudaTile_SinHOp : CudaTileFloatingPointOpDef<"sinh", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise hyperbolic sine";
  let description = !strconcat([{
    The :code:`sinh` operation computes the element-wise hyperbolic sine of the input
    floating-point tile.

    .. math::

      \text{sinh}(x)_i = \sinh(x_i)
  }], floating_point_math_suffix);

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input float tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The hyperbolic sine of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// StoreOpBase (abstract)
//===----------------------------------------------------------------------===//

def StoreOpBaseDoc {
  string summary =
      "Store and scatter data from pointer of tile to global memory";
  string description = [{
    The :code:`store` operation performs a scatter by storing a tile of data
    from a tile into global memory.

    The :code:`destination` operand is a tile of pointers indicating the global
    memory locations where data from the :code:`value` tile will be stored. Each
    pointer must be aligned to the element type byte size. Even stronger
    alignment guarantees can be specified with the :code:`cuda_tile.assume`
    operation.

    When storing i1 values, each value occupies a full byte in memory. Any
    nonzero byte is canonicalized to 0x01, and zero bytes become 0x00.
    Direct pointer-tile stores do not support :code:`f4E2M1FN`. Use a tile view
    to store f4 memory. Alternatively, convert the f4 values with
    :code:`cuda_tile.pack` and store the resulting :code:`i8` bytes.

    Additionally, the operation supports an optional :code:`mask` operand, which allows
    selective scattering of elements. If provided, only the elements specified by
    the :code:`mask` are stored. The shape of the :code:`mask` must align with the shape of
    the :code:`value` tile.
  }];
}

class CudaTile_StoreOpBase<string mnemonic, string version,
                           list<Trait> traits = []>
    : CudaTileMemOpDef<
          mnemonic, version,
          traits#[TypesMatchWith<
                      "`destination` type is expected a pointer type of `value` type",
                      "value", "destination", "$_self",
                      "mlir::OpTrait::cuda_tile::impl::verifyLoadStoreType">,
                  OptionalTypesMatchWith<
                      "shape of 'destination' must match the shape of 'mask'",
                      "mask", "destination", "$_self",
                      "mlir::OpTrait::cuda_tile::impl::verifyLoadStoreMask">]> {}

//===----------------------------------------------------------------------===//
// StorePtrTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_StorePtrTkoOp
    : CudaTile_StoreOpBase<"store_ptr_tko",
                           "13.1", [AttrSizedOperandSegments]> {
  let summary =
      !strconcat(StoreOpBaseDoc.summary, " without ordering guarantees");
  let description = !strconcat(StoreOpBaseDoc.description, [{
    When :code:`memory_ordering_semantics` is :code:`weak`,
    :code:`memory_scope` must be omitted; when it is :code:`relaxed` or :code:`release`,
    :code:`memory_scope` is required.
  }]);

  let arguments = (ins
      CudaTileArg<
        CudaTile_MemoryOrderingSemanticsAttr,
        "The memory ordering semantics.",
        "13.1",
        [OnlyVariants<["WEAK", "RELAXED", "RELEASE"]>]>:$memory_ordering_semantics,
      CudaTileArg<OptionalAttr<CudaTile_MemoryScopeAttr>, "The optional memory scope.", "13.1">:$memory_scope,
      CudaTileArg<CudaTile_PointerTileType, "The destination pointer tile.", "13.1">:$destination,
      CudaTileArg<CudaTile_TileType, "The value tile to store.", "13.1">:$value,
      CudaTileArg<Optional<CudaTile_TileOf<[CudaTile_Int1]>>, "The optional mask for selective storage.", "13.1">:$mask,
      CudaTileArg<Optional<CudaTile_TokenType>, token_desc, "13.1">:$token,
      CudaTileArg<OptionalAttr<CudaTile_OptimizationHintsAttr>, "Optimization hints for operation", "13.1">:$optimization_hints);

  let results = (outs CudaTileArg<CudaTile_TokenType, "The result token for synchronization.", "13.1">:$result_token);

  let assemblyFormat = [{
    $memory_ordering_semantics
    ($memory_scope^)?
    $destination `,` $value
    (`,` $mask^)? (`token` `` `=` `` $token^)?
    (`optimization_hints` `=` $optimization_hints^)?
    attr-dict `:`
    custom<CudaTileType>(type($destination)) `,` custom<CudaTileType>(type($value))
    (`,` custom<CudaTileType>(type($mask))^)?
    `->` custom<CudaTileType>(type($result_token))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StoreViewTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_StoreViewTkoOp : CudaTileViewOpDef<"store_view_tko", "13.1",
    [AttrSizedOperandSegments]> {
  let summary = "Stores a tile into a tile view";
  let description = [{
    The :code:`store_view_tko` operation stores a tile to a view indexing into a
    tile view.

    A view is mapping from view-space indices to a particular element in the view, each
    view type has a defined mapping from view-space indices to tiles produced from elements
    of the view.

    For example, the :ref:`type-partition_view` partitions a :ref:`type-tensor_view` into
    a grid of equally sized tiles. The view indexes one of the partitioned tiles in the grid.

    For a given view the rank of the indices must match the rank of the view's index
    space. The space of valid indices depends on which view is passed to the operation.
    For example the index space of a :ref:`type-partition_view` is equal to the
    rank of the partitioned tiles.

    The index space of the view is computed a function of the requested tile
    size and the shape of the view.

    The :code:`index` operands are interpreted as unsigned integers.

    The :code:`inbounds` attr lets the user (or a preceding analysis) declare
    that selected tile-coordinate dimensions are statically known to address
    only valid memory. It is required and must contain exactly one bool per
    index dimension; it is interpreted element-wise:

    - element :code:`i = true` is a guarantee that the i-th tile coordinate
      stays inside the parent tensor along dimension :code:`i`. Violating the
      guarantee is undefined behavior.
    - element :code:`i = false` carries no such guarantee: the i-th tile coordinate
      is conservatively treated as potentially out-of-bounds and handled
      according to the semantics of the concrete view type
      (see for example :ref:`type-partition_view`).

    In textual MLIR the :code:`inbounds = [...]` clause may be omitted, in
    which case the parser synthesizes an all-:code:`false` vector sized to
    the index rank (every dimension conservatively treated as potentially
    out-of-bounds).

    When :code:`memory_ordering_semantics` is :code:`weak`, :code:`memory_scope` must be
    omitted; when it is :code:`relaxed` or :code:`release`, :code:`memory_scope` is required.

    .. note::

      The pointer of the underlying :code:`tensor_view` must be aligned to the
      element type byte size (rounded to the next full byte for
      sub-byte types). Even stronger alignment guarantees can be
      specified with the :code:`cuda_tile.assume` operation.

      An :code:`f4E2M1FN` access additionally requires the underlying tensor
      view to identify exactly one statically encoded unit-stride packing
      dimension. The tensor view and concrete tile-view types specify
      additional evenness preconditions on extents, strides, and origins. Some
      may be runtime values, while tile extents and traversal strides are
      static type attributes. If an access executes with any applicable value
      odd, behavior is undefined.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%ptr: tile<ptr<f32>>) {
          %tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128,1] :
            tensor_view<8192x128xf32, strides=[128,1]>

          // This example uses the PartitionView on a 8192x128xf32 tensor_view,
          // dividing the tensor_view in tiles of 64x64.
          %view = make_partition_view %tensor_view :
            partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

          %c0 = constant <i32: 0> : tile<i32>
          %c1 = constant <i32: 1> : tile<i32>

          %tile = constant <f32: 0.0> : tile<64x64xf32>

          // Store a tile at index (0, 0) in the view's index space.
          // For this TilePartitionView, this is the rectangular tile such that
          // X=[0,64) and Y=[0,64), in the coordinates of tiles.
          %res_token0 = store_view_tko weak %tile, %view[%c0, %c0]
            : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

          // Store a tile at index (0, 1) in the view's index space.
          // For this PartitionView, this is the rectangular tile such that
          // X=[0,64) and Y=[64,128), in the coordinates of tiles.
          %res_token1 = store_view_tko weak %tile, %view[%c0, %c1]
            : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

          // Same example as above but with input token.
          %token = make_token : token
          %res_token2 = store_view_tko weak %tile, %view[%c0, %c1] token = %token
            : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token
        # }
      # }
  }],
  [{
    # cuda_tile.module @module {
    #   entry @example_scatter(%ptr: tile<ptr<f32>>) {
          // Create a tensor view
          %tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
            : tensor_view<8192x128xf32, strides=[128,1]>

          // Create a gather/scatter view with tile size (64x64) along dimension 0
          %gsview = make_gather_scatter_view %tensor_view :
            gather_scatter_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>, sparse_dim=0>

          // Tile to store
          %tile = constant <f32: 1.0> : tile<64x64xf32>

          // The 1D tensor index MUST be at position `sparse_dim` (here sparse_dim=0)
          // Index at sparse_dim=0: 1D tensor containing scatter indices
          %scatter_indices = constant <i32: [5, 15, 25, 35, 45, 55, 65, 75]> : tile<8xi32>
          // Index at sparse_dim=1: scalar for the other dimension
          %col_idx = constant <i32: 1> : tile<i32>

          // Store tiles using scatter semantics
          // Index order: [%scatter_indices (at sparse_dim=0), %col_idx (at sparse_dim=1)]
          %res_token0 = store_view_tko weak %tile, %gsview[%scatter_indices, %col_idx]
            : tile<64x64xf32>, gather_scatter_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>, sparse_dim=0>, tile<8xi32>, tile<i32>
            -> token

          // Store with input token
          %token_in = make_token : token
          %res_token1 = store_view_tko weak %tile, %gsview[%scatter_indices, %col_idx] token = %token_in
            : tile<64x64xf32>, gather_scatter_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>, sparse_dim=0>, tile<8xi32>, tile<i32>
            -> token
        # }
      # }
  }]];

  let arguments = (ins
    CudaTileArg<
        CudaTile_MemoryOrderingSemanticsAttr,
        "The memory scope for the store operation.",
        "13.1",
        [OnlyVariants<["WEAK", "RELAXED", "RELEASE"]>]>:$memory_ordering_semantics,
    CudaTileArg<OptionalAttr<CudaTile_MemoryScopeAttr>, "The memory scope for the store operation.", "13.1">:$memory_scope,
    CudaTileArg<CudaTile_TileType, "The tile to store.", "13.1">:$tile,
    CudaTileArg<CudaTile_TileView, "The view to store the tile to.", "13.1">:$view,
    CudaTileArg<Variadic<CudaTile_IntTileType>, "The indices of the desired target tile within the view. For GatherScatterView, the first index can be a 1D tensor.", "13.1">:$index,
    CudaTileArg<Optional<CudaTile_TokenType>, token_desc, "13.1">:$token,
    CudaTileArg<OptionalAttr<CudaTile_OptimizationHintsAttr>, "Optimization hints for operation", "13.1">:$optimization_hints,
    CudaTileArg<DenseBoolArrayAttr, inbounds_desc, "13.4",
                [RequireSameOperandRank<"index">]>:$inbounds);

  let results = (outs CudaTileArg<CudaTile_TokenType, "The result token for synchronization.", "13.1">:$result_token);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    // Convenience builder (attr form of MemoryOrderingSemantics) that
    // synthesizes the required `inbounds` attribute as an all-`false` vector
    // sized to the index rank, so callers that do not have per-dimension
    // inbounds information do not have to construct the attribute themselves.
    OpBuilder<(ins "::mlir::Type":$result_token,
                   "::mlir::cuda_tile::MemoryOrderingSemanticsAttr":$memory_ordering_semantics,
                   "::mlir::cuda_tile::MemoryScopeAttr":$memory_scope,
                   "::mlir::Value":$tile,
                   "::mlir::Value":$view,
                   "::mlir::ValueRange":$index,
                   "::mlir::Value":$token,
                   "::mlir::cuda_tile::OptimizationHintsAttr":$optimization_hints), [{
      auto inbounds = ::mlir::DenseBoolArrayAttr::get(
          $_builder.getContext(),
          ::llvm::SmallVector<bool>(index.size(), /*value=*/false));
      build($_builder, $_state, result_token, memory_ordering_semantics,
            memory_scope, tile, view, index, token, optimization_hints, inbounds);
    }]>,
    // Same convenience builder using the enum form of MemoryOrderingSemantics.
    OpBuilder<(ins "::mlir::Type":$result_token,
                   "::mlir::cuda_tile::MemoryOrderingSemantics":$memory_ordering_semantics,
                   "::mlir::cuda_tile::MemoryScopeAttr":$memory_scope,
                   "::mlir::Value":$tile,
                   "::mlir::Value":$view,
                   "::mlir::ValueRange":$index,
                   "::mlir::Value":$token,
                   "::mlir::cuda_tile::OptimizationHintsAttr":$optimization_hints), [{
      auto semAttr = ::mlir::cuda_tile::MemoryOrderingSemanticsAttr::get(
          $_builder.getContext(), memory_ordering_semantics);
      auto inbounds = ::mlir::DenseBoolArrayAttr::get(
          $_builder.getContext(),
          ::llvm::SmallVector<bool>(index.size(), /*value=*/false));
      build($_builder, $_state, result_token, semAttr, memory_scope, tile,
            view, index, token, optimization_hints, inbounds);
    }]>,
    // Convenience builder (enum form, no explicit result_token type) for
    // callers that do not need to supply the token type; it is always
    // inferred as cuda_tile.token.
    OpBuilder<(ins "::mlir::cuda_tile::MemoryOrderingSemantics":$memory_ordering_semantics,
                   "::mlir::cuda_tile::MemoryScopeAttr":$memory_scope,
                   "::mlir::Value":$tile,
                   "::mlir::Value":$view,
                   "::mlir::ValueRange":$index,
                   "::mlir::Value":$token,
                   "::mlir::cuda_tile::OptimizationHintsAttr":$optimization_hints), [{
      auto resultTokenType =
          ::mlir::cuda_tile::TokenType::get($_builder.getContext());
      auto semAttr = ::mlir::cuda_tile::MemoryOrderingSemanticsAttr::get(
          $_builder.getContext(), memory_ordering_semantics);
      auto inbounds = ::mlir::DenseBoolArrayAttr::get(
          $_builder.getContext(),
          ::llvm::SmallVector<bool>(index.size(), /*value=*/false));
      build($_builder, $_state, resultTokenType, semAttr, memory_scope, tile,
            view, index, token, optimization_hints, inbounds);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

def CudaTile_SubFOp : CudaTileFloatingPointOpDef<"subf", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise floating-point subtraction";
  let description = !strconcat([{
    The :code:`subf` operation computes the element-wise subtraction of the input floating-point tiles.

    .. math::
      \text{subf}(x, y)_i = x_i - y_i
  }], floating_point_arith_suffix);

  let descriptionTables = [
    Table<":code:`subf` Modifiers", "The below table shows the supported modifiers and rounding modes for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["flush_to_zero", "yes", "no", "no", "no"]>,
       TableRow<["rounding<nearest_even>", "yes", "yes", "yes", "yes"]>,
       TableRow<["rounding<zero>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<negative_inf>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<positive_inf>", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_BaseFloatTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                                   [OnlyCudaTileIEEERoundingModes]>:$rounding_mode,
                       CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);


  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The result of the subtraction.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs
    custom<IEEERoundingMode>($rounding_mode)
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def CudaTile_SubIOp : CudaTileIntegerOpDef<"subi", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise integer subtraction";
  let description = !strconcat([{
    The :code:`subi` operation computes the element-wise subtraction of two input integer tiles.

    .. math::
      \text{subi}(x, y)_i = x_i - y_i
  }], integer_arith_suffix);


  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs,
                       CudaTileArg<DefaultValuedAttr<CudaTile_IntegerOverflowAttr, "::mlir::cuda_tile::IntegerOverflow::NONE">, overflow_desc, "13.1">:$overflow);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The result of the subtraction.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`overflow` `` $overflow^)? attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// TanOp
//===----------------------------------------------------------------------===//

def CudaTile_TanOp : CudaTileFloatingPointOpDef<"tan", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise tangent";
  let description = !strconcat([{
    The :code:`tan` operation computes the element-wise tangent of
    the input floating-point tile.

    .. math::

      \text{tan}(x)_i = \tan(x_i)
  }], floating_point_math_suffix);

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input floating-point tile.", "13.1">:$source);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The tangent of the input floating-point tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// TanHOp
//===----------------------------------------------------------------------===//

def CudaTile_TanHOp : CudaTileFloatingPointOpDef<"tanh", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise hyperbolic tangent";
  let description = !strconcat([{
    The :code:`tanh` operation computes the element-wise hyperbolic tangent of the
    input floating-point tile. Default rounding mode is `full`.

    The :code:`approx` rounding mode implements a fast approximation to hyperbolic tangent.
    Subnormal results of this fast approximation are not flushed to zero.

    The :code:`full` rounding mode implements a relatively fast full-range approximation.
    The maximum ulp error is 2 across the full range of inputs in FP32 and 1 in FP64.

    .. math::

      \text{tanh}(x)_i = \tanh(x_i)
  }], floating_point_math_suffix);

  let descriptionTables = [
    Table<":code:`tanh` Modifiers", "The below table shows the supported modifiers for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["approx", "yes", "no", "no", "no"]>,
       TableRow<["full", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments = (ins
    CudaTileArg<CudaTile_BaseFloatTileType, "The input floating-point tile.", "13.1">:$source,
    CudaTileArg<DefaultValuedAttr<CudaTile_RoundingModeAttr, "RoundingMode::FULL">, rounding_mode_desc, "13.2",
                [OnlyCudaTileApproxFullRoundingModes]>:$rounding_mode);

  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The hyperbolic tangent of the input floating-point tile.", "13.1">:$result);

  let assemblyFormat = [{
    $source
    custom<TanHOpRoundingMode>($rounding_mode)
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_tanh() {
        %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
        %res0 = tanh %in : tile<4xf32>

        // tanh with approx modifier
        %res1 = tanh %in rounding<approx> : tile<4xf32>
      # }
    # }
  }]];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnpackOp
//===----------------------------------------------------------------------===//

// NumberTileType + Int4
def UnpackOp_UnpackedType : CudaTile_TileOf<
  !listconcat(CudaTile_NumberTileType.allowedElementTypes, [CudaTile_Int4])>;

def CudaTile_UnpackOp : CudaTileCoreOpDef<"unpack", "13.3", [
    Pure, AllRanksMatch<["source", "result"]>]> {
  let summary = "Unpack a byte array into a tile";
  let description = [{
    The :code:`unpack` operation takes a rank-1 :code:`tile<i8>` and produces a rank-1 numeric tile.

    Similar to :code:`bitcast`, underlying bit-values are not changed. However, :code:`unpack` does not
    operate elementwise, instead reinterpreting the entire tile as a numeric tile of different element type.

    Input and output tiles must be rank-1 to eliminate unpacking ambiguity. The size of the input
    tile must match the number of bytes in the output tile. The output tile cannot be an 8-bit type
    (bitcast should be used instead).
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %arg0 = constant <i8: 0> : tile<64xi8>
          %0 = unpack %arg0 : tile<64xi8> -> tile<32xf16>
    #   }
    # }
  }], [{
    # cuda_tile.module @module {
    #   entry @example() {
          %arg0 = constant <i8: 0> : tile<64xi8>
          %0 = unpack %arg0 : tile<64xi8> -> tile<128xF4E2M1FN>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TileOf<[CudaTile_Int8]>, "The input i8 tile.", "13.3">:$source);
  let results = (outs CudaTileArg<UnpackOp_UnpackedType, "The output unpacked tile.", "13.3">:$result);
  let assemblyFormat = [{
    $source attr-dict `:` custom<CudaTileType>(type($source)) `->` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MakePartitionViewOp
//===----------------------------------------------------------------------===//

def CudaTile_MakePartitionViewOp
      : CudaTileViewOpDef<"make_partition_view", "13.1",
    [Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a partition view from a tensor view";
  let description = [{
    The :code:`make_partition_view` operation creates a :tileirty:`partition_view` from a
    :tileirty:`tensor_view`. For more details about partition views see :ref:`type-partition_view`.

    The operation uses the type constraints of the input tensor view and the annotated return type
    to perform the partitioning, so the type of the :code:`tensor_view` operand must exactly match
    the :code:`tensor_view<...>` embedded in the result :code:`partition_view<...>`. The tensor
    view's type contains its physical layout in the form of shapes and strides and the partition
    view contains the logical size of a single tile.

    The resulting partition view can be loaded from using :ref:`op-cuda_tile.load_view_tko` and
    stored to using :ref:`op-cuda_tile.store_view_tko`.

    The view memory options act on the computed index space of the partition view, see
    :ref:`type-tensor_view` and :ref:`type-partition_view` for detailed semantics.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%ptr: tile<ptr<f32>>) {

          %tensor_view0 = make_tensor_view %ptr, shape=[8192, 8192, 64], strides=[524288,64,1]
            : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>

          // Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
          // the provided tensor_view.
          make_partition_view %tensor_view0 :
            partition_view<
              tile=(1024x1x32),
              tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
            >

          %s0 = constant <i32: 8192> : tile<i32>
          %str0 = constant <i32: 524288> : tile<i32>

          %tensor_view1 = make_tensor_view %ptr, shape=[%s0, 8192, 64], strides=[%str0, 64, 1]
            : tile<i32> -> tensor_view<?x8192x64xf32, strides=[?,64,1]>

          // Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
          // the provided tensor_view. The provided tensor_view has a
          // dynamically-sized dimension.
          make_partition_view %tensor_view1 :
            partition_view<tile=(1024x1x32), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TensorViewType, "The source tensor view to create a partition view from.", "13.1">:$tensor_view);
  let results = (outs CudaTileArg<CudaTile_PartitionViewType, "The created partition view.", "13.1">:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MakeStridedViewOp
//===----------------------------------------------------------------------===//

def CudaTile_MakeStridedViewOp
      : CudaTileViewOpDef<"make_strided_view", "13.3",
    [Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a strided view from a tensor view";
  let description = [{
    The :code:`make_strided_view` operation creates a :tileirty:`strided_view` from a
    :tileirty:`tensor_view`. For more details about strided views see :ref:`type-strided_view`.

    The operation uses the type constraints of the input tensor view and the annotated return type
    to perform the tiling, so the type of the :code:`tensor_view` operand must exactly match the
    :code:`tensor_view<...>` embedded in the result :code:`strided_view<...>`. The tensor view's
    type contains its physical layout in the form of shapes and strides and the strided view
    contains the logical size of a single tile, along with the traversal striding factors.

    The resulting strided view can be loaded from using :ref:`op-cuda_tile.load_view_tko` and
    stored to using :ref:`op-cuda_tile.store_view_tko`.

    The view memory options act on the computed index space of the strided view, see
    :ref:`type-tensor_view` and :ref:`type-strided_view` for detailed semantics.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%ptr: tile<ptr<f32>>) {

          %tensor_view0 = make_tensor_view %ptr, shape=[8192, 8192, 64], strides=[524288,64,1]
            : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>

          // Creates a strided view with 32-bit-indexed tiles of size (1024x1x32) and
          // a traversal of strides [512,1,10] over the provided tensor_view.
          make_strided_view %tensor_view0 :
            strided_view<
              tile=(1024x1x32),
              traversal_strides=[512,1,10],
              tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
            >

          %s0 = constant <i32: 8192> : tile<i32>
          %str0 = constant <i32: 524288> : tile<i32>

          %tensor_view1 = make_tensor_view %ptr, shape=[%s0, 8192, 64], strides=[%str0, 64, 1]
            : tile<i32> -> tensor_view<?x8192x64xf32, strides=[?,64,1]>

          // Creates a strided view with 32-bit-indexed tiles of size (1024x1x32) and
          // a traversal of strides [512,1,10] over the provided tensor_view. The
          // provided tensor_view has a dynamically-sized dimension.
          make_strided_view %tensor_view1 :
            strided_view<
              tile=(1024x1x32),
              traversal_strides=[512,1,10],
              tensor_view<?x8192x64xf32, strides=[?,64,1]>
            >
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TensorViewType, "The source tensor view to create a strided view from.", "13.3">:$tensor_view);
  let results = (outs CudaTileArg<CudaTile_StridedViewType, "The created strided view.", "13.3">:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MakeGatherScatterViewOp
//===----------------------------------------------------------------------===//

def CudaTile_MakeGatherScatterViewOp
      : CudaTileViewOpDef<"make_gather_scatter_view", "13.3",
    [Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a gather/scatter view from a tensor view";
  let description = [{
    The :code:`make_gather_scatter_view` operation creates a :tileirty:`gather_scatter_view` from a
    :tileirty:`tensor_view`. For more details about gather/scatter views see :ref:`type-gather_scatter_view`.

    The operation uses the type constraints of the input tensor view and the annotated return type
    to perform the gathering/scattering, so the type of the :code:`tensor_view` operand must
    exactly match the :code:`tensor_view<...>` embedded in the result
    :code:`gather_scatter_view<...>`. The tensor view's type contains its physical layout in the
    form of shapes and strides and the gather/scatter view contains the logical size of a single
    tile and the dimension along which to gather/scatter.

    The resulting gather/scatter view can be loaded from using :ref:`op-cuda_tile.load_view_tko` and
    stored to using :ref:`op-cuda_tile.store_view_tko`.

    The view memory options act on the computed index space of the gather/scatter view see
    :ref:`type-tensor_view` and :ref:`type-gather_scatter_view` for detailed semantics.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example(%ptr: tile<ptr<f32>>) {

          %tensor_view0 = make_tensor_view %ptr, shape=[8192, 8192], strides=[8192, 1]
            : tensor_view<8192x8192xf32, strides=[8192, 1]>

          // Creates a gather/scatter view with tiles of size (128x128) over
          // the provided tensor_view, gathering/scattering along dimension 0.
          make_gather_scatter_view %tensor_view0 :
            gather_scatter_view<
              tile=(128x128),
              tensor_view<8192x8192xf32, strides=[8192, 1]>,
              sparse_dim=0
            >

    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_TensorViewType, "The source tensor view to create a gather/scatter view from.", "13.1">:$tensor_view);
  let results = (outs CudaTileArg<CudaTile_GatherScatterViewType, "The created gather/scatter view.", "13.3">:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

def CudaTile_XOrIOp : CudaTileBitwiseOpDef<"xori", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise bitwise XOR";
  let description = !strconcat([{
    The :code:`xori` operation computes the element-wise bitwise exclusive or (XOR)
    of two tile values with integer element types.

    .. math::
      \text{xori}(x, y)_i = x_i \oplus y_i
  }], integer_arith_suffix);

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
          %rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
          // This computes the bitwise XOR of each element in `%lhs` and `%rhs`, which
          // are tiles of shape `4xi32`, and returns the result as `%result`.
          %result = xori %lhs, %rhs : tile<4xi32>
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The bitwise XOR of the input tiles.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def CudaTile_YieldOp : CudaTileControlFlowOpDef<"yield", "13.1", [
    Pure, ReturnLike, Terminator, ParentOneOf<[
      "IfOp", "ReduceOp", "ScanOp"
  ]>]> {
  let summary = "Yield a value from the block";

  let description = [{
    The :code:`yield` operation terminates a block that must yield control back to the parent operation
    such as :code:`if`, :code:`scan`, :code:`reduce`.

    The operation may yield any number of :code:`$operands` to the parent upon termination. The number of values yielded
    and the execution semantics of how they are yielded are determined by the parent operation.

    .. note::

      Unlike standard MLIR control flow dialects :code:`yield` is not used for loop control flow, see
      :ref:`op-cuda_tile.break`, :ref:`op-cuda_tile.continue`, and :ref:`op-cuda_tile.return` for loop control flow.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          %condition = constant <i1: true> : tile<i1>
          // Yield from the body of an if conditional.
          if %condition  {
              yield
          }

          // Yield values from within an if conditional.
          %x, %y = if %condition -> (tile<f32>, tile<f32>) {
              %x_then = constant <f32: 0.0> : tile<f32>
              %y_then = constant <f32: 1.0> : tile<f32>
              yield %x_then, %y_then : tile<f32>, tile<f32>
          } else {
              %x_else = constant <f32: 2.0> : tile<f32>
              %y_else = constant <f32: 3.0> : tile<f32>
              yield %x_else, %y_else : tile<f32>, tile<f32>
          }
    #   }
    # }
  }]];

  let arguments = (ins CudaTileArg<Variadic<AnyType>, "The operands to yield to the parent operation.", "13.1">:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict ($operands^ `:` custom<CudaTileType>(type($operands)))?
  }];
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

def CudaTile_OrIOp : CudaTileBitwiseOpDef<"ori", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise bitwise OR";
  let description = !strconcat([{
    The :code:`ori` operation computes the element-wise bitwise OR of two tiles with
    integer element types.

    .. math::
      \text{ori}(x, y)_i = x_i | y_i
  }], integer_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_IntTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_IntTileType, "The right hand side operand.", "13.1">:$rhs);
  let results = (outs CudaTileArg<CudaTile_IntTileType, "The bitwise OR of the input tiles.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// RemFOp
//===----------------------------------------------------------------------===//

def CudaTile_RemFOp : CudaTileFloatingPointOpDef<"remf", "13.1",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Element-wise floating-point remainder";
  let description = !strconcat([{
    The :code:`remf` operation computes the element-wise floating-point remainder using
    truncated division (rounding towards zero).

    .. math::
      \text{remf}(x, y)_i = x_i - \text{trunc}(x_i / y_i) \times y_i

    The result has the same sign as the dividend (:code:`lhs`) and its magnitude is
    less than the magnitude of divisor (:code:`rhs`).
    
    .. note::
      "Remainder" and "modulus" are used inconsistently across different languages.
      Tile IR's :code:`remf` matches the semantics of LLVM's :code:`frem`
      and C's :code:`fmodf`, which does *not* match C :code:`remainderf`.

    **Special cases:**

    - If :code:`x` is :code:`+0` or :code:`-0` and :code:`y` is nonzero, returns :code:`x`
    - If :code:`x` is finite and :code:`y` is infinite, returns :code:`x`
    - If :code:`x` is infinite or :code:`y` is zero, returns :code:`NaN`
    - If either argument is :code:`NaN`, returns :code:`NaN`
  }], floating_point_arith_suffix);

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The left hand side operand.", "13.1">:$lhs,
                       CudaTileArg<CudaTile_BaseFloatTileType, "The right hand side operand.", "13.1">:$rhs);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The remainder after division.", "13.1">:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<CudaTileType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

def CudaTile_RsqrtOp : CudaTileFloatingPointOpDef<"rsqrt", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise reciprocal square root";
  let description = !strconcat([{
    The :code:`rsqrt` operation computes the element-wise reciprocal square root
    of the input floating-point tile.

    .. math::

      \text{rsqrt}(x)_i = \frac{1}{\sqrt{x_i}}
  }], floating_point_math_suffix);

  let descriptionTables = [
    Table<":code:`rsqrt` Modifiers", "The below table shows the supported modifiers for each data type.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">],
      [TableRow<["flush_to_zero", "yes", "no"]>]
    >
  ];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input tile to compute the reciprocal square root of.", "13.1">:$source,
                       CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The reciprocal square root of the input tile.", "13.1">:$result);

  let assemblyFormat = [{
    $source
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @ex_module {
      # entry @example_rsqrt() {
        %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
        %res = rsqrt %in : tile<4xf32>

        // Rsqrt op with flush to zero modifier
        %ftz_res = rsqrt %in flush_to_zero : tile<4xf32>
      # }
    # }
  }]];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

def CudaTile_SqrtOp : CudaTileFloatingPointOpDef<"sqrt", "13.1",
    [Pure, AllTypesMatch<["source", "result"]>]> {
  let summary = "Element-wise square root";
  let description = [{
    The :code:`sqrt` operation computes the element-wise square root of a floating-point tile.

    .. math::

      \text{sqrt}(x)_i = \sqrt{x_i}
  }];

  let descriptionTables = [
    Table<":code:`sqrt` Modifiers", "The below table shows the supported modifiers and rounding modes for each data type. Entries with '*' are emulated in f32.",
      [TableHeader<"Modifier", "code">, TableHeader<"Float32">, TableHeader<"Float64">, TableHeader<"BFloat16">, TableHeader<"Float16">],
      [TableRow<["flush_to_zero", "yes", "no", "no", "no"]>,
       TableRow<["approx", "yes", "no", "no", "no"]>,
       TableRow<["rounding<nearest_even>", "yes", "yes", "yes", "yes"]>,
       TableRow<["rounding<zero>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<negative_inf>", "yes", "yes", "yes*", "yes*"]>,
       TableRow<["rounding<positive_inf>", "yes", "yes", "yes*", "yes*"]>]
    >
  ];

  let arguments = (ins CudaTileArg<CudaTile_BaseFloatTileType, "The input tile to compute the square root of.", "13.1">:$source,
      CudaTileArg<CudaTile_RoundingModeAttr, rounding_mode_desc, "13.1",
                  [OnlyCudaTileSqrtRoundingModes]>:$rounding_mode,
      CudaTileArg<UnitAttr, flush_to_zero_desc, "13.1">:$flush_to_zero);
  let results = (outs CudaTileArg<CudaTile_BaseFloatTileType, "The square root of the input tile.", "13.1">:$result);
  let assemblyFormat = [{
    $source
    custom<SqrtOpRoundingMode>($rounding_mode)
    (`flush_to_zero` $flush_to_zero^)?
    attr-dict `:` custom<CudaTileType>(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GdcLaunchDependentsTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_GdcLaunchDependentsTkoOp
    : CudaTileMiscOpDef<"gdc_launch_dependents_tko", "13.4", []> {
  let summary = "Signal dependent kernels can be scheduled (PDL)";
  let description = [{
    Signals that dependent kernels may be scheduled for Programmatic
    Dependent Launch (PDL).

    Grid-wide: every CTA in the grid must execute this operation (or
    complete via EXIT) before the dependent kernel is scheduled.

    This operation controls scheduling only and does not provide
    memory ordering guarantees.

    On SM<90 targets this operation is a no-op.

    .. note::
      PDL overlap requires the kernel to be launched with
      ``CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION``.
      Without it, this operation is a no-op.

    Pair with ``gdc_wait_tko`` in the dependent kernel.
  }];

  let arguments = (ins
    CudaTileArg<Optional<CudaTile_TokenType>,
        "Optional input token for ordering constraints.", "13.4">:$token);

  let results = (outs
    CudaTileArg<CudaTile_TokenType,
        "Result token for ordering constraints.", "13.4">:$result_token);

  let assemblyFormat = [{
    (`token` `=` $token^)? attr-dict `->` custom<CudaTileType>(type($result_token))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          // Standalone — no prior token dependency
          %t0 = gdc_launch_dependents_tko -> token

          // Chained — ordered after a prior operation
          // %prior = ...
          // %t1 = gdc_launch_dependents_tko token = %prior -> token
    #   }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// GdcWaitTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_GdcWaitTkoOp
    : CudaTileMiscOpDef<"gdc_wait_tko", "13.4", []> {
  let summary = "Wait for predecessor kernel completion (PDL)";
  let description = [{
    Waits for the predecessor kernel to complete for Programmatic
    Dependent Launch (PDL). Acquire semantics: all stores from the
    predecessor kernel are visible after this operation completes.

    On SM<90 targets this operation is a no-op.

    .. note::
      PDL overlap requires the kernel to be launched with
      ``CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION``.
      Without it, this operation is a no-op.

    Pair with ``gdc_launch_dependents_tko`` in the predecessor kernel.
  }];

  let arguments = (ins
    CudaTileArg<Optional<CudaTile_TokenType>,
        "Optional input token for ordering constraints.", "13.4">:$token);

  let results = (outs
    CudaTileArg<CudaTile_TokenType,
        "Result token for ordering constraints.", "13.4">:$result_token);

  let assemblyFormat = [{
    (`token` `=` $token^)? attr-dict `->` custom<CudaTileType>(type($result_token))
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @example() {
          // Standalone — wait for predecessor kernel
          %t0 = gdc_wait_tko -> token

          // Chained — ordered after a prior operation
          // %prior = ...
          // %t1 = gdc_wait_tko token = %prior -> token
    #   }
    # }
  }]];
}

//===----------------------------------------------------------------------===//
// MultimemLoadReduceViewTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_MultimemLoadReduceViewTkoOp : CudaTileViewOpDef<"multimem_load_reduce_view_tko",
    "13.5", [MemoryEffects<[MemRead]>, AttrSizedOperandSegments]> {
  let summary = "Load and reduce from a multimem tile view";
  let description = [{
    The :code:`multimem_load_reduce_view_tko` operation loads from the tile view :code:`$view` and
    returns a single reduced result tile using a multimem load-reduce access path.

    .. note::

      The underlying pointer for :code:`$view` must be a multicast global pointer (for
      example :code:`tile<ptr<T, #cuda_tile.ptr_attr<multicast>>>`),
      and the view must be built over that multicast-backed storage. A plain unicast
      global pointer is invalid.

      This operation requires the target GPUs to be connected through an
      NVLink SHARP-supported NVSwitch.

      Currently, compiler support is limited to the following:

      * :code:`sm_100`, :code:`sm_103`, and :code:`sm_107` targets.
      * :ref:`type-partition_view` operands.

    The :code:`$index` operands address :code:`$view` in view index space. There must be
    one index per index dimension of the view, the number of indices must match the
    view's index rank, and together they select which
    tile (or indexed region, depending on the view type) is loaded and reduced. The
    indices are interpreted as unsigned integers. Which indices are valid and how
    out-of-bounds accesses are handled are defined by the concrete view type of
    :code:`$view`.

    The :code:`$memory_ordering_semantics` attribute specifies the concurrency
    assumption between memory accesses in different threads. When it is :code:`weak`,
    the compiler may assume there are no concurrent accesses to the loaded locations and
    :code:`$memory_scope` must not be specified. When it is :code:`relaxed` or
    :code:`acquire`, :code:`$memory_scope` must be specified.

    The :code:`$reduction` attribute selects the reduction combiner.
    Supported element types per reduction kind:

    .. note::

      ================  =====  =====  =====  ======  =====  =====
      Reduction         i32    i64    f16    bf16    f32    f64
      ================  =====  =====  =====  ======  =====  =====
      ADD               ✓      ✓      ✓      ✓       ✓      ✓
      MIN (signed)      ✓      ✓      ✓      ✓       —      —
      MAX (signed)      ✓      ✓      ✓      ✓       —      —
      UMIN (unsigned)   ✓      ✓      —      —       —      —
      UMAX (unsigned)   ✓      ✓      —      —       —      —
      AND / OR / XOR    ✓      ✓      —      —       —      —
      ================  =====  =====  =====  ======  =====  =====

    The :code:`inbounds` attribute lets the user (or a preceding
    analysis) declare that selected tile-coordinate dimensions are statically
    known to address only valid memory. When set, it must contain one bool per
    index dimension; element :code:`i = true` is a guarantee that for every
    program instance the i-th tile coordinate, combined with the tile size and
    the view's traversal stride, stays inside the parent tensor along that
    dimension. Violating that guarantee is undefined behavior. When absent, all
    dimensions are conservatively treated as potentially out-of-bounds and
    handled according to the semantics of the concrete view type
    (see for example :ref:`type-partition_view`).
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @multimem_load_reduce_example(
    #       %mcptr: tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
          %tensor_view = make_tensor_view %mcptr, shape=[8192, 128],
            strides=[128, 1]
            : tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>
              -> tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>

          %view = make_partition_view %tensor_view :
            partition_view<tile=(64x64),
              tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>>

          %c0 = constant <i32: 0> : tile<i32>
          %c1 = constant <i32: 1> : tile<i32>

          // Weak load-reduce at tile index (0, 0); no memory scope (default add).
          %reduced, %tok0 = multimem_load_reduce_view_tko weak %view[%c0, %c0]
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>>,
              tile<i32>
            -> tile<64x64xf32>, token

          // Acquire load-reduce with explicit device scope and an input token.
          %tok_in = make_token : token
          %reduced2, %tok1 = multimem_load_reduce_view_tko acquire device
              %view[%c0, %c1]
            token = %tok_in
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>>,
              tile<i32>, token
            -> tile<64x64xf32>, token
    #   }
    # }
  }]];

  let arguments = (ins
    CudaTileArg<
      CudaTile_MemoryOrderingSemanticsAttr,
      "The memory ordering semantics for the multimem load-reduce operation.", "13.5",
      [OnlyVariants<["WEAK", "RELAXED", "ACQUIRE"]>]>:$memory_ordering_semantics,
    CudaTileArg<OptionalAttr<CudaTile_MemoryScopeAttr>,
      "The memory scope for the load-reduce operation.", "13.5">:$memory_scope,
    CudaTileArg<CudaTile_TileView,
      "The view from which the tile will be loaded.", "13.5">:$view,
    CudaTileArg<Variadic<CudaTile_IntTileType>,
      "The n-dimensional index of the desired tile in the view's index space.", "13.5">:$index,
    CudaTileArg<Optional<CudaTile_TokenType>,
      "Optional token for operation ordering.", "13.5">:$token,
    CudaTileArg<
        DefaultValuedAttr<CudaTile_MultimemReductionKindAttr,
                          "::mlir::cuda_tile::MultimemReductionKind::ADD">,
      "Reduction kind (add, min, max, umin, umax, and, or, xor); default add.", "13.5">:$reduction,
    CudaTileArg<DenseBoolArrayAttr, inbounds_desc, "13.5",
                [RequireSameOperandRank<"index">]>:$inbounds
  );

  let results = (outs
    CudaTileArg<CudaTile_TileType, "The reduced values.", "13.5">:$reduced_values,
    CudaTileArg<CudaTile_TokenType, "The result token.", "13.5">:$result_token
  );

  let assemblyFormat = [{
    $memory_ordering_semantics
    ($memory_scope^)?
    $view `[` $index `]`
    (`token` `=` $token^)?
    custom<Inbounds>($inbounds, ref($index))
    attr-dict-with-keyword
    `:` custom<CudaTileType>(type($view)) `,`
        custom<CudaTileTypeSplat>(type($index), ref($index))
    (`,` custom<CudaTileType>(type($token))^)? `->`
        custom<CudaTileType>(type($reduced_values)) `,`
        custom<CudaTileType>(type($result_token))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MultimemReduceViewTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_MultimemReduceViewTkoOp : CudaTileViewOpDef<"multimem_reduce_view_tko",
    "13.5", [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "Atomic reduce into a multimem tile view";
  let description = [{
    The :code:`multimem_reduce_view_tko` operation performs an atomic read-modify-write at
    the memory locations reached through the tile view :code:`$view`. Each element of
    :code:`$tile` is combined with the existing memory at the matching location using
    :code:`$reduction`; the update is atomic with respect to other accesses to those same
    locations.

    .. note::

      The underlying pointer for :code:`$view` must be a multicast global pointer (for
      example :code:`tile<ptr<T, #cuda_tile.ptr_attr<multicast>>>`),
      and the view must be built over that multicast-backed storage. A plain unicast
      global pointer is invalid.

      This operation requires the target GPUs to be connected through an
      NVLink SHARP-supported NVSwitch.

      Currently, compiler support is limited to the following:

      * :code:`sm_100`, :code:`sm_103`, and :code:`sm_107` targets.
      * :ref:`type-partition_view` operands.

    The :code:`$index` operands address :code:`$view` in view index space. There must be
    one index per index dimension of the view, the number of indices must match the
    view's index rank, and together they select the destination tile (or region) that is
    updated together with :code:`$tile`. The indices are interpreted as unsigned
    integers. Which indices are valid and how out-of-bounds
    accesses are handled are defined by the concrete view type of :code:`$view`.

    The :code:`$memory_ordering_semantics` attribute must be :code:`relaxed` or
    :code:`release`, and :code:`$memory_scope` is always required.

    This op has no :code:`weak` ordering: it is an atomic multimem reduction, which is
    only defined together with an explicit memory scope (:code:`tl_blk`,
    :code:`device`, or :code:`sys`) alongside :code:`relaxed` or :code:`release`.

    The :code:`$reduction` attribute selects the reduction combiner used in the atomic
    update. Supported element types per reduction kind:

    .. note::

      ================  =====  =====  =====  ======  =====  =====
      Reduction         i32    i64    f16    bf16    f32    f64
      ================  =====  =====  =====  ======  =====  =====
      ADD               ✓      ✓      ✓      ✓       ✓      ✓
      MIN (signed)      ✓      ✓      —      —       —      —
      MAX (signed)      ✓      ✓      —      —       —      —
      UMIN (unsigned)   ✓      ✓      —      —       —      —
      UMAX (unsigned)   ✓      ✓      —      —       —      —
      AND / OR / XOR    ✓      ✓      —      —       —      —
      ================  =====  =====  =====  ======  =====  =====

    The :code:`inbounds` attribute lets the user (or a preceding
    analysis) declare that selected tile-coordinate dimensions are statically
    known to address only valid memory. When set, it must contain one bool per
    index dimension; element :code:`i = true` is a guarantee that for every
    program instance the i-th tile coordinate, combined with the tile size and
    the view's traversal stride, stays inside the parent tensor along that
    dimension. Violating that guarantee is undefined behavior. When absent, all
    dimensions are conservatively treated as potentially out-of-bounds and
    handled according to the semantics of the concrete view type
    (see for example :ref:`type-partition_view`).
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @multimem_reduce_example(
    #       %mcptr: tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>) {
          %tensor_view = make_tensor_view %mcptr, shape=[2], strides=[1]
            : tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>
              -> tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>

          %view = make_partition_view %tensor_view :
            partition_view<tile=(1), tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>

          %c0 = constant <i32: 0> : tile<i32>
          %partials = constant <i32: 1> : tile<1xi32>
          %tok_in = make_token : token

          // Atomic add into the tile at index %c0 (default reduction = add).
          %tok_out = multimem_reduce_view_tko relaxed device %view, %partials[%c0]
            token = %tok_in
            : partition_view<tile=(1), tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
              tile<1xi32>, tile<i32>, token
            -> token
    #   }
    # }
  }]];

  let arguments = (ins
    CudaTileArg<
      CudaTile_MemoryOrderingSemanticsAttr,
      "The memory ordering semantics for the multimem reduce operation.", "13.5",
      [OnlyVariants<["RELAXED", "RELEASE"]>]>:$memory_ordering_semantics,
    CudaTileArg<CudaTile_MemoryScopeAttr,
      "The memory scope for the multimem reduce operation.", "13.5">:$memory_scope,
    CudaTileArg<CudaTile_TileView, "The view into which the reduction is performed.", "13.5">:$view,
    CudaTileArg<CudaTile_TileType, "The tile of values to combine at the destinations.", "13.5">:$tile,
    CudaTileArg<Variadic<CudaTile_IntTileType>,
      "The n-dimensional index of the destination tile in the view's index space.", "13.5">:$index,
    CudaTileArg<Optional<CudaTile_TokenType>, "Optional token for operation ordering.", "13.5">:$token,
    CudaTileArg<DefaultValuedAttr<CudaTile_MultimemReductionKindAttr, "::mlir::cuda_tile::MultimemReductionKind::ADD">,
      "Reduction kind (add, min, max, umin, umax, and, or, xor); default add.", "13.5">:$reduction,
    CudaTileArg<DenseBoolArrayAttr, inbounds_desc, "13.5",
                [RequireSameOperandRank<"index">]>:$inbounds
  );

  let results = (outs
    CudaTileArg<CudaTile_TokenType, "The result token.", "13.5">:$result_token
  );

  let assemblyFormat = [{
    $memory_ordering_semantics $memory_scope
    $view `,` $tile `[` $index `]`
    (`token` `=` $token^)?
    custom<Inbounds>($inbounds, ref($index))
    attr-dict-with-keyword
    `:` custom<CudaTileType>(type($view)) `,` custom<CudaTileType>(type($tile)) `,`
        custom<CudaTileTypeSplat>(type($index), ref($index))
    (`,` custom<CudaTileType>(type($token))^)? `->`
        custom<CudaTileType>(type($result_token))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MultimemStoreViewTkoOp
//===----------------------------------------------------------------------===//

def CudaTile_MultimemStoreViewTkoOp : CudaTileViewOpDef<"multimem_store_view_tko",
    "13.5", [MemoryEffects<[MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "Store to a multimem tile view";
  let description = [{
    The :code:`multimem_store_view_tko` operation writes the tile :code:`$tile` to the memory
    locations reached through the tile view :code:`$view` using a multimem store path.
    It is intended for multicast global paths where the same logical store is visible to
    multiple participants while still using a single view-shaped store in |cuda_tile|.

    .. note::

      The underlying pointer for :code:`$view` must be a multicast global pointer (for
      example :code:`tile<ptr<T, #cuda_tile.ptr_attr<multicast>>>`),
      and the view must be built over that multicast-backed storage. A plain unicast
      global pointer is invalid.

      This operation requires the target GPUs to be connected through an
      NVLink SHARP-supported NVSwitch.

      Currently, compiler support is limited to the following:

      * :code:`sm_100`, :code:`sm_103`, and :code:`sm_107` targets.
      * :ref:`type-partition_view` operands.

    The :code:`$index` operands address :code:`$view` in view index space. There must be
    one index per index dimension of the view, the number of indices must match the
    view's index rank, and together they select the destination tile (or region) that
    receives :code:`$tile`. The indices are interpreted as unsigned integers. Which
    indices are valid and how out-of-bounds accesses are
    handled are defined by the concrete view type of :code:`$view`.

    The :code:`$memory_ordering_semantics` attribute follows the same weak, relaxed, and
    release split as other token-ordered stores: when :code:`weak`, :code:`$memory_scope`
    must not be specified; when :code:`relaxed` or :code:`release`,
    :code:`$memory_scope` must be specified.

    The values in :code:`$tile` replace the contents at the destination selected by
    :code:`$index` on the multimem store path.

    Supported element types:

    .. note::

      ================  =====  =====  =====  ======  =====  =====
      Operation         i32    i64    f16    bf16    f32    f64
      ================  =====  =====  =====  ======  =====  =====
      multimem_store    ✓      ✓      ✓      ✓       ✓      ✓
      ================  =====  =====  =====  ======  =====  =====

    The :code:`inbounds` attribute lets the user (or a preceding
    analysis) declare that selected tile-coordinate dimensions are statically
    known to address only valid memory. When set, it must contain one bool per
    index dimension; element :code:`i = true` is a guarantee that for every
    program instance the i-th tile coordinate, combined with the tile size and
    the view's traversal stride, stays inside the parent tensor along that
    dimension. Violating that guarantee is undefined behavior. When absent, all
    dimensions are conservatively treated as potentially out-of-bounds and
    handled according to the semantics of the concrete view type
    (see for example :ref:`type-partition_view`).
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
    #   entry @multimem_store_example(
    #       %mcptr: tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
          %tensor_view = make_tensor_view %mcptr, shape=[8192, 128],
            strides=[128, 1]
            : tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>
              -> tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>

          %view = make_partition_view %tensor_view :
            partition_view<tile=(64x64),
              tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>>

          %c0 = constant <i32: 0> : tile<i32>
          %c1 = constant <i32: 1> : tile<i32>
          %tile = constant <f32: 0.0> : tile<64x64xf32>

          // Weak store at tile index (0, 1); no memory scope.
          %tok_st = multimem_store_view_tko weak %view, %tile[%c0, %c1]
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>>,
              tile<64x64xf32>, tile<i32>
            -> token

          // Relaxed store with device scope and an input token.
          %tok_in = make_token : token
          %tok_st2 = multimem_store_view_tko relaxed device %view, %tile[%c0, %c0]
            token = %tok_in
            : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1], #cuda_tile.ptr_attr<multicast>>>,
              tile<64x64xf32>, tile<i32>, token
            -> token
    #   }
    # }
  }]];

  let arguments = (ins
    CudaTileArg<
      CudaTile_MemoryOrderingSemanticsAttr,
      "The memory ordering semantics for the multimem store operation.", "13.5",
      [OnlyVariants<["WEAK", "RELAXED", "RELEASE"]>]>:$memory_ordering_semantics,
    CudaTileArg<OptionalAttr<CudaTile_MemoryScopeAttr>,
      "The memory scope for the multimem store operation.", "13.5">:$memory_scope,
    CudaTileArg<CudaTile_TileView, "The view to store into.", "13.5">:$view,
    CudaTileArg<CudaTile_TileType, "The tile to store.", "13.5">:$tile,
    CudaTileArg<Variadic<CudaTile_IntTileType>,
      "The n-dimensional index of the destination tile in the view's index space.", "13.5">:$index,
    CudaTileArg<Optional<CudaTile_TokenType>, "Optional token for operation ordering.", "13.5">:$token,
    CudaTileArg<DenseBoolArrayAttr, inbounds_desc, "13.5",
                [RequireSameOperandRank<"index">]>:$inbounds
  );

  let results = (outs
    CudaTileArg<CudaTile_TokenType, "The result token.", "13.5">:$result_token
  );

  let assemblyFormat = [{
    $memory_ordering_semantics
    ($memory_scope^)?
    $view `,` $tile `[` $index `]`
    (`token` `=` $token^)?
    custom<Inbounds>($inbounds, ref($index))
    attr-dict-with-keyword
    `:` custom<CudaTileType>(type($view)) `,` custom<CudaTileType>(type($tile)) `,`
        custom<CudaTileTypeSplat>(type($index), ref($index))
    (`,` custom<CudaTileType>(type($token))^)? `->`
        custom<CudaTileType>(type($result_token))
  }];

  let hasVerifier = 1;
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_OPS_TD
