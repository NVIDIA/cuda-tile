//===- PreviewOps.td - CUDA Tile Public-Preview Ops --------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public-preview operations.
//
// Preview ops ship publicly. They are not part of the forever-portable wire
// format: a preview op is only accepted by the exact bytecode version that
// produced it, and only when the consumer opts in to the operation's named
// feature.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_PREVIEW_OPS_TD
#define CUDATILE_DIALECT_CUDATILE_IR_PREVIEW_OPS_TD

include "mlir/IR/OpBase.td"

include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Interfaces.td"
include "cuda_tile/Dialect/CudaTile/IR/Types.td"
include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td"

// Argument wrapper for public-preview operations. Preview operations carry no
// version information, because their membership is not tracked per bytecode
// version.
class CudaTilePreviewArg<Constraint constraint, string desc,
                         list<OpVariableDecorator> decorators = []>
    : Arg<constraint, desc, decorators # [CudaTileArgMetadata<"", desc>]>;

//===----------------------------------------------------------------------===//
// Preview Operations
//===----------------------------------------------------------------------===//
// New preview ops are declared here with a matching PreviewOpcode assignment in
// BytecodeOpcodes.td. Preview ops carry no version metadata (use
// CudaTilePreviewArg for their arguments).

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def CudaTile_FuncOp
    : CudaTilePreviewOpDef<"func", CudaTile_SIMTPreviewFeature,
                           [SymbolName, SymbolVisibility, FunctionOpInterface,
                            IsolatedFromAbove, OpAsmOpInterface, SingleBlock]> {
  let summary = "Declare an external function for SIMT interop";
  let description = [{
    The :code:`preview$func` operation declares an external function at the
    SIMT interop boundary. It must carry the :code:`extern` attribute and must
    not have a body.

    The function signature uses :code:`!cuda_tile.tile<...>` ABI boundary
    types, which may contain symbolic shapes and element types. The declaration
    may include compile-time constant parameters and must have at least one
    input or result. Calls to the function use :code:`preview$call`.
  }];

  let mlirExamples = [[{
    # cuda_tile.module @module {
        cuda_tile.preview$func extern @scale_tile<[
            M: index, T: !cuda_tile.preview$Datatype]>(
            %arg0: !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>)
          -> !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>
    # }
  }]];
  let arguments = (ins
      CudaTilePreviewArg<
          SymbolNameAttr, "The name of the function.">:$sym_name,
      CudaTilePreviewArg<
          UnitAttr, "Declare an extern function without a body.">:$is_extern,
      CudaTilePreviewArg<
          TypeAttrOf<FunctionType>, "The type of the function.">:$function_type,
      CudaTilePreviewArg<
          OptionalAttr<StrAttr>, "The visibility of the function.">:$sym_visibility,
      CudaTilePreviewArg<
          OptionalAttr<DictArrayAttr>,
          "The argument attributes of the function.">:$arg_attrs,
      CudaTilePreviewArg<
          OptionalAttr<DictArrayAttr>,
          "The result attributes of the function.">:$res_attrs,
      CudaTilePreviewArg<
          CudaTileArrayOf<CudaTile_ConstParamAttr>,
          "Array of compile-time constant parameters for an extern function.">:$const_params);
  let regions = (region MaxSizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl#[{
    // FunctionOpInterface Methods

    /// Returns the region on the current operation
    ::mlir::Region *getCallableRegion() {
      return getIsExtern() ? nullptr : &getBody();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    bool isDeclaration() { return getIsExtern() && getBody().empty(); }

    static void ensureTerminator(::mlir::Region &region,
                                 ::mlir::Builder &builder,
                                 ::mlir::Location loc);
  }];
}

//===----------------------------------------------------------------------===//
// External Function Interface Operations
//===----------------------------------------------------------------------===//

def CudaTile_ExternImplOp
    : CudaTilePreviewOpDef<
          "extern_impl", CudaTile_SIMTPreviewFeature,
          [SymbolName, SymbolVisibility, Symbol, FunctionOpInterface,
           OpAsmOpInterface,
           DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "External function implementation";
  let description = [{
    The :code:`preview$extern_impl` operation associates a tile-level
    :code:`preview$func extern` declaration with a native device function.
    Multiple SIMT implementations may back the same declaration.

    The implementation specifies:
    - Which extern function it implements (via symbol reference)
    - Optional constant parameters for this implementation
    - Optional constant arguments to instantiate the extern function's parameters
    - Optional named parameter list with types
    - Fragment types carrying memory spaces and layouts
    - Target architectures (arch)
    - An optional signed provider candidate score

    The implementation's `sym_name` names the native device function.
    Load tile libraries containing implementations with ``-l=<path>`` and
    ``--preview=simt`` in tileiras.

    `synchronization` records the widest cooperative synchronization domain
    used by the implementation. It defaults to `none`. It is a separate
    legality contract from the fragment map's physical participant count and
    does not add an entry barrier. Concrete fragment maps must satisfy the
    participant-count constraint of the selected synchronization domain;
    symbolic counts are checked after concretization.

    The `config` dictionary is reserved and unsupported on extern
    implementations. It must be absent; verification rejects even an empty
    dictionary. Express declaration specialization through constant parameters
    and arguments, target restrictions through `arch`, and invocation
    requirements through fragment maps and `synchronization`.

    `provider_score` is an optional signed preference among implementations.
    A larger value indicates a stronger preference; absence means zero.
    It does not change the callable ABI or implementation compatibility
    requirements.

    The `implements` symbol reference may be unresolved. When it resolves,
    it must name an extern :code:`preview$func` in the same symbol scope,
    and the implementation must satisfy that declaration's parameter and
    fragment compatibility requirements. Checks that require the declaration
    apply only when the symbol resolves.

    Example:

    .. code-block:: mlir

      cuda_tile.preview$extern_impl @my_simt_func_sm80
        implements @my_simt_func

      cuda_tile.preview$extern_impl @my_simt_func_sm80
        implements @my_simt_func(
          %arg0: !cuda_tile.preview$fragment<1xf32, local,
            mem_layout = #cuda_tile.preview$layout<(1):(1)>,
            fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>> {
              cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
  }];

  let arguments = (ins
    CudaTilePreviewArg<SymbolNameAttr, "The symbol name of this implementation.">:$sym_name,
    CudaTilePreviewArg<OptionalAttr<StrAttr>, "The visibility of this implementation.">:$sym_visibility,
    CudaTilePreviewArg<FlatSymbolRefAttr, "Reference to the extern preview function.">:$implements,
    CudaTilePreviewArg<OptionalAttr<CudaTileArrayOf<CudaTile_ConstParamAttr>>, "Array of compile-time constant parameters for this implementation.">:$const_params,
    CudaTilePreviewArg<OptionalAttr<CudaTileArrayOf<CudaTile_SymbolicExprAttrConstraint>>, "Array of symbolic expression arguments to instantiate extern function parameters.">:$const_args,
    CudaTilePreviewArg<TypeAttrOf<FunctionType>, "The function type signature for this implementation.">:$function_type,
    CudaTilePreviewArg<OptionalAttr<DictArrayAttr>, "The argument attributes of this implementation.">:$arg_attrs,
    CudaTilePreviewArg<OptionalAttr<DictArrayAttr>, "The result attributes of this implementation.">:$res_attrs,
    CudaTilePreviewArg<OptionalAttr<CudaTile_ScratchRequestAttr>, "Shared-memory scratch request.">:$scratch_smem,
    CudaTilePreviewArg<OptionalAttr<DictionaryAttr>, "Reserved and unsupported; must be absent.">:$config,
    CudaTilePreviewArg<OptionalAttr<StrArrayAttr>, "Target architectures (array of StringAttr).">:$arch,
    CudaTilePreviewArg<OptionalAttr<DictionaryAttr>, "Frontend-specific ABI constraints.">:$frontendABI,
    CudaTilePreviewArg<
        DefaultValuedOptionalAttr<
            CudaTile_SynchronizationAttr,
            "::mlir::cuda_tile::Synchronization::None">,
        "Widest synchronization domain used by this implementation.">:$synchronization,
    CudaTilePreviewArg<OptionalAttr<I64Attr>, "Signed implementation preference; larger values are preferred, absent means zero.">:$provider_score
  );
  let regions = (region SizedRegion<0>:$body);
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    bool isDeclaration() { return true; }

    /// Whether this implementation uses the native fragment/SIMT ABI.
    bool isNativeSimtImplementation();

    /// Get the number of constant parameters.
    size_t getNumConstParams() {
      if (auto params = getConstParams())
        return params->size();
      return 0;
    }

    /// Get a specific constant parameter by index.
    ConstParamAttr getConstParam(size_t index) {
      if (auto params = getConstParams()) {
        if (index < params->size())
          return mlir::cast<ConstParamAttr>((*params)[index]);
      }
      return nullptr;
    }

    /// Get a specific constant argument by index.
    Attribute getConstArg(size_t index) {
      if (auto args = getConstArgs()) {
        if (index < args->size())
          return (*args)[index];
      }
      return nullptr;
    }

    /// Keep an implementation live while its declaration has a call user.
    bool canDiscardOnUseEmpty();

  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CudaTile_CallOp
    : CudaTilePreviewOpDef<"call", CudaTile_SIMTPreviewFeature,
        [CallOpInterface, DeclareOpInterfaceMethods<
            SymbolUserOpInterface>]> {
  let summary = "SIMT interop call to an external function";
  let description = [{
    The :code:`preview$call` operation represents a SIMT interop call to an
    external function. When resolved, its callee must be an extern
    :code:`preview$func` in the same symbol scope. The callee is encoded
    as a symbol reference attribute named :code:`callee`. An unresolved callee
    is permitted. Checks that require the declaration apply only when the
    symbol resolves.

    Both declarations and call sites use
    :code:`!cuda_tile.tile<...>`; declaration types may contain symbolic shapes
    or element types, while call-site tiles must be concrete. External calls may
    additionally carry :code:`!cuda_tile.token` operands and results
    anywhere in the call signature. These tokens are not passed to or returned
    by the external function. Instead, they are call-boundary ordering edges:
    token operands make the call depend on the represented effects, and token
    results are continuation tokens for effects modeled at the call boundary.
    Multiple tokens are allowed so independent ordering chains can cross the
    same external call without forcing the verifier to join them.

  }]
  ;

  let mlirExamples = [[{
    cuda_tile.module @my_module {
    preview$func extern @my_add(
      %0: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      },
      %1: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<f32>

    entry @my_entry() {
      %0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
      %1 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
      %2 = cuda_tile.preview$call @my_add(%0, %1)
      : (!cuda_tile.tile<f32>, !cuda_tile.tile<f32>) -> !cuda_tile.tile<f32>
      }
    }
  }]];

  let arguments = (ins
      CudaTilePreviewArg<
          FlatSymbolRefAttr,
          "The symbol reference to the extern function.">:$callee,
      CudaTilePreviewArg<
          Variadic<AnyType>,
          "The operands to pass to the extern function.">:$operands,
      CudaTilePreviewArg<
          CudaTileArrayOf<CudaTile_SymbolicExprAttrConstraint>,
          "Constant arguments supplied to an extern function.">:$const_args,
      CudaTilePreviewArg<
          OptionalAttr<DictArrayAttr>,
          "Optional attributes for the arguments.">:$arg_attrs,
      CudaTilePreviewArg<
          OptionalAttr<DictArrayAttr>,
          "Optional attributes for the results.">:$res_attrs);
  let results = (outs
      CudaTilePreviewArg<
          Variadic<AnyType>,
          "The results returned by the extern function.">:$results);

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function-like symbol.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  // The const_args attribute is required, but empty const_args elide the
  // compact `<...>` spelling. The empty literals keep that case printed as
  // `@callee(...)`, without spaces around the elided custom directive.
  let assemblyFormat = [{
    $callee `` custom<OptionalCompactConstArgs>($const_args) `` `(`
    $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

#ifdef TILE_IR_INCLUDE_TESTS
//===----------------------------------------------------------------------===//
// Preview_BytecodeTestOp
//===----------------------------------------------------------------------===//
// Exercises the public-preview bytecode path. This example op is
// test-only; genuine product preview ops are declared ungated.

def CudaTile_Preview_BytecodeTestOp
    : CudaTilePreviewOpDef<"bytecode_test_op",
                           CudaTile_SIMTPreviewFeature> {
  let summary = "Example public-preview op for exercising preview bytecode.";
  let description = [{
    The :code:`preview$bytecode_test_op` operation exists solely to exercise the
    public-preview bytecode serialization path. It has a required attribute and
    an optional attribute.
  }];

  let arguments = (ins
    CudaTilePreviewArg<I32Attr, "Required attribute.">:$req_attr,
    CudaTilePreviewArg<OptionalAttr<I32Attr>, "Optional attribute.">:$opt_attr);

  let assemblyFormat = [{
    `req` `=` $req_attr (`opt` `=` $opt_attr^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BytecodeTest_PreviewAttributeOp
//===----------------------------------------------------------------------===//
// Exercises public-preview attribute serialization.

def CudaTile_BytecodeTest_PreviewAttributeOp
    : CudaTilePreviewOpDef<"bytecode_test_preview_attribute",
                           CudaTile_SIMTPreviewFeature> {
  let summary = "Testing operation carrying public-preview attributes";
  let arguments = (ins
    CudaTilePreviewArg<CudaTile_PreviewBytecodeTestAttr,
                       "A directly encoded public-preview attribute.">:$preview_attr,
    CudaTilePreviewArg<OptionalAttr<ArrayAttr>,
                       "Optional container used to test nested preview attributes.">:$nested_attrs);

  let assemblyFormat = [{
    `preview_attr` `=` $preview_attr
    (`nested_attrs` `=` $nested_attrs^)?
    attr-dict
  }];
}
#endif // TILE_IR_INCLUDE_TESTS

#endif // CUDATILE_DIALECT_CUDATILE_IR_PREVIEW_OPS_TD
