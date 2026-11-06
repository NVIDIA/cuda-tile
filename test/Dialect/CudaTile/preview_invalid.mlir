// RUN: cuda-tile-opt --preview=simt %s -split-input-file -verify-diagnostics \
// RUN:   -allow-unregistered-dialect

// Verifier diagnostics for the public-preview SIMT surface. Diagnostics that
// are not attached to an operation location live in preview_diagnostics.mlir.

//===----------------------------------------------------------------------===//
// preview$func
//===----------------------------------------------------------------------===//

// Tokens are ordering values, not extern-function data operands.

// expected-error @below {{input type 0 must be a cuda_tile.tile, got '!cuda_tile.token'}}
cuda_tile.preview$func extern @token_input(
    %token: !cuda_tile.token {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> !cuda_tile.tile<f32>

// -----

// expected-error @below {{result type 0 must be a cuda_tile.tile, got '!cuda_tile.token'}}
cuda_tile.preview$func extern @token_result(
    %value: !cuda_tile.tile<f32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> !cuda_tile.token

// -----

// A writable argument binds to the leading result with the same type.

// expected-error @below {{extern function argument 0 must match result 0}}
cuda_tile.preview$func extern @bad_inout_result_type(
    %arg0: !cuda_tile.tile<128xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<inout>
    })
  -> !cuda_tile.tile<128xf16>

// -----

// Every writable argument requires a corresponding result.

// expected-error @below {{number of bound inouts exceeded the number of results}}
cuda_tile.preview$func extern @bad_out_no_result(
    %arg0: !cuda_tile.tile<128xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<out>
    })

// -----

// Result binding is one-to-one even for a contiguous writable run.

// expected-error @below {{number of bound inouts exceeded the number of results}}
cuda_tile.preview$func extern @bad_too_many_inouts(
    %arg0: !cuda_tile.tile<128xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<inout>
    },
    %arg1: !cuda_tile.tile<128xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<inout>
    })
  -> !cuda_tile.tile<128xf32>

// -----

cuda_tile.module @bad_extern_input {
  // expected-error @below {{input type 0 must be a cuda_tile.tile}}
  preview$func extern @bad(%arg0: !cuda_tile.ptr<f32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

cuda_tile.module @bad_extern_result {
  // expected-error @below {{result type 0 must be a cuda_tile.tile}}
  preview$func extern @bad(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.ptr<f32>
}

// -----

cuda_tile.module @bad_extern_body {
  // expected-error @below {{extern functions must not have a body}}
  "cuda_tile.preview$func"() ({
  ^bb0:
    cuda_tile.return
  }) {const_params = [], function_type = () -> (), is_extern, sym_name = "bad"} : () -> ()
}

// -----

cuda_tile.module @bad_non_extern_empty {
  // expected-error @below {{non-extern functions must have a body}}
  "cuda_tile.preview$func"() ({
  }) {const_params = [], function_type = () -> (), sym_name = "bad"} : () -> ()
}

// -----

// Non-extern preview$func rejects const_params.

// expected-error @below {{constant parameters are only valid on extern functions}}
"cuda_tile.preview$func"() ({
  cuda_tile.return
}) {
  const_params = [#cuda_tile.preview$const_param<name = "M", type = index>],
  function_type = () -> (),
  sym_name = "bad_nonextern_func_const_params"
} : () -> ()

// -----

// preview$func rejects fragments in either form: a fragment describes one
// invocation's slice, which only preview$extern_impl signatures carry.

cuda_tile.module @bad_func_fragment_input {
  // expected-error @below {{input type 0 must not be a cuda_tile.preview$fragment}}
  cuda_tile.preview$func extern @bad(
      %arg0: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(1, 1, 1):(1, 0, 0)>> {
          cuda_tile.preview$access = #cuda_tile.preview$access<in>})
}

// -----

cuda_tile.module @bad_func_fragment_result {
  // expected-error @below {{result type 0 must not be a cuda_tile.preview$fragment}}
  cuda_tile.preview$func extern @bad(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(1, 1, 1):(1, 0, 0)>>
}

// -----

//===----------------------------------------------------------------------===//
// preview$func: symbolic parameter scope
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'cuda_tile.preview$func' op input type references undeclared symbolic variable 'Undefined'}}
cuda_tile.preview$func extern @undefined_dimension<[M: index]>(
  %arg0: !cuda_tile.tile<[Undefined]xf32> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[Undefined]xf32>

// -----

// expected-error @+1 {{'cuda_tile.preview$func' op input type references undeclared symbolic variable 'T'}}
cuda_tile.preview$func extern @undefined_element_type<[M: index]>(
  %arg0: !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>

// -----

// expected-error @+1 {{'cuda_tile.preview$func' op input type references undeclared symbolic variable 'N'}}
cuda_tile.preview$func extern @undefined_shape_expression<[M: index]>(
  %arg0: !cuda_tile.tile<[(M * N)]x!cuda_tile.preview$sym_t<T>> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>

// -----

// expected-error @+1 {{'cuda_tile.preview$func' op result type references undeclared symbolic variable 'T'}}
cuda_tile.preview$func extern @undefined_result_element_type<[M: index]>(
  %arg0: !cuda_tile.tile<[M]xf32> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>

// -----

cuda_tile.preview$func extern @declared_symbols<[
    M: index, N: index, T: !cuda_tile.preview$Datatype]>(
  %arg0: !cuda_tile.tile<[(M * N)]x!cuda_tile.preview$sym_t<T>> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[(M * N)]x!cuda_tile.preview$sym_t<T>>

// -----

// Execution-context builtins are not user-declared symbolic variables.
cuda_tile.preview$func extern @context_builtin(
  %arg0: !cuda_tile.tile<[context.num_threads]xf32> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[context.num_threads]xf32>

// -----

cuda_tile.preview$func extern @declared_shape_expression<[N: index]>(
  %arg0: !cuda_tile.tile<[(N + 6)]xf32> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[(N + 6)]xf32>

// -----

// expected-error @+1 {{'cuda_tile.preview$func' op input type references undeclared symbolic variable 'N'}}
cuda_tile.preview$func extern @undefined_shape_expression_add<[M: index]>(
  %arg0: !cuda_tile.tile<[(N + 6)]xf32> {
    cuda_tile.preview$access = #cuda_tile.preview$access<inout>
  }
) -> !cuda_tile.tile<[(N + 6)]xf32>

// -----

// expected-error @+1 {{'cuda_tile.preview$func' op input type references undeclared symbolic variable 'N'}}
cuda_tile.preview$func @non_extern_unbound_dimension(
    %arg0: !cuda_tile.tile<[N]xf32>) -> !cuda_tile.tile<[N]xf32> {
  cuda_tile.return %arg0 : !cuda_tile.tile<[N]xf32>
}

// -----

// This example is invalid for multiple reasons:
// 1. The symbol is not declared.
// 2. sym_t cannot be used as a standalone type.

// expected-error @+1 {{'cuda_tile.preview$func' op input type references undeclared symbolic variable 'T'}}
cuda_tile.preview$func @non_extern_unbound_bare_symbol(
    %arg0: !cuda_tile.preview$sym_t<T>) -> !cuda_tile.preview$sym_t<T> {
  cuda_tile.return %arg0 : !cuda_tile.preview$sym_t<T>
}

// -----

cuda_tile.module @entry_unbound_symbol {
  // expected-error @below {{'cuda_tile.entry' op input type references undeclared symbolic variable 'T'}}
  cuda_tile.entry @kernel(
      %arg0: !cuda_tile.tile<!cuda_tile.preview$sym_t<T>>) {
    cuda_tile.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// constant: symbolic shapes
//===----------------------------------------------------------------------===//

// A dense value has no statically known element count for a symbolic shape.
// The type is rejected while parsing, before the attribute is built.

cuda_tile.module @constant_symbolic_shape {
  cuda_tile.entry @f() {
    // expected-error @below {{constant tile type must have a static shape}}
    %0 = cuda_tile.constant <f32: 1.0> : tile<[N]xf32>
    cuda_tile.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// preview$call
//===----------------------------------------------------------------------===//

// preview$call constant arguments are only valid for extern
// preview$func.

cuda_tile.module @bad_func_call_const_args {
  cuda_tile.preview$func @target(
      %arg0: !cuda_tile.tile<128xf32>)
      -> !cuda_tile.tile<128xf32> {
    cuda_tile.return %arg0 : !cuda_tile.tile<128xf32>
  }

  cuda_tile.preview$func @caller(
      %arg0: !cuda_tile.tile<128xf32>)
      -> !cuda_tile.tile<128xf32> {
    // expected-error @below {{constant arguments are only valid when calling an extern preview$func}}
    %result = cuda_tile.preview$call @target<[128]>(%arg0)
      : (!cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>
    cuda_tile.return %result : !cuda_tile.tile<128xf32>
  }
}

// -----

// preview$call constant argument count must match the extern function.

cuda_tile.module @bad_call_const_arg_count {
  cuda_tile.preview$func extern @callee<
      [M: index, T: !cuda_tile.preview$Datatype]>(
      %arg0: !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func @caller(
      %arg0: !cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32> {
    // expected-error @below {{constant argument count (1) must match extern function constant parameter count (2)}}
    %result = cuda_tile.preview$call @callee<[128]>(%arg0)
      : (!cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>
    cuda_tile.return %result : !cuda_tile.tile<128xf32>
  }
}

// -----

// preview$call must supply arguments for all extern constant parameters.

cuda_tile.module @bad_call_missing_const_args {
  cuda_tile.preview$func extern @callee<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<[M]xf32>

  cuda_tile.preview$func @caller(
      %arg0: !cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32> {
    // expected-error @below {{constant argument count (0) must match extern function constant parameter count (1)}}
    %result = cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>
    cuda_tile.return %result : !cuda_tile.tile<128xf32>
  }
}

// -----

// preview$call constant arguments must match their parameter kinds.

cuda_tile.module @bad_call_const_arg_kind {
  cuda_tile.preview$func extern @callee<
      [T: !cuda_tile.preview$Datatype]>(
      %arg0: !cuda_tile.tile<128x!cuda_tile.preview$sym_t<T>> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<128x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func @caller(
      %arg0: !cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32> {
    // expected-error @below {{constant argument #0 for constant parameter 'T' must be a valid datatype expression: expected datatype expression, got #cuda_tile<preview$expr_int<128>>}}
    %result = cuda_tile.preview$call @callee<[128]>(%arg0)
      : (!cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>
    cuda_tile.return %result : !cuda_tile.tile<128xf32>
  }
}

// -----

// A scalar integer is not a shape expression.

cuda_tile.module @bad_call_shape_const_arg_kind {
  cuda_tile.preview$func extern @callee<
      [S: !cuda_tile.preview$shape]>(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<1xf32>) {
    // expected-error @below {{constant argument #0 for constant parameter 'S' must be a valid shape expression: expected tuple expression, got #cuda_tile<preview$expr_int<128>>}}
    cuda_tile.preview$call @callee<[128]>(%arg0)
      : (!cuda_tile.tile<1xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// A datatype is not a string expression.

cuda_tile.module @bad_call_string_const_arg_kind {
  cuda_tile.preview$func extern @callee<
      [Tag: !cuda_tile.preview$string]>(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<1xf32>) {
    // expected-error @below {{constant argument #0 for constant parameter 'Tag' must be a valid string expression: expected string expression, got f32}}
    cuda_tile.preview$call @callee<[f32]>(%arg0)
      : (!cuda_tile.tile<1xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// preview$call rejects unsupported integer constant expressions.

cuda_tile.module @bad_call_const_arg_unsupported_expr {
  cuda_tile.preview$func extern @callee<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(
      %arg0: !cuda_tile.tile<64xf32>) {
    // expected-error @below {{constant argument #0 for constant parameter 'M' must be a valid integer expression: unsupported expression in integer context}}
    "cuda_tile.preview$call"(%arg0) <{
      callee = @callee,
      const_args = [
        #cuda_tile.preview$expr_div<
          #cuda_tile.preview$expr_int<128>,
          #cuda_tile.preview$expr_int<2>>
      ]
    }> : (!cuda_tile.tile<64xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// preview$call recursively validates integer constant expressions.

cuda_tile.module @bad_call_const_arg_child_kind {
  cuda_tile.preview$func extern @callee<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(
      %arg0: !cuda_tile.tile<128xf32>) {
    // expected-error @below {{constant argument #0 for constant parameter 'M' must be a valid integer expression: expected integer expression, got f32}}
    "cuda_tile.preview$call"(%arg0) <{
      callee = @callee,
      const_args = [
        #cuda_tile.preview$expr_add<
          #cuda_tile.preview$concrete_type<f32>,
          #cuda_tile.preview$expr_int<1>>
      ]
    }> : (!cuda_tile.tile<128xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// preview$call constant arguments cannot forward unbounded declaration
// parameters, even when their expression kind is otherwise valid.

cuda_tile.module @bad_call_const_arg_symbolic_var {
  cuda_tile.preview$func extern @callee<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<64xf32>) {
    // expected-error @below {{constant argument #0 for constant parameter 'M' must be statically known; references symbolic variable 'M'}}
    cuda_tile.preview$call @callee<[M]>(%arg0)
      : (!cuda_tile.tile<64xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// The staticness requirement applies recursively to symbolic expressions.

cuda_tile.module @bad_call_const_arg_symbolic_expr {
  cuda_tile.preview$func extern @callee<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<64xf32>) {
    // expected-error @below {{constant argument #0 for constant parameter 'M' must be statically known; references symbolic variable 'M'}}
    cuda_tile.preview$call @callee<[(M + N)]>(%arg0)
      : (!cuda_tile.tile<64xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// Literal expressions and compiler-provided context builtins are statically
// known at a call site.

cuda_tile.module @ok_call_const_arg_static_exprs {
  cuda_tile.preview$func extern @callee<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<64xf32>) {
    cuda_tile.preview$call @callee<[(1 + 3 + 4)]>(%arg0)
      : (!cuda_tile.tile<64xf32>) -> ()
    cuda_tile.preview$call @callee<[context.num_threads]>(%arg0)
      : (!cuda_tile.tile<64xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// preview$call constant arguments cannot be empty tuples.

cuda_tile.module @bad_call_const_arg_empty_tuple {
  cuda_tile.preview$func extern @callee<[S: !cuda_tile.preview$shape]>(
      %arg0: !cuda_tile.tile<64xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      })

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<64xf32>) {
    // expected-error @below {{constant argument #0 for constant parameter 'S' must be statically known; empty tuple is not supported}}
    cuda_tile.preview$call @callee<[()]>(%arg0)
      : (!cuda_tile.tile<64xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// preview$call operands must be concrete tiles.

cuda_tile.module @bad_call_operand {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<128xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<128xf32>

  cuda_tile.testing$func @caller(
      %arg0: !cuda_tile.tile<[M]xf32>) -> !cuda_tile.tile<128xf32> {
    // expected-error @below {{operand 0 must be a concrete 'cuda_tile.tile' at call sites, got '!cuda_tile.tile<[M]xf32>'}}
    %result = cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.tile<[M]xf32>) -> !cuda_tile.tile<128xf32>
    cuda_tile.return %result : !cuda_tile.tile<128xf32>
  }
}

// -----

// preview$call results must be concrete tiles.

cuda_tile.module @bad_call_result {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<1xf32>

  cuda_tile.testing$func @caller(
      %arg0: !cuda_tile.tile<1xf32>)
      -> !cuda_tile.tile<[M]xf32> {
    // expected-error @below {{result 0 must be a concrete 'cuda_tile.tile' at call sites, got '!cuda_tile.tile<[M]xf32>'}}
    %result = cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.tile<1xf32>) -> !cuda_tile.tile<[M]xf32>
    cuda_tile.return %result : !cuda_tile.tile<[M]xf32>
  }
}

// -----

// preview$call must reference an preview$func.

cuda_tile.module @bad_call_callee_kind {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<1xf32>

  cuda_tile.preview$extern_impl @impl
    implements @callee
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, 1, 1):(1, 0, 0)>> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>})

  cuda_tile.preview$func @caller() -> !cuda_tile.tile<f32> {
    // expected-error @below {{'impl' does not reference a valid preview$func}}
    %result = cuda_tile.preview$call @impl()
      : () -> !cuda_tile.tile<f32>
    cuda_tile.return %result : !cuda_tile.tile<f32>
  }
}

// -----

// preview$call data operand count must match the extern function.

cuda_tile.module @bad_call_arity {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<1xf32>

  cuda_tile.preview$func @caller() -> !cuda_tile.tile<f32> {
    // expected-error @below {{input count (0) must match extern preview$func input count (1)}}
    %result = cuda_tile.preview$call @callee()
      : () -> !cuda_tile.tile<f32>
    cuda_tile.return %result : !cuda_tile.tile<f32>
  }
}

// -----

// preview$call data result count must match the extern function.

cuda_tile.module @bad_call_result_arity {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<1xf32>

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<1xf32>) {
    // expected-error @below {{result count (0) must match extern preview$func result count (1)}}
    cuda_tile.preview$call @callee(%arg0) : (!cuda_tile.tile<1xf32>) -> ()
    cuda_tile.return
  }
}

// -----

// preview$call rejects a non-tile cuda_tile type at an extern call site.

cuda_tile.module @bad_call_pointer_operand {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.ptr<f32>) {
    // expected-error @below {{operand 0 must be a concrete 'cuda_tile.tile' at call sites, got '!cuda_tile.ptr<f32>'}}
    cuda_tile.preview$call @callee(%arg0) : (!cuda_tile.ptr<f32>) -> ()
    cuda_tile.return
  }
}

// -----

// Fragment values are implementation details, not call-site data operands.

cuda_tile.module @bad_call_fragment_operand {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})

  cuda_tile.testing$func @caller(
      %arg0: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(1, 1, 1):(1, 0, 0)>>) {
    // expected-error @below {{operand 0 must be a concrete 'cuda_tile.tile' at call sites}}
    cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(1, 1, 1):(1, 0, 0)>>) -> ()
    cuda_tile.return
  }
}

// -----

// Fragment values are implementation details, not call-site data results.

cuda_tile.module @bad_call_fragment_result {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<1xf32>) {
    // expected-error @below {{result 0 must be a concrete 'cuda_tile.tile' at call sites}}
    %result = cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.tile<1xf32>)
        -> !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(1, 1, 1):(1, 0, 0)>>
    cuda_tile.return
  }
}

// -----

// preview$call rejects an incompatible concrete element type for an extern
// declaration (e.g. tile-of-pointers vs tile-of-scalars).

cuda_tile.module @bad_call_pointer_tile_operand {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below {{operand type mismatch at index 0}}
    // expected-note @below {{op operand types: '!cuda_tile.tile<ptr<f32>>'}}
    // expected-note @below {{function operand types: '!cuda_tile.tile<f32>'}}
    cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.tile<!cuda_tile.ptr<f32>>) -> ()
    cuda_tile.return
  }
}

// -----

// preview$call rejects an incompatible result element type for an extern
// declaration.

cuda_tile.module @bad_call_result_element_type {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<f32>

  cuda_tile.preview$func @caller(%arg0: !cuda_tile.tile<f32>)
      -> !cuda_tile.tile<i32> {
    // expected-error @below {{result type mismatch at index 0}}
    // expected-note @below {{op result types: '!cuda_tile.tile<i32>'}}
    // expected-note @below {{function result types: '!cuda_tile.tile<f32>'}}
    %result = cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.tile<f32>) -> !cuda_tile.tile<i32>
    cuda_tile.return %result : !cuda_tile.tile<i32>
  }
}

// -----

// preview$call still accepts a concrete specialization of a symbolic extern
// declaration when the concrete parts agree.

cuda_tile.module @ok_call_symbolic_compatible {
  cuda_tile.preview$func extern @callee<
      [M: index, T: !cuda_tile.preview$Datatype]>(
      %arg0: !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func @caller(
      %arg0: !cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32> {
    %result = cuda_tile.preview$call @callee<[128, f32]>(%arg0)
      : (!cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>
    cuda_tile.return %result : !cuda_tile.tile<128xf32>
  }
}

// -----

// preview$call results with symbolic element types are not concrete.

cuda_tile.module @bad_call_symbolic_result_type {
  cuda_tile.preview$func extern @callee(
      %arg0: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<1xf32>

  cuda_tile.testing$func @caller(
      %arg0: !cuda_tile.tile<1xf32>)
      -> !cuda_tile.tile<!cuda_tile.preview$sym_t<T>> {
    // expected-error @below {{result 0 must be a concrete 'cuda_tile.tile' at call sites, got '!cuda_tile.tile<preview$sym_t<T>>'}}
    %result = cuda_tile.preview$call @callee(%arg0)
      : (!cuda_tile.tile<1xf32>)
        -> !cuda_tile.tile<!cuda_tile.preview$sym_t<T>>
    cuda_tile.return %result
      : !cuda_tile.tile<!cuda_tile.preview$sym_t<T>>
  }
}

// -----

//===----------------------------------------------------------------------===//
// preview$extern_impl
//===----------------------------------------------------------------------===//

// A scalar integer cannot bind an extern function shape parameter.

cuda_tile.preview$func extern @shape_decl<
    [S: !cuda_tile.preview$shape]>(
    %arg0: !cuda_tile.tile<1xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>
    })

// expected-error @below {{constant argument #0 for constant parameter 'S' must be a valid shape expression: expected tuple expression, got #cuda_tile<preview$expr_int<128>>}}
cuda_tile.preview$extern_impl @bad_shape_impl<[?]>
  implements @shape_decl<[128]>(
    %arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 0, 1)>>
    {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// A datatype cannot bind an extern function string parameter.

cuda_tile.preview$func extern @string_decl<
    [Tag: !cuda_tile.preview$string]>(
    %arg0: !cuda_tile.tile<1xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>
    })

// expected-error @below {{constant argument #0 for constant parameter 'Tag' must be a valid string expression: expected string expression, got f32}}
cuda_tile.preview$extern_impl @bad_string_impl<[?]>
  implements @string_decl<[f32]>(
    %arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 0, 1)>>
    {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Implementation parameters share the declaration's call-site evaluation
// context, so non-wildcard parameter names must match positionally.

cuda_tile.preview$func extern @renamed_param_decl<[M: index]>(
    %arg0: !cuda_tile.tile<[M]xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{const_params[0] name 'P' must match implemented function const_param name 'M'}}
cuda_tile.preview$extern_impl @bad_renamed_param<[P: index]>
  implements @renamed_param_decl<[P]>(
    %arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(P, 1, 1):(1, 0, 0)>>
    {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Implementation and declaration access contracts must agree exactly.

cuda_tile.preview$func extern @access_decl(
    %arg0: !cuda_tile.tile<1xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>
    })

// expected-error @below {{simt-path impl argument 0 access (out) must match extern function argument access (in)}}
cuda_tile.preview$extern_impl @bad_access
  implements @access_decl(
    %arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, 1, 1):(0, 0, 0)>>
    {cuda_tile.preview$access = #cuda_tile.preview$access<out>}) -> ()

// -----

// Scratch alignment must describe a valid power-of-two ABI boundary.

cuda_tile.preview$func extern @scratch_decl(
    %arg0: !cuda_tile.tile<1xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>
    })

cuda_tile.preview$extern_impl @bad_scratch_alignment
  implements @scratch_decl(
    %arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, 1, 1):(0, 0, 0)>>
    {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  // expected-error @below {{scratch alignment must be a power of two}}
  scratch {
    smem.size = 128,
    smem.alignment = 192
  }

// -----

//===----------------------------------------------------------------------===//
// preview$extern_impl: scratch requests
//===----------------------------------------------------------------------===//

cuda_tile.module @extern_impl_scratch_invalid {
  cuda_tile.preview$func extern @decl(
      %arg0: !cuda_tile.tile<128xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> ()

  cuda_tile.preview$extern_impl @bad_scratch_alignment
    implements @decl
    (%arg0: !cuda_tile.preview$fragment<128xf32, local,
       mem_layout = #cuda_tile.preview$layout<(128):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, 128):(0, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    // expected-error @below {{scratch alignment must be a power of two}}
    scratch {
      smem.size = #cuda_tile.preview$expr_int<128>,
      smem.alignment = #cuda_tile.preview$expr_int<192>
    }
}

// -----

// A scratch field written as a plain integer attribute can carry any width the
// attribute grammar allows, but the expression it becomes holds an int64_t.

cuda_tile.module @extern_impl_scratch_wide {
  cuda_tile.preview$func extern @decl(
      %arg0: !cuda_tile.tile<128xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> ()

  cuda_tile.preview$extern_impl @wide_scratch_size
    implements @decl
    (%arg0: !cuda_tile.preview$fragment<128xf32, local,
       mem_layout = #cuda_tile.preview$layout<(128):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, 128):(0, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    scratch {
      // expected-error @below {{scratch field 'smem.size' does not fit in a signed 64-bit integer}}
      smem.size = 9223372036854775808 : i128,
      smem.alignment = 16
    }
}

// -----

//===----------------------------------------------------------------------===//
// preview$extern_impl: fragment ABI
//===----------------------------------------------------------------------===//

// Nullary extern functions have no legal native SIMT implementation.
// expected-error @below {{extern functions must declare at least one input}}
cuda_tile.preview$func extern @empty_extern() -> ()

// -----

// Results do not make a zero-input extern function implementable.
// expected-error @below {{extern functions must declare at least one input}}
cuda_tile.preview$func extern @result_only_extern()
  -> !cuda_tile.tile<1xf32>

// -----

// Results must not be declared; outputs are inout/out fragment args.

cuda_tile.preview$func extern @results_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{extern_impl must not declare results}}
cuda_tile.preview$extern_impl @bad_results
  implements @results_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> (!cuda_tile.preview$fragment<[64] x f32, local,
      mem_layout = #cuda_tile.preview$layout<(64):(1)>,
      fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>)

// -----

// Every fragment arg must carry cuda_tile.preview$access.

cuda_tile.preview$func extern @missing_access_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %arg1: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{extern_impl argument 1 must carry a 'cuda_tile.preview$access' arg attr}}
cuda_tile.preview$extern_impl @bad_missing_access
  implements @missing_access_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
   %arg1: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>) -> ()

// -----

// out/inout args must form a single contiguous run.
// Pattern: in, out, in, out — two out args with an in arg between them.

cuda_tile.preview$func extern @noncontiguous_decl(
    %a: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %b: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<out>},
    %c: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %d: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
  -> (!cuda_tile.tile<64xf32>, !cuda_tile.tile<64xf32>)

// expected-error @below {{out/inout fragment args must form a single contiguous run; found 2 out/inout arg(s) spanning indices 1..3 with an 'in' arg in between}}
cuda_tile.preview$extern_impl @bad_noncontiguous
  implements @noncontiguous_decl
  (%a: !cuda_tile.preview$fragment<[64] x f32, local,
       mem_layout = #cuda_tile.preview$layout<(64):(1)>,
       fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
   %b: !cuda_tile.preview$fragment<[64] x f32, local,
       mem_layout = #cuda_tile.preview$layout<(64):(1)>,
       fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<out>},
   %c: !cuda_tile.preview$fragment<[64] x f32, local,
       mem_layout = #cuda_tile.preview$layout<(64):(1)>,
       fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
   %d: !cuda_tile.preview$fragment<[64] x f32, local,
       mem_layout = #cuda_tile.preview$layout<(64):(1)>,
       fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<out>}) -> ()

// -----

// Symbolic variable in a fragment must be declared in const_params.

cuda_tile.preview$func extern @undeclared_var_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{argument 0 field shape references an undeclared symbolic variable N}}
cuda_tile.preview$extern_impl @bad_undeclared_var<[M: index]>
  implements @undeclared_var_decl
  (%arg0: !cuda_tile.preview$fragment<[N] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(N):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, N):(N, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Fragmap must have exactly 3 shape modes: (invocation, participant, footprint).

cuda_tile.preview$func extern @wrong_modes_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{fragment arg 0 fragmap must have 3 shape modes (invocation, participant, footprint), got 2}}
cuda_tile.preview$extern_impl @bad_fragmap_modes
  implements @wrong_modes_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 64):(64, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Fragmap footprint must equal the fragment element count.

cuda_tile.preview$func extern @footprint_decl(
    %arg0: !cuda_tile.tile<32xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{fragment arg 0 fragmap footprint mode size (64) must equal the fragment element count (32)}}
cuda_tile.preview$extern_impl @bad_footprint
  implements @footprint_decl
  (%arg0: !cuda_tile.preview$fragment<[32] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(32):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Distinct (non-broadcast) invocations and participants must tile the extern
// function argument exactly.

cuda_tile.preview$func extern @coverage_decl(
    %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{simt-path argument 0 fragmap covers 100 distinct invocation(s) * 1 distinct participant(s) * 1 footprint element(s), but the extern function tile has 128 element(s)}}
cuda_tile.preview$extern_impl @bad_coverage
  implements @coverage_decl
  (%arg0: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(100, 1, 1):(1, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Every non-unit SMEM participant submode must broadcast the shared buffer.

cuda_tile.preview$func extern @smem_stride_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{smem fragmap participant mode must have zero stride for every non-unit submode}}
cuda_tile.preview$extern_impl @bad_smem_participant_stride
  implements @smem_stride_decl
  (%arg0: !cuda_tile.preview$fragment<64xf32, smem,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 32, 64):(0, 1, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// All fragment args must agree on participant count.

cuda_tile.preview$func extern @part_mismatch_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %arg1: !cuda_tile.tile<16xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{fragment arg 1 fragmap participant mode size (4) must match arg 0's (1); all args of one invocation use the same participants}}
cuda_tile.preview$extern_impl @bad_participant_mismatch
  implements @part_mismatch_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
   %arg1: !cuda_tile.preview$fragment<[16] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(16):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 4, 16):(64, 16, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// All fragment args must agree on invocation count.

cuda_tile.preview$func extern @inv_mismatch_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %arg1: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{fragment arg 1 fragmap invocation mode size (256) must match arg 0's (128)}}
cuda_tile.preview$extern_impl @bad_invocation_mismatch
  implements @inv_mismatch_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
   %arg1: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(256, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// config is not permitted on the simt (fragment) path.

cuda_tile.preview$func extern @config_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{extern_impl must not carry 'config'}}
cuda_tile.preview$extern_impl @bad_config
  implements @config_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  {config = {}}

// -----

// An implementation must have the same number of args as the extern func.

cuda_tile.preview$func extern @count_mismatch_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %arg1: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{input count (1) must match extern function input count (2)}}
cuda_tile.preview$extern_impl @bad_count_mismatch
  implements @count_mismatch_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Each implementation arg's access must match the corresponding extern func
// arg's access.

cuda_tile.preview$func extern @access_mismatch_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{simt-path impl argument 0 access (out) must match extern function argument access (in)}}
cuda_tile.preview$extern_impl @bad_access_mismatch
  implements @access_mismatch_decl
  (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<out>}) -> ()

// -----

// Tile-typed implementation inputs are not supported.
cuda_tile.preview$func extern @tile_impl_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{function_type must have at least one input and all inputs must be fragment types}}
cuda_tile.preview$extern_impl @bad_tile_impl
  implements @tile_impl_decl
  (%arg0: !cuda_tile.tile<1xf32>) -> ()

// -----

// Nullary implementations are not supported.
cuda_tile.preview$func extern @nullary_impl_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{function_type must have at least one input and all inputs must be fragment types}}
cuda_tile.preview$extern_impl @bad_nullary_impl
  implements @nullary_impl_decl
  () -> ()

// -----

// Nested participant shape and stride structures must match.

cuda_tile.preview$func extern @smem_structure_decl(
    %arg0: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @+5 {{mode 1 is a tuple in shapes and an integer in strides}}
cuda_tile.preview$extern_impl @bad_smem_participant_structure
  implements @smem_structure_decl
  (%arg0: !cuda_tile.preview$fragment<64xf32, smem,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, (2, 16), 64):(0, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Matching participant structures must still simplify to signed 64-bit shape
// and stride integers. An arithmetic expression over literals is not folded
// when its result leaves the range, so it stays a stride the SMEM rule cannot
// evaluate.

cuda_tile.preview$func extern @smem_stride_range_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{smem fragmap participant mode must simplify to signed 64-bit shape and stride integers}}
cuda_tile.preview$extern_impl @bad_smem_participant_stride_range
  implements @smem_stride_range_decl
  (%arg0: !cuda_tile.preview$fragment<1xf32, smem,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 9223372036854775807 * 2, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Every fragmap shape leaf must be positive.

cuda_tile.preview$func extern @negative_leaf_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @+5 {{shape extents must be positive, but mode 0.0 is -2}}
cuda_tile.preview$extern_impl @bad_negative_fragmap_leaves
  implements @negative_leaf_decl
  (%arg0: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<((-2, -2), 1, 1):((1, 2), 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Concrete mode products must fit in a signed 64-bit integer.

cuda_tile.preview$func extern @overflow_mode_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{fragmap modes must contain signed 64-bit shape leaves with representable products}}
cuda_tile.preview$extern_impl @bad_fragmap_product_overflow
  implements @overflow_mode_decl
  (%arg0: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<((3037000500, 3037000500), 1, 1):((1, 3037000500), 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// The layout attribute admits an extent that no longer folds to a literal
// because the fold left the signed 64-bit range, so the fragmap modes are still
// checked for integer leaves here.

cuda_tile.preview$func extern @wide_leaf_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{fragmap modes must contain signed 64-bit shape leaves with representable products}}
cuda_tile.preview$extern_impl @bad_fragmap_wide_leaf
  implements @wide_leaf_decl
  (%arg0: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 9223372036854775807 + 1, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// A concrete fragment shape gets its element count from the fragment verifier.

cuda_tile.preview$func extern @overflow_shape_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

cuda_tile.preview$extern_impl @bad_fragment_shape_overflow
  implements @overflow_shape_decl
  // expected-error @below {{fragment element count must fit in a signed 64-bit integer}}
  (%arg0: !cuda_tile.preview$fragment<3037000500x3037000500xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 1, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Warp synchronization requires whole physical warps.

cuda_tile.preview$func extern @warp_p33_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{synchronization 'warp' requires the fragmap participant mode size (33) to be a positive multiple of 32}}
cuda_tile.preview$extern_impl @bad_warp_p33
  implements @warp_p33_decl
  (%arg0: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 33, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  synchronization = #cuda_tile.preview$synchronization<warp>

// -----

// Warpgroup synchronization requires whole 128-thread warpgroups.

cuda_tile.preview$func extern @warpgroup_p64_decl(
    %arg0: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// expected-error @below {{synchronization 'warpgroup' requires the fragmap participant mode size (64) to be a positive multiple of 128}}
cuda_tile.preview$extern_impl @bad_warpgroup_p64
  implements @warpgroup_p64_decl
  (%arg0: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 64, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  synchronization = #cuda_tile.preview$synchronization<warpgroup>

// -----

// The compact const_params syntax must reject the same parameter types the
// generic #cuda_tile.preview$const_param spelling does, rather than building an
// unverified attribute.

// expected-error @below {{constant parameter type must be index, !cuda_tile.preview$shape, !cuda_tile.preview$Datatype, or !cuda_tile.preview$string, got 'i64'}}
cuda_tile.preview$func extern @bad_const_param_type<[M: i64]>(
    %arg0: !cuda_tile.tile<[M]xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// -----

//===----------------------------------------------------------------------===//
// Library linking
//===----------------------------------------------------------------------===//

// A resolved implementation target must be an extern preview function.
cuda_tile.module @bad_impl_target_kind {
  cuda_tile.entry @not_an_extern_func() {
    cuda_tile.return
  }

  // expected-error @below {{'not_an_extern_func' does not reference a valid extern preview$func}}
  cuda_tile.preview$extern_impl @bad_target_kind
    implements @not_an_extern_func
    (%arg0: !cuda_tile.preview$fragment<[1] x f32, local,
       mem_layout = #cuda_tile.preview$layout<(1):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, 1):(1, 0, 0)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// A resolved call target must be a preview function, not an
// implementation symbol.
cuda_tile.module @bad_call_target_kind {
  cuda_tile.preview$func extern @decl_for_wrong_kind(
      %arg0: !cuda_tile.tile<2xi16> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})

  cuda_tile.preview$extern_impl @not_a_func
    implements @decl_for_wrong_kind
    (%arg0: !cuda_tile.preview$fragment<[2] x i16, local,
       mem_layout = #cuda_tile.preview$layout<(2):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, 2):(2, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$func @caller() {
    %0 = cuda_tile.constant <i16: 1> : !cuda_tile.tile<2xi16>
    // expected-error @below {{'not_a_func' does not reference a valid preview$func}}
    cuda_tile.preview$call @not_a_func(%0)
      : (!cuda_tile.tile<2xi16>) -> ()
  }
}

// -----

//===----------------------------------------------------------------------===//
// preview$fragment type
//===----------------------------------------------------------------------===//

// Unknown memory space keyword.
// expected-error @+1 {{unknown memory space 'gmem'}}
%0 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, gmem, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>>

// -----

// Tensor memory is not a valid CUDA Tile memory space keyword (the enum only
// has `local` and `smem`), so `tmem` is rejected at parse time.
// expected-error @+1 {{unknown memory space 'tmem'}}
%1 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, tmem, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>>

// -----

// The `mem_layout` keyword is mandatory.
// expected-error @+1 {{expected 'mem_layout'}}
%2 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>>

// -----

// The `fragmap` keyword is mandatory.
// expected-error @+1 {{expected 'fragmap'}}
%3 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, <(8, 1, 1):(1, 0, 0)>>

// -----

// When the optional alignment suffix is present, `alignment` must be the keyword.
// expected-error @+1 {{expected 'alignment'}}
%4 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>, badkey = 4>

// -----

// Concrete fragment shape dimensions must be positive.
// expected-error @+1 {{fragment shape dimension 1 must be positive, got 0}}
%5 = "test.frag"() : () -> !cuda_tile.preview$fragment<8x0xf32, local, mem_layout = <(8, 1):(1, 1)>, fragmap = <(8, 1, 1):(1, 0, 0)>>

// -----

// A concrete alignment must be a positive number of elements.
// expected-error @+1 {{fragment alignment must be a positive number of elements}}
%6 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>, alignment = 0>

// -----

// A parenthesized negation is the negative literal, so it is rejected the same
// way `alignment = -4` is. Accepting it here would print a type that fails to
// verify when it is read back.
// expected-error @+1 {{fragment alignment must be a positive number of elements}}
%neg_group_alignment = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>, alignment = -(4)>

// -----

// The check runs on the simplified alignment, so an expression that folds to a
// non-positive number of elements is rejected as well.
// expected-error @+1 {{fragment alignment must be a positive number of elements}}
%folded_alignment = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>, alignment = 4 - 8>

// -----

// An alignment goes through the same expression grammar as a layout, so a
// literal it cannot hold is rejected while parsing rather than stored.
// expected-error @+2 {{integer literal does not fit in a signed 64-bit integer}}
// expected-error @+1 {{expected alignment expression}}
%wide_alignment = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>, alignment = 9223372036854775808>

// -----

// An alignment is a single number of elements, so a tuple is not admissible.
// expected-error @+1 {{fragment alignment: expected integer expression, got tuple}}
%7 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>, alignment = (16,)>

// -----

// expected-error @+1 {{fragment alignment must be a power of two number of elements, got 3}}
%8 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>, alignment = 3>

// -----

// A mode with no elements addresses nothing, which the layout attribute
// rejects before the fragment sees it.
// expected-error @+1 {{shape extents must be positive, but mode 0 is 0}}
%9 = "test.frag"() : () -> !cuda_tile.preview$fragment<1xf32, local, mem_layout = <(0):(1)>, fragmap = <(1, 1, 1):(1, 0, 0)>>

// -----

// The domain of an otherwise-valid layout must stay countable.
// expected-error @+1 {{mem_layout domain size must be representable as a signed 64-bit integer}}
%overflow_mem_layout = "test.frag"() : () -> !cuda_tile.preview$fragment<1xf32, local, mem_layout = <(3037000500, 3037000500):(1, 3037000500)>, fragmap = <(1, 1, 1):(1, 0, 0)>>

// -----

// The layout's domain must cover the whole fragment.
// expected-error @+1 {{mem_layout domain size (1) must equal the fragment element count (8)}}
%10 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(1):(1)>, fragmap = <(8, 1, 1):(1, 0, 0)>>

// -----

// A padded layout addresses 15 slots to hold 8 elements.
// expected-error @+1 {{mem_layout must be a bijection on the footprint elements (a permutation of [0, 8)), but its codomain spans 15 element(s)}}
%11 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(2)>, fragmap = <(8, 1, 1):(1, 0, 0)>>

// -----

// A 2-D layout is padded in the same way when its strides leave holes.
// expected-error @+1 {{mem_layout must be a bijection on the footprint elements (a permutation of [0, 32)), but its codomain spans 35 element(s)}}
%12 = "test.frag"() : () -> !cuda_tile.preview$fragment<4x8xf32, local, mem_layout = <(4, 8):(9, 1)>, fragmap = <(1, 1, (4, 8)):(0, 0, (9, 1))>>

// -----

// A mode cannot walk backwards, which the layout attribute rejects before the
// fragment sees it. See preview_attrs_invalid.mlir.
// expected-error @+1 {{strides must be non-negative, but mode 0 is -1}}
%13 = "test.frag"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(-1)>, fragmap = <(8, 1, 1):(1, 0, 0)>>

// -----

// The codomain of an otherwise-valid layout must stay addressable.
// expected-error @+1 {{mem_layout storage span must be representable as a signed 64-bit integer}}
%overflow_mem_layout_span = "test.frag"() : () -> !cuda_tile.preview$fragment<4xf32, local, mem_layout = <(2, 2):(1, 9223372036854775807)>, fragmap = <(4, 1, 1):(1, 0, 0)>>

// -----

// Shared-memory offsets are held in a signed 32-bit field.
// expected-error @+1 {{smem mem_layout storage span (4290772993) exceeds the current signed 32-bit offset ABI limit (2147483647)}}
%14 = "test.frag"() : () -> !cuda_tile.preview$fragment<1024xf32, smem, mem_layout = <(1024):(4194304)>, fragmap = <(1, 1, 1024):(0, 0, 1)>>

// -----

// A symbolic fragment shape leaves the element count unknown, but a static
// layout is still checked for the properties that do not depend on it. A
// stride-zero mode broadcasts several elements onto one slot.
// expected-error @+1 {{mem_layout must be a bijection on the footprint elements, but its modes overlap or leave gaps}}
%15 = "test.frag"() : () -> !cuda_tile.preview$fragment<[N]xf32, local, mem_layout = <(4):(0)>, fragmap = <(1, 1, N):(N, 0, 1)>>

// -----

// Two modes sharing a stride alias onto the same offsets.
// expected-error @+1 {{mem_layout must be a bijection on the footprint elements, but its modes overlap or leave gaps}}
%16 = "test.frag"() : () -> !cuda_tile.preview$fragment<[N]xf32, local, mem_layout = <(2, 2):(1, 1)>, fragmap = <(1, 1, N):(N, 0, 1)>>

// -----

// A padded layout can never be a permutation of [0, N), so the holes are
// caught while the element count is still unknown. A concrete shape reports
// them through the cosize instead.
// expected-error @+1 {{mem_layout must be a bijection on the footprint elements, but its modes overlap or leave gaps}}
%17 = "test.frag"() : () -> !cuda_tile.preview$fragment<[N]xf32, local, mem_layout = <(2, 2):(1, 8)>, fragmap = <(1, 1, N):(N, 0, 1)>>

// -----

//===----------------------------------------------------------------------===//
// preview$sym_t and symbolic tile shapes
//===----------------------------------------------------------------------===//

cuda_tile.module @kernels {
  // expected-error @+2 {{symbolic type expression must be a symbolic expression attribute}}
  testing$func @bad_symbolic_type_plain_integer(
      %arg0: !cuda_tile.preview$sym_t<10>) {
    return
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @+2 {{symbolic type expression must be a SymbolicExprVarAttr, got #cuda_tile<preview$expr_int<1>>}}
  testing$func @bad_symbolic_type_int(
      %arg0: !cuda_tile.preview$sym_t<#cuda_tile.preview$expr_int<1>>) {
    return
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @+2 {{symbolic type expression must be a SymbolicExprVarAttr, got #cuda_tile<preview$expr_add<#cuda_tile<preview$expr_var<"T">>, #cuda_tile<preview$expr_int<1>>>>}}
  testing$func @bad_symbolic_type_expr(
      %arg0: !cuda_tile.preview$sym_t<#cuda_tile.preview$expr_add<#cuda_tile.preview$expr_var<"T">, #cuda_tile.preview$expr_int<1>>>) {
    return
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @+2 {{symbolic type expression must be a SymbolicExprVarAttr, got #cuda_tile<preview$concrete_type<f32>>}}
  testing$func @bad_symbolic_type_concrete(
      %arg0: !cuda_tile.preview$sym_t<#cuda_tile.preview$concrete_type<f32>>) {
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// PreviewOps.cpp: parser error paths
//===----------------------------------------------------------------------===//

// PreviewOps.cpp: parseCompactConstParams / parseCompactConstArgs
// remaining error early-returns.  Each split section exercises one specific
// `if (parser.parse...)` failure in the compact-list parsers.

// Test: parseCompactConstParams compact form missing colon —
// parser.parseColon fails in `name : type`.
cuda_tile.module @test {
  // expected-error @+1 {{expected ':'}}
  cuda_tile.preview$func extern @decl_missing_colon<[M i64]>(
      %arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: parseCompactConstParams missing closing ']' —
// parser.parseRSquare fails.
cuda_tile.module @test {
  // expected-error @+1 {{expected ']'}}
  cuda_tile.preview$func extern @decl_missing_rsquare<[M: index(
      %arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: parseCompactConstParams missing closing '>' —
// the outer parser.parseGreater fails after constParams.
cuda_tile.module @test {
  // expected-error @+1 {{expected '>'}}
  cuda_tile.preview$func extern @decl_missing_gt<[M: index]
      (%arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: parseCompactConstParams empty `<[]>` — exercises the immediate
// parseOptionalRSquare success path returning an empty ArrayAttr.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_no_params<[]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: parseCompactConstParams non-integer type parameter.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_dtype_param<[Q: !cuda_tile.preview$Datatype]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: parseCompactConstArgs missing '[' after the opening '<'.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_bad_args<[M: index]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_missing_lsquare
    // expected-error @below {{expected '['}}
    implements @decl_for_bad_args<128>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: parseCompactConstArgs missing its closing ']'.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_no_rsquare<[M: index]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_no_rsquare
    // expected-error @below {{expected ']'}}
    implements @decl_for_no_rsquare<[128
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: parseCompactConstArgs rejects an unrecognized leading token.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_bad_arg<[M: index]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_bad_arg<[K: index]>
    // expected-error @+1 {{expected integer, identifier, or '('}}
    implements @decl_for_bad_arg<[+]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: parseCompactConstArgs accepts a bare symbolic variable.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_keyword_arg<[N: index]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_keyword_arg<[N: index]>
    implements @decl_keyword_arg<[N]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: parseCompactConstArgs accepts a bare type.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_type_arg<[
      T: !cuda_tile.preview$Datatype]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_type_arg<[?]>
    implements @decl_type_arg<[f32]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: parseCompactConstArgs accepts the verbose attribute form.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_verbose_arg<[T: index]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_verbose_arg<[?]>
    implements @decl_verbose_arg<[
      #cuda_tile.preview$expr_int<32>]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: parseCompactConstArgs accepts an explicit empty list.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_no_const_args(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_no_const_args
    implements @decl_no_const_args<[]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: extern FuncOp parser `parser.parseSymbolName` failure — drop the
// `@name` symbol entirely.
cuda_tile.module @test {
  // expected-error @below {{expected valid '@'-identifier for symbol name}}
  cuda_tile.preview$func extern (
      %arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: ExternImplOp::parse `parser.parseOptionalAttrDict` failure — feed
// a malformed trailing attribute dict after a native fragment signature.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_bad_trailing_attrs(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  cuda_tile.preview$extern_impl @impl_bad_trailing_attrs_decl
    implements @decl_bad_trailing_attrs
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    // expected-error @below {{expected attribute value}}
    {foo = +}
}

// -----

// Test: ExternImplOp::parse `parser.parseSymbolName` failure — drop the
// `@name` for the impl op.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_no_sym_impl(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  // expected-error @below {{expected valid '@'-identifier for symbol name}}
  cuda_tile.preview$extern_impl implements @decl_for_no_sym_impl
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: ExternImplOp::parse `parser.parseAttribute(implementsAttr)` failure
// path — feed a non-attribute token where the `implements @name` symbol-ref
// is expected.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_bad_implements(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  cuda_tile.preview$extern_impl @impl_bad_implements
    // expected-error @below {{expected attribute value}}
    implements +
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: ExternImplOp::parse — `arch` keyword followed by no `=`. Drives the
// `if (parser.parseEqual())` failure inside the arch optional clause.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_arch_no_eq(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  cuda_tile.preview$extern_impl @impl_arch_no_eq
    implements @decl_for_arch_no_eq
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    // expected-error @below {{expected '='}}
    arch ["sm_90"]
}

// -----

// Test: ExternImplOp::parse — `arch = <bad attribute>`. Drives the
// `if (parser.parseAttribute(attr))` failure inside the arch clause.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_arch_bad(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  cuda_tile.preview$extern_impl @impl_arch_bad
    implements @decl_for_arch_bad
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    // expected-error @below {{expected attribute value}}
    arch = +
}

// -----

// Test: ExternImplOp::parse — trailing `parser.parseOptionalAttrDict`
// failure after a well-formed `arch` clause. Drives the
// `if (failed(parser.parseOptionalAttrDict(result.attributes)))` branch.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_for_impl_bad_attrs(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  cuda_tile.preview$extern_impl @impl_bad_trailing_attrs
    implements @decl_for_impl_bad_attrs
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    // expected-error @below {{expected attribute value}}
    {foo = +}
}

// -----

//===----------------------------------------------------------------------===//
// parseExternAttrDict — retained native signature attribute-dictionary paths.
//===----------------------------------------------------------------------===//

// Test: an explicit empty argument dictionary parses successfully before a
// deliberately malformed later clause terminates parsing.
cuda_tile.module @test {
  cuda_tile.preview$extern_impl @empty_attr_dict_impl
    implements @unresolved_attr_dict_decl
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>> {}) -> ()
    // expected-error @below {{expected attribute value}}
    arch = +
}

// -----

// Test: a key without '=' exercises parseExternAttrDict's parseEqual failure.
cuda_tile.module @test {
  cuda_tile.preview$extern_impl @attr_no_equal_impl
    implements @unresolved_attr_dict_decl
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      // expected-error @below {{expected '='}}
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>> {foo}) -> ()
}

// -----

// Test: an unparseable value exercises parseExternAttrDict's value failure.
cuda_tile.module @test {
  cuda_tile.preview$extern_impl @attr_bad_value_impl
    implements @unresolved_attr_dict_decl
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      // expected-error @below {{expected attribute value}}
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>> {foo = +}) -> ()
}

// -----

// Test: a missing closing brace exercises parseExternAttrDict's final check.
cuda_tile.module @test {
  cuda_tile.preview$extern_impl @attr_no_rbrace_impl
    implements @unresolved_attr_dict_decl
    // expected-error @+3 {{expected '}'}}
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>> {foo = 42) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// parseCompactConstParams — missing leading '['.
//===----------------------------------------------------------------------===//

// -----

// Test: parseCompactConstParams `parser.parseLSquare` failure — open `<` but
// no leading `[` before the first param entry.
cuda_tile.module @test {
  // expected-error @below {{expected '['}}
  cuda_tile.preview$func extern @decl_no_lsquare<M: index>(
      %arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

// PreviewOps.cpp: verifier error paths that were previously
// uncovered. Each split section exercises one specific verifier error message
// inside CallOp::verifySymbolUses or an extern FuncOp verifier.

//===----------------------------------------------------------------------===//
// CallOp::verifySymbolUses — error paths against preview$func target.
//===----------------------------------------------------------------------===//

// Test: preview$call with operand-count mismatch against the
// preview$func target — drives the
// `if (fnType.getNumInputs() != getNumOperands())` error.
cuda_tile.module @test_call_operand_count_mismatch {
  preview$func @callee_one_arg(%arg0: !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
    return %arg0 : tile<2xi16>
  }
  preview$func @caller(%a: !cuda_tile.tile<2xi16>, %b: !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
    // expected-error @below {{op incorrect number of operands for callee}}
    %0 = preview$call @callee_one_arg(%a, %b) : (!cuda_tile.tile<2xi16>, !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16>
    return %0 : tile<2xi16>
  }
}

// -----

// Test: preview$call with a result-count mismatch — drives the
// `if (fnType.getNumResults() != getNumResults())` error.
cuda_tile.module @test_call_result_count_mismatch {
  preview$func @callee_one_result(%arg0: !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
    return %arg0 : tile<2xi16>
  }
  preview$func @caller(%a: !cuda_tile.tile<2xi16>) {
    // expected-error @below {{op incorrect number of results for callee}}
    preview$call @callee_one_result(%a) : (!cuda_tile.tile<2xi16>) -> ()
  }
}

// -----

// Test: extern call with a non-CudaTile operand type at the call site —
// drives the `if (!llvm::isa<CudaTileType>(type))` error inside the lambda.
cuda_tile.module @test_extern_call_non_cuda_tile_operand {
  cuda_tile.preview$func extern @decl_one_arg(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>
  preview$func @caller(%a: index) -> !cuda_tile.tile<128xf32> {
    // expected-error @below {{operand 0 must be a cuda_tile type or token when calling an extern preview$func}}
    %0 = preview$call @decl_one_arg(%a) : (index) -> !cuda_tile.tile<128xf32>
    return %0 : tile<128xf32>
  }
}

// -----

// Test: extern call with mismatched data-operand count vs declaration —
// drives `declType.getNumInputs() != dataOperandCount`.
cuda_tile.module @test_extern_call_input_count_mismatch {
  cuda_tile.preview$func extern @decl_two_args(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %arg1: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>
  preview$func @caller(%a: !cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32> {
    // expected-error @below {{input count (1) must match extern preview$func input count (2)}}
    %0 = preview$call @decl_two_args(%a) : (!cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>
    return %0 : tile<128xf32>
  }
}

// -----

// Test: extern call with mismatched data-result count vs declaration —
// drives `declType.getNumResults() != dataResultCount`.
cuda_tile.module @test_extern_call_result_count_mismatch {
  cuda_tile.preview$func extern @decl_one_arg_one_res(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>
  preview$func @caller(%a: !cuda_tile.tile<128xf32>) {
    // expected-error @below {{result count (0) must match extern preview$func result count (1)}}
    preview$call @decl_one_arg_one_res(%a) : (!cuda_tile.tile<128xf32>) -> ()
  }
}

//===----------------------------------------------------------------------===//
// Extern FuncOp verification — duplicate const_params.
//===----------------------------------------------------------------------===//

// -----

// Test: extern function with duplicate const_params names — drives the duplicate
// detection path `if (!inserted)` in the const_params loop.
cuda_tile.module @test_extern_func_dup_params {
  // expected-error @below {{duplicate constant parameter name: 'M'}}
  cuda_tile.preview$func extern @dup_params<[M: index, M: index]>(
      %arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

cuda_tile.module @kernels {
  preview$func @callee_two_args(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xf32>) {}

  preview$func @test_operand_count_mismatch() {
    %0 = constant <i16: 1> : !cuda_tile.tile<2xi16>
    // expected-error @below {{incorrect number of operands for callee}}
    cuda_tile.preview$call @callee_two_args(%0) : (!cuda_tile.tile<2xi16>) -> ()
  }
}

// -----

cuda_tile.module @kernels {
  preview$func @callee_returns_one() -> !cuda_tile.tile<2xf32> {
    %0 = constant <f32: 2.0> : !cuda_tile.tile<2xf32>
    return %0 : !cuda_tile.tile<2xf32>
  }

  preview$func @test_result_count_mismatch() {
    // expected-error @below {{incorrect number of results for callee}}
    %0:2 = cuda_tile.preview$call @callee_returns_one() : () -> (!cuda_tile.tile<2xf32>, !cuda_tile.tile<2xi16>)
  }
}

// -----

// Test: call to an extern function with a non-CudaTileType result.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_three(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  preview$func @test_call_extern_non_cudatile_result(%a: !cuda_tile.tile<128xf32>) {
    // expected-error @below {{result 0 must be a cuda_tile type or token when calling an extern preview$func, got 'i32'}}
    %0 = cuda_tile.preview$call @decl_three(%a) : (!cuda_tile.tile<128xf32>) -> i32
  }
}

// -----

// Test: call to an extern function with an input-count mismatch.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_five(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  preview$func @test_call_extern_input_count_mismatch(
      %a: !cuda_tile.tile<128xf32>, %b: !cuda_tile.tile<128xf32>) {
    // expected-error @below {{input count (2) must match extern preview$func input count (1)}}
    %0 = cuda_tile.preview$call @decl_five(%a, %b) : (!cuda_tile.tile<128xf32>, !cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>
  }
}

// -----

// Test: call to an extern function with a result-count mismatch.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_six(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  preview$func @test_call_extern_result_count_mismatch(%a: !cuda_tile.tile<128xf32>) {
    // expected-error @below {{result count (2) must match extern preview$func result count (1)}}
    %0:2 = cuda_tile.preview$call @decl_six(%a) : (!cuda_tile.tile<128xf32>) -> (!cuda_tile.tile<128xf32>, !cuda_tile.tile<128xf32>)
  }
}

// -----

// Test: call to an extern function with token operands (tokens are skipped).
// Valid case: 1 data operand + 1 token, decl expects 1 input
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_with_token(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  preview$func @test_call_extern_with_token(
      %a: !cuda_tile.tile<128xf32>, %tok: !cuda_tile.token) {
    %0 = cuda_tile.preview$call @decl_with_token(%a, %tok) : (!cuda_tile.tile<128xf32>, !cuda_tile.token) -> !cuda_tile.tile<128xf32>
  }
}

// -----

// Test: extern function with duplicate const parameter names.
cuda_tile.module @test {
  // expected-error @below {{duplicate constant parameter name: 'M'}}
  cuda_tile.preview$func extern @decl_dup_param<[M: index, M: index]>(
      %arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: extern_impl const_args count mismatch (decl has 2 params, impl has 1).
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_two_params<[M: index, N: index]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  // expected-error @below {{constant argument count (1) must match extern function constant parameter count (2)}}
  cuda_tile.preview$extern_impl @impl_args_mismatch
    implements @decl_two_params<[#cuda_tile.preview$expr_int<128>]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: extern_impl input count mismatch with its declaration.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_one_input(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  // expected-error @below {{input count (2) must match extern function input count (1)}}
  cuda_tile.preview$extern_impl @impl_input_mismatch
    implements @decl_one_input
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
     %arg1: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: extern_impl with duplicate const parameter names.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_impl_dup(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  // expected-error @below {{duplicate constant parameter name: 'X'}}
  cuda_tile.preview$extern_impl @impl_dup_param<[X: index, X: index]>
    implements @decl_impl_dup
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: valid native extern_impl with const_params, const_args,
// synchronization, and architecture constraints.
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_valid<[M: index]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<128xf32>

  cuda_tile.preview$extern_impl @impl_valid<[M: index]>
    implements @decl_valid<[M]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, 128, 1):(0, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
    synchronization = #cuda_tile.preview$synchronization<cta> arch = ["sm90"]
}

// -----

// PreviewOps.cpp: parseCompactConstParams / parseCompactConstArgs —
// coverage for the verbose-attribute parse-failure branch
// `optRes.has_value() && succeeded(*optRes)` returning false because
// `parseOptionalAttribute` set has_value but the inner parse failed.

// Test: parseCompactConstParams — a `#cuda_tile.preview$const_param<...>` attribute
// with a malformed body. `parseOptionalAttribute` sees the leading `#`,
// attempts the attribute parse, and the custom op parser reports the bad
// compact parameter.
cuda_tile.module @t {
  // expected-error @+1 {{custom op 'cuda_tile.preview$func' expected valid keyword}}
  cuda_tile.preview$func extern @decl_bad_verbose_const_param<[#cuda_tile.preview$const_param<name = "Q", type = i64]>(
      %arg0: !cuda_tile.tile<128xf32>)
    -> !cuda_tile.tile<128xf32>
}

// -----

// Test: parseCompactConstArgs — a `#cuda_tile.preview$expr_int<...>` attribute with a
// malformed body (missing the integer value). `parseOptionalAttribute` sees
// `#` then fails on the inner integer parse and reports the malformed
// symbolic integer attribute.
cuda_tile.module @t {
  cuda_tile.preview$func extern @decl_ok_for_bad_compact_arg<[M: index]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_bad_compact_arg
    // expected-error @+2 {{expected integer value}}
    // expected-error @+1 {{failed to parse CudaTile_SymbolicExprIntAttr}}
    implements @decl_ok_for_bad_compact_arg<[#cuda_tile.preview$expr_int<>]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// PreviewOps.cpp: ExternImplOp::parse error paths in the optional
// const_params/const_args clauses, the `implements` keyword/attribute, the
// optional synchronization and arch clauses. Each split section drives a
// single specific early-return in the custom parser.

// Test: ExternImplOp::parse — const_params is started with `<` but the
// closing `>` after the const_params list is missing. Drives the
// `parseCompactConstParams(...) || parser.parseGreater()` early-return path.
cuda_tile.module @t {
  cuda_tile.preview$func extern @decl_for_impl_no_gt(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  // expected-error @+1 {{expected '>'}}
  cuda_tile.preview$extern_impl @impl_no_gt<[T: index]
    implements @decl_for_impl_no_gt
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: ExternImplOp::parse — missing `implements` keyword. Drives the
// `parser.parseKeyword("implements")` early-return path.
cuda_tile.module @t {
  cuda_tile.preview$func extern @decl_for_no_implements(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_no_implements
    // expected-error @+1 {{expected 'implements'}}
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// ExternImplOp::parse: `implements` must name a symbol.
cuda_tile.module @test {
  cuda_tile.preview$extern_impl @bad_implements
    // expected-error @+1 {{invalid kind of attribute specified}}
    implements 7
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: ExternImplOp::parse — const_args starts with `<` but the closing `>`
// is missing. Drives the `parseCompactConstArgs(...) || parser.parseGreater()`
// early-return on the const_args path.
cuda_tile.module @t {
  cuda_tile.preview$func extern @decl_for_const_args_no_gt<[M: index]>(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @const_arg_missing_greater
    // expected-error @+1 {{expected '>'}}
    implements @decl_for_const_args_no_gt<[128]
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
}

// -----

// Test: ExternImplOp::parse — native signature parse failure surfaced from
// inside ExternImplOp::parse. We make the argument list fail by omitting the
// `(` after `implements <symbol>`, which makes parseFunctionArgumentList fail.
cuda_tile.module @t {
  cuda_tile.preview$func extern @decl_for_impl_bad_sig(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_bad_sig
    // expected-error @+1 {{expected '('}}
    implements @decl_for_impl_bad_sig
    %arg0
}

// -----

// Test: ExternImplOp::parse — `arch` without the `=`. Drives the
// `parser.parseEqual()` failure inside the optional arch clause.
cuda_tile.module @t {
  cuda_tile.preview$func extern @decl_for_arch_no_eq(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_arch_no_eq
    implements @decl_for_arch_no_eq
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    // expected-error @+1 {{expected '='}}
    arch ["sm90"]
}

// -----

// Test: ExternImplOp::parse — `arch =` followed by something that isn't an
// attribute. Drives the `parser.parseAttribute(attr)` failure inside the
// optional arch clause.
cuda_tile.module @t {
  cuda_tile.preview$func extern @decl_for_arch_bad_attr(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @impl_arch_bad_attr
    implements @decl_for_arch_bad_attr
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    // expected-error @+1 {{expected attribute value}}
    arch = +
}

// -----

// PreviewOps.cpp: ExternImplOp::verify

// Test: extern_impl const_args with invalid attribute type (not a symbolic expression)
// Use generic format to bypass the custom parser so ODS constraint triggers at verify.
cuda_tile.module @test {
  "cuda_tile.preview$func"() <{arg_attrs = [{cuda_tile.preview$access = #cuda_tile.preview$access<in>}], const_params = [#cuda_tile.preview$const_param<name = "M", type = index>], function_type = (!cuda_tile.tile<128xf32>) -> (), is_extern, sym_name = "decl_with_param"}> ({}) {sym_visibility = "private"} : () -> ()

  // expected-error @below {{attribute 'const_args' failed to satisfy constraint: array of symbolic expression attribute values}}
  "cuda_tile.preview$extern_impl"() <{
    arg_attrs = [{cuda_tile.preview$access = #cuda_tile.preview$access<in>}],
    const_args = [#cuda_tile.ptr_attr<multicast>],
    function_type = (!cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>) -> (),
    implements = @decl_with_param, res_attrs = [],
    sym_name = "impl_bad_const_arg"}> ({}) {sym_visibility = "private"} : () -> ()
}

// -----

// Test: extern_impl with invalid arch element
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_arch(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  // expected-error @below {{arch[1] must be a non-empty string}}
  cuda_tile.preview$extern_impl @impl_bad_arch
    implements @decl_arch
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    arch = ["sm90", ""]
}

// -----

//===----------------------------------------------------------------------===//
// Types.cpp: TileType::verify on a symbolic shape
//===----------------------------------------------------------------------===//

// Types.cpp: TileType::verify on a symbolic shape. A public-preview extern
// declaration is the only way to spell one, so this case needs the opt-in; the
// concrete-shape cases live in Types_cpp_coverage_tile_shapes.mlir.

// F4E2M1FN with symbolic dimension (valid, skip even check).
cuda_tile.module @test {
  cuda_tile.preview$func extern @decl_f4_sym<[N: index]>(
      %arg0: !cuda_tile.tile<[N]xf4E2M1FN> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<[N]xf4E2M1FN>
}

// -----

//===----------------------------------------------------------------------===//
// CudaTile.cpp: ReturnOp::verify, preview$func arm
//===----------------------------------------------------------------------===//

// ReturnOp::verify() keys off the enclosing function op, and the
// cuda_tile.preview$func branch is a separate code path from the
// cuda_tile.entry and cuda_tile.testing$func ones. Keep these cases on
// preview$func (and therefore behind --preview=simt): rewriting them to
// testing$func would silently retarget them at the test-build-only branch.
// The entry and testing$func branches are covered by
// CudaTile_cpp_coverage_v4.mlir and Dialect/CudaTile/invalid.mlir.

// Coverage target: lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers ReturnOp::verify() - preview$func return operand count mismatch
// Covers ReturnOp::verify() - preview$func return type mismatch

// Return operand count mismatch in preview$func
cuda_tile.module @func_return_count_mismatch {
  preview$func @test_wrong_return_count() -> !cuda_tile.tile<i32> {
    // expected-error @below{{has 0 operands, but enclosing function (@test_wrong_return_count) returns 1}}
    cuda_tile.return
  }
}

// -----

// Return type mismatch in preview$func
cuda_tile.module @func_return_type_mismatch {
  preview$func @test_wrong_return_type(%a : !cuda_tile.tile<f32>) -> !cuda_tile.tile<i32> {
    // expected-error @below{{type of return operand 0 ('!cuda_tile.tile<f32>') doesn't match function result type ('!cuda_tile.tile<i32>') in function @test_wrong_return_type}}
    cuda_tile.return %a : tile<f32>
  }
}

// -----

// Return with too many operands in preview$func
cuda_tile.module @func_return_too_many {
  preview$func @test_too_many_return(%a : !cuda_tile.tile<i32>, %b : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    // expected-error @below{{has 2 operands, but enclosing function (@test_too_many_return) returns 1}}
    cuda_tile.return %a, %b : tile<i32>, tile<i32>
  }
}

// -----

// Return inside if inside preview$func (tests the IfOp traversal in verify)
cuda_tile.module @func_return_in_if {
  preview$func @test_return_in_if(%cond : !cuda_tile.tile<i1>, %a : !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    cuda_tile.if %cond {
      cuda_tile.return %a : tile<i32>
    }
    cuda_tile.return %a : tile<i32>
  }
}
