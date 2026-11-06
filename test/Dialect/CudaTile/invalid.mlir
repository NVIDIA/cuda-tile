// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// expected-error @below{{expected '<'}}
%0 = cuda_tile.constant "foo" : !cuda_tile.tile<i8>

// -----

// expected-error @below{{expected '<'}}
%0 = cuda_tile.constant 10.0 : f32

// -----

// No MLIR tensor types. Only !cuda_tile.tile is allowed
// expected-error @below{{custom op 'cuda_tile.constant' result #0 must be tile of i1 or i8 or i16 or i32 or i64 or f16 or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 or f8E8M0FNU or f4E2M1FN values, but got 'tensor<f32>'}}
%0 = cuda_tile.constant <f32: 10.0> : tensor<f32>

// -----

// expected-error @below{{expected integer value}}
%0 = cuda_tile.constant <i8: true> : tile<i8>

// -----

// expected-error @below{{expected integer value}}
%0 = cuda_tile.constant <i8: false> : tile<i8>

// -----

cuda_tile.module @kernels {
  // expected-error-re @below{{failed to verify 'pointeeType': f16 or bf16 or f32 or tf32 or f64 or f8E4M3FN or f8E5M2 or f8E8M0FNU or f4E2M1FN{{( or f8E5M3FNU)?}} or i1 or i8 or i16 or i32 or i64}}
  testing$func @kernel(%arg0: !cuda_tile.tile<ptr<tile<2x2xf32>>>) {
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @below{{failed to verify constraint: region with 1 blocks}}
  "cuda_tile.testing$func"() ({ }) {function_type = () -> (), sym_name = "foo"} : () -> ()
}

// -----

// expected-error @below{{expects parent op to be one of 'cuda_tile.for, cuda_tile.if, cuda_tile.loop'}}
cuda_tile.continue

// -----


cuda_tile.module @kernels {
// expected-note @below{{see unexpected ancestor operation}}
cuda_tile.entry @kernel() {
  %cond = "cond"() : () -> !cuda_tile.tile<i1>
  cuda_tile.if %cond {
    // expected-error @below{{op can only be nested within a ancestor chain of 'cuda_tile.for', 'cuda_tile.loop', 'cuda_tile.if' operations}}
    cuda_tile.continue
  }
}
}

// -----

%c4_i32 = cuda_tile.constant <i32: 4> : !cuda_tile.tile<i32>
// expected-error @below{{operand #0 must be 0D tile of i1 values, but got '!cuda_tile.tile<i32>'}}
"cuda_tile.if"(%c4_i32) ({
  cuda_tile.yield
}, {
}) : (!cuda_tile.tile<i32>) -> ()

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
%c1_i32 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
cuda_tile.for %iv in (%c0_i32 to %c1_i32, step %c1_i32) : !cuda_tile.tile<i32> {
  // expected-error @below{{`for` is missing a valid terminator. `continue` op should have operand types that match the parent loop return types: (), but found: ('!cuda_tile.tile<i32>')}}
  cuda_tile.continue %c0_i32 : !cuda_tile.tile<i32>
}

// -----

%0 = cuda_tile.constant <i16: 1> : !cuda_tile.tile<i16>
// expected-error @below{{'cuda_tile.negi' op 'no_unsigned_wrap' overflow flag is not supported}}
%1 = cuda_tile.negi %0 overflow<no_unsigned_wrap> : !cuda_tile.tile<i16>

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
// expected-error @below{{`loop` is missing a valid terminator. `continue` op should have operand types that match the parent loop iter_values: ('!cuda_tile.tile<i32>'), but found: ()}}
cuda_tile.loop iter_values(%arg0 = %c0_i32) : tile<i32> { }

// -----

// expected-error @below{{expects parent op to be one of 'cuda_tile.if, cuda_tile.loop'}}
cuda_tile.break

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
%c1_i32 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-note @below{{see unexpected ancestor operation}}
cuda_tile.for %iv in (%c0_i32 to %c1_i32, step %c1_i32) : !cuda_tile.tile<i32> {
  %cond = "cond"() : () -> !cuda_tile.tile<i1>
  cuda_tile.if %cond {
    // expected-error @below{{op can only be nested within a ancestor chain of 'cuda_tile.loop', 'cuda_tile.if' operations}}
    cuda_tile.break
  }
}

// -----


%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
cuda_tile.loop {
  // expected-error @below{{operand types must correspond to the parent loop result types}}
  cuda_tile.break %c0_i32 : !cuda_tile.tile<i32>
}

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<1xi32>

// expected-error@+1 {{op operand #0 must be 0D tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<1xi32>'}}
"cuda_tile.for"(%c0_i32, %c0_i32, %c0_i32) ({
  ^bb0(%i0 : !cuda_tile.tile<1xf32>):
    cuda_tile.continue
}) : (!cuda_tile.tile<1xi32>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<1xi32>) -> ()

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>

// expected-error@+1 {{expected induction variable to be same type as bounds}}
"cuda_tile.for"(%c0_i32, %c0_i32, %c0_i32) ({
  ^bb0(%i0 : !cuda_tile.tile<f32>):
    cuda_tile.continue
}) : (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>) -> ()

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
%init = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>

// expected-error @below{{init value 0 and region iter_value 0 have different type: '!cuda_tile.tile<f32>' != '!cuda_tile.tile<f64>'}}
"cuda_tile.for"(%c0_i32, %c0_i32, %c0_i32, %init) ({
  ^bb0(%i0 : !cuda_tile.tile<i32>, %iter: !cuda_tile.tile<f64>):
    cuda_tile.continue %init : !cuda_tile.tile<f32>
}) : (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<f32>) -> (!cuda_tile.tile<f32>)

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
%init = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>

// expected-error @below{{mismatch in number of region iterator values and loop iterator inits: 2 vs 1}}
%x = "cuda_tile.for"(%c0_i32, %c0_i32, %c0_i32, %init) ({
  ^bb0(%i0 : !cuda_tile.tile<i32>, %iter: !cuda_tile.tile<f32>, %iter2: !cuda_tile.tile<f32>):
    cuda_tile.continue %iter : !cuda_tile.tile<f32>
}) : (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<f32>) -> (!cuda_tile.tile<f32>)

// -----

// expected-error @below{{incorrect number of operands: expected 1, found 0}}
cuda_tile.print_tko "Expect one parameter %i" -> !cuda_tile.token

// -----

// expected-error @below{{expected valid keyword}}
%1 = "use_type"() : () -> !cuda_tile.tile<5x?xf32>

// -----

// expected-error @below{{all dimensions must be positive constants, got -1}}
%1 = "use_type"() : () -> !cuda_tile.tile<-1xf32>

// -----

// expected-error @below{{shape dimension 0 must be an integer-valued expression}}
%1 = "use_type"() : () -> !cuda_tile.tile<[M/2]xf32>

// -----

// expected-error @below{{shape dimension 0 must be an integer-valued expression}}
%1 = "use_type"() : () -> !cuda_tile.tile<[M/+2]xf32>

// -----

// expected-error-re @below{{failed to verify 'elementType': f16 or bf16 or f32 or tf32 or f64 or f8E4M3FN or f8E5M2 or f8E8M0FNU or f4E2M1FN{{( or f8E5M3FNU)?}} or i1 or i8 or i16 or i32 or i64 or Pointer type{{.*}}}}
%1 = "use_type"() : () -> !cuda_tile.tile<8x4xi28>

// -----

%0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
// expected-note @below{{prior use here}}
%1 = cuda_tile.constant <f64: 2.0> : !cuda_tile.tile<f64>
// expected-error @below{{expects different type than prior uses: '!cuda_tile.tile<f32>' vs '!cuda_tile.tile<f64>'}}
cuda_tile.maxf %0, %1 : !cuda_tile.tile<f32>

// -----

// expected-error @below{{expects result type to be 1-d tile}}
cuda_tile.iota : !cuda_tile.tile<i64>

// -----

// expected-error @below{{expects result type to be 1-d tile}}
cuda_tile.iota : !cuda_tile.tile<32x64xi64>

// -----

// expected-error @below{{the number of elements 512 exceeds the maximum value of element type 'i8'}}
cuda_tile.iota : !cuda_tile.tile<512xi8>

// -----

%0 = cuda_tile.constant <i16: 1> : !cuda_tile.tile<i16>
// expected-error @below{{requires the same element type for all operands and results}}
%1 = cuda_tile.reshape %0 : !cuda_tile.tile<i16> -> !cuda_tile.tile<1xi32>

// -----

%0 = cuda_tile.constant <i16: 1> : !cuda_tile.tile<i16>
// expected-error @below{{expected source tile and result tile to have the same number of elements}}
%1 = cuda_tile.reshape %0 : !cuda_tile.tile<i16> -> !cuda_tile.tile<1x2x1xi16>

// -----

%0 = cuda_tile.constant <f32: [[1.0, 2.0], [4.0, 5.0]]> : !cuda_tile.tile<2x2xf32>
// expected-error @below{{expected source tile and result tile to have the same number of elements}}
%1 = cuda_tile.reshape %0 : !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<8xf32>

// -----

%0 = cuda_tile.constant <f32: [[1.0, 2.0], [4.0, 5.0]]> : !cuda_tile.tile<2x2xf32>
// expected-error @below{{expected source tile and result tile to have the same number of elements}}
%1 = cuda_tile.reshape %0 : !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<f32>

// -----

%0 = cuda_tile.constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
// expected-error @below{{requires the same element type for all operands and results}}
%1 = cuda_tile.reshape %0 : !cuda_tile.tile<1xf32> -> !cuda_tile.tile<i32>

// -----

cuda_tile.module @kernels  {
  // expected-error @+1{{non-symbol operations are not allowed in a module body}}
  %0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
}

// -----

cuda_tile.module @kernels {
  testing$func @bcast_type_cast(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{requires the same element type for all operands and results}}
    %0 = cuda_tile.broadcast %arg0 : tile<2x2xf32> -> tile<2x2xf64>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @bcast_different_rank(%arg0: !cuda_tile.tile<2xf32>) {
    // expected-error @below{{failed to verify that all of {source, result} have same rank}}
    %0 = cuda_tile.broadcast %arg0 : tile<2xf32> -> tile<2x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @bcast_different_rank(%arg0: !cuda_tile.tile<4x4xf32>) {
    // expected-error @below{{expects the shape of source tile to be compatible with that of the result tile, but got: 4, 4 and 2, 4}}
    %0 = cuda_tile.broadcast %arg0 : tile<4x4xf32> -> tile<2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @bcast_invalid_dyn_dim1(%arg0: !cuda_tile.tile<1x4x4xf32>) {
    // expected-error @below{{expected valid keyword}}
    %0 = cuda_tile.broadcast %arg0 : tile<1x4x4xf32> -> tile<4x?x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @below{{expected valid keyword}}
  testing$func @bcast_invalid_dyn_dim2(%arg0: !cuda_tile.tile<1x?x4xf32>) {
    %0 = cuda_tile.broadcast %arg0 : tile<1x?x4xf32> -> tile<4x?x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @below{{all dimensions must be positive constants, got 1, 0, 2}}
  testing$func @bcast_empty_tile1(%arg0: !cuda_tile.tile<1x0x2xf32>) {
    %0 = cuda_tile.broadcast %0 : tile<1x0x2xi32> -> tile<4x0x2xi32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @bcast_empty_tile2(%arg0: !cuda_tile.tile<1x2x2xf32>) {
    // expected-error @below{{all dimensions must be positive constants, got 0, 2, 2}}
    %0 = cuda_tile.broadcast %0 : tile<1x2x2xi32> -> tile<0x2x2xi32>
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @below{{all dimensions must be positive constants, got 1, -1, 4}}
  testing$func @bcast_invalid_neg_dim(%arg0: !cuda_tile.tile<1x-1x4xf32>) {
    %0 = cuda_tile.broadcast %arg0 : tile<1x-1x4xf32> -> tile<4x-1x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @bcast_invalid_neg_dim2(%arg0: !cuda_tile.tile<4x1x4xf32>) {
    // expected-error @below{{all dimensions must be positive constants, got 4, -4, 4}}
    %0 = cuda_tile.broadcast %arg0 : tile<4x1x4xf32> -> tile<4x-4x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @bcast_invalid_non_power_2(%arg0: !cuda_tile.tile<1x1x1xf32>) {
    // expected-error @below{{all dimensions must be powers of two, got 3, 5, 9}}
    %0 = cuda_tile.broadcast %arg0 : tile<1x1x1xf32> -> tile<3x5x9xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @tile_size_overflow(%arg0: !cuda_tile.tile<1x1x1xf32>) {
    // expected-error @below{{tile would exceed the maximum of 16777216 elements}}
    %0 = cuda_tile.broadcast %arg0 : tile<1x1x1xf32> -> tile<1024x1024x1024xf32>
  }
}

// -----

// expected-error @below{{all dimensions must be powers of two, got 5, 5}}
%1 = "use_type"() : () -> !cuda_tile.tile<5x5xf32>

// -----

cuda_tile.module @kernels {
  testing$func @extract(%t: !cuda_tile.tile<8xf32>, %idx: !cuda_tile.tile<i32>) {
    // TODO: Enable this test case when non-power-of-2 tiles are supported.
    // TODO: error {{result dim size must divide source dim size evenly}}
    // %0 = cuda_tile.extract %t[%idx] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<3xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @extract(%t: !cuda_tile.tile<8xf32>, %idx: !cuda_tile.tile<i32>) {
    // expected-error@below {{source and result element type do not match}}
    %0 = cuda_tile.extract %t[%idx] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @extract(%t: !cuda_tile.tile<8xf32>, %idx: !cuda_tile.tile<i32>) {
    // expected-error@below {{failed to verify that all of {source, result} have same rank}}
    %0 = cuda_tile.extract %t[%idx] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<2x1xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @extract(%t: !cuda_tile.tile<8xf32>, %idx: !cuda_tile.tile<i32>) {
    // expected-error@below {{expected 1 indices, but got 2}}
    %0 = cuda_tile.extract %t[%idx, %idx] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  // expected-note @below{{prior use here}}
  testing$func @extract(%t: !cuda_tile.tile<8x8xf32>, %idx: !cuda_tile.tile<2xi32>) {
    // expected-error@below {{use of value '%idx' expects different type than prior uses: '!cuda_tile.tile<i32>' vs '!cuda_tile.tile<2xi32>'}}
    %0 = cuda_tile.extract %t[%idx] : !cuda_tile.tile<8x8xf32> -> !cuda_tile.tile<4x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @extract_invalid_index_count(%arg0: !cuda_tile.tile<4x8xf32>) {
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 2 indices, but got 1}}
    %0 = cuda_tile.extract %arg0[%c1] : !cuda_tile.tile<4x8xf32> -> !cuda_tile.tile<2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @extract_invalid_result_division(%arg0: !cuda_tile.tile<8x4xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{source dimension 1 size (4) must be evenly divisible by result dimension 1 size (8)}}
    %0 = cuda_tile.extract %arg0[%c0, %c1] : !cuda_tile.tile<8x4xf32> -> !cuda_tile.tile<4x8xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @extract_invalid_element_type(%arg0: !cuda_tile.tile<4x8xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{source and result element type do not match}}
    %0 = cuda_tile.extract %arg0[%c0, %c1] : !cuda_tile.tile<4x8xf32> -> !cuda_tile.tile<2x4xi32>
  }
}

// -----

cuda_tile.testing$func @test_insert(%src: !cuda_tile.tile<4x2xi32>, %dst: !cuda_tile.tile<32x8xf32>, %idx: !cuda_tile.tile<i32>) {
  // expected-error @+1 {{op source and result element type do not match}}
  %r = cuda_tile.insert %src, %dst[%idx, %idx] : !cuda_tile.tile<4x2xi32>, !cuda_tile.tile<32x8xf32>
}

// -----

cuda_tile.testing$func @test_insert(%src: !cuda_tile.tile<4x16xf32>, %dst: !cuda_tile.tile<32x8xf32>, %idx: !cuda_tile.tile<i32>) {
  // expected-error @+1 {{result dimension 1 size (8) must be evenly divisible by source dimension 1 size (16)}}
  %r = cuda_tile.insert %src, %dst[%idx, %idx] : !cuda_tile.tile<4x16xf32>, !cuda_tile.tile<32x8xf32>
}

// -----

cuda_tile.testing$func @test_insert(%src: !cuda_tile.tile<4x2xf32>, %dst: !cuda_tile.tile<32x8xf32>, %idx: !cuda_tile.tile<i32>) {
  // expected-error @+1 {{expected 2 indices, but got 1}}
  %r = cuda_tile.insert %src, %dst[%idx] : !cuda_tile.tile<4x2xf32>, !cuda_tile.tile<32x8xf32>
}

// -----

cuda_tile.testing$func @test_insert_rank_mismatch(%src: !cuda_tile.tile<4xf32>, %dst: !cuda_tile.tile<32x8xf32>, %idx: !cuda_tile.tile<i32>) {
  // expected-error @+1 {{op source and result must have the same rank}}
  %r = cuda_tile.insert %src, %dst[%idx, %idx] : !cuda_tile.tile<4xf32>, !cuda_tile.tile<32x8xf32>
}

// -----

// expected-note @below{{prior use here}}
cuda_tile.testing$func @test_insert_non_scalar_index(%src: !cuda_tile.tile<4x4xf32>, %dst: !cuda_tile.tile<8x8xf32>, %idx: !cuda_tile.tile<2xi32>) {
  // expected-error @+1 {{use of value '%idx' expects different type than prior uses: '!cuda_tile.tile<i32>' vs '!cuda_tile.tile<2xi32>'}}
  %r = cuda_tile.insert %src, %dst[%idx, %idx] : !cuda_tile.tile<4x4xf32>, !cuda_tile.tile<8x8xf32>
}

// -----

// expected-note @below{{prior use here}}
cuda_tile.testing$func @test_insert_index_element_type_mismatch(%src: !cuda_tile.tile<4xf32>, %dst: !cuda_tile.tile<8xf32>, %idx: !cuda_tile.tile<i64>) {
  // expected-error @+1 {{use of value '%idx' expects different type than prior uses: '!cuda_tile.tile<i32>' vs '!cuda_tile.tile<i64>'}}
  %r = cuda_tile.insert %src, %dst[%idx] : !cuda_tile.tile<4xf32>, !cuda_tile.tile<8xf32>
}

// -----

cuda_tile.testing$func @test_insert_negative_constant_index(%src: !cuda_tile.tile<4x2xf32>, %dst: !cuda_tile.tile<32x8xf32>) {
  %cn1 = cuda_tile.constant <i32: -1> : !cuda_tile.tile<i32>
  %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
  // expected-warning @+1 {{insert index 0 has negative value -1, which may cause undefined behavior}}
  %r = cuda_tile.insert %src, %dst[%cn1, %c0] : !cuda_tile.tile<4x2xf32>, !cuda_tile.tile<32x8xf32>
}

// -----

cuda_tile.testing$func @test_insert_out_of_bounds_constant_index(%src: !cuda_tile.tile<2x2xf32>, %dst: !cuda_tile.tile<2x2xf32>) {
  %c2 = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
  %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
  // expected-warning @+1 {{insert index 0 value 2 may be out of bounds (max valid index: 0)}}
  %r = cuda_tile.insert %src, %dst[%c2, %c0] : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32>
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_lhs_rhs_type_mismatch(%arg0: !cuda_tile.tile<4x8xf32>, %arg1: !cuda_tile.tile<8x16xf16>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // expected-error @below{{op failed to verify that all of {lhs, rhs} have the same element type}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<4x8xf32>, !cuda_tile.tile<8x16xf16>, !cuda_tile.tile<4x16xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_shape_mismatch(%arg0: !cuda_tile.tile<4x16xf32>, %arg1: !cuda_tile.tile<8x16xf32>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // expected-error @below{{dim 1 of lhs (16) and dim 0 of rhs (8) must match, but got lhs shape (4, 16) and rhs shape (8, 16)}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<4x16xf32>, !cuda_tile.tile<8x16xf32>, !cuda_tile.tile<4x16xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_shape_mismatch(%arg0: !cuda_tile.tile<16x8xf32>, %arg1: !cuda_tile.tile<8x16xf32>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // expected-error @below{{dim 0 of lhs (16) and dim 0 of acc (4) must match, but got lhs shape (16, 8) and acc shape (4, 16)}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<16x8xf32>, !cuda_tile.tile<8x16xf32>, !cuda_tile.tile<4x16xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_shape_mismatch(%arg0: !cuda_tile.tile<4x8xf32>, %arg1: !cuda_tile.tile<8x16xf32>, %arg2: !cuda_tile.tile<4x32xf32>) {
    // expected-error @below{{dim 1 of rhs (16) and dim 1 of acc (32) must match, but got rhs shape (8, 16) and acc shape (4, 32)}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<4x8xf32>, !cuda_tile.tile<8x16xf32>, !cuda_tile.tile<4x32xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_rank_mismatch(%arg0: !cuda_tile.tile<4xf32>, %arg1: !cuda_tile.tile<8x16xf32>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // expected-error @below{{op failed to verify that all of {lhs, rhs, acc} have same rank}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<4xf32>, !cuda_tile.tile<8x16xf32>, !cuda_tile.tile<4x16xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_rank_mismatch(%arg0: !cuda_tile.tile<4x8xf32>, %arg1: !cuda_tile.tile<8xf32>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // expected-error @below{{op failed to verify that all of {lhs, rhs, acc} have same rank}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<4x8xf32>, !cuda_tile.tile<8xf32>, !cuda_tile.tile<4x16xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_batch_mismatch(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x8x16xf32>, %arg2: !cuda_tile.tile<4x4x16xf32>) {
    // expected-error @below{{dim 0 of lhs (2) and dim 0 of acc (4) must match, but got lhs shape (2, 4, 8) and acc shape (4, 4, 16)}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x8x16xf32>, !cuda_tile.tile<4x4x16xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_rank_mismatch(%arg0: !cuda_tile.tile<4x8xf32>, %arg1: !cuda_tile.tile<8x16xf32>, %arg2: !cuda_tile.tile<4xf32>) {
    // expected-error @below{{op failed to verify that all of {lhs, rhs, acc} have same rank}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<4x8xf32>, !cuda_tile.tile<8x16xf32>, !cuda_tile.tile<4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_type_mismatch(%arg0: !cuda_tile.tile<4x8xf32>, %arg1: !cuda_tile.tile<8x16xf64>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // expected-error @below{{op failed to verify that all of {lhs, rhs} have the same element type}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<4x8xf32>, !cuda_tile.tile<8x16xf64>, !cuda_tile.tile<4x16xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_unsigned_float(%arg0: !cuda_tile.tile<4x8xf32>, %arg1: !cuda_tile.tile<8x16xf32>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // expected-error @below{{expected ':'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 signed signed : !cuda_tile.tile<4x8xf32>, !cuda_tile.tile<8x16xf32>, !cuda_tile.tile<4x16xf32>
  }
}

// -----
cuda_tile.module @kernels {
  testing$func @mmaf_int_types(%arg0: !cuda_tile.tile<2x2xi8>, %arg1: !cuda_tile.tile<2x2xi8>, %arg2: !cuda_tile.tile<2x2xi32>) {
    // expected-error @below{{op operand #0 must be mmaf operand tile type of f16 or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 values, but got '!cuda_tile.tile<2x2xi8>'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xi8>, !cuda_tile.tile<2x2xi8>, !cuda_tile.tile<2x2xi32>
  }
}

// -----
cuda_tile.module @kernels {
  testing$func @mmai_float_types(%arg0: !cuda_tile.tile<2x2xf32>, %arg1: !cuda_tile.tile<2x2xf32>, %arg2: !cuda_tile.tile<2x2xi32>) {
    // expected-error @below{{op operand #0 must be mmai operand tile type of i8 values, but got '!cuda_tile.tile<2x2xf32>'}}
    %0 = cuda_tile.mmai %arg0, %arg1, %arg2 signed signed : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xi32>
  }
}


// -----

cuda_tile.module @kernels {
  testing$func @mma_rank_mismatch(%arg0: !cuda_tile.tile<2x2x2xf32>, %arg1: !cuda_tile.tile<2x2xf32>, %arg2: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{op failed to verify that all of {lhs, rhs, acc} have same rank}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2x2xf32>, !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mmaf_scaled_fp8e4m3_scale(%arg0: !cuda_tile.tile<128x128xf8E5M2>, %arg1: !cuda_tile.tile<128x128xf8E5M2>, %arg2: !cuda_tile.tile<128x128xf32>, %arg3: !cuda_tile.tile<128x4xf8E4M3FN>, %arg4: !cuda_tile.tile<4x128xf8E4M3FN>) {
    // expected-error @below {{op unsupported combination of element types. Scale type 'f8E4M3FN' expects lhs and rhs element types to be 'f4E2M1FN', but got 'f8E5M2'}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E4M3FN>, !cuda_tile.tile<4x128xf8E4M3FN>
  }
}

// -----

// -----

cuda_tile.module @kernels {
  testing$func @mmaf_scaled_fp16_acc(%arg0: !cuda_tile.tile<128x128xf8E5M2>, %arg1: !cuda_tile.tile<128x128xf8E5M2>, %arg2: !cuda_tile.tile<128x128xf16>, %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>, %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>) {
    // expected-error @below {{op operand #2 must be mmaf_scaled result tile type of f32 values, but got '!cuda_tile.tile<128x128xf16>'}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf16>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }
}

// -----


cuda_tile.module @kernels {
  testing$func @mmaf_scaled_mixed_scale_types(%arg0: !cuda_tile.tile<128x128xf4E2M1FN>, %arg1: !cuda_tile.tile<128x128xf4E2M1FN>, %arg2: !cuda_tile.tile<128x128xf32>, %arg3: !cuda_tile.tile<128x8xf8E8M0FNU>, %arg4: !cuda_tile.tile<8x128xf8E4M3FN>) {
    // expected-error @below {{op failed to verify that all of {lhs_scale, rhs_scale} have the same element type}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E8M0FNU>, !cuda_tile.tile<8x128xf8E4M3FN>
  }
}

// -----

// Test mmaf_scaled with M dimension mismatch between lhs and lhs_scale
cuda_tile.module @kernels {
  testing$func @mmaf_scaled_m_dim_mismatch(%arg0: !cuda_tile.tile<128x128xf4E2M1FN>, %arg1: !cuda_tile.tile<128x128xf4E2M1FN>, %arg2: !cuda_tile.tile<128x128xf32>, %arg3: !cuda_tile.tile<64x4xf8E8M0FNU>, %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>) {
    // expected-error @below {{shape error: dim 0 of lhs (128) and dim 0 of lhs_scale (64) must match}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<64x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }
}

// -----

// Test mmaf_scaled with N dimension mismatch between rhs and rhs_scale
cuda_tile.module @kernels {
  testing$func @mmaf_scaled_n_dim_mismatch(%arg0: !cuda_tile.tile<128x128xf4E2M1FN>, %arg1: !cuda_tile.tile<128x128xf4E2M1FN>, %arg2: !cuda_tile.tile<128x128xf32>, %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>, %arg4: !cuda_tile.tile<4x64xf8E8M0FNU>) {
    // expected-error @below {{shape error: dim 1 of rhs (128) and dim 1 of rhs_scale (64) must match}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x64xf8E8M0FNU>
  }
}

// -----

// Test mmaf_scaled with batch dimension mismatch (3D tiles)
cuda_tile.module @kernels {
  testing$func @mmaf_scaled_batch_dim_mismatch(%arg0: !cuda_tile.tile<4x64x64xf4E2M1FN>, %arg1: !cuda_tile.tile<4x64x64xf4E2M1FN>, %arg2: !cuda_tile.tile<4x64x64xf32>, %arg3: !cuda_tile.tile<2x64x2xf8E8M0FNU>, %arg4: !cuda_tile.tile<4x2x64xf8E8M0FNU>) {
    // expected-error @below {{shape error: dim 0 of lhs (4) and dim 0 of lhs_scale (2) must match}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<4x64x64xf4E2M1FN>, !cuda_tile.tile<4x64x64xf4E2M1FN>, !cuda_tile.tile<4x64x64xf32>, !cuda_tile.tile<2x64x2xf8E8M0FNU>, !cuda_tile.tile<4x2x64xf8E8M0FNU>
  }
}

// -----

// Test mmaf_scaled with f8E8M0FNU scale and unsupported f16 operand
cuda_tile.module @kernels {
  testing$func @mmaf_scaled_unsupported_f16_operand(%arg0: !cuda_tile.tile<128x128xf16>, %arg1: !cuda_tile.tile<128x128xf16>, %arg2: !cuda_tile.tile<128x128xf32>, %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>, %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>) {
    // expected-error @below {{op operand #0 must be mmaf_scaled operand tile type of f8E4M3FN or f8E5M2 or f4E2M1FN values, but got '!cuda_tile.tile<128x128xf16>'}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<128x128xf16>, !cuda_tile.tile<128x128xf16>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }
}

// -----

// Test mmaf_scaled with unsupported f32 scale type
cuda_tile.module @kernels {
  testing$func @mmaf_scaled_unsupported_f32_scale(%arg0: !cuda_tile.tile<128x128xf4E2M1FN>, %arg1: !cuda_tile.tile<128x128xf4E2M1FN>, %arg2: !cuda_tile.tile<128x128xf32>, %arg3: !cuda_tile.tile<128x4xf32>, %arg4: !cuda_tile.tile<4x128xf32>) {
    // expected-error @below {{op operand #3 must be mmaf_scaled scale tile type of f8E4M3FN or f8E8M0FNU or f8E5M3FNU values, but got '!cuda_tile.tile<128x4xf32>'}}
    %0 = cuda_tile.mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf32>, !cuda_tile.tile<4x128xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mmaf_scaled_different_K_dimensions(%A: !cuda_tile.tile<128x128xf4E2M1FN>, %B: !cuda_tile.tile<128x128xf4E2M1FN>, %C: !cuda_tile.tile<128x128xf32>, %sfA: !cuda_tile.tile<128x8xf8E8M0FNU>, %sfB: !cuda_tile.tile<4x128xf8E8M0FNU>) {
    // expected-error @below {{shape error: dim 1 of lhs_scale (8) and dim 0 of rhs_scale (4) must match, but got lhs_scale shape (128, 8) and rhs_scale shape (4, 128)}}
    %0 = cuda_tile.mmaf_scaled %A, %B, %C, %sfA, %sfB : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }
}
// -----

cuda_tile.module @kernels {
  testing$func @mmaf_scaled_invalid_block_scale_factor(%A: !cuda_tile.tile<128x128xf8E4M3FN>, %B: !cuda_tile.tile<128x128xf8E4M3FN>, %C: !cuda_tile.tile<128x128xf32>, %sfA: !cuda_tile.tile<128x8xf8E8M0FNU>, %sfB: !cuda_tile.tile<8x128xf8E8M0FNU>) {
    // expected-error @below {{shape error: f8 element type requires block scale factor of lhs_scale and rhs_scale to be 32, but got 16}}
    %0 = cuda_tile.mmaf_scaled %A, %B, %C, %sfA, %sfB : !cuda_tile.tile<128x128xf8E4M3FN>, !cuda_tile.tile<128x128xf8E4M3FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E8M0FNU>, !cuda_tile.tile<8x128xf8E8M0FNU>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mmaf_scaled_invalid_block_scale_factor(%A: !cuda_tile.tile<128x128xf4E2M1FN>, %B: !cuda_tile.tile<128x128xf4E2M1FN>, %C: !cuda_tile.tile<128x128xf32>, %sfA: !cuda_tile.tile<128x16xf8E8M0FNU>, %sfB: !cuda_tile.tile<16x128xf8E8M0FNU>) {
    // expected-error @below {{shape error: f4E2M1FN element type with f8E8M0FNU scales requires block scale factor of lhs_scale and rhs_scale to be 16 or 32, but got 8}}
    %0 = cuda_tile.mmaf_scaled %A, %B, %C, %sfA, %sfB : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x16xf8E8M0FNU>, !cuda_tile.tile<16x128xf8E8M0FNU>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mmaf_scaled_invalid_block_scale_factor(%A: !cuda_tile.tile<128x128xf4E2M1FN>, %B: !cuda_tile.tile<128x128xf4E2M1FN>, %C: !cuda_tile.tile<128x128xf32>, %sfA: !cuda_tile.tile<128x16xf8E4M3FN>, %sfB: !cuda_tile.tile<16x128xf8E4M3FN>) {
    // expected-error @below {{shape error: f4E2M1FN element type with f8E4M3FN scales requires block scale factor of lhs_scale and rhs_scale to be 16, but got 8}}
    %0 = cuda_tile.mmaf_scaled %A, %B, %C, %sfA, %sfB : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x16xf8E4M3FN>, !cuda_tile.tile<16x128xf8E4M3FN>
  }
}


// -----

cuda_tile.module @kernels {
  testing$func @mma_i16(%arg0: !cuda_tile.tile<2x2xi16>, %arg1: !cuda_tile.tile<2x2xi16>, %arg2: !cuda_tile.tile<2x2xi32>) {
    // expected-error @below{{op operand #0 must be mmai operand tile type of i8 values, but got '!cuda_tile.tile<2x2xi16>'}}
    %0 = cuda_tile.mmai %arg0, %arg1, %arg2 signed signed : !cuda_tile.tile<2x2xi16>, !cuda_tile.tile<2x2xi16>, !cuda_tile.tile<2x2xi32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_i32(%arg0: !cuda_tile.tile<2x2xi32>, %arg1: !cuda_tile.tile<2x2xi32>, %arg2: !cuda_tile.tile<2x2xi32>) {
    // expected-error @below{{op operand #0 must be mmai operand tile type of i8 values, but got '!cuda_tile.tile<2x2xi32>'}}
    %0 = cuda_tile.mmai %arg0, %arg1, %arg2 signed signed : !cuda_tile.tile<2x2xi32>, !cuda_tile.tile<2x2xi32>, !cuda_tile.tile<2x2xi32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_i64(%arg0: !cuda_tile.tile<2x2xi64>, %arg1: !cuda_tile.tile<2x2xi64>, %arg2: !cuda_tile.tile<2x2xi64>) {
    // expected-error @below{{op operand #0 must be mmai operand tile type of i8 values, but got '!cuda_tile.tile<2x2xi64>'}}
    %0 = cuda_tile.mmai %arg0, %arg1, %arg2 signed signed : !cuda_tile.tile<2x2xi64>, !cuda_tile.tile<2x2xi64>, !cuda_tile.tile<2x2xi64>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_mixed_f8(%arg0: !cuda_tile.tile<2x2xf8E4M3FN>, %arg1: !cuda_tile.tile<2x2xf8E5M2>, %arg2: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{op failed to verify that all of {lhs, rhs} have the same element type}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xf8E4M3FN>, !cuda_tile.tile<2x2xf8E5M2>, !cuda_tile.tile<2x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_f8_f8(%arg0: !cuda_tile.tile<2x2xf8E4M3FN>, %arg1: !cuda_tile.tile<2x2xf8E4M3FN>, %arg2: !cuda_tile.tile<2x2xf8E4M3FN>) {
    // expected-error @below{{op operand #2 must be mmaf acc/result tile type of f16 or f32 or f64 values, but got '!cuda_tile.tile<2x2xf8E4M3FN>'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xf8E4M3FN>, !cuda_tile.tile<2x2xf8E4M3FN>, !cuda_tile.tile<2x2xf8E4M3FN>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_f8_f64(%arg0: !cuda_tile.tile<2x2xf8E4M3FN>, %arg1: !cuda_tile.tile<2x2xf8E4M3FN>, %arg2: !cuda_tile.tile<2x2xf64>) {
    // expected-error @below{{op unsupported combination of element types. Input type 'f8E4M3FN' expects accumulator/result type to be one of {'f16', 'f32'}, but got 'f64'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xf8E4M3FN>, !cuda_tile.tile<2x2xf8E4M3FN>, !cuda_tile.tile<2x2xf64>
  }
}
// -----

cuda_tile.module @kernels {
  testing$func @mma_bf16_bf16(%arg0: !cuda_tile.tile<2x2xbf16>, %arg1: !cuda_tile.tile<2x2xbf16>, %arg2: !cuda_tile.tile<2x2xbf16>) {
    // expected-error @below{{op operand #2 must be mmaf acc/result tile type of f16 or f32 or f64 values, but got '!cuda_tile.tile<2x2xbf16>'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xbf16>, !cuda_tile.tile<2x2xbf16>, !cuda_tile.tile<2x2xbf16>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_bf16_f16(%arg0: !cuda_tile.tile<2x2xbf16>, %arg1: !cuda_tile.tile<2x2xbf16>, %arg2: !cuda_tile.tile<2x2xf16>) {
    // expected-error @below{{op unsupported combination of element types. Input type 'bf16' expects accumulator/result type to be one of {'f32'}, but got 'f16'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xbf16>, !cuda_tile.tile<2x2xbf16>, !cuda_tile.tile<2x2xf16>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_tf32_tf32(%arg0: !cuda_tile.tile<2x2xtf32>, %arg1: !cuda_tile.tile<2x2xtf32>, %arg2: !cuda_tile.tile<2x2xtf32>) {
    // expected-error @below{{op operand #2 must be mmaf acc/result tile type of f16 or f32 or f64 values, but got '!cuda_tile.tile<2x2xtf32>'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xtf32>, !cuda_tile.tile<2x2xtf32>, !cuda_tile.tile<2x2xtf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @mma_tf32_f16(%arg0: !cuda_tile.tile<2x2xtf32>, %arg1: !cuda_tile.tile<2x2xtf32>, %arg2: !cuda_tile.tile<2x2xf16>) {
    // expected-error @below{{op unsupported combination of element types. Input type 'tf32' expects accumulator/result type to be one of {'f32'}, but got 'f16'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xtf32>, !cuda_tile.tile<2x2xtf32>, !cuda_tile.tile<2x2xf16>
  }
}


// -----

cuda_tile.module @kernels {
  testing$func @mma_f16_f64(%arg0: !cuda_tile.tile<2x2xf16>, %arg1: !cuda_tile.tile<2x2xf16>, %arg2: !cuda_tile.tile<2x2xf64>) {
    // expected-error @below{{op unsupported combination of element types. Input type 'f16' expects accumulator/result type to be one of {'f16', 'f32'}, but got 'f64'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xf16>, !cuda_tile.tile<2x2xf16>, !cuda_tile.tile<2x2xf64>
  }
}
// -----

cuda_tile.module @kernels {
  testing$func @mma_f32_f64(%arg0: !cuda_tile.tile<2x2xf32>, %arg1: !cuda_tile.tile<2x2xf32>, %arg2: !cuda_tile.tile<2x2xf64>) {
    // expected-error @below{{op unsupported combination of element types. Input type 'f32' expects accumulator/result type to be one of {'f32'}, but got 'f64'}}
    %0 = cuda_tile.mmaf %arg0, %arg1, %arg2 : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf64>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_different_element_type_in_result(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{failed to verify that all of {lhs, rhs, result} have the same element type}}
    %0 = cuda_tile.cat %arg0, %arg0 dim = 1
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x4xf64>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_different_element_type_in_lhs(%arg0: !cuda_tile.tile<2x2xf64>, %arg1: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{failed to verify that all of {lhs, rhs, result} have the same element type}}
    %0 = cuda_tile.cat %arg0, %arg1 dim = 1
      : !cuda_tile.tile<2x2xf64>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_different_element_type_in_rhs(%arg0: !cuda_tile.tile<2x2xf32>, %arg1: !cuda_tile.tile<2x2xf64>) {
    // expected-error @below{{failed to verify that all of {lhs, rhs, result} have the same element type}}
    %0 = cuda_tile.cat %arg0, %arg1 dim = 1
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf64> -> !cuda_tile.tile<2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_different_rank_in_result(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{failed to verify that all of {lhs, rhs, result} have same rank}}
    %0 = cuda_tile.cat %arg0, %arg0 dim = 1
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x4x1xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_different_rank_in_lhs(%arg0: !cuda_tile.tile<1x2x2xf32>, %arg1: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{failed to verify that all of {lhs, rhs, result} have same rank}}
    %0 = cuda_tile.cat %arg0, %arg1 dim = 1
      : !cuda_tile.tile<1x2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x4x1xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_different_rank_in_rhs(%arg0: !cuda_tile.tile<2x2xf32>, %arg1: !cuda_tile.tile<1x2x2xf32>) {
    // expected-error @below{{failed to verify that all of {lhs, rhs, result} have same rank}}
    %0 = cuda_tile.cat %arg0, %arg1 dim = 1
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<1x2x2xf32> -> !cuda_tile.tile<2x4x1xf32>
  }
}


// -----

cuda_tile.module @kernels {
  testing$func @cat_invalid_dim(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{expect dim to be [0, 2), but got: -1}}
    %0 = cuda_tile.cat %arg0, %arg0 dim = -1
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_invalid_dim(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{expect dim to be [0, 2), but got: 2}}
    %0 = cuda_tile.cat %arg0, %arg0 dim = 2
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_invalid_dim(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{expect dim to be [0, 2), but got: 10}}
    %0 = cuda_tile.cat %arg0, %arg0 dim = 10
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @cat_invalid_concatenation(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{invalid concat at position 1, expected: 4 but got: 16}}
    %0 = cuda_tile.cat %arg0, %arg0 dim = 1
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<2x16xf32>
  }
}

// -----
cuda_tile.module @kernels {
  testing$func @cat_invalid_non_concatenating_dim(%arg0: !cuda_tile.tile<2x2xf32>) {
    // expected-error @below{{expect lhs and result shapes to match at non-concat position 0, expected: 2 but got: 4}}
    %0 = cuda_tile.cat %arg0, %arg0 dim = 1
      : !cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32> -> !cuda_tile.tile<4x4xf32>
  }
}

// -----
%init = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>

// expected-error @below{{init value 0 and region iter_value 0 have different type: '!cuda_tile.tile<f32>' != '!cuda_tile.tile<f64>'}}
"cuda_tile.loop"(%init) ({
  ^bb0(%iter: !cuda_tile.tile<f64>):
    cuda_tile.continue %init : !cuda_tile.tile<f32>
}) : (!cuda_tile.tile<f32>) -> (!cuda_tile.tile<f32>)

// -----

%init = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>

// expected-error @below{{mismatch in number of region iterator values and loop iterator inits: 2 vs 1}}
%x = "cuda_tile.loop"(%init) ({
  ^bb0(%iter: !cuda_tile.tile<f32>, %iter2: !cuda_tile.tile<f32>):
    cuda_tile.continue %iter : !cuda_tile.tile<f32>
}) : (!cuda_tile.tile<f32>) -> (!cuda_tile.tile<f32>)

// -----

%init = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>
// expected-error @below{{found different number of iter_values and types}}
cuda_tile.loop iter_values(%arg0 = %init) : !cuda_tile.tile<f32>, !cuda_tile.tile<f32> {
  cuda_tile.continue %arg1
}
// -----

%init0 = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>
// expected-note @below{{prior use here}}
%init1 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
// expected-error @below{{use of value '%init1' expects different type than prior uses: '!cuda_tile.tile<f32>' vs '!cuda_tile.tile<i32>'}}
cuda_tile.loop iter_values(%arg0 = %init0, %arg1 = %init1) : !cuda_tile.tile<f32>, !cuda_tile.tile<f32> {}

// -----

// expected-error @below{{expected valid keyword}}
cuda_tile.loop : {}

// -----

// expected-error @below{{expected valid keyword}}
cuda_tile.loop iter_values(%arg0=%init0) : {}

// -----

// expected-error @below{{expected valid keyword}}
%result = cuda_tile.loop iter_values(%arg0=%init0) : !cuda_tile.tile<f32> -> {}

// -----

%0 = cuda_tile.constant <i16: 1> : !cuda_tile.tile<i16>
// expected-error @below{{'cuda_tile.exp' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i16>'}}
cuda_tile.exp %0 : !cuda_tile.tile<i16>

// -----

%0 = cuda_tile.constant <i8: 1> : !cuda_tile.tile<i8>
// expected-error @below{{'cuda_tile.exp2' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i8>'}}
cuda_tile.exp2 %0 : !cuda_tile.tile<i8>

// -----

cuda_tile.module @kernels {
  testing$func @select_operation(%condition: !cuda_tile.tile<4xi32>, %trueval: !cuda_tile.tile<4xi32>, %falseval: !cuda_tile.tile<4xi32>) {
    // expected-error @below{{op operand #0 must be tile of i1 values}}
    %0 = cuda_tile.select %condition, %trueval, %falseval : !cuda_tile.tile<4xi32>, !cuda_tile.tile<4xi32>
  }
}

// -----

cuda_tile.module @kernels {
  // expected-note @below{{prior use here}}
  testing$func @select_operation(%condition: !cuda_tile.tile<4xi1>, %trueval: !cuda_tile.tile<4xi32>, %falseval: !cuda_tile.tile<4xi16>) {
    // expected-error @below{{use of value '%falseval' expects different type than prior uses}}
    %0 = cuda_tile.select %condition, %trueval, %falseval : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xi32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @select_operation(%condition: !cuda_tile.tile<i1>, %trueval: !cuda_tile.tile<4xi32>, %falseval: !cuda_tile.tile<4xi32>) {
    // expected-error @below{{op failed to verify that all of {cond, val_if_true, val_if_false, result} have same shape}}
    %0 = cuda_tile.select %condition, %trueval, %falseval : !cuda_tile.tile<i1>, !cuda_tile.tile<4xi32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @select_f4(%condition: !cuda_tile.tile<4xi1>, %trueval: !cuda_tile.tile<4xf4E2M1FN>, %falseval: !cuda_tile.tile<4xf4E2M1FN>) {
    // expected-error @below{{cannot operate on sub-byte type F4E2M1FN}}
    %0 = cuda_tile.select %condition, %trueval, %falseval : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xf4E2M1FN>
  }
}

// -----

%0 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{'cuda_tile.log' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i32>'}}
cuda_tile.log %0 : !cuda_tile.tile<i32>

// -----

%0 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{'cuda_tile.log2' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i32>'}}
cuda_tile.log2 %0 : !cuda_tile.tile<i32>

// -----

cuda_tile.module @kernels {
  entry @bitcast_different_width() {
    %c0_i32 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{op types must be equal width}}
    %c1_i16 = cuda_tile.bitcast %c0_i32 : !cuda_tile.tile<i32> -> !cuda_tile.tile<i16>
  }
}

// -----

cuda_tile.module @kernels {
  entry @bitcast_different_shape() {
    %c0_i16 = cuda_tile.constant <i16: [1, 2, 3, 4]> : !cuda_tile.tile<4xi16>
    // expected-error @below{{op failed to verify that all of {source, result} have same shape}}
    %c1_i32 = cuda_tile.bitcast %c0_i16 : !cuda_tile.tile<4xi16> -> !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @kernel {
  testing$func @bitcast_pointer_to_int_invalid(%arg0 : !cuda_tile.tile<!cuda_tile.ptr<i8>>) {
    // expected-error @below{{result #0 must be tile of i64 values, but got '!cuda_tile.tile<i32>'}}
    %c0_i32 = cuda_tile.ptr_to_int %arg0 : !cuda_tile.tile<!cuda_tile.ptr<i8>> -> !cuda_tile.tile<i32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @div_by(%arg0: !cuda_tile.tile<f32>) {
    // expected-error @below{{'cuda_tile.div_by' is valid only for tile of integer/pointer or tensor_view values}}
    cuda_tile.assume #cuda_tile.div_by<16>, %arg0 : !cuda_tile.tile<f32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @div_by(%arg0: !cuda_tile.tile<i8>) {
    // expected-error @+1{{'cuda_tile.div_by' divisor is too large}}
    cuda_tile.assume #cuda_tile.div_by<9223372036854775808>, %arg0 : !cuda_tile.tile<i8>
  }
}

// -----

cuda_tile.module @module {
  testing$func @div_by(%arg0: !cuda_tile.tile<!cuda_tile.ptr<f16>>) {
    // expected-error @below{{'cuda_tile.div_by' 'every'/'along' cannot be used if the constrained value is a 0D tile}}
    cuda_tile.assume #cuda_tile.div_by<1, every 8 along 0>, %arg0 : !cuda_tile.tile<!cuda_tile.ptr<f16>>
  }
}

// -----

cuda_tile.module @module {
  testing$func @div_by(%arg0: !cuda_tile.tensor_view<64x64xf16, strides=[1,1]>) {
    // expected-error @below{{'cuda_tile.div_by' 'every'/'along' cannot be used if the constrained value is a tensor_view}}
    cuda_tile.assume #cuda_tile.div_by<1, every 8 along 0>, %arg0 : !cuda_tile.tensor_view<64x64xf16, strides=[1,1]>
  }
}

// -----

cuda_tile.module @module {
  testing$func @div_by(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected 'cuda_tile.div_by' every_dim to be within 0 and the size of the respective dimension (16)}}
    cuda_tile.assume #cuda_tile.div_by<1, every 24 along 0>, %arg0 : !cuda_tile.tile<16xi32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @div_by(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{'cuda_tile.div_by' every_dim (1) must be >= 0 and < tile rank (1)}}
    cuda_tile.assume #cuda_tile.div_by<1, every 2 along 1>, %arg0 : !cuda_tile.tile<16xi32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @div_by(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{'cuda_tile.div_by' divisor must be a power of 2}}
    cuda_tile.assume #cuda_tile.div_by<7>, %arg0 : !cuda_tile.tile<16xi32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @same_elements(%arg0: !cuda_tile.tile<!cuda_tile.ptr<f16>>) {
    // expected-error @below{{expected number of values in 'cuda_tile.same_elements' (1) to match rank of constrained tile (0)}}
    cuda_tile.assume #cuda_tile.same_elements<[8]>, %arg0 : !cuda_tile.tile<!cuda_tile.ptr<f16>>
  }
}

// -----

cuda_tile.module @module {
  testing$func @same_elements(%arg0: !cuda_tile.tile<16xf32>) {
    // expected-error @below{{'cuda_tile.same_elements' is valid only for tile of integer/pointer values}}
    cuda_tile.assume #cuda_tile.same_elements<[8]>, %arg0 : !cuda_tile.tile<16xf32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @same_elements(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected 'cuda_tile.same_elements' value 0 to be within 0 and the size of the respective dimension (16)}}
    cuda_tile.assume #cuda_tile.same_elements<[24]>, %arg0 : !cuda_tile.tile<16xi32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @bounded(%arg0: !cuda_tile.tile<16xf32>) {
    // expected-error @below{{'cuda_tile.bounded' is valid only for tile of integer values}}
    cuda_tile.assume #cuda_tile.bounded<0, 0>, %arg0 : !cuda_tile.tile<16xf32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @bounded(%arg0: !cuda_tile.tile<16xi8>) {
    // expected-error @below{{'cuda_tile.bounded' expects upper bound to be within [-128, 127]}}
    cuda_tile.assume #cuda_tile.bounded<0, 128>, %arg0 : !cuda_tile.tile<16xi8>
  }
}

// -----

cuda_tile.module @module {
  testing$func @bounded(%arg0: !cuda_tile.tile<16xi8>) {
    // expected-error @below{{'cuda_tile.bounded' expects lower bound to be within [-128, 127]}}
    cuda_tile.assume #cuda_tile.bounded<-129, 6>, %arg0 : !cuda_tile.tile<16xi8>
  }
}

// -----

cuda_tile.module @module {
  testing$func @bounded(%arg0: !cuda_tile.tile<16xi8>) {
    // expected-error @below{{'cuda_tile.bounded' expects lower bound to be less than or equal to upper bound}}
    cuda_tile.assume #cuda_tile.bounded<8, 6>, %arg0 : !cuda_tile.tile<16xi8>
  }
}

// -----

cuda_tile.module @module {
  testing$func @invalid_predicate(%arg0: !cuda_tile.tile<f32>) {
    // expected-error @below{{expected assume predicate attribute}}
    cuda_tile.assume 32 : i32, %arg0 : !cuda_tile.tile<f32>
  }
}

// -----

cuda_tile.module @test_func_with_operand_but_no_result {
  // expected-error @below{{op has 0 operands, but enclosing function (@kernel) returns 1}}
  testing$func @kernel(%arg0: !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {}
}

// -----

cuda_tile.module @test_func_with_operand_and_wrong_result {
  testing$func @kernel(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xf32>) -> !cuda_tile.tile<2xi16> {
    // expected-error @below{{type of return operand 0 ('!cuda_tile.tile<2xf32>') doesn't match function result type ('!cuda_tile.tile<2xi16>') in function @kernel}}
    cuda_tile.return %arg1: !cuda_tile.tile<2xf32>
  }
}

// -----

cuda_tile.module @test_kernel_scope {
  // expected-error @below{{expected valid '@'-identifier for symbol name}}
  entry pluto @func_with_kernel_scope_global() {}
}

// -----

cuda_tile.module @test_kernel_scope {
  // expected-error @below{{entry op must not return values}}
  cuda_tile.entry @entry_with_result(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xf32> {
    cuda_tile.return %arg0 : !cuda_tile.tile<2x2xf32>
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                  %arg1: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{'addf' works only with floats f16, bf16, f32, and f64}}
    cuda_tile.atomic_rmw_tko relaxed device %arg0, addf, %arg1
        : !cuda_tile.tile<2x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                  %arg1: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'mode' [and, or, xor, add, addf, max, min, umax, umin, xchg]}}
    cuda_tile.atomic_rmw_tko relaxed device %arg0, foo, %arg1
        : !cuda_tile.tile<2x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw(%arg0: !cuda_tile.tile<4x!cuda_tile.ptr<i32>>,
                                  %arg1: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{failed to verify that all of {pointers, arg, result} have same shape}}
    cuda_tile.atomic_rmw_tko relaxed device %arg0, add, %arg1
        : !cuda_tile.tile<4x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<f32>>,
                                  %arg1: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{expected pointee type ('f32') to match element type of 'arg' ('i32')}}
    cuda_tile.atomic_rmw_tko relaxed device %arg0, add, %arg1
        : !cuda_tile.tile<2x!cuda_tile.ptr<f32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                  %arg1: !cuda_tile.tile<2xi32>, %arg2: !cuda_tile.tile<4xi1>) {
    // expected-error @below {{failed to verify that all of {pointers, arg, mask} have same shape}}
    %0, %t = cuda_tile.atomic_rmw_tko relaxed device %arg0, and, %arg1, %arg2
        : !cuda_tile.tile<2x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32>, !cuda_tile.tile<4xi1> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_cas_tko(%arg0: !cuda_tile.tile<4x!cuda_tile.ptr<i32>>,
                                  %arg1: !cuda_tile.tile<2xi32>,
                                  %arg2: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{failed to verify that all of {pointers, cmp, val, result} have same shape}}
    %0, %t = cuda_tile.atomic_cas_tko relaxed device %arg0, %arg1, %arg2
        : !cuda_tile.tile<4x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_cas_tko(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<f32>>,
                                  %arg1: !cuda_tile.tile<2xi32>,
                                  %arg2: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{expected pointee type ('f32') to match element type of 'val' ('i32')}}
    %0, %t = cuda_tile.atomic_cas_tko relaxed device %arg0, %arg1, %arg2
        : !cuda_tile.tile<2x!cuda_tile.ptr<f32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_cas_tko(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i8>>,
                       %arg1: !cuda_tile.tile<2xi8>,
                       %arg2: !cuda_tile.tile<2xi8>) {
  // expected-error @below{{expect only float or integer types with 32 or 64 bit}}
  %0, %t = atomic_cas_tko relaxed device %arg0, %arg1, %arg2
      : !cuda_tile.tile<2x!cuda_tile.ptr<i8>>, !cuda_tile.tile<2xi8> -> !cuda_tile.tile<2xi8>, !cuda_tile.token
}
}

// -----

cuda_tile.module @test_global {
  cuda_tile.global @g1 <f16: [1.0, 2.0]> : !cuda_tile.tile<2xf16>
  entry @kernel() {
    // expected-error @below{{pointee type of result type '!cuda_tile.ptr<f32>' does not match type 'f16' of the global @g1}}
    %0 = cuda_tile.get_global @g1 : !cuda_tile.tile<!cuda_tile.ptr<f32>>
  }
}

// -----

cuda_tile.module @test_global {
  entry @kernel() {
    // expected-error @below{{'g1' does not reference a valid global}}
    %0 = cuda_tile.get_global @g1 : !cuda_tile.tile<!cuda_tile.ptr<f32>>
  }
}

// -----

cuda_tile.module @test_global_non_scalar {
  entry @kernel() {
    // expected-error @below{{op result #0 must be 0D tile of Pointer type values, but got '!cuda_tile.tile<4xptr<f32>>}}
    %0 = cuda_tile.get_global @g1 : !cuda_tile.tile<4x!cuda_tile.ptr<f32>>
  }
}
// -----

cuda_tile.module @test_global {
  // expected-error @below{{type must have rank 1}}
  cuda_tile.global @g1 <f16: [[1.0, 2.0]]> : !cuda_tile.tile<1x2xf16>
}

// -----

cuda_tile.module @test_global_invalid_visibility {
  // expected-error @below{{expected valid '@'-identifier for symbol name}}
  cuda_tile.global invalid @g1 <i32: [42]> : !cuda_tile.tile<1xi32>
}

// -----

cuda_tile.module @test_global_visibility_typo {
  // expected-error @below{{expected valid '@'-identifier for symbol name}}
  cuda_tile.global Public @g1 <i32: [42]> : !cuda_tile.tile<1xi32>
}

// -----

cuda_tile.module @test_global_visibility_wrong_syntax {
  // expected-error @below{{expected valid '@'-identifier for symbol name}}
  cuda_tile.global "private" @g1 <i32: [42]> : !cuda_tile.tile<1xi32>
}

// -----

cuda_tile.module @test_entry_nested_visibility {
  // expected-error @below{{entry op must have public symbol visibility}}
  cuda_tile.entry @kernel() {
    cuda_tile.return
  } {sym_visibility = "nested"}
}

// -----

cuda_tile.module @test_entry_private_visibility {
  // expected-error @below{{entry op must have public symbol visibility}}
  cuda_tile.entry @kernel() {
    cuda_tile.return
  } {sym_visibility = "private"}
}

// -----

cuda_tile.module @test_global_builtin_visibility {
  // expected-error @below{{expected visibility to be specified with `symbol_visibility`, not `sym_visibility`}}
  cuda_tile.global @g1 {sym_visibility = "nested"} <i32: [42]> : !cuda_tile.tile<1xi32>
}

// -----

cuda_tile.module @test_kernel_scope {
  // expected-error @below{{entry op must have scalar types (rank 0 !cuda_tile.tile)}}
  cuda_tile.entry @entry_with_result(%arg0: !cuda_tile.tile<2x2xf32>) {}
}

// -----

cuda_tile.module @test_fpowf {
  testing$func @kernel(%arg0: !cuda_tile.tile<2xi32>) {
    // expected-error @below{{'cuda_tile.fpowf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2xi32>'}}
    %0 = cuda_tile.fpowf %arg0, %arg0 : !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @test_negf {
  testing$func @kernel(%arg0: !cuda_tile.tile<2xi32>) {
    // expected-error @below{{'cuda_tile.negf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2xi32>'}}
    %0 = cuda_tile.negf %arg0 : !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @test_get_tensor_shape_tensor_view_oob {
  testing$func @kernel(%tensor_view : !cuda_tile.tensor_view<64x64xf16, strides=[1,1]>) {
    // expected-error @below{{operation defines 2 results but was provided 3 to bind}}
    %0, %1, %2 = cuda_tile.get_tensor_shape %tensor_view : !cuda_tile.tensor_view<64x64xf16, strides=[1,1]> -> !cuda_tile.tile<i32>
  }
}

// -----

// Test that get_tensor_shape op has the right amount of results.
// This test uses generic format to specifically test the verifier.
cuda_tile.module @test_get_tensor_shape_tensor_view_oob {
  testing$func @kernel(%tensor_view : !cuda_tile.tensor_view<64x64xf16, strides=[1,1]>) {
    // expected-error @below{{expected 2 results due to tensor rank, but got 3}}
    %0:3 = "cuda_tile.get_tensor_shape"(%tensor_view) : (!cuda_tile.tensor_view<64x64xf16, strides=[1,1]>) -> (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>)
  }
}

// -----

cuda_tile.module @test_get_tensor_shape_invalid_input_type {
  testing$func @kernel(%value : !cuda_tile.tile<8x8x!cuda_tile.ptr<i32>>) {
    // expected-error @below{{expected tensor_view, got '!cuda_tile.tile<8x8xptr<i32>>'}}
    %0, %1 = cuda_tile.get_tensor_shape %value : !cuda_tile.tile<8x8x!cuda_tile.ptr<i32>> -> !cuda_tile.tile<i32>
  }
}

// -----

cuda_tile.module @test_get_tensor_shape_invalid_output_type {
  testing$func @kernel(%tensor_view : !cuda_tile.tensor_view<64x64xi32, strides=[1,1]>) {
    // expected-error @below{{op result #0 must be variadic of 0D tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<2xi32>}}
    %0, %1 = cuda_tile.get_tensor_shape %tensor_view : !cuda_tile.tensor_view<64x64xi32, strides=[1,1]> -> !cuda_tile.tile<2xi32>
  }
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%value = cuda_tile.constant <i64: [1, 2, 7, 8]> : !cuda_tile.tile<4xi64>
cuda_tile.loop {
  // expected-error @below{{op type does not match yield type, else branch yields '!cuda_tile.tile<i1>' but op result type is '!cuda_tile.tile<4xi64>'}}
  cuda_tile.if %cond -> (!cuda_tile.tile<4xi64>) {
    cuda_tile.yield %value : !cuda_tile.tile<4xi64>
  }
  else {
    cuda_tile.yield %cond : !cuda_tile.tile<i1>
  }
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op has non-empty return type, must define else branch}}
%if_val = cuda_tile.if %cond -> (!cuda_tile.tile<i32>) {
  cuda_tile.yield %value : !cuda_tile.tile<i32>
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op has return type of '!cuda_tile.tile<i32>' but else branch does not yield anything}}
%if_val = cuda_tile.if %cond -> (!cuda_tile.tile<i32>) {
  cuda_tile.yield %value : !cuda_tile.tile<i32>
} else {
  cuda_tile.print_tko "if else" -> !cuda_tile.token
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op does not return a value, but then branch yields '!cuda_tile.tile<i32>'}}
cuda_tile.if %cond {
  cuda_tile.yield %value : !cuda_tile.tile<i32>
} else {
  cuda_tile.print_tko "if else" -> !cuda_tile.token
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op does not return a value, but else branch yields '!cuda_tile.tile<i32>'}}
cuda_tile.if %cond {
  cuda_tile.print_tko "if then" -> !cuda_tile.token
} else {
  cuda_tile.yield %value : !cuda_tile.tile<i32>
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%i64value = cuda_tile.constant <i64: 1> : !cuda_tile.tile<i64>
%i32value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op type does not match yield type, then branch yields '!cuda_tile.tile<i32>' but op result type is '!cuda_tile.tile<i64>'}}
%if_value = cuda_tile.if %cond -> (!cuda_tile.tile<i64>) {
  cuda_tile.yield %i32value : !cuda_tile.tile<i32>
} else {
  cuda_tile.yield %i64value : !cuda_tile.tile<i64>
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%i64value = cuda_tile.constant <i64: 1> : !cuda_tile.tile<i64>
%i32value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op type does not match yield type, else branch yields '!cuda_tile.tile<i32>' but op result type is '!cuda_tile.tile<i64>'}}
%if_value = cuda_tile.if %cond -> (!cuda_tile.tile<i64>) {
  cuda_tile.yield %i64value : !cuda_tile.tile<i64>
} else {
  cuda_tile.yield %i32value : !cuda_tile.tile<i32>
}

// -----

cuda_tile.module @test_early_exit_loop_break_control_flow {
  entry @kernel() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    %value = cuda_tile.constant <i64: [1, 2, 7, 8]> : !cuda_tile.tile<4xi64>
    cuda_tile.loop {
      // expected-error @below{{op does not return a value, but else branch yields '!cuda_tile.tile<4xi64>'}}
      cuda_tile.if %cond {
        cuda_tile.break
      }
      else {
        cuda_tile.yield %value : !cuda_tile.tile<4xi64>
      }
    }
  }
}

// -----

// Test: 1D condition for if op (expecting scalar)
// expected-note @below{{prior use here}}
%cond_1d = cuda_tile.constant <i1: [true, false, true, false]> : !cuda_tile.tile<4xi1>
%i64value = cuda_tile.constant <i64: 1> : !cuda_tile.tile<i64>
%i32value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{use of value '%cond_1d' expects different type than prior uses: '!cuda_tile.tile<i1>' vs '!cuda_tile.tile<4xi1>}}
%if_value = cuda_tile.if %cond_1d -> (!cuda_tile.tile<i64>) {
  cuda_tile.yield %i32value : !cuda_tile.tile<i32>
} else {
  cuda_tile.yield %i64value : !cuda_tile.tile<i64>
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%i64value = cuda_tile.constant <i64: 1> : !cuda_tile.tile<i64>
%i32value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op type does not match yield type, then branch yields '!cuda_tile.tile<i32>' but op result type is '!cuda_tile.tile<i64>'}}
%if_value = cuda_tile.if %cond -> (!cuda_tile.tile<i64>) {
  cuda_tile.yield %i32value : !cuda_tile.tile<i32>
} else {
  cuda_tile.yield %i64value : !cuda_tile.tile<i64>
}

// -----

%cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%i64value = cuda_tile.constant <i64: 1> : !cuda_tile.tile<i64>
%i32value = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{op type does not match yield type, else branch yields '!cuda_tile.tile<i32>' but op result type is '!cuda_tile.tile<i64>'}}
%if_value = cuda_tile.if %cond -> (!cuda_tile.tile<i64>) {
  cuda_tile.yield %i64value : !cuda_tile.tile<i64>
} else {
  cuda_tile.yield %i32value : !cuda_tile.tile<i32>
}

// -----

cuda_tile.module @test_early_exit_loop_break_control_flow {
  testing$func @kernel() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    %value = cuda_tile.constant <i64: [1, 2, 7, 8]> : !cuda_tile.tile<4xi64>
    cuda_tile.loop {
      // expected-error @below{{op does not return a value, but else branch yields '!cuda_tile.tile<4xi64>'}}
      cuda_tile.if %cond {
        cuda_tile.break
      }
      else {
        cuda_tile.yield %value : !cuda_tile.tile<4xi64>
      }
    }
  }
}

// -----

// Direct return in for loop should fail
cuda_tile.module @test_return_in_for_loop {
  testing$func @kernel(%arg0: !cuda_tile.tile<4xi64>) -> !cuda_tile.tile<4xi64> {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c10 = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    cuda_tile.for %i in (%c0 to %c10, step %c1) : !cuda_tile.tile<i32> {
      // expected-error @below{{'cuda_tile.return' op expects parent op to be one of 'cuda_tile.entry, cuda_tile.preview$func, cuda_tile.if, cuda_tile.loop}}
      cuda_tile.return %arg0 : !cuda_tile.tile<4xi64>
    }
  }
}

// -----

// Return nested in if inside for loop should fail
cuda_tile.module @test_return_nested_in_if_in_for_loop {
  testing$func @kernel(%arg0: !cuda_tile.tile<4xi64>) -> !cuda_tile.tile<4xi64> {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c10 = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.for %i in (%c0 to %c10, step %c1) : !cuda_tile.tile<i32> {
      cuda_tile.if %cond {
        // expected-error @below{{'cuda_tile.return' op must be used within a}}
        cuda_tile.return %arg0 : !cuda_tile.tile<4xi64>
      }
      cuda_tile.continue
    }
  }
}

// -----

// Return nested in loop inside for loop should fail
cuda_tile.module @test_return_nested_in_loop_in_for_loop {
  testing$func @kernel(%arg0: !cuda_tile.tile<4xi64>) -> !cuda_tile.tile<4xi64> {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c10 = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    cuda_tile.for %i in (%c0 to %c10, step %c1) : !cuda_tile.tile<i32> {
      cuda_tile.loop {
        // expected-error @below{{'cuda_tile.return' op must be used within a}}
        cuda_tile.return %arg0 : !cuda_tile.tile<4xi64>
      }
      cuda_tile.continue
    }
  }
}

// -----

// Return inside reduce body should fail
cuda_tile.module @test_return_in_reduce {
  testing$func @kernel(%arg0: !cuda_tile.tile<8xf32>) -> !cuda_tile.tile<f32> {
    %identity = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>
    %result = cuda_tile.reduce %arg0 dim=0 identities=[0.0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    (%val: !cuda_tile.tile<f32>, %acc: !cuda_tile.tile<f32>) {
      // expected-error @below{{'cuda_tile.return' op expects parent op to be one of 'cuda_tile.entry, cuda_tile.preview$func, cuda_tile.if, cuda_tile.loop}}
      cuda_tile.return %identity : !cuda_tile.tile<f32>
    }
    cuda_tile.return %result : !cuda_tile.tile<f32>
  }
}

// -----

// Return inside scan body should fail
cuda_tile.module @test_return_in_scan {
  testing$func @kernel(%arg0: !cuda_tile.tile<8xf32>) -> !cuda_tile.tile<8xf32> {
    %identity = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>
    %result = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
    (%val: !cuda_tile.tile<f32>, %acc: !cuda_tile.tile<f32>) {
      // expected-error @below{{'cuda_tile.return' op expects parent op to be one of 'cuda_tile.entry, cuda_tile.preview$func, cuda_tile.if, cuda_tile.loop}}
      cuda_tile.return %identity : !cuda_tile.tile<f32>
    }
    cuda_tile.return %result : !cuda_tile.tile<8xf32>
  }
}

// -----

// expected-error @below{{use of undeclared SSA value name}}
%loop_result = cuda_tile.loop iter_values(%var0 = %foo) : !cuda_tile.tile<i32> -> !cuda_tile.tile<i32>  {
  %foo = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>
}

// -----

// expected-error @below{{cannot name an operation with no results}}
%loop_result = cuda_tile.loop  {}

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
// expected-error @below{{use of undeclared SSA value name}}
%for_result = cuda_tile.for %iv in (%c0_i32 to %c1_i32, step %c1_i32) : !cuda_tile.tile<i32>
                                    iter_values(%var0 = %c0_i32) -> (!cuda_tile.tile<i32>) {
  %c1_i32 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
}

// -----

%c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
%c1_i32 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
%for_result = cuda_tile.for %iv in (%c0_i32 to %c1_i32, step %c1_i32) : !cuda_tile.tile<i32>
// expected-error @below{{use of undeclared SSA value name}}
                                    iter_values(%var0 = %c2_i32) -> (!cuda_tile.tile<i32>) {
  %c2_i32 = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
}

// -----

%c0_i32_float_test = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
  // expected-note @below{{prior use here}}
%c1_f32_float_test = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32> // Float upper bound
%c1_i32_float_test = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{expects different type than prior uses: '!cuda_tile.tile<i32>' vs '!cuda_tile.tile<f32>'}}
%for_result_float_test = cuda_tile.for %iv in (%c0_i32_float_test to %c1_f32_float_test, step %c1_i32_float_test) : !cuda_tile.tile<i32> {
  // Loop body
}

// -----

// expected-error @below{{use of undeclared SSA value name}}
cuda_tile.if %c1_i32 {
  %c1_i32 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
}

// -----

cuda_tile.module @kernel {
  entry @flush_to_zero_modifier_add() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    %1 = cuda_tile.constant <f64: 2.0> : !cuda_tile.tile<f64>
    // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f64'}}
    addf %0, %1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<f64>
  }
}

// -----

cuda_tile.module @kernel {
  entry @modifiers_divf() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    %1 = cuda_tile.constant <f64: 2.0> : !cuda_tile.tile<f64>
  // Just make sure we allow only one rounding.
    // expected-error @below{{expected '>'}}
    divf %0, %1 rounding<approx, full> : !cuda_tile.tile<f64>
  }
}

// -----

cuda_tile.module @kernel {
  entry @flush_to_zero_modifier() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    %1 = cuda_tile.constant <f64: 2.0> : !cuda_tile.tile<f64>
    // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f64'}}
    divf %0, %1 rounding<approx> flush_to_zero : !cuda_tile.tile<f64>
  }
}

// -----

cuda_tile.module @test_absf {
  testing$func @kernel(%arg0 : !cuda_tile.tile<4x4xi16>) {
    // expected-error @below{{'cuda_tile.absf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<4x4xi16>'}}
    %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<4x4xi16>
  }
}

// -----

cuda_tile.module @kernel {
  entry @approx_modifier() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    %1 = cuda_tile.constant <f64: 2.0> : !cuda_tile.tile<f64>
    // expected-error @below{{approx modifier only supported for f32 data type, but got: 'f64'}}
    divf %0, %1 rounding<approx> : !cuda_tile.tile<f64>
  }
}

// -----

cuda_tile.module @test_absf {
  // expected-note @below{{prior use here}}
  testing$func @kernel(%arg0 : !cuda_tile.tile<f32>) {
    // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
    %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<1xf32>
  }
}

// -----

cuda_tile.module @kernel {
  entry @full_modifier() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    %1 = cuda_tile.constant <f64: 2.0> : !cuda_tile.tile<f64>
    // expected-error @below{{full modifier only supported for f32 data type, but got: 'f64'}}
    divf %0, %1 rounding<full> : !cuda_tile.tile<f64>
  }
}

// -----

cuda_tile.module @test_absf {
  testing$func @kernel(%arg0 : !cuda_tile.tile<4x4xtf32>) {
    // expected-error @below{{'cuda_tile.absf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<4x4xtf32>'}}
    %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<4x4xtf32>
  }
}
// -----

cuda_tile.module @kernel {
  entry @rounding_mode_and_approx_modifier() {
    %0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    %1 = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    // expected-error @below{{expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', 'approx', 'full'}}
    divf %0, %1 rounding<near_exact> : !cuda_tile.tile<f32>
  }
}

// -----

cuda_tile.module @test_rsqrt {
  testing$func @i16_input(%arg0 : !cuda_tile.tile<4xi16>) {
    // expected-error @below{{'cuda_tile.rsqrt' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<4xi16>'}}
    %0 = cuda_tile.rsqrt %arg0 : !cuda_tile.tile<4xi16>
  }
}

// -----

cuda_tile.module @test_sqrt {
  testing$func @i16_input(%arg0 : !cuda_tile.tile<4xi16>) {
    // expected-error @below{{'cuda_tile.sqrt' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<4xi16>'}}
    %0 = cuda_tile.sqrt %arg0 rounding<nearest_even> : !cuda_tile.tile<4xi16>
  }
}
// -----

cuda_tile.module @test_ceil {
  testing$func @i16_input(%arg0: !cuda_tile.tile<i16>) {
    // expected-error @below{{'cuda_tile.ceil' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i16>'}}
    %0 = cuda_tile.ceil %arg0 : !cuda_tile.tile<i16>
  }
}

// -----

cuda_tile.module @test_remf {
  testing$func @kernel(%arg0 : !cuda_tile.tile<4xi16>, %arg1 : !cuda_tile.tile<4xi16>) {
    // expected-error @below{{'cuda_tile.remf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<4xi16>'}}
    %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<4xi16>
  }
}

// -----

cuda_tile.module @test_mulf_modifiers {
  testing$func @kernel(%arg0: !cuda_tile.tile<2x4x8xbf16>) {
    // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'bf16'}}
    %0 = mulf %arg0, %arg0 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<2x4x8xbf16>
  }
}
// -----

cuda_tile.module @kernel {
  testing$func @invalid_exp2() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f64'}}
    exp2 %0 flush_to_zero : !cuda_tile.tile<f64>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @add_ptr_shape_mismatch(%ptr: !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, %idx: !cuda_tile.tile<i32>) {
    // expected-error @below{{op requires the same shape for all operands and results}}
    %0 = cuda_tile.offset %ptr, %idx : !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, !cuda_tile.tile<i32> -> !cuda_tile.tile<8x!cuda_tile.ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  // expected-note @below{{prior use here}}
  testing$func @add_ptr_invalid_operand_types(%arg0: !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<8x!cuda_tile.ptr<f32>>) {
    // expected-error @below{{use of value '%arg1' expects different type}}
    %0 = cuda_tile.offset %arg0, %arg1 : !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, !cuda_tile.tile<i32> -> !cuda_tile.tile<8x!cuda_tile.ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @add_ptr_invalid_offset_type(%arg0: !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<8xf32>) {
    // expected-error @below {{'cuda_tile.offset' op operand #1 must be tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<8xf32>'}}
    %0 = cuda_tile.offset %arg0, %arg1 : !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, !cuda_tile.tile<8xf32> -> !cuda_tile.tile<16x!cuda_tile.ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @add_ptr_invalid_result_type(%arg0: !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below {{'cuda_tile.offset' op failed to verify that all of {result, ptr} have same type}}
    %0 = cuda_tile.offset %arg0, %arg1 : !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<8x!cuda_tile.ptr<f64>>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @add_ptr_invalid_result_shape(%arg0: !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below {{'cuda_tile.offset' op failed to verify that all of {result, ptr} have same type}}
    %0 = cuda_tile.offset %arg0, %arg1 : !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<16x!cuda_tile.ptr<f32>>
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_cas(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                  %arg1: !cuda_tile.tile<2xi32>,
                                  %arg2: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'memory_ordering_semantics' [weak, relaxed, acquire, release, acq_rel]}}
    %0, %t = cuda_tile.atomic_rmw_tko invalid_sem %arg0, %arg1, %arg2
        : !cuda_tile.tile<2x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw_invalid_sem(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                          %arg1: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{memory ordering semantics must be one of: relaxed, acquire, release, acq_rel}}
    %0, %t = cuda_tile.atomic_rmw_tko weak device %arg0, add, %arg1
        : !cuda_tile.tile<2x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw_invalid_sem_seq_cst(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                                  %arg1: !cuda_tile.tile<2xi32>) {
    // expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'memory_ordering_semantics' [weak, relaxed, acquire, release, acq_rel]}}
    %0, %t = cuda_tile.atomic_rmw_tko seq_cst device %arg0, add, %arg1
        : !cuda_tile.tile<2x!cuda_tile.ptr<i32>>, !cuda_tile.tile<2xi32> -> !cuda_tile.tile<2xi32>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_rmw(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<tf32>>,
                                  %arg1: !cuda_tile.tile<2xtf32>) {
    // expected-error @below {{'xchg' works only with integers or float of 32 or 64 bitwidth}}
    %0, %t = cuda_tile.atomic_rmw_tko relaxed device %arg0, xchg, %arg1
        : !cuda_tile.tile<2x!cuda_tile.ptr<tf32>>, !cuda_tile.tile<2xtf32> -> !cuda_tile.tile<2xtf32>, !cuda_tile.token
  }
}


// -----

cuda_tile.module @get_tile_block_id_invalid_shape {
  cuda_tile.entry @func() {
    // expected-error @below{{op result #0 must be 0D tile of i32 values, but got '!cuda_tile.tile<1xi32>'}}
    cuda_tile.get_tile_block_id : !cuda_tile.tile<1xi32>
  }
}

// -----

cuda_tile.module @get_tile_block_id_invalid_type {
  cuda_tile.entry @func() {
    // expected-error @below{{op result #0 must be 0D tile of i32 values, but got '!cuda_tile.tile<i64>'}}
    cuda_tile.get_tile_block_id : !cuda_tile.tile<i64>
  }
}

// -----

cuda_tile.module @get_num_tile_blocks_invalid_shape {
  cuda_tile.entry @func() {
    // expected-error @below{{op result #0 must be 0D tile of i32 values, but got '!cuda_tile.tile<1xi32>'}}
    cuda_tile.get_num_tile_blocks : !cuda_tile.tile<1xi32>
  }
}

// -----

cuda_tile.module @get_num_tile_blocks_invalid_type {
  cuda_tile.entry @func() {
    // expected-error @below{{op result #0 must be 0D tile of i32 values, but got '!cuda_tile.tile<i64>'}}
    cuda_tile.get_num_tile_blocks : !cuda_tile.tile<i64>
  }
}

// -----

cuda_tile.module @print_expected_attribute_value {
  cuda_tile.entry @func() {
    // expected-error @below{{expected attribute value}}
    cuda_tile.print_tko : !cuda_tile.tile<2xf16> -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @print_invalid_operand {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.1, 2.2]> : !cuda_tile.tile<2xf16>
    // expected-error @below{{incorrect number of operands: expected 2, found 1}}
    cuda_tile.print_tko "hello_world, %f, %f", %0 : !cuda_tile.tile<2xf16> -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @print_invalid_format_string {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.1, 2.2]> : !cuda_tile.tile<2xf16>
    // expected-error @below{{found unterminated format expression}}
    cuda_tile.print_tko "hello_world, %", %0 : !cuda_tile.tile<2xf16> -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @print_invalid_token_type {
  cuda_tile.entry @func() {
    // expected-note @below{{prior use here}}
    %val = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    // expected-error @below{{expects different type}}
    cuda_tile.print_tko "test: %i\n", %val token = %val : !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

// Test that get_index_space_shape op fails when the amount of results is out of bounds for the tile view.
cuda_tile.module @test_get_index_space_shape_oob {
  testing$func @kernel(%view: !cuda_tile.partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>>) {
    // expected-error @below{{operation defines 2 results but was provided 3 to bind}}
    %0, %1, %2 = get_index_space_shape %view : partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>> -> tile<i32>
  }
}

// -----

// Test that get_index_space_shape op fails when the amount of results is out of bounds for the tile view.
// This test uses generic format to specifically test the verifier.
cuda_tile.module @test_get_index_space_shape_oob_generic {
  testing$func @kernel(%view: !cuda_tile.partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>>) {
    // expected-error @below{{expected 2 results due to view index space rank, but got 3}}
    %0:3 = "cuda_tile.get_index_space_shape"(%view) : (!cuda_tile.partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>>) -> (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>)
  }
}

// -----

// Test that a tensor_view is not allowed to be returned by a loop.
cuda_tile.testing$func @test_tensor_view_returned_by_loop(%arg0: !cuda_tile.tensor_view<2x2xf32, strides=[1,1]>) {
  // expected-error @below {{result type 0 is a tensor_view, which is not supported}}
  %0 = loop : tensor_view<2x2xf32, strides=[1,1]> {
    break %arg0 : tensor_view<2x2xf32, strides=[1,1]>
  }
}

// -----

// Test that a partition_view is not allowed to be returned by a loop.
cuda_tile.testing$func @test_partition_view_returned_by_loop(%arg0: !cuda_tile.partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>) {
  // expected-error @below {{result type 0 is a tile view, which is not supported}}
  %0 = loop : partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>> {
    break %arg0 : partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>
  }
}

// -----

// Test that a tensor_view is not allowed as a block argument of a loop.
cuda_tile.testing$func @test_tensor_view_as_block_argument(%arg0: !cuda_tile.tensor_view<2x2xf32, strides=[1,1]>) {
  // expected-error @below {{loop-carried value 0 is a tensor_view, which is not supported}}
  loop iter_values(%x = %arg0) : tensor_view<2x2xf32, strides=[1,1]> {
    continue %x : tensor_view<2x2xf32, strides=[1,1]>
  }
}

// -----

// Test that a partition_view is not allowed as a block argument of a loop.
cuda_tile.testing$func @test_partition_view_as_block_argument(%arg0: !cuda_tile.partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>) {
  // expected-error @below {{loop-carried value 0 is a tile view, which is not supported}}
  loop iter_values(%x = %arg0) : partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>> {
    continue %x : partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>
  }
}

// -----

// Test that a tensor_view is not allowed as a result of a for-loop.
cuda_tile.testing$func @test_tensor_view_as_result_of_for_loop(%arg0: !cuda_tile.tensor_view<2x2xf32, strides=[1,1]>) {
  %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
  %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
  %c2 = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
  // expected-error @below {{op loop-carried value 0 is a tensor_view, which is not supported}}
  %0 = for %i in (%c0 to %c2, step %c1) : tile<i32> iter_values(%x = %arg0) -> (tensor_view<2x2xf32, strides=[1,1]>) {
    continue %x : tensor_view<2x2xf32, strides=[1,1]>
  }
}

// -----

// Test that a partition_view is not allowed as a result of a for-loop.
cuda_tile.testing$func @test_partition_view_as_result_of_for_loop(%arg0: !cuda_tile.partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>) {
  %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
  %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
  %c2 = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
  // expected-error @below {{op loop-carried value 0 is a tile view, which is not supported}}
  %0 = for %i in (%c0 to %c2, step %c1) : tile<i32> iter_values(%x = %arg0) -> (partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>) {
    continue %x : partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>
  }
}

// -----

// Test that a tensor_view is not allowed as a result of an if statement.
cuda_tile.testing$func @test_tensor_view_as_result_of_if(%cond: !cuda_tile.tile<i1>, %arg0: !cuda_tile.tensor_view<2x2xf32, strides=[1,1]>) {
  // expected-error @below {{op result type 0 is a tensor_view, which is not supported}}
  %0 = if %cond -> (tensor_view<2x2xf32, strides=[1,1]>) {
    cuda_tile.return %arg0 : tensor_view<2x2xf32, strides=[1,1]>
  } else {
    cuda_tile.return %arg0 : tensor_view<2x2xf32, strides=[1,1]>
  }
}

// -----

// Test that a partition_view is not allowed as a result of an if statement.
cuda_tile.testing$func @test_partition_view_as_result_of_if(%cond: !cuda_tile.tile<i1>, %arg0: !cuda_tile.partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>) {
  // expected-error @below {{op result type 0 is a tile view, which is not supported}}
  %0 = if %cond -> (partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>) {
    cuda_tile.return %arg0 : partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>
  } else {
    cuda_tile.return %arg0 : partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[1,1]>>
  }
}

// -----

cuda_tile.testing$func @itof_test(%arg0: !cuda_tile.tile<2x2xi32>) -> !cuda_tile.tile<2x2xf32> {
  // expected-error @below {{expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', 'nearest_away', got: 'foo'}}
  %f = itof %arg0 unsigned rounding<foo> : tile<2x2xi32> -> tile<2x2xf32>
  cuda_tile.return %f : tile<2x2xf32>
}

// -----

cuda_tile.testing$func @itof_test(%arg0: !cuda_tile.tile<2x2xi32>) -> !cuda_tile.tile<2x2xf32> {
  // expected-error @below {{expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', 'nearest_away', got: 'nearest_int_to_positive_inf'}}
  %f = itof %arg0 unsigned rounding<nearest_int_to_positive_inf> : tile<2x2xi32> -> tile<2x2xf32>
  cuda_tile.return %f : tile<2x2xf32>
}

// -----

cuda_tile.testing$func @ftoi_test(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xi64> {
 // expected-error @below {{expected rounding mode to be one of: 'nearest_int_to_zero', got: 'foo'}}
  %f = ftoi %arg0 unsigned rounding<foo> : tile<2x2xf32> -> tile<2x2xi64>
  cuda_tile.return %f : tile<2x2xi64>
}

// -----

cuda_tile.testing$func @ftoi_test(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xi64> {
 // expected-error @below {{expected rounding mode to be one of: 'nearest_int_to_zero', got: 'nearest_even'}}
  %f = ftoi %arg0 unsigned rounding<nearest_even> : tile<2x2xf32> -> tile<2x2xi64>
  cuda_tile.return %f : tile<2x2xi64>
}

// -----

cuda_tile.testing$func @itof_test(%arg0: !cuda_tile.tile<2x2xi32>) -> !cuda_tile.tile<2x2xf32> {
  // expected-error @below {{op invalid rounding mode specified. Only 'nearest_even' is supported}}
  %f = itof %arg0 unsigned rounding<negative_inf> : tile<2x2xi32> -> tile<2x2xf32>
  cuda_tile.return %f : tile<2x2xf32>
}

// -----

cuda_tile.testing$func @ftof(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xf8E8M0FNU> {
  // expected-error @below {{invalid rounding mode specified for conversion to f8E8M0FNU. Only 'zero' and 'positive_inf' are supported}}
  %f = ftof %arg0 rounding<nearest_even> : tile<2x2xf32> -> tile<2x2xf8E8M0FNU>
  cuda_tile.return %f : tile<2x2xf8E8M0FNU>
}

// -----

cuda_tile.testing$func @ftof_rna_bf16(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xbf16> {
  // expected-error @below {{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
  %f = ftof %arg0 rounding<nearest_away> : tile<2x2xf32> -> tile<2x2xbf16>
  cuda_tile.return %f : tile<2x2xbf16>
}

// -----

cuda_tile.testing$func @ftof_rna_f64_to_f32(%arg0: !cuda_tile.tile<2x2xf64>) -> !cuda_tile.tile<2x2xf32> {
  // expected-error @below {{invalid rounding mode specified for conversion from f64 to f32. Only 'nearest_even', 'zero', 'negative_inf', and 'positive_inf' are supported}}
  %f = ftof %arg0 rounding<nearest_away> : tile<2x2xf64> -> tile<2x2xf32>
  cuda_tile.return %f : tile<2x2xf32>
}

// -----

cuda_tile.testing$func @ftof_rp_f32_to_tf32(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xtf32> {
  // expected-error @below {{invalid rounding mode specified for conversion from f32 to tf32. Only 'nearest_even', 'zero', and 'nearest_away' are supported}}
  %f = ftof %arg0 rounding<positive_inf> : tile<2x2xf32> -> tile<2x2xtf32>
  cuda_tile.return %f : tile<2x2xtf32>
}

// -----

cuda_tile.testing$func @ftof_zero_to_fp8(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xf8E4M3FN> {
  // expected-error @below {{invalid rounding mode specified for conversion to low-precision type. Only 'nearest_even' is supported}}
  %f = ftof %arg0 rounding<zero> : tile<2x2xf32> -> tile<2x2xf8E4M3FN>
  cuda_tile.return %f : tile<2x2xf8E4M3FN>
}

// -----

cuda_tile.testing$func @ftof_rp_f32_to_f16(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xf16> {
  // expected-error @below {{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
  %f = ftof %arg0 rounding<positive_inf> : tile<2x2xf32> -> tile<2x2xf16>
  cuda_tile.return %f : tile<2x2xf16>
}

// -----

cuda_tile.testing$func @ftof_rm_f64_to_f16(%arg0: !cuda_tile.tile<2x2xf64>) -> !cuda_tile.tile<2x2xf16> {
  // expected-error @below {{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
  %f = ftof %arg0 rounding<negative_inf> : tile<2x2xf64> -> tile<2x2xf16>
  cuda_tile.return %f : tile<2x2xf16>
}

// -----

cuda_tile.testing$func @ftof_rna_f64_to_bf16(%arg0: !cuda_tile.tile<2x2xf64>) -> !cuda_tile.tile<2x2xbf16> {
  // expected-error @below {{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
  %f = ftof %arg0 rounding<nearest_away> : tile<2x2xf64> -> tile<2x2xbf16>
  cuda_tile.return %f : tile<2x2xbf16>
}

// -----

cuda_tile.testing$func @ftof_rm_bf16_to_f16(%arg0: !cuda_tile.tile<2x2xbf16>) -> !cuda_tile.tile<2x2xf16> {
  // expected-error @below {{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
  %f = ftof %arg0 rounding<negative_inf> : tile<2x2xbf16> -> tile<2x2xf16>
  cuda_tile.return %f : tile<2x2xf16>
}

// -----

cuda_tile.testing$func @ftof_rp_f16_to_bf16(%arg0: !cuda_tile.tile<2x2xf16>) -> !cuda_tile.tile<2x2xbf16> {
  // expected-error @below {{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
  %f = ftof %arg0 rounding<positive_inf> : tile<2x2xf16> -> tile<2x2xbf16>
  cuda_tile.return %f : tile<2x2xbf16>
}

// -----

cuda_tile.entry @tensor_view_store_dynamic(%tensor_view: !cuda_tile.tensor_view<?x4096xf64, strides=[4096,1]>) {
  %view = make_partition_view %tensor_view 
  // expected-error @below {{'cuda_tile.make_partition_view' expected 'partition_view' type, but got '!cuda_tile.tensor_view<?x4096xf64, strides=[4096,1]>'}}
    : !cuda_tile.tensor_view<?x4096xf64, strides=[4096,1]> 
    -> !cuda_tile.partition_view<tile=(1024x1024), tensor_view<?x4096xf64, strides=[4096,1]>>
}

// -----

cuda_tile.testing$func @make_partition_view_invalid_type(%arg0: !cuda_tile.partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>) {
  // expected-error @+1 {{expected 'tensor_view' type, but got '!cuda_tile.partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>'}}
  %view = make_partition_view %arg0 : !cuda_tile.partition_view<tile=(4x4), partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>>
}

// -----

cuda_tile.testing$func @make_strided_view_invalid_type(%arg0: !cuda_tile.partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>) {
  // expected-error @+1 {{expected 'tensor_view' type, but got '!cuda_tile.partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>'}}
  %view = make_strided_view %arg0 : strided_view<tile=(4x4), traversal_strides=[4,4], partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>>
}

// -----

cuda_tile.testing$func @gather_scatter_view_invalid_type(%arg0: !cuda_tile.partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>) {
  // expected-error @+1 {{expected 'tensor_view' type, but got '!cuda_tile.partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>'}}
  %view = make_gather_scatter_view %arg0 : gather_scatter_view<tile=(4x4), partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>, sparse_dim=0>
}

// -----

// -----
// expected-error @below {{F4E2M1FN tiles must have an even number of elements}}
cuda_tile.testing$func @f4e2m1fn_invalid_scalar(%arg0 : !cuda_tile.tile<f4E2M1FN>) {
  return
}

// -----
// expected-error @below {{F4E2M1FN tiles must have an even number of elements}}
cuda_tile.testing$func @f4e2m1fn_invalid_shape(%arg0 : !cuda_tile.tile<3x3x3xf4E2M1FN>) {
  return
}

// -----

// expected-error @+2 {{tiles of ptr<f4E2M1FN> must have rank 0}}
cuda_tile.testing$func @f4e2m1fn_invalid_pointer_tile(
    %arg0 : !cuda_tile.tile<2x!cuda_tile.ptr<f4E2M1FN>>) {
  return
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf16>) {
  // expected-error @below {{op failed to verify that all of {source, result} have same rank}}
  %0 = pack %arg0 : tile<32xf16> -> tile<32x2xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32x32xf16>) {
  // expected-error @below {{op expects source and result to be rank-1 tiles}}
  %0 = pack %arg0 : tile<32x32xf16> -> tile<32x64xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf16>) {
  // expected-error @below {{op expects source and result to have the same size in bytes, but got source tile size 64 bytes and result tile size 32 bytes}}
  %0 = pack %arg0 : tile<32xf16> -> tile<32xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf8E4M3FN>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = pack %arg0 : tile<32xf8E4M3FN> -> tile<32xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf8E5M2>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = pack %arg0 : tile<32xf8E5M2> -> tile<32xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf8E8M0FNU>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = pack %arg0 : tile<32xf8E8M0FNU> -> tile<32xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xi8>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = pack %arg0 : tile<32xi8> -> tile<32xi8>
}

// -----


cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
  // expected-error @below {{op failed to verify that all of {source, result} have same rank}}
  %0 = unpack %arg0 : tile<64xi8> -> tile<32x2xf16>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64x64xi8>) {
  // expected-error @below {{op expects source and result to be rank-1 tiles}}
  %0 = unpack %arg0 : tile<64x64xi8> -> tile<64x32xf16>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
  // expected-error @below {{op expects source and result to have the same size in bytes, but got source tile size 64 bytes and result tile size 128 bytes}}
  %0 = unpack %arg0 : tile<64xi8> -> tile<64xf16>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<32xi8>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = unpack %arg0 : tile<32xi8> -> tile<32xf8E4M3FN>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<32xi8>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = unpack %arg0 : tile<32xi8> -> tile<32xf8E5M2>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<32xi8>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = unpack %arg0 : tile<32xi8> -> tile<32xf8E8M0FNU>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<32xi8>) {
  // expected-error @below {{op expects source and result to have different element type widths}}
  %0 = unpack %arg0 : tile<32xi8> -> tile<32xi8>
}


// -----


cuda_tile.module @test_static_alloca {
  cuda_tile.entry @test_static_alloca(%c64: tile<i64>) {
    // expected-error @below {{failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0}}
    %0 = alloca num_elem = -10, alignment = 16 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.entry @test_mixed_alloca(%c64: tile<i64>) {
    // expected-error @below {{failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0}}
    %0 = alloca num_elem = 32, alignment = -10 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.entry @test_alloc_align() {
    // expected-error @below {{op 'alignment' must be power of two}}
    %0 = alloca num_elem = 64, alignment = 3 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.entry @test_alloc_align() {
    // expected-error @below {{'alignment' (2) must be at least the natural size (4 bytes) for element type 'f32'}}
    %0 = alloca num_elem = 64, alignment = 2 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_invalid_weak(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{only 'relaxed' memory ordering is supported for view-based atomic reductions}}
    %t = atomic_red_view_tko weak device %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_red_view_tko_xchg(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{atomic_red_view_tko op cannot use xchg operation}}
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], xchg, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
      }
}

// -----

cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_invalid_sys_scope(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{is not supported for view-based atomic reductions; use 'tl_blk' or 'device' for TMA compatibility}}
    %t = atomic_red_view_tko relaxed sys %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }
}

// -----

// Check: view element type must match value element type.
cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_invalid_elem_type_mismatch(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xf32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{view element type ('i32') must match value element type ('f32')}}
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], addf, %value
        : tile<2x2xf32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }
}

// -----

// Check: view tile shape must match value shape.
cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_invalid_shape_mismatch(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<4x4xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{view tile shape 2, 2 must match value shape 4, 4 for element-wise atomic operations}}
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<4x4xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }
}

// -----

// Check: integer RMW mode requires integer element type.
cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_invalid_int_mode_with_float(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xf32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{'add' works only with integers i32 and i64}}
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<2x2xf32>, partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[2, 1]>>, tile<i32> -> token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_red_view_tko_shape_mismatch(
      %view: !cuda_tile.partition_view<tile=(4x4), !cuda_tile.tensor_view<4x4xi32, strides=[4, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{view tile shape 4, 4 must match value shape 2, 2 for element-wise atomic operations}}
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(4x4), tensor_view<4x4xi32, strides=[4, 1]>>, tile<i32> -> token
  }
}

// -----

// Check: float RMW mode (addf) requires float element type.
cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_addf_on_integers(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{'addf' works only with floats f16, bf16, f32, and f64}}
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], addf, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }
}

// -----

// Check: index count must match view tile rank.
cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_invalid_index_count(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{expected 2 index operand(s) for view with tile rank 2, got 1}}
    %t = atomic_red_view_tko relaxed device %view[%c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }
}

// -----

// Check: each index operand must be a scalar (rank-0) tile.
cuda_tile.module @kernels {
  testing$func @test_atomic_red_view_tko_invalid_index_not_scalar(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>,
      %idx: !cuda_tile.tile<1xi32>) {
    // expected-error @below {{index operand 0 must be a scalar tile (rank 0), got shape 1}}
    %t = atomic_red_view_tko relaxed device %view[%idx, %idx], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<1xi32> -> token
  }
}

// -----

cuda_tile.module @module {
  testing$func @test_atomic_red_view_tko_gather_scatter_view(
      %view: !cuda_tile.gather_scatter_view<tile=(16x16), !cuda_tile.tensor_view<1024x16xi32, strides=[16, 1]>, sparse_dim=0>,
      %value: !cuda_tile.tile<16x16xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below {{gather_scatter_view is not supported; use partition_view instead}}
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<16x16xi32>, gather_scatter_view<tile=(16x16), tensor_view<1024x16xi32, strides=[16, 1]>, sparse_dim=0>, tile<i32> -> token
  }
}

// -----

// Verify that type checking on control-flow edges is preserved when operands
// carry types not defined by the cuda_tile dialect (e.g. builtin `index`,
// `f32`).

// yield: non-CudaTile type mismatch between yield operand and if-op result.
%cond_ft = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
%fval = "unknown.const"() : () -> f32
%ival = "unknown.const"() : () -> index
// expected-error @below {{op type does not match yield type, else branch yields 'index' but op result type is 'f32'}}
%r1 = cuda_tile.if %cond_ft -> (f32) {
  cuda_tile.yield %fval : f32
} else {
  cuda_tile.yield %ival : index
}

// -----

// break: non-CudaTile type mismatch between break operand and loop result type.
%ival2 = "unknown.const"() : () -> index
%r2 = cuda_tile.loop : f32 {
  // expected-error @below {{operand types must correspond to the parent loop result types}}
  cuda_tile.break %ival2 : index
}

// -----

// continue: non-CudaTile type mismatch between continue operand and loop iter
// type.
%idx_init = "unknown.const"() : () -> index
%fval2 = "unknown.const"() : () -> f32
cuda_tile.loop iter_values(%arg = %idx_init) : index {
  // expected-error @below {{`loop` is missing a valid terminator. `continue` op should have operand types that match the parent loop iter_values}}
  cuda_tile.continue %fval2 : f32
}

