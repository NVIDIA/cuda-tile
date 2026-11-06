// Test bytecode writer with version 13.3 exercising producer section,
// global extended fields, and diverse operations together.
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.3 %s -o %t.tilebc
// RUN: cuda-tile-translate -cudatilebc-to-mlir %t.tilebc -o %t.roundtrip.mlir
// RUN: cuda-tile-opt %s -no-implicit-module -o %t.ref.mlir
// RUN: %PYTHON %S/../../diff_helper.py %t.ref.mlir %t.roundtrip.mlir

cuda_tile.module @kernels attributes {producer = "test-coverage-v133"} {
  cuda_tile.global private @g_priv <i32: [10]> : !cuda_tile.tile<1xi32>
  cuda_tile.global constant @g_const <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
  cuda_tile.global private constant @g_both <i64: [100]> : !cuda_tile.tile<1xi64>

  // Entry with optimization hints containing various attribute types
  cuda_tile.entry @rich_kernel(%a: !cuda_tile.tile<f32>,
                                %b: !cuda_tile.tile<i32>) optimization_hints=<sm_100={occupancy=128, label="fast", dims=[1, 2], scale=0.5 : f32, flag=true, data_type=!cuda_tile.tile<f32>}> {
    %c = cuda_tile.constant <f32: 3.14> : tile<f32>
    %sum = cuda_tile.addf %a, %c rounding<nearest_even> : tile<f32>
    cuda_tile.return
  }

  // Entry without hints using diverse ops
  cuda_tile.entry @diverse_ops(%x: !cuda_tile.tile<i32>,
                                %y: !cuda_tile.tile<i32>,
                                %cond: !cuda_tile.tile<i1>) {
    // Arithmetic ops
    %add = cuda_tile.addi %x, %y : tile<i32>
    %sub = cuda_tile.subi %x, %y : tile<i32>
    %mul = cuda_tile.muli %x, %y : tile<i32>

    // Control flow with regions
    %if_r = cuda_tile.if %cond -> (!cuda_tile.tile<i32>) {
      cuda_tile.yield %add : !cuda_tile.tile<i32>
    } else {
      cuda_tile.yield %sub : !cuda_tile.tile<i32>
    }

    // Loop with iter values
    %c0 = cuda_tile.constant <i32: 0> : tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : tile<i32>
    %c4 = cuda_tile.constant <i32: 4> : tile<i32>
    %loop_r = cuda_tile.for %iv in (%c0 to %c4, step %c1) : tile<i32> iter_values(%acc = %if_r) -> (tile<i32>) {
      %next = cuda_tile.addi %acc, %iv : tile<i32>
      cuda_tile.continue %next : tile<i32>
    }
    cuda_tile.return
  }
}
