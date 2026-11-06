// Roundtrip test for miscellaneous operations and features to exercise
// BytecodeReader generated readers for operations not covered by other tests.
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  cuda_tile.global @g1 <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
  cuda_tile.global @g2 alignment = 128 <i32: [10, 20, 30, 40]> : !cuda_tile.tile<4xi32>
  cuda_tile.global private @g3 <f64: [3.14]> : !cuda_tile.tile<1xf64>
  cuda_tile.global private constant @g4 <i32: [42]> : !cuda_tile.tile<1xi32>
  cuda_tile.global constant @g5 <i16: [7, 8]> : !cuda_tile.tile<2xi16>

  // Test iota operation
  cuda_tile.entry @iota_op() {
    %0 = cuda_tile.iota : tile<32xi32>
    %1 = cuda_tile.iota : tile<4xi64>
    cuda_tile.return
  }

  // Test bitcast, inttoptr, ptrtoint
  cuda_tile.entry @ptr_and_bitcast_ops(%iarg: !cuda_tile.tile<i64>,
                                        %parg: !cuda_tile.tile<ptr<i8>>) {
    %0 = cuda_tile.bitcast %iarg : tile<i64> -> tile<f64>
    %1 = cuda_tile.int_to_ptr %iarg : tile<i64> -> tile<ptr<i8>>
    %2 = cuda_tile.ptr_to_int %parg : tile<ptr<i8>> -> tile<i64>
    cuda_tile.return
  }

  // Test mulhii
  cuda_tile.entry @mulhii_op(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %0 = cuda_tile.mulhii %a, %b : tile<i32>
    cuda_tile.return
  }

  // Test reshape, broadcast, and cat
  cuda_tile.entry @shape_ops(%a: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.reshape %a : tile<f32> -> tile<1xf32>
    %1 = cuda_tile.broadcast %0 : tile<1xf32> -> tile<4xf32>
    %2 = cuda_tile.cat %1, %1 dim = 0 : tile<4xf32>, tile<4xf32> -> tile<8xf32>
    cuda_tile.return
  }

  // Test loop with continue
  cuda_tile.entry @loop_continue_ops(%a: !cuda_tile.tile<i32>) {
    %c0 = cuda_tile.constant <i32: 0> : tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : tile<i32>
    cuda_tile.loop iter_values(%iv = %c0) : tile<i32> {
      %next = cuda_tile.addi %iv, %c1 : tile<i32>
      cuda_tile.continue %next : tile<i32>
    }
    cuda_tile.return
  }

  // Test if without else
  cuda_tile.entry @if_no_else(%cond: !cuda_tile.tile<i1>) {
    cuda_tile.if %cond {
      %0 = cuda_tile.constant <i32: 1> : tile<i32>
      cuda_tile.yield
    }
    cuda_tile.return
  }

  // Test nested for loops
  cuda_tile.entry @nested_for(%a: !cuda_tile.tile<i32>) {
    %c0 = cuda_tile.constant <i32: 0> : tile<i32>
    %c4 = cuda_tile.constant <i32: 4> : tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : tile<i32>
    %outer = cuda_tile.for %oi in (%c0 to %c4, step %c1) : tile<i32> iter_values(%oagg = %a) -> (tile<i32>) {
      %inner = cuda_tile.for %ii in (%c0 to %c4, step %c1) : tile<i32> iter_values(%iagg = %oagg) -> (tile<i32>) {
        %sum = cuda_tile.addi %iagg, %ii : tile<i32>
        cuda_tile.continue %sum : tile<i32>
      }
      cuda_tile.continue %inner : tile<i32>
    }
    cuda_tile.return
  }
}
