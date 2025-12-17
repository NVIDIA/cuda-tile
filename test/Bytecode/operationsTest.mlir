// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  cuda_tile.global @my_test_global <f32: 1.23> : !cuda_tile.tile<1xf32>

  // Test addi operation
  cuda_tile.entry @addi_op(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %0 = cuda_tile.addi %a, %b : tile<i32>
  }

  // Test addf operation
  cuda_tile.entry @addf_op(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<nearest_even> : tile<f32>
  }

  // Test return operation
  cuda_tile.entry @return_op(%a: !cuda_tile.tile<i32>) {
    cuda_tile.return
  }

  // Test constant operation
  cuda_tile.entry @constant_op() {
    %0 = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
  }

  // Test multiple operations chained together
  cuda_tile.entry @multiple_ops(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %0 = cuda_tile.addi %a, %b : tile<i32>
    %1 = cuda_tile.addi %0, %a : tile<i32>
    %2 = cuda_tile.constant <i32: 5> : !cuda_tile.tile<i32>
    %3 = cuda_tile.addi %1, %2 : tile<i32>
  }

  // Test get_global operation
  cuda_tile.entry @get_global_op_test() {
    %0 = cuda_tile.get_global @my_test_global : tile<ptr<f32>>
  }

  // Test for operation with iter_values
  cuda_tile.entry @for_op(%a: !cuda_tile.tile<i32>) {
    %lower = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %upper = cuda_tile.constant <i32: 5> : !cuda_tile.tile<i32>
    %step = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %result = cuda_tile.for %iv in (%lower to %upper, step %step) : tile<i32> iter_values(%value = %a) -> (tile<i32>) {
      %new_value = cuda_tile.addi %value, %iv : tile<i32>
      cuda_tile.continue %new_value : tile<i32>
    }
    cuda_tile.return
  }

  cuda_tile.entry @join_tokens_op(%tok0: !cuda_tile.token, %tok1: !cuda_tile.token) {
    %0 = cuda_tile.join_tokens %tok0, %tok1 : token
  }

  entry @assume(%arg0: !cuda_tile.tile<i16>,
                %arg1: !cuda_tile.tile<ptr<f32>>,
                %arg2: !cuda_tile.tile<i1>,
                %arg3: !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>,
                %arg4: !cuda_tile.tile<i16>,
                %arg5: !cuda_tile.tile<i64>) {
    %0 = cuda_tile.assume #cuda_tile.div_by<32>, %arg0 : tile<i16>
    %1 = cuda_tile.assume #cuda_tile.div_by<32>, %arg1 : tile<ptr<f32>>
    %3 = cuda_tile.assume #cuda_tile.div_by<32>, %arg3 : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
    %5 = cuda_tile.assume #cuda_tile.div_by<1>, %arg4 : tile<i16>
    %6 = cuda_tile.assume #cuda_tile.div_by<1>, %arg5 : tile<i64>
    %7 = cuda_tile.assume #cuda_tile.same_elements<[]>, %arg4 : tile<i16>

    // CHECK: assume bounded<0, 42>, %{{.*}} : tile<i16>
    %9 = cuda_tile.assume #cuda_tile.bounded<0, 42>, %arg4 : tile<i16>
    // CHECK: assume bounded<?, 42>, %{{.*}} : tile<i16>
    %10 = cuda_tile.assume #cuda_tile.bounded<?, 42>, %arg4 : tile<i16>
    // CHECK: assume bounded<-4, ?>, %{{.*}} : tile<i16>
    %11 = cuda_tile.assume #cuda_tile.bounded<-4, ?>, %arg4 : tile<i16>
    // CHECK: assume bounded<?, ?>, %{{.*}} : tile<i16>
    %12 = cuda_tile.assume #cuda_tile.bounded<?, ?>, %arg4 : tile<i16>
  }

  // Test if-else operation
  cuda_tile.entry @if_else_op_test(%cond: !cuda_tile.tile<i1>, %a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %result = cuda_tile.if %cond -> (!cuda_tile.tile<i32>) {
      cuda_tile.yield %a : !cuda_tile.tile<i32>
    } else {
      cuda_tile.yield %b : !cuda_tile.tile<i32>
    }
    cuda_tile.return
  }

  entry @store_ptr_tko(%arg0: !cuda_tile.tile<!cuda_tile.ptr<i32>>, %arg1: !cuda_tile.tile<i32>, %arg2: !cuda_tile.tile<f64>) {
    %0 = make_token : !cuda_tile.token
    %result, %result_token = load_ptr_tko weak %arg0 token=%0 : !cuda_tile.tile<!cuda_tile.ptr<i32>> -> !cuda_tile.tile<i32>, !cuda_tile.token
    %1 = constant <i32: 25> : !cuda_tile.tile<i32>
    %2 = store_ptr_tko weak %arg0, %1 token=%result_token : !cuda_tile.tile<!cuda_tile.ptr<i32>>, !cuda_tile.tile<i32> -> !cuda_tile.token
    print "\0Ahello % from the tile world !\0A\00", %result : !cuda_tile.tile<i32>
    return
  }
}
