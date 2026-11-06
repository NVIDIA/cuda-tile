// Test bytecode roundtrip with rich optimization hints to exercise
// writeSelfContainedAttribute paths for various attribute types.
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  // Hints with string, bool, float, type, array, and nested dictionary values
  cuda_tile.entry @rich_hints(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={name="my_kernel", flag=true, ratio=1.5 : f32, dims=[1, 2, 3], data_type=!cuda_tile.tile<f32>, nested={inner_val=42}}> {
    cuda_tile.return
  }

  // Hints with only string values to exercise StringAttr self-contained path
  cuda_tile.entry @string_hints(%a: !cuda_tile.tile<i32>) optimization_hints=<sm_100={label="fast_path"}> {
    cuda_tile.return
  }

  // Hints with only bool values to exercise BoolAttr self-contained path
  cuda_tile.entry @bool_hints(%a: !cuda_tile.tile<i32>) optimization_hints=<sm_100={enable=true, disable=false}> {
    cuda_tile.return
  }

  // Hints with array of mixed values to exercise ArrayAttr self-contained path
  cuda_tile.entry @array_hints(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={schedule=[1, 2, 4, 8]}> {
    cuda_tile.return
  }

  // Hints with float value to exercise FloatAttr self-contained path
  cuda_tile.entry @float_hints(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={threshold=0.5 : f32, scale=2.0 : f64}> {
    cuda_tile.return
  }
}
