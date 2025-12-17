// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  // Test operations with optional attributes
  cuda_tile.entry @optional_attrs_test(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    // Operation with optional flush_to_zero attribute present
    %0 = cuda_tile.addf %a, %b rounding<nearest_even> flush_to_zero : tile<f32>
    
    // Operation with optional flush_to_zero attribute absent
    %1 = cuda_tile.addf %a, %b rounding<nearest_even> : tile<f32>
    
    // Operation with different optional attributes
    %2 = cuda_tile.addf %a, %b rounding<zero> : tile<f32>
    
    // Operation with flush_to_zero attribute present
    %3 = cuda_tile.addf %a, %b rounding<zero> flush_to_zero : tile<f32>
  }

  // Test operations with UnitAttr (presence-only attributes)
  cuda_tile.entry @unit_attrs_test(%cond: !cuda_tile.tile<i1>, %a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    // Test if-else operation which may have optional attributes
    %0 = cuda_tile.if %cond -> (!cuda_tile.tile<i32>) {
      cuda_tile.yield %a : !cuda_tile.tile<i32>
    } else {
      cuda_tile.yield %b : !cuda_tile.tile<i32>
    }
    cuda_tile.return
  }

  // Test operations with AttrSizedOperandSegments and optional operands
  cuda_tile.entry @optional_operands_test(%ptr: !cuda_tile.tile<ptr<f32>>, %mask: !cuda_tile.tile<i1>, %padding: !cuda_tile.tile<f32>) {
    %token0 = cuda_tile.make_token : token
    %0, %res_token0 = cuda_tile.load_ptr_tko weak %ptr, %mask, %padding token=%token0
        : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

    // Test with some optional operands absent
    %1, %res_token1 = cuda_tile.load_ptr_tko weak %ptr
        : tile<ptr<f32>> -> tile<f32>, token

    // Test with mask but no padding or token
    %2, %res_token2 = cuda_tile.load_ptr_tko weak %ptr, %mask
        : tile<ptr<f32>>, tile<i1> -> tile<f32>, token
  }

  // Test mixed optional attributes and operands
  cuda_tile.entry @mixed_optional_test(%ptr: !cuda_tile.tile<ptr<f32>>, %mask: !cuda_tile.tile<i1>) {
    // Test with optional attribute and optional operand
    %0, %res_token0 = cuda_tile.load_ptr_tko relaxed device %ptr, %mask
        : tile<ptr<f32>>, tile<i1> -> tile<f32>, token
    
    // Test with optional attribute but no optional operands
    %1, %res_token1 = cuda_tile.load_ptr_tko relaxed device %ptr
        : tile<ptr<f32>> -> tile<f32>, token
        
    // Test with no optional attribute but with optional operand
    %2, %res_token2 = cuda_tile.load_ptr_tko weak %ptr, %mask
        : tile<ptr<f32>>, tile<i1> -> tile<f32>, token
  }
}
