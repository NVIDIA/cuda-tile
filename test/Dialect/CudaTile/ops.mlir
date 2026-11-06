// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s

cuda_tile.module @kernels {

  // CHECK: global constant @g1 <f32: [1.000000e+00, 2.000000e+00]> : tile<2xf32>
  global constant @g1 <f32 : [1.0, 2.0]> : !cuda_tile.tile<2xf32>
  // CHECK: global @g2 alignment = 256 <f32: [1.000000e+00, 2.000000e+00]> : tile<2xf32>
  global @g2 alignment = 256 <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
  
  // Test sym_visibility attribute - default (public, omitted in output)
  // CHECK: global @g3 <i32: 42> : tile<1xi32>
  global @g3 <i32: [42]> : !cuda_tile.tile<1xi32>
  
  // Test sym_visibility attribute - explicit public (omitted in output)
  // CHECK: global @g4 <i32: 42> : tile<1xi32>
  global public @g4 <i32: [42]> : !cuda_tile.tile<1xi32>
  
  // Test sym_visibility attribute - private
  // CHECK: global private @g5 <i32: 10> : tile<1xi32>
  global private @g5 <i32: [10]> : !cuda_tile.tile<1xi32>
  
  // Test sym_visibility with alignment
  // CHECK: global private @g7 alignment = 128 <i64: 100> : tile<1xi64>
  global private @g7 alignment = 128 <i64: [100]> : !cuda_tile.tile<1xi64>
  
  // Test constant qualifier
  // CHECK: global constant @g8 <f32: 3.140000e+00> : tile<1xf32>
  global constant @g8 <f32: [3.14]> : !cuda_tile.tile<1xf32>
  
  // Test constant with visibility
  // CHECK: global private constant @g9 <i32: 255> : tile<1xi32>
  global private constant @g9 <i32: [255]> : !cuda_tile.tile<1xi32>
  
  // Test constant with alignment
  // CHECK: global constant @g10 alignment = 128 <f64: 2.718000e+00> : tile<1xf64>
  global constant @g10 alignment = 128 <f64: [2.718]> : !cuda_tile.tile<1xf64>
  
  entry @kernel8() {
    // CHECK: get_global @g1 : tile<ptr<f32>>
    %0 = get_global @g1 : tile<ptr<f32>>
  }

  entry @test() {
  // CHECK: %[[c1:.*]] = constant <i1: true> : tile<i1>
  %c1 = constant <i1: true> : !cuda_tile.tile<i1>

  // CHECK: %[[c42:.*]] = constant <i8: 42> : tile<i8>
  %c42 = constant <i8: 42> : !cuda_tile.tile<i8>

  // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
  %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>

  // CHECK: %[[c5:.*]] = constant <bf16: 5.500000e+00> : tile<bf16>
  %c5 = constant <bf16: 5.5> : !cuda_tile.tile<bf16>

  // CHECK: %[[c4_i32:.*]] = constant <i32: 4> : tile<i32>
  %c4_i32 = constant <i32: 4> : !cuda_tile.tile<i32>

  // CHECK: %[[c4_i64:.*]] = constant <i64: 4> : tile<i64>
  %c4_i64 = constant <i64: 4> : !cuda_tile.tile<i64>

  // CHECK: %[[c_tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
  %c_tensor = constant <f32: [[1.0, 2.0], [4.0, 5.0]]> : !cuda_tile.tile<2x2xf32>

  // CHECK: %[[cf16_tensor:.*]] = constant <f16: {{\[}}[2.000000e+00, 1.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
  %cf16_tensor = constant <f16: [[2.0, 1.0], [4.0, 5.0]]> : !cuda_tile.tile<2x2xf16>

  // CHECK: %[[cf4_tensor:.*]] = constant <f4E2M1FN: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 6.000000e+00]]> : tile<2x2xf4E2M1FN>
  %cf4_tensor = constant <f4E2M1FN: [[1.0, 2.0], [4.0, 6.0]]> : !cuda_tile.tile<2x2xf4E2M1FN>

  // CHECK: %[[c_itensor:.*]] = constant <i32: {{\[}}[1, 2], [4, 5]]> : tile<2x2xi32>
  %c_itensor = constant <i32: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi32>

  // CHECK: %[[c_i64tensor:.*]] = constant <i64: {{\[}}[1, 2], [4, 5]]> : tile<2x2xi64>
  %c_i64tensor = constant <i64: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi64>

  // CHECK: if %[[c1]] {
  if %c1 {
    // CHECK-NOT: yield
    yield
  }
  // CHECK: if %[[c1]] -> (tile<i1>) {
  %if_result = if %c1 -> (tile<i1>) {
    // CHECK: yield %[[c1]]
    yield %c1 : tile<i1>
  } else {
    // CHECK: yield %[[c1]]
    yield %c1 : tile<i1>
  }

  // CHECK: for {{.*}} in ({{.*}} to {{.*}}, step {{.*}}) : tile<i32>
  %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
  %c1_i32 = constant <i32: 1> : !cuda_tile.tile<i32>
  for %iv in (%c0_i32 to %c1_i32, step %c1_i32) : tile<i32> {
    // CHECK-NOT: continue
    continue
  }

  // CHECK: for unsigned {{.*}} in ({{.*}} to {{.*}}, step {{.*}}) : tile<i32>
  for unsigned %iv_u in (%c0_i32 to %c1_i32, step %c1_i32) : tile<i32> {
    // CHECK-NOT: continue
    continue
  }

  // CHECK: for {{.*}} in ({{.*}} to {{.*}}, step {{.*}}) : tile<i32> iter_values({{.*}}) -> (tile<i32>)
  %for_result = for %iv in (%c0_i32 to %c1_i32, step %c1_i32) : tile<i32>
                              iter_values(%var0 = %c0_i32) -> (tile<i32>) {
    // CHECK: if %[[c1]] {
    if %c1 {
      // CHECK: continue %{{.*}} : tile<i32>
      continue %iv : tile<i32>
    }

    // CHECK: continue %{{.*}} : tile<i32>
    continue %iv : tile<i32>
  }

  // CHECK: for unsigned {{.*}} in ({{.*}} to {{.*}}, step {{.*}}) : tile<i32> iter_values({{.*}}) -> (tile<i32>)
  %for_result_u = for unsigned %iv_u in (%c0_i32 to %c1_i32, step %c1_i32) : tile<i32>
                              iter_values(%var0_u = %c0_i32) -> (tile<i32>) {
    // CHECK: continue %{{.*}} : tile<i32>
    continue %iv_u : tile<i32>
  }

  // CHECK: loop {
  loop {
    // CHECK-NOT: continue
    continue
  }

  // CHECK: loop iter_values({{.*}}) : tile<i32> {
  loop iter_values(%var0 = %c0_i32) : tile<i32> {
    // CHECK: if %[[c1]] {
    if %c1 {
      // CHECK: break
      break
    }

    // CHECK: continue %{{.*}} : tile<i32>
    continue %var0 : tile<i32>
  }

  // CHECK: loop iter_values({{.*}}) : tile<i32>
  loop iter_values(%arg1 = %c0_i32) : tile<i32> {
    if %c1 {
      // CHECK: continue %{{.*}} : tile<i32>
      continue %arg1 : tile<i32>
    }
    // CHECK: break
    break
  }

  // CHECK: loop : tile<i32>
  %loop1 = loop : tile<i32> {}

  // CHECK: loop iter_values({{.*}}, {{.*}}) : tile<i32>, tile<i16> -> tile<2x2xf16>, tile<2x2xf32>, tile<bf16>
  %loop2:3 = loop iter_values(%arg1 = %c0_i32, %arg2 = %c42_i16) : tile<i32>, tile<i16> -> tile<2x2xf16>, tile<2x2xf32>, tile<bf16> {
    if %c1 {
      continue %arg1, %arg2 : tile<i32>, tile<i16>
    }
    break %cf16_tensor, %c_tensor, %c5 : tile<2x2xf16>, tile<2x2xf32>, tile<bf16>
  }

  // CHECK: loop iter_values({{.*}}) : tile<i32>
  loop iter_values(%arg1 = %c0_i32) : tile<i32> {
    if %c1 {
      // CHECK: continue %{{.*}} : tile<i32>
      continue %arg1 : tile<i32>
    }
    // CHECK: break
    break
  }

  // CHECK: loop iter_values({{.*}}, {{.*}}) : tile<i32>, tile<i16> -> tile<2x2xf16>, tile<2x2xf32>, tile<bf16>
  %loop4:3 = loop iter_values(%arg1 = %c0_i32, %arg2 = %c42_i16) : tile<i32>, tile<i16> -> tile<2x2xf16>, tile<2x2xf32>, tile<bf16> {
    if %c1 {
      continue %arg1, %arg2 : tile<i32>, tile<i16>
    }
    break %cf16_tensor, %c_tensor, %c5 : tile<2x2xf16>, tile<2x2xf32>, tile<bf16>
  }

  // ============================================================================
  // Loop with Return - Void Returns Only (entry op constraint)
  // ============================================================================
  
  // Void return inside conditional in loop
  // CHECK: loop {
  loop {
    if %c1 {
      // CHECK: return
      return 
    }
    continue
  }

  // Direct unconditional return in loop
  // CHECK: loop {
  loop { 
    // CHECK: return
    return
  }

  // Void return from typed loop
  // CHECK: loop iter_values
  %ret_loop1 = loop iter_values(%arg1 = %c0_i32) : tile<i32> -> tile<i32> {
    if %c1 {
      // CHECK: return
      return
    }
    break %arg1 : tile<i32>
  }

  // Test 4: Multiple sequential if-returns in same loop
  // CHECK: loop {
  loop {
    if %c1 {
      // CHECK: return
      return
    }
    if %c1 {
      // CHECK: return
      return
    }
    continue
  }

  // Both branches return
  // CHECK: loop {
  loop {
    if %c1 {
      // CHECK: return
      return
    } else {
      // CHECK: return
      return
    }
    continue
  }

  // Return in else branch
  // CHECK: loop {
  loop {
    if %c1 {
      break
    } else {
      // CHECK: return
      return
    }
    continue
  }

  // Two-level nested loops with return in inner
  // CHECK: loop {
  loop {
    // CHECK: loop {
    loop {
      if %c1 {
        // CHECK: return
        return
      }
      continue
    }
    continue
  }

  // Three-level nested loops with return in deepest
  // CHECK: loop {
  loop {
    // CHECK: loop {
    loop {
      // CHECK: loop {
      loop {
        if %c1 {
          // CHECK: return
          return
        }
        continue
      }
      continue
    }
    continue
  }

  // Sequential loops, first with return
  // CHECK: loop {
  loop {
    if %c1 {
      // CHECK: return
      return
    }
    continue
  }
  // CHECK: loop {
  loop {
    continue
  }

  // Nested loop with return in outer, not inner
  // CHECK: loop {
  loop {
    // CHECK: loop {
    loop {
      break
    }
    if %c1 {
      // CHECK: return
      return
    }
    continue
  }

  // Multiple iteration arguments with void return
  // CHECK: loop iter_values
  loop iter_values(%arg1 = %c0_i32, %arg2 = %c42_i16) : tile<i32>, tile<i16> {
    if %c1 {
      // CHECK: return
      return
    }
    continue %arg1, %arg2 : tile<i32>, tile<i16>
  }

  // Nested conditionals with return
  // CHECK: loop {
  loop {
    if %c1 {
      if %c1 {
        // CHECK: return
        return
      }
    }
    continue
  }

  // Loop with if-else where only one branch has return
  // CHECK: loop {
  loop {
    if %c1 {
      // CHECK: return
      return
    } else {
      continue
    }
    break
  }

  // CHECK: print_tko "hello_world"
  print_tko "hello_world" -> !cuda_tile.token

  // CHECK: print_tko "hello_world, %i, %f", %[[c1]], %[[c5]] : tile<i1>, tile<bf16>
  print_tko "hello_world, %i, %f", %c1, %c5 : tile<i1>, tile<bf16> -> !cuda_tile.token

  // CHECK: print_tko "hello_world2, %lld, %+08.3f %%", %[[c_i64tensor]], %[[c5]] : tile<2x2xi64>, tile<bf16>
  print_tko "hello_world2, %lld, %+08.3f %%", %c_i64tensor, %c5 : !cuda_tile.tile<2x2xi64>, tile<bf16> -> !cuda_tile.token

  // CHECK: print_tko "%f%f"
  print_tko "%f%f", %c5, %c5 : tile<bf16>, tile<bf16> -> !cuda_tile.token

  // CHECK: print_tko "%f", %[[cf4_tensor]] : tile<2x2xf4E2M1FN>
  print_tko "%f", %cf4_tensor : tile<2x2xf4E2M1FN> -> !cuda_tile.token

  // CHECK: print_tko "%%%%"
  print_tko "%%%%" -> !cuda_tile.token

  // Token chaining with args - tests result token capture and reuse
  // CHECK: %[[TOK3:.*]] = print_tko "val: %i", %[[c4_i32]] : tile<i32>
  %tok3 = print_tko "val: %i", %c4_i32 : tile<i32> -> !cuda_tile.token
  // CHECK: print_tko "next: %i", %[[c4_i32]] token=%[[TOK3]] : tile<i32>
  print_tko "next: %i", %c4_i32 token = %tok3 : tile<i32> -> !cuda_tile.token

  // CHECK: addi %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %addi = addi %c42_i16, %c42_i16 : tile<i16>
  // CHECK: addi %[[c42_i16]], %[[c42_i16]] overflow<no_signed_wrap>  : tile<i16>
  %addi2 = addi %c42_i16, %c42_i16 overflow<no_signed_wrap> : tile<i16>
  // CHECK: addi %[[c42_i16]], %[[c42_i16]] overflow<no_unsigned_wrap>  : tile<i16>
  %addi3 = addi %c42_i16, %c42_i16 overflow<no_unsigned_wrap> : tile<i16>
  // CHECK: addi %[[c42_i16]], %[[c42_i16]] overflow<no_wrap>  : tile<i16>
  %addi4 = addi %c42_i16, %c42_i16 overflow<no_wrap> : tile<i16>
  // CHECK: addi %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %addi5 = addi %c42_i16, %c42_i16 overflow<none> : tile<i16>

  // CHECK: subi %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %subi = subi %c42_i16, %c42_i16 : tile<i16>
  // CHECK: subi %[[c42_i16]], %[[c42_i16]] overflow<no_signed_wrap>  : tile<i16>
  %subi2 = subi %c42_i16, %c42_i16 overflow<no_signed_wrap> : tile<i16>
  // CHECK: subi %[[c42_i16]], %[[c42_i16]] overflow<no_unsigned_wrap>  : tile<i16>
  %subi3 = subi %c42_i16, %c42_i16 overflow<no_unsigned_wrap> : tile<i16>
  // CHECK: subi %[[c42_i16]], %[[c42_i16]] overflow<no_wrap>  : tile<i16>
  %subi4 = subi %c42_i16, %c42_i16 overflow<no_wrap> : tile<i16>
  // CHECK: subi %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %subi5 = subi %c42_i16, %c42_i16 overflow<none> : tile<i16>

  // CHECK: muli %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %muli = muli %c42_i16, %c42_i16 : tile<i16>
  // CHECK: muli %[[c42_i16]], %[[c42_i16]] overflow<no_signed_wrap>  : tile<i16>
  %muli2 = muli %c42_i16, %c42_i16 overflow<no_signed_wrap> : tile<i16>
  // CHECK: muli %[[c42_i16]], %[[c42_i16]] overflow<no_unsigned_wrap>  : tile<i16>
  %muli3 = muli %c42_i16, %c42_i16 overflow<no_unsigned_wrap> : tile<i16>
  // CHECK: muli %[[c42_i16]], %[[c42_i16]] overflow<no_wrap>  : tile<i16>
  %muli4 = muli %c42_i16, %c42_i16 overflow<no_wrap> : tile<i16>
  // CHECK: muli %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %muli5 = muli %c42_i16, %c42_i16 overflow<none> : tile<i16>

  // CHECK: shli %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %shli = shli %c42_i16, %c42_i16 : tile<i16>
  // CHECK: shli %[[c42_i16]], %[[c42_i16]] overflow<no_signed_wrap>  : tile<i16>
  %shli2 = shli %c42_i16, %c42_i16 overflow<no_signed_wrap> : tile<i16>
  // CHECK: shli %[[c42_i16]], %[[c42_i16]] overflow<no_unsigned_wrap>  : tile<i16>
  %shli3 = shli %c42_i16, %c42_i16 overflow<no_unsigned_wrap> : tile<i16>
  // CHECK: shli %[[c42_i16]], %[[c42_i16]] overflow<no_wrap>  : tile<i16>
  %shli4 = shli %c42_i16, %c42_i16 overflow<no_wrap> : tile<i16>
  // CHECK: shli %[[c42_i16]], %[[c42_i16]] : tile<i16>
  %shli5 = shli %c42_i16, %c42_i16 overflow<none> : tile<i16>

  // CHECK: addf %[[c_tensor]], %[[c_tensor]] rounding<negative_inf> : tile<2x2xf32>
  %add2 = addf %c_tensor, %c_tensor rounding<negative_inf> : tile<2x2xf32>

  // CHECK: addf %[[c_tensor]], %[[c_tensor]] : tile<2x2xf32>
  %add3 = addf %c_tensor, %c_tensor : tile<2x2xf32>

  // CHECK: subf %[[c_tensor]], %[[c_tensor]] : tile<2x2xf32>
  %sub3 = subf %c_tensor, %c_tensor : tile<2x2xf32>

  // CHECK: addf %[[c_tensor]], %[[c_tensor]] flush_to_zero : tile<2x2xf32>
  %add4 = addf %c_tensor, %c_tensor flush_to_zero : tile<2x2xf32>

  // CHECK: remf %[[c_tensor]], %[[c_tensor]] : tile<2x2xf32>
  %remf1 = remf %c_tensor, %c_tensor : tile<2x2xf32>

  // CHECK: mulf %[[c_tensor]], %[[c_tensor]] rounding<zero> : tile<2x2xf32>
  %mul2 = mulf %c_tensor, %c_tensor rounding<zero> : tile<2x2xf32>

  // CHECK: maxf %[[c5]], %[[c5]] : tile<bf16>
  %maxf1 = maxf %c5, %c5 : tile<bf16>

  // CHECK: maxf %[[c_tensor]], %[[c_tensor]] : tile<2x2xf32>
  %maxf2 = maxf %c_tensor, %c_tensor : tile<2x2xf32>

  // CHECK: maxf %[[c_tensor]], %[[c_tensor]] flush_to_zero : tile<2x2xf32>
  %maxf3 = maxf %c_tensor, %c_tensor flush_to_zero : tile<2x2xf32>

  // CHECK: maxf %[[c_tensor]], %[[c_tensor]] propagate_nan : tile<2x2xf32>
  %maxf4 = maxf %c_tensor, %c_tensor propagate_nan : tile<2x2xf32>

  // CHECK: maxf %[[c_tensor]], %[[c_tensor]] flush_to_zero propagate_nan : tile<2x2xf32>
  %maxf5 = maxf %c_tensor, %c_tensor flush_to_zero propagate_nan : tile<2x2xf32>

  // CHECK: maxf %[[cf16_tensor]], %[[cf16_tensor]] propagate_nan : tile<2x2xf16>
  %maxf6 = maxf %cf16_tensor, %cf16_tensor propagate_nan : tile<2x2xf16>

  // CHECK: minf %[[c5]], %[[c5]] : tile<bf16>
  %minf1 = minf %c5, %c5 : tile<bf16>

  // CHECK: minf %[[c_tensor]], %[[c_tensor]] : tile<2x2xf32>
  %minf2 = minf %c_tensor, %c_tensor : tile<2x2xf32>

  // CHECK: minf %[[c_tensor]], %[[c_tensor]] flush_to_zero : tile<2x2xf32>
  %minf3 = minf %c_tensor, %c_tensor flush_to_zero : tile<2x2xf32>

  // CHECK: minf %[[c_tensor]], %[[c_tensor]] propagate_nan : tile<2x2xf32>
  %minf4 = minf %c_tensor, %c_tensor propagate_nan : tile<2x2xf32>

  // CHECK: minf %[[c_tensor]], %[[c_tensor]] flush_to_zero propagate_nan : tile<2x2xf32>
  %minf5 = minf %c_tensor, %c_tensor flush_to_zero propagate_nan : tile<2x2xf32>

  // CHECK: mini %[[c42_i16]], %[[c42_i16]] signed : tile<i16>
  %mini1 = mini %c42_i16, %c42_i16 signed : tile<i16>

  // CHECK: mini %[[c_itensor]], %[[c_itensor]] signed : tile<2x2xi32>
  %mini2 = mini %c_itensor, %c_itensor signed : tile<2x2xi32>

  // CHECK: mini %[[c_itensor]], %[[c_itensor]] unsigned : tile<2x2xi32>
  %mini3 = mini %c_itensor, %c_itensor unsigned : tile<2x2xi32>

  // CHECK: negi %[[c42_i16]] : tile<i16>
  %negi1 = negi %c42_i16 : tile<i16>
  // CHECK: negi %[[c42_i16]] overflow<no_signed_wrap> : tile<i16>
  %negi2 = negi %c42_i16 overflow<no_signed_wrap> : tile<i16>
  // CHECK: negi %[[c42]] : tile<i8>
  %negi3 = negi %c42 : tile<i8>
  // CHECK: negi %[[c4_i32]] overflow<no_signed_wrap> : tile<i32>
  %negi4 = negi %c4_i32 overflow<no_signed_wrap> : tile<i32>
  // CHECK: negi %[[c4_i64]] : tile<i64>
  %negi5 = negi %c4_i64 : tile<i64>
  // CHECK: negi %[[c_itensor]] : tile<2x2xi32>
  %negi6 = negi %c_itensor : tile<2x2xi32>

  // CHECK: exp2 %[[c_tensor]] : tile<2x2xf32>
  %exp2 = exp2 %c_tensor : tile<2x2xf32>

  // CHECK: exp2 %[[c_tensor]] flush_to_zero : tile<2x2xf32>
  %exp2_1 = exp2 %c_tensor flush_to_zero : tile<2x2xf32>

  // CHECK: exp %[[c_tensor]] : tile<2x2xf32>
  %exp = exp %c_tensor : tile<2x2xf32>

  // CHECK: exp %[[c_tensor]] rounding<approx> : tile<2x2xf32>
  %exp_approx = exp %c_tensor rounding<approx> : tile<2x2xf32>

  // CHECK: reshape %[[c42]] : tile<i8> -> tile<1xi8>
  %c_tensor_42 = reshape %c42 : tile<i8> -> tile<1xi8>

  // CHECK: reshape %{{.*}} : tile<1xi8> -> tile<i8>
  %c_tensor_reshaped = reshape %c_tensor_42 : tile<1xi8> -> tile<i8>

  // CHECK: reshape %[[c_tensor]] : tile<2x2xf32> -> tile<4xf32>
  %c_tensor_reshaped2 = reshape %c_tensor : tile<2x2xf32> -> tile<4xf32>

  // CHECK: divf %[[c_tensor]], %[[c_tensor]] flush_to_zero : tile<2x2xf32>
  %divf = divf %c_tensor, %c_tensor flush_to_zero : tile<2x2xf32>

  // CHECK: divf %[[c_tensor]], %[[c_tensor]] rounding<approx> : tile<2x2xf32>
  %divf1 = divf %c_tensor, %c_tensor rounding<approx> : tile<2x2xf32>

  // CHECK: divf %[[c_tensor]], %[[c_tensor]] rounding<full> : tile<2x2xf32>
  %divf2 = divf %c_tensor, %c_tensor rounding<full> : tile<2x2xf32>

  // CHECK: divf %[[c_tensor]], %[[c_tensor]] : tile<2x2xf32>
  %divf3 = divf %c_tensor, %c_tensor : tile<2x2xf32>

  // CHECK: log %[[c_tensor]] : tile<2x2xf32>
  %log_1 = log %c_tensor : tile<2x2xf32>

  // CHECK: log2 %[[c_tensor]] : tile<2x2xf32>
  %log2_1 = log2 %c_tensor : tile<2x2xf32>

  // CHECK: rsqrt %[[c_tensor]] : tile<2x2xf32>
  %rsqrt = rsqrt %c_tensor : tile<2x2xf32>

  // CHECK: sqrt %[[c_tensor]] rounding<approx> : tile<2x2xf32>
  %sqrt = sqrt %c_tensor rounding<approx> : tile<2x2xf32>

  // CHECK: trunci %[[c42_i16]] : tile<i16> -> tile<i8>
  %trunci1 = trunci %c42_i16 : tile<i16> -> tile<i8>
  // CHECK: trunci %[[c42_i16]] overflow<no_signed_wrap> : tile<i16> -> tile<i8>
  %trunci2 = trunci %c42_i16 overflow<no_signed_wrap> : tile<i16> -> tile<i8>
  // CHECK: trunci %[[c42_i16]] overflow<no_unsigned_wrap> : tile<i16> -> tile<i8>
  %trunci3 = trunci %c42_i16 overflow<no_unsigned_wrap> : tile<i16> -> tile<i8>
  // CHECK: trunci %[[c42_i16]] overflow<no_wrap> : tile<i16> -> tile<i8>
  %trunci4 = trunci %c42_i16 overflow<no_wrap> : tile<i16> -> tile<i8>
  // CHECK: trunci %[[c42_i16]] : tile<i16> -> tile<i8>
  %trunci5 = trunci %c42_i16 overflow<none> : tile<i16> -> tile<i8>

  // i4 trunci/exti round-trip
  // CHECK: trunci %[[c42_i16]] : tile<i16> -> tile<i4>
  %trunci_i4 = trunci %c42_i16 : tile<i16> -> tile<i4>
  // CHECK: trunci %[[c42_i16]] overflow<no_wrap> : tile<i16> -> tile<i4>
  %trunci_i4_nw = trunci %c42_i16 overflow<no_wrap> : tile<i16> -> tile<i4>
  // CHECK: exti %{{.+}} signed : tile<i4> -> tile<i8>
  %exti_i4_s = exti %trunci_i4 signed : tile<i4> -> tile<i8>
  // CHECK: exti %{{.+}} unsigned : tile<i4> -> tile<i16>
  %exti_i4_u = exti %trunci_i4 unsigned : tile<i4> -> tile<i16>
  // CHECK: exti %{{.+}} signed : tile<i4> -> tile<i32>
  %exti_i4_i32 = exti %trunci_i4 signed : tile<i4> -> tile<i32>
  }

    // ============================================================================
    // Loop with Return - Typed Returns (testing$func supports return values)
    // ============================================================================

    // CHECK-LABEL: test_loop_return_single_operand
    testing$func @test_loop_return_single_operand() -> !cuda_tile.tile<i32> {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        // CHECK: loop iter_values
        %result = loop iter_values(%arg1 = %c0_i32) : tile<i32> -> tile<i32> {
            if %c1 {
                // CHECK: return %{{.*}} : tile<i32>
                return %arg1 : tile<i32>
            }
            break %arg1 : tile<i32>
        }
        return %result : !cuda_tile.tile<i32>
    }

    // CHECK-LABEL: test_loop_return_multiple_operands
    testing$func @test_loop_return_multiple_operands() -> (!cuda_tile.tile<i32>, !cuda_tile.tile<i16>) {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
        // CHECK: loop iter_values
        %result:2 = loop iter_values(%arg1 = %c0_i32) : tile<i32> -> tile<i32>, tile<i16> {
            if %c1 {
                // CHECK: return %{{.*}}, %{{.*}} : tile<i32>, tile<i16>
                return %arg1, %c42_i16 : tile<i32>, tile<i16>
            }
            break %arg1, %c42_i16 : tile<i32>, tile<i16>
        }
        return %result#0, %result#1 : !cuda_tile.tile<i32>, !cuda_tile.tile<i16>
    }

    // CHECK-LABEL: test_loop_return_type_mismatch
    testing$func @test_loop_return_type_mismatch() -> !cuda_tile.tile<i16> {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
        // Return type differs from loop result type
        // CHECK: loop iter_values
        %result = loop iter_values(%arg1 = %c0_i32) : tile<i32> -> tile<i32> {
            if %c1 {
                // CHECK: return %{{.*}} : tile<i16>
                return %c42_i16 : tile<i16>
            }
            break %arg1 : tile<i32>
        }
        return %c42_i16 : !cuda_tile.tile<i16>
    }

    // CHECK-LABEL: test_loop_mixed_return_break
    testing$func @test_loop_mixed_return_break() -> !cuda_tile.tile<i32> {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        // CHECK: loop iter_values
        %result = loop iter_values(%arg1 = %c0_i32) : tile<i32> -> tile<i32> {
            if %c1 {
                // CHECK: return %{{.*}} : tile<i32>
                return %arg1 : tile<i32>
            } else {
                // CHECK: break %{{.*}} : tile<i32>
                break %arg1 : tile<i32>
            }
            continue %arg1 : tile<i32>
        }
        return %result : !cuda_tile.tile<i32>
    }

    // CHECK-LABEL: test_loop_return_tensor_types
    testing$func @test_loop_return_tensor_types() -> !cuda_tile.tile<2x2xf16> {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %cf16_tensor = constant <f16: [[2.0, 1.0], [4.0, 5.0]]> : !cuda_tile.tile<2x2xf16>
        // CHECK: loop iter_values
        %result = loop iter_values(%arg1 = %cf16_tensor) : tile<2x2xf16> -> tile<2x2xf16> {
            if %c1 {
                // CHECK: return %{{.*}} : tile<2x2xf16>
                return %arg1 : tile<2x2xf16>
            }
            break %arg1 : tile<2x2xf16>
        }
        return %result : !cuda_tile.tile<2x2xf16>
    }

    // CHECK-LABEL: test_loop_return_fewer_operands
    testing$func @test_loop_return_fewer_operands() -> !cuda_tile.tile<bf16> {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
        %c5 = constant <bf16: 5.5> : !cuda_tile.tile<bf16>
        // Loop has multiple results, return has different type
        // CHECK: loop iter_values
        %result:2 = loop iter_values(%arg1 = %c0_i32) : tile<i32> -> tile<i32>, tile<i16> {
            if %c1 {
                // CHECK: return %{{.*}} : tile<bf16>
                return %c5 : tile<bf16>
            }
            break %arg1, %c42_i16 : tile<i32>, tile<i16>
        }
        return %c5 : !cuda_tile.tile<bf16>
    }

    // CHECK-LABEL: test_void_loop_typed_return
    testing$func @test_void_loop_typed_return() -> !cuda_tile.tile<i32> {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        // Void loop (no results) but return has operand
        // CHECK: loop iter_values
        loop iter_values(%arg1 = %c0_i32) : tile<i32> {
            if %c1 {
                // CHECK: return %{{.*}} : tile<i32>
                return %arg1 : tile<i32>
            }
            break
        }
        return %c0_i32 : !cuda_tile.tile<i32>
    }

    // CHECK-LABEL: test_loop_return_nested_with_results
    testing$func @test_loop_return_nested_with_results() -> !cuda_tile.tile<i32> {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>
        %c0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        // CHECK: loop {
        loop {
            // CHECK: loop iter_values
            %inner = loop iter_values(%arg1 = %c0_i32) : tile<i32> -> tile<i32> {
                if %c1 {
                    // CHECK: return %{{.*}} : tile<i32>
                    return %arg1 : tile<i32>
                }
                break %arg1 : tile<i32>
            }
            break
        }
        return %c0_i32 : !cuda_tile.tile<i32>
    }

  // CHECK: entry @entry_early_exit
  entry @entry_early_exit() {
    %c1 = constant <i1: true> : !cuda_tile.tile<i1>

    // CHECK: if
    if %c1 {
      if %c1 {
        // CHECK: return
        return
      } else {
        // CHECK: return
        return
      }
      // CHECK: return
      return
    }
  }

  // CHECK-LABEL: test_broadcast_1
  testing$func @test_broadcast_1(%arg0: !cuda_tile.tile<1x2xf32>) {
    // CHECK: %{{.+}} = broadcast %{{.+}} : tile<1x2xf32> -> tile<2x2xf32>
    %0 = broadcast %arg0 : tile<1x2xf32> -> tile<2x2xf32>
  }
  // CHECK-LABEL: test_broadcast_2
  testing$func @test_broadcast_2(%arg0: !cuda_tile.tile<2x1xf32>) {
    // CHECK: %{{.+}} = broadcast %{{.+}} : tile<2x1xf32> -> tile<2x2xf32>
    %0 = broadcast %arg0 : tile<2x1xf32> -> tile<2x2xf32>
  }
  // CHECK-LABEL: test_broadcast_3
  testing$func @test_broadcast_3(%arg0: !cuda_tile.tile<1x1xf32>) {
    // CHECK: broadcast %{{.+}} : tile<1x1xf32> -> tile<2x2xf32>
    %0 = broadcast %arg0 : tile<1x1xf32> -> tile<2x2xf32>
  }

  // CHECK-LABEL: func_permute
  testing$func @func_permute(%arg0: !cuda_tile.tile<1x2xf32>) {
    // CHECK: permute %{{.+}} [1, 0] : tile<1x2xf32> -> tile<2x1xf32>
    %0 = permute %arg0 [1,0] : tile<1x2xf32> -> tile<2x1xf32>
    // CHECK: permute %{{.+}} [0, 1] : tile<1x2xf32> -> tile<1x2xf32>
    %1 = permute %arg0 [0,1] : tile<1x2xf32> -> tile<1x2xf32>
  }


  // CHECK-LABEL: @extract
  testing$func @extract(%t: !cuda_tile.tile<8xf32>, %idx: !cuda_tile.tile<i32>) {
    // CHECK: extract %{{.*}}[%{{.*}}] : tile<8xf32> -> tile<4xf32>
    %0 = extract %t[%idx] : tile<8xf32> -> tile<4xf32>
  }

  // CHECK-LABEL: extract_1d
  testing$func @extract_1d(%arg0: !cuda_tile.tile<8xf32>) {
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: extract %{{.+}}[%{{.+}}] : tile<8xf32> -> tile<4xf32>
    %0 = extract %arg0[%c0] : tile<8xf32> -> tile<4xf32>
  }

  // CHECK-LABEL: extract_2d
  testing$func @extract_2d(%arg0: !cuda_tile.tile<4x8xf32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    // CHECK: extract %{{.+}}[%{{.+}}, %{{.+}}] : tile<4x8xf32> -> tile<2x4xf32>
    %0 = extract %arg0[%c0, %c1] : tile<4x8xf32> -> tile<2x4xf32>
  }

  // CHECK-LABEL: extract_3d
  testing$func @extract_3d(%arg0: !cuda_tile.tile<8x4x16xf32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c2 = constant <i32: 0> : tile<i32>
    // CHECK: extract %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tile<8x4x16xf32> -> tile<4x2x8xf32>
    %0 = extract %arg0[%c0, %c1, %c2] : tile<8x4x16xf32> -> tile<4x2x8xf32>
  }

  // CHECK-LABEL: extract_scalar
  testing$func @extract_scalar(%arg0: !cuda_tile.tile<f32>) {
    // CHECK: extract %{{.+}}[] : tile<f32> -> tile<f32>
    %0 = extract %arg0[] : tile<f32> -> tile<f32>
  }

  // CHECK-LABEL: extract_boundary_valid
  testing$func @extract_boundary_valid(%arg0: !cuda_tile.tile<8xf32>) {
    %boundary_idx = constant <i32: 1> : tile<i32>
    // CHECK: extract %{{.+}}[%{{.+}}] : tile<8xf32> -> tile<4xf32>
    %0 = extract %arg0[%boundary_idx] : tile<8xf32> -> tile<4xf32>
  }

  // CHECK-LABEL: insert_1d
  testing$func @insert_1d(%src: !cuda_tile.tile<4xf32>, %dst: !cuda_tile.tile<8xf32>) -> !cuda_tile.tile<8xf32> {
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: insert %{{.+}}, %{{.+}}[%{{.+}}] : tile<4xf32>, tile<8xf32>
    %r = insert %src, %dst[%c0] : tile<4xf32>, tile<8xf32>
    return %r : tile<8xf32>
  }

  // CHECK-LABEL: func @insert_2d(
  //  CHECK-SAME:     %[[src:.*]]: tile<4x2xf32>,
  //  CHECK-SAME:     %[[dst:.*]]: tile<32x8xf32>,
  //  CHECK-SAME:     %[[idx:.*]]: tile<i32>)
  testing$func @insert_2d(%src: !cuda_tile.tile<4x2xf32>, %dst: !cuda_tile.tile<32x8xf32>, %idx: !cuda_tile.tile<i32>) -> !cuda_tile.tile<32x8xf32> {
    // CHECK: insert %[[src]], %[[dst]][%[[idx]], %[[idx]]] : tile<4x2xf32>, tile<32x8xf32>
    %r = insert %src, %dst[%idx, %idx] : tile<4x2xf32>, tile<32x8xf32>
    return %r : tile<32x8xf32>
  }

  // CHECK-LABEL: insert_3d
  testing$func @insert_3d(%src: !cuda_tile.tile<4x2x8xf32>, %dst: !cuda_tile.tile<8x4x16xf32>) -> !cuda_tile.tile<8x4x16xf32> {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    // CHECK: insert %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tile<4x2x8xf32>, tile<8x4x16xf32>
    %r = insert %src, %dst[%c0, %c1, %c0] : tile<4x2x8xf32>, tile<8x4x16xf32>
    return %r : tile<8x4x16xf32>
  }

  // CHECK-LABEL: insert_scalar
  testing$func @insert_scalar(%src: !cuda_tile.tile<f32>, %dst: !cuda_tile.tile<f32>) -> !cuda_tile.tile<f32> {
    // CHECK: insert %{{.+}}, %{{.+}}[] : tile<f32>, tile<f32>
    %r = insert %src, %dst[] : tile<f32>, tile<f32>
    return %r : tile<f32>
  }

  // CHECK-LABEL: insert_boundary_valid
  testing$func @insert_boundary_valid(%src: !cuda_tile.tile<4xf32>, %dst: !cuda_tile.tile<8xf32>) -> !cuda_tile.tile<8xf32> {
    %boundary_idx = constant <i32: 1> : tile<i32>
    // CHECK: insert %{{.+}}, %{{.+}}[%{{.+}}] : tile<4xf32>, tile<8xf32>
    %r = insert %src, %dst[%boundary_idx] : tile<4xf32>, tile<8xf32>
    return %r : tile<8xf32>
  }
  
  // CHECK-LABEL: add_ptr_i8
  testing$func @add_ptr_i8(%ptr: !cuda_tile.tile<8x!cuda_tile.ptr<f32>>, %idx: !cuda_tile.tile<8xi8>) {
    // CHECK:  %{{.+}} = offset %{{.+}}, %{{.+}} : tile<8xptr<f32>>, tile<8xi8> -> tile<8xptr<f32>>
    %0 = offset %ptr, %idx : tile<8xptr<f32>>, tile<8xi8> -> tile<8xptr<f32>>
  }

  // CHECK-LABEL: add_ptr_i16
  testing$func @add_ptr_i16(%ptr: !cuda_tile.tile<8xptr<f32>>, %idx: !cuda_tile.tile<8xi16>) {
    // CHECK:  %{{.+}} = offset %{{.+}}, %{{.+}} : tile<8xptr<f32>>, tile<8xi16> -> tile<8xptr<f32>>
    %0 = offset %ptr, %idx : tile<8xptr<f32>>, tile<8xi16> -> tile<8xptr<f32>>
  }

  // CHECK-LABEL: add_ptr_i32
  testing$func @add_ptr_i32(%ptr: !cuda_tile.tile<8xptr<f32>>, %idx: !cuda_tile.tile<8xi32>) {
    // CHECK:  %{{.+}} = offset %{{.+}}, %{{.+}} : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
    %0 = offset %ptr, %idx : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
  }

  // CHECK-LABEL: add_ptr_i64
  testing$func @add_ptr_i64(%ptr: !cuda_tile.tile<8xptr<f32>>, %idx: !cuda_tile.tile<8xi64>) {
    // CHECK:  %{{.+}} = offset %{{.+}}, %{{.+}} : tile<8xptr<f32>>, tile<8xi64> -> tile<8xptr<f32>>
    %0 = offset %ptr, %idx : tile<8xptr<f32>>, tile<8xi64> -> tile<8xptr<f32>>
  }

  // CHECK-LABEL: make_tensor_view
  // CHECK-SAME: (%[[BASE:.+]]: tile<ptr<f32>>, %[[CI64:.+]]: tile<i64>, %[[CI32:.+]]: tile<i32>, %[[CI16:.+]]: tile<i16>, %[[CI8:.+]]: tile<i8>, %[[CI1:.+]]: tile<i1>)
  testing$func @make_tensor_view(%base: !cuda_tile.tile<ptr<f32>>, %ci64: !cuda_tile.tile<i64>, %ci32: !cuda_tile.tile<i32>, %ci16: !cuda_tile.tile<i16>, %ci8: !cuda_tile.tile<i8>, %ci1: !cuda_tile.tile<i1>) {
    // CHECK: make_tensor_view %[[BASE]], shape = [], strides = [] : tensor_view<f32>
    make_tensor_view %base, shape = [], strides = [] : tensor_view<f32>

    // CHECK: make_tensor_view %[[BASE]], shape = [], strides = [] : tensor_view<f32>
    make_tensor_view %base, shape = [], strides = [] : tensor_view<f32>

    // CHECK: make_tensor_view %[[BASE]], shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf32, strides=[32,1]>
    make_tensor_view %base, shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf32, strides=[32,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI64]], 32], strides = [32, 1] : tile<i64> -> tensor_view<?x32xf32, strides=[32,1]>
    make_tensor_view %base, shape = [%ci64, 32], strides = [32, 1] : tile<i64> -> tensor_view<?x32xf32, strides=[32,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [32, 32], strides = [%[[CI64]], 1] : tile<i64> -> tensor_view<32x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [32, 32], strides = [%ci64, 1] : tile<i64> -> tensor_view<32x32xf32, strides=[?,1]>
    
    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI64]], %[[CI64]]], strides = [%[[CI64]], 1] : tile<i64> -> tensor_view<?x?xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci64, %ci64], strides = [%ci64, 1] : tile<i64> -> tensor_view<?x?xf32, strides=[?,1]>

    // Type coverage for bitwidth 32

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI64]], 32], strides = [%[[CI64]], 1] : tile<i64> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci64, 32], strides = [%ci64, 1] : tile<i64> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI32]], 32], strides = [%[[CI32]], 1] : tile<i32> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci32, 32], strides = [%ci32, 1] : tile<i32> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI16]], 32], strides = [%[[CI16]], 1] : tile<i16> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci16, 32], strides = [%ci16, 1] : tile<i16> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI8]], 32], strides = [%[[CI8]], 1] : tile<i8> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci8, 32], strides = [%ci8, 1] : tile<i8> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI1]], 32], strides = [%[[CI1]], 1] : tile<i1> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci1, 32], strides = [%ci1, 1] : tile<i1> -> tensor_view<?x32xf32, strides=[?,1]>

    // Type coverage for bitwidth 64

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI64]], 32], strides = [%[[CI64]], 1] : tile<i64> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci64, 32], strides = [%ci64, 1] : tile<i64> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI32]], 32], strides = [%[[CI32]], 1] : tile<i32> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci32, 32], strides = [%ci32, 1] : tile<i32> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI16]], 32], strides = [%[[CI16]], 1] : tile<i16> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci16, 32], strides = [%ci16, 1] : tile<i16> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI8]], 32], strides = [%[[CI8]], 1] : tile<i8> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci8, 32], strides = [%ci8, 1] : tile<i8> -> tensor_view<?x32xf32, strides=[?,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [%[[CI1]], 32], strides = [%[[CI1]], 1] : tile<i1> -> tensor_view<?x32xf32, strides=[?,1]>
    make_tensor_view %base, shape = [%ci1, 32], strides = [%ci1, 1] : tile<i1> -> tensor_view<?x32xf32, strides=[?,1]>
  }

  // CHECK-LABEL: make_tensor_view_f4e2m1fn
  // CHECK-SAME: (%[[BASE:.+]]: tile<ptr<f4E2M1FN>>)
  testing$func @make_tensor_view_f4e2m1fn(%base: !cuda_tile.tile<!cuda_tile.ptr<f4E2M1FN>>) {
    // CHECK: make_tensor_view %[[BASE]], shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf4E2M1FN, strides=[32,1]>
    make_tensor_view %base, shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf4E2M1FN, strides=[32,1]>

    // CHECK: make_tensor_view %[[BASE]], shape = [32, 32], strides = [1, 32] : tensor_view<32x32xf4E2M1FN, strides=[1,32]>
    make_tensor_view %base, shape = [32, 32], strides = [1, 32] : tensor_view<32x32xf4E2M1FN, strides=[1,32]>

    // CHECK: make_tensor_view %[[BASE]], shape = [1, 32], strides = [32, 1] : tensor_view<1x32xf4E2M1FN, strides=[32,1]>
    make_tensor_view %base, shape = [1, 32], strides = [32, 1] : tensor_view<1x32xf4E2M1FN, strides=[32,1]>
  }

  // CHECK-LABEL: get_tensor_shape
  // CHECK-SAME: (%[[VIEW:.+]]: tensor_view<64x64xi32, strides=[1,1]>)
  testing$func @get_tensor_shape(%tensor_view: !cuda_tile.tensor_view<64x64xi32, strides=[1,1]>) {
    // CHECK: %[[SIZE_I32:.*]]:2 = get_tensor_shape %[[VIEW]] : tensor_view<64x64xi32, strides=[1,1]> -> tile<i32>
    %size_i32:2 = get_tensor_shape %tensor_view : tensor_view<64x64xi32, strides=[1,1]> -> tile<i32>

    // CHECK: %[[SIZE_I16:.*]]:2 = get_tensor_shape %[[VIEW]] : tensor_view<64x64xi32, strides=[1,1]> -> tile<i16>
    %size_i16:2 = get_tensor_shape %tensor_view : tensor_view<64x64xi32, strides=[1,1]> -> tile<i16>

    // CHECK: %[[SIZE_I64:.*]]:2 = get_tensor_shape %[[VIEW]] : tensor_view<64x64xi32, strides=[1,1]> -> tile<i64>
    %size_i64:2 = get_tensor_shape %tensor_view : tensor_view<64x64xi32, strides=[1,1]> -> tile<i64>
  }

  // CHECK-LABEL: make_partition_view
  // CHECK-SAME: (%[[TENSOR_VIEW:.+]]: tensor_view<8192x8192x64xf32, strides=[524288,64,1]>,
  // CHECK-SAME (DISABLED): %[[TENSOR_VIEW_SCALAR:.+]]: tensor_view<f32>,
  // CHECK-SAME: %[[TENSOR_VIEW_DYN:.+]]: tensor_view<?x8192x64xf32, strides=[?,64,1]>)
  testing$func @make_partition_view(%tensor_view: !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>,
                            //%tensor_view_scalar: !cuda_tile.tensor_view<f32>,
                             %tensor_view_dyn: !cuda_tile.tensor_view<?x8192x64xf32, strides=[?,64,1]>) {
    // FIXME: Once 0-d tiled views are supported, enable this test.
    // CHECK (DISABLED): make_partition_view %[[TENSOR_VIEW_SCALAR]] : partition_view<tile=(), tensor_view<f32>>
    //make_partition_view %tensor_view_scalar : partition_view<tile=(), tensor_view<f32>>

    // CHECK: make_partition_view %[[TENSOR_VIEW]] : partition_view<tile=(1x1x1), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
    make_partition_view %tensor_view : partition_view<tile=(1x1x1), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>

    // CHECK: make_partition_view %[[TENSOR_VIEW]] : partition_view<tile=(1x1x1), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
    make_partition_view %tensor_view : partition_view<tile=(1x1x1), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
    // CHECK: make_partition_view %[[TENSOR_VIEW]] : partition_view<tile=(1024x8192x2), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
    make_partition_view %tensor_view : partition_view<tile=(1024x8192x2), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
    // CHECK: make_partition_view %[[TENSOR_VIEW]] : partition_view<tile=(1024x8x1024), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>, dim_map=[0, 2, 1]>
    make_partition_view %tensor_view : partition_view<tile=(1024x8x1024), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>, dim_map=[0, 2, 1]>

    // CHECK: make_partition_view %[[TENSOR_VIEW_DYN]] : partition_view<tile=(1x1x1), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
    make_partition_view %tensor_view_dyn : partition_view<tile=(1x1x1), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
    // CHECK: make_partition_view %[[TENSOR_VIEW_DYN]] : partition_view<tile=(1024x8192x2), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
    make_partition_view %tensor_view_dyn : partition_view<tile=(1024x8192x2), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
    // CHECK: make_partition_view %[[TENSOR_VIEW_DYN]] : partition_view<tile=(1024x8x1024), tensor_view<?x8192x64xf32, strides=[?,64,1]>, dim_map=[0, 2, 1]>
    make_partition_view %tensor_view_dyn : partition_view<tile=(1024x8x1024), tensor_view<?x8192x64xf32, strides=[?,64,1]>, dim_map=[0, 2, 1]>
  }

  // CHECK-LABEL: get_index_space_shape_partition_view
  // CHECK-SAME: (%[[VIEW:.*]]: partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>>)
  testing$func @get_index_space_shape_partition_view(%partition_view: !cuda_tile.partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>>) {
    // CHECK: %[[SIZE_I32:.*]]:3 = get_index_space_shape %[[VIEW]] : partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i32>
    %size_i32:3 = get_index_space_shape %partition_view : partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i32>

    // CHECK: %[[SIZE_I16:.*]]:3 = get_index_space_shape %[[VIEW]] : partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i16>
    %size_i16:3 = get_index_space_shape %partition_view : partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i16>

    // CHECK: %[[SIZE_I64:.*]]:3 = get_index_space_shape %[[VIEW]] : partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i64>
    %size_i64:3 = get_index_space_shape %partition_view : partition_view<tile=(8x1x16), tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i64>
  }

  // CHECK-LABEL: load_store_tile_partition
  // CHECK-SAME: (%[[VIEW1:.+]]: partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>
  // CHECK-SAME:  %[[VIEW3:.+]]: partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
  // CHECK-SAME:  %[[T1:.+]]: tile<8xf32>, %[[T3:.+]]: tile<1024x1024x8xf32>
  testing$func @load_store_tile_partition(%view1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>,
                             %view3: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>,
                             %t1: !cuda_tile.tile<8xf32>, %t3: !cuda_tile.tile<1024x1024x8xf32>) {
    // CHECK: %[[C0I64:.+]] = constant <i64: 0> : tile<i64>
    %c0i64 = constant <i64: 0> : !cuda_tile.tile<i64>
    // CHECK: %[[C0I32:.+]] = constant <i32: 0> : tile<i32>
    %c0i32 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %[[C0I16:.+]] = constant <i16: 0> : tile<i16>
    %c0i16 = constant <i16: 0> : !cuda_tile.tile<i16>
    // CHECK: %[[C0I8:.+]] = constant <i8: 0> : tile<i8>
    %c0i8 = constant <i8: 0> : !cuda_tile.tile<i8>
    // CHECK: %[[C0I1:.+]] = constant <i1: false> : tile<i1>
    %c0i1 = constant <i1: false> : !cuda_tile.tile<i1>
    
    // Stores

    // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I64]]] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i64> -> token
    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I64]], %[[C0I64]], %[[C0I64]]] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> token
    %s1i64 = store_view_tko weak %t1, %view1[%c0i64] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i64> -> token
    %s2i64 = store_view_tko weak %t3, %view3[%c0i64, %c0i64, %c0i64] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> token
  
    // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I32]]] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
    %s1i32 = store_view_tko weak %t1, %view1[%c0i32] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    %s2i32 = store_view_tko weak %t3, %view3[%c0i32, %c0i32, %c0i32] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
  
    // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I16]]] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i16> -> token
    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I16]], %[[C0I16]], %[[C0I16]]] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> token
    %s1i16 = store_view_tko weak %t1, %view1[%c0i16] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i16> -> token
    %s2i16 = store_view_tko weak %t3, %view3[%c0i16, %c0i16, %c0i16] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> token

    // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I8]]] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i8> -> token
    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I8]], %[[C0I8]], %[[C0I8]]] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> token
    %s1i8 = store_view_tko weak %t1, %view1[%c0i8] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i8> -> token
    %s2i8 = store_view_tko weak %t3, %view3[%c0i8, %c0i8, %c0i8] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> token

    // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I1]]] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i1> -> token
    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I1]], %[[C0I1]], %[[C0I1]]] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> token
    %s1i1 = store_view_tko weak %t1, %view1[%c0i1] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i1> -> token
    %s2i1 = store_view_tko weak %t3, %view3[%c0i1, %c0i1, %c0i1] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> token

    // Stores with explicit inbounds annotation (round-trip).

    // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I32]]] inbounds = [true] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    %s1ib1 = store_view_tko weak %t1, %view1[%c0i32] inbounds = [true] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token

    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] inbounds = [true, false, true] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
    %s2ib1 = store_view_tko weak %t3, %view3[%c0i32, %c0i32, %c0i32] inbounds = [true, false, true] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token

    // An all-`false` inbounds vector is the same value the parser
    // synthesizes for an omitted `inbounds = [...]` clause, so the printer
    // elides it for textual round-trip stability.
    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
    %s2ib0 = store_view_tko weak %t3, %view3[%c0i32, %c0i32, %c0i32] inbounds = [false, false, false] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token

    // Omitting the `inbounds = [...]` clause entirely must parse successfully
    // and round-trip to the same elided form as the all-`false` case above.
    // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
    %s2ibomit = store_view_tko weak %t3, %view3[%c0i32, %c0i32, %c0i32] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token

    // Loads

    // CHECK: %[[T1_I64:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I64]]] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i64> -> tile<8xf32>, token
    // CHECK: %[[T3_I64:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I64]], %[[C0I64]], %[[C0I64]]] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> tile<1024x1024x8xf32>, token
    %t1i64, %tok0i64 = load_view_tko weak %view1[%c0i64] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i64> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t3i64, %tok1i64 = load_view_tko weak %view3[%c0i64, %c0i64, %c0i64] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> tile<1024x1024x8xf32>, token

    // CHECK: %[[T1_I32:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I32]]] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> tile<8xf32>, token
    // CHECK: %[[T3_I32:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
    %t1i32, %tok0i32 = load_view_tko weak %view1[%c0i32] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t3i32, %tok1i32 = load_view_tko weak %view3[%c0i32, %c0i32, %c0i32] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token

    // CHECK: %[[T1_I16:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I16]]] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i16> -> tile<8xf32>, token
    // CHECK: %[[T3_I16:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I16]], %[[C0I16]], %[[C0I16]]] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> tile<1024x1024x8xf32>, token
    %t1i16, %tok0i16 = load_view_tko weak %view1[%c0i16] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i16> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t3i16, %tok1i16 = load_view_tko weak %view3[%c0i16, %c0i16, %c0i16] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> tile<1024x1024x8xf32>, token

    // CHECK: %[[T1_I8:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I8]]] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i8> -> tile<8xf32>, token
    // CHECK: %[[T3_I8:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I8]], %[[C0I8]], %[[C0I8]]] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> tile<1024x1024x8xf32>, token
    %t1i8, %tok0i8 = load_view_tko weak %view1[%c0i8] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i8> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t3i8, %tok1i8 = load_view_tko weak %view3[%c0i8, %c0i8, %c0i8] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> tile<1024x1024x8xf32>, token

    // CHECK: %[[T1_I1:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I1]]] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i1> -> tile<8xf32>, token
    // CHECK: %[[T3_I1:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I1]], %[[C0I1]], %[[C0I1]]] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> tile<1024x1024x8xf32>, token
    %t1i1, %tok0i1 = load_view_tko weak %view1[%c0i1] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i1> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t3i1, %tok1i1 = load_view_tko weak %view3[%c0i1, %c0i1, %c0i1] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> tile<1024x1024x8xf32>, token

    // Loads with explicit inbounds annotation (round-trip).

    // CHECK: %{{.+}}, %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I32]]] inbounds = [true] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> tile<8xf32>, token
    %t1ib1, %tok0ib1 = load_view_tko weak %view1[%c0i32] inbounds = [true] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> tile<8xf32>, token

    // CHECK: %{{.+}}, %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] inbounds = [true, false, true] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
    %t3ib1, %tok1ib1 = load_view_tko weak %view3[%c0i32, %c0i32, %c0i32] inbounds = [true, false, true] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token

    // An all-`false` inbounds vector is the same value the parser
    // synthesizes for an omitted `inbounds = [...]` clause, so the printer
    // elides it for textual round-trip stability.
    // CHECK: %{{.+}}, %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
    %t3ib0, %tok1ib0 = load_view_tko weak %view3[%c0i32, %c0i32, %c0i32] inbounds = [false, false, false] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token

    // Omitting the `inbounds = [...]` clause entirely must parse successfully
    // and round-trip to the same elided form as the all-`false` case above.
    // CHECK: %{{.+}}, %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
    %t3ibomit, %tok1ibomit = load_view_tko weak %view3[%c0i32, %c0i32, %c0i32] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
  }

  // CHECK-LABEL: @mma1
  testing$func @mma1(%arg0: !cuda_tile.tile<4x8xf32>, %arg1: !cuda_tile.tile<8x16xf32>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // CHECK: %{{.+}} = mmaf %{{.+}} : tile<4x8xf32>, tile<8x16xf32>, tile<4x16xf32>
    %0 = mmaf %arg0, %arg1, %arg2 : tile<4x8xf32>, tile<8x16xf32>, tile<4x16xf32>
  }

  // CHECK-LABEL: @mma1_fast_accum
  testing$func @mma1_fast_accum(%arg0: !cuda_tile.tile<4x8xf8E4M3FN>, %arg1: !cuda_tile.tile<8x16xf8E4M3FN>, %arg2: !cuda_tile.tile<4x16xf32>) {
    // CHECK: %{{.+}} = mmaf %{{.+}}, %{{.+}}, %{{.+}} fast_acc : tile<4x8xf8E4M3FN>, tile<8x16xf8E4M3FN>, tile<4x16xf32>
    %0 = mmaf %arg0, %arg1, %arg2 fast_acc : tile<4x8xf8E4M3FN>, tile<8x16xf8E4M3FN>, tile<4x16xf32>
  }

  // CHECK-LABEL: @mma2
  testing$func @mma2(%arg0: !cuda_tile.tile<4x8xi8>, %arg1: !cuda_tile.tile<8x16xi8>, %arg2: !cuda_tile.tile<4x16xi32>) {
    // CHECK: %{{.+}} = mmai %{{.+}}, %{{.+}}, %{{.+}} signed signed : tile<4x8xi8>, tile<8x16xi8>, tile<4x16xi32>
    %0 = mmai %arg0, %arg1, %arg2 signed signed : tile<4x8xi8>, tile<8x16xi8>, tile<4x16xi32>
  }

  // CHECK-LABEL: @mma3
  testing$func @mma3(%arg0: !cuda_tile.tile<4x8xi8>, %arg1: !cuda_tile.tile<8x16xi8>, %arg2: !cuda_tile.tile<4x16xi32>) {
    // CHECK: %{{.+}} = mmai %{{.+}}, %{{.+}}, %{{.+}} unsigned unsigned : tile<4x8xi8>, tile<8x16xi8>, tile<4x16xi32>
    %0 = mmai %arg0, %arg1, %arg2 unsigned unsigned : tile<4x8xi8>, tile<8x16xi8>, tile<4x16xi32>
  }

  // CHECK-LABEL: @mma4
  testing$func @mma4(%arg0: !cuda_tile.tile<2x4x8xi8>, %arg1: !cuda_tile.tile<2x8x16xi8>, %arg2: !cuda_tile.tile<2x4x16xi32>) {
    // CHECK: %{{.+}} = mmai %{{.+}}, %{{.+}}, %{{.+}} unsigned unsigned : tile<2x4x8xi8>, tile<2x8x16xi8>, tile<2x4x16xi32>
    %0 = mmai %arg0, %arg1, %arg2 unsigned unsigned : tile<2x4x8xi8>, tile<2x8x16xi8>, tile<2x4x16xi32>
  }

  // CHECK-LABEL: test_mmaf_scaled_fp8e5m2
  // CHECK-SAME: (%[[LHS:.+]]: tile<128x128xf8E5M2>, %[[RHS:.+]]: tile<128x128xf8E5M2>, %[[ACC:.+]]: tile<128x128xf32>, %[[LHS_SCALE:.+]]: tile<128x4xf8E8M0FNU>, %[[RHS_SCALE:.+]]: tile<4x128xf8E8M0FNU>)
  testing$func @test_mmaf_scaled_fp8e5m2(%lhs: !cuda_tile.tile<128x128xf8E5M2>, %rhs: !cuda_tile.tile<128x128xf8E5M2>, %acc: !cuda_tile.tile<128x128xf32>, %lhs_scale: !cuda_tile.tile<128x4xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<4x128xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled %[[LHS]], %[[RHS]], %[[ACC]], %[[LHS_SCALE]], %[[RHS_SCALE]] : tile<128x128xf8E5M2>, tile<128x128xf8E5M2>, tile<128x128xf32>, tile<128x4xf8E8M0FNU>, tile<4x128xf8E8M0FNU>
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }

  // CHECK-LABEL: test_mmaf_scaled_fp8e4m3
  // CHECK-SAME: (%[[LHS:.+]]: tile<128x128xf8E4M3FN>, %[[RHS:.+]]: tile<128x128xf8E4M3FN>, %[[ACC:.+]]: tile<128x128xf32>, %[[LHS_SCALE:.+]]: tile<128x4xf8E8M0FNU>, %[[RHS_SCALE:.+]]: tile<4x128xf8E8M0FNU>)
  testing$func @test_mmaf_scaled_fp8e4m3(%lhs: !cuda_tile.tile<128x128xf8E4M3FN>, %rhs: !cuda_tile.tile<128x128xf8E4M3FN>, %acc: !cuda_tile.tile<128x128xf32>, %lhs_scale: !cuda_tile.tile<128x4xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<4x128xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled %[[LHS]], %[[RHS]], %[[ACC]], %[[LHS_SCALE]], %[[RHS_SCALE]] : tile<128x128xf8E4M3FN>, tile<128x128xf8E4M3FN>, tile<128x128xf32>, tile<128x4xf8E8M0FNU>, tile<4x128xf8E8M0FNU>
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf8E4M3FN>, !cuda_tile.tile<128x128xf8E4M3FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }

  // CHECK-LABEL: test_mmaf_scaled_mxfp4
  // CHECK-SAME: (%[[LHS:.+]]: tile<128x128xf4E2M1FN>, %[[RHS:.+]]: tile<128x128xf4E2M1FN>, %[[ACC:.+]]: tile<128x128xf32>, %[[LHS_SCALE:.+]]: tile<128x4xf8E8M0FNU>, %[[RHS_SCALE:.+]]: tile<4x128xf8E8M0FNU>)
  testing$func @test_mmaf_scaled_mxfp4(%lhs: !cuda_tile.tile<128x128xf4E2M1FN>, %rhs: !cuda_tile.tile<128x128xf4E2M1FN>, %acc: !cuda_tile.tile<128x128xf32>, %lhs_scale: !cuda_tile.tile<128x4xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<4x128xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled %[[LHS]], %[[RHS]], %[[ACC]], %[[LHS_SCALE]], %[[RHS_SCALE]] : tile<128x128xf4E2M1FN>, tile<128x128xf4E2M1FN>, tile<128x128xf32>, tile<128x4xf8E8M0FNU>, tile<4x128xf8E8M0FNU>
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }

  // CHECK-LABEL: test_mmaf_scaled_mxfp4_vec16
  // CHECK-SAME: (%[[LHS:.+]]: tile<128x128xf4E2M1FN>, %[[RHS:.+]]: tile<128x128xf4E2M1FN>, %[[ACC:.+]]: tile<128x128xf32>, %[[LHS_SCALE:.+]]: tile<128x8xf8E8M0FNU>, %[[RHS_SCALE:.+]]: tile<8x128xf8E8M0FNU>)
  testing$func @test_mmaf_scaled_mxfp4_vec16(%lhs: !cuda_tile.tile<128x128xf4E2M1FN>, %rhs: !cuda_tile.tile<128x128xf4E2M1FN>, %acc: !cuda_tile.tile<128x128xf32>, %lhs_scale: !cuda_tile.tile<128x8xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<8x128xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled %[[LHS]], %[[RHS]], %[[ACC]], %[[LHS_SCALE]], %[[RHS_SCALE]] : tile<128x128xf4E2M1FN>, tile<128x128xf4E2M1FN>, tile<128x128xf32>, tile<128x8xf8E8M0FNU>, tile<8x128xf8E8M0FNU>
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E8M0FNU>, !cuda_tile.tile<8x128xf8E8M0FNU>
  }

  // CHECK-LABEL: test_mmaf_scaled_nvfp4
  // CHECK-SAME: (%[[LHS:.+]]: tile<128x128xf4E2M1FN>, %[[RHS:.+]]: tile<128x128xf4E2M1FN>, %[[ACC:.+]]: tile<128x128xf32>, %[[LHS_SCALE:.+]]: tile<128x8xf8E4M3FN>, %[[RHS_SCALE:.+]]: tile<8x128xf8E4M3FN>)
  testing$func @test_mmaf_scaled_nvfp4_f4e2m1_f8e4m3(%lhs: !cuda_tile.tile<128x128xf4E2M1FN>, %rhs: !cuda_tile.tile<128x128xf4E2M1FN>, %acc: !cuda_tile.tile<128x128xf32>, %lhs_scale: !cuda_tile.tile<128x8xf8E4M3FN>, %rhs_scale: !cuda_tile.tile<8x128xf8E4M3FN>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled %[[LHS]], %[[RHS]], %[[ACC]], %[[LHS_SCALE]], %[[RHS_SCALE]] : tile<128x128xf4E2M1FN>, tile<128x128xf4E2M1FN>, tile<128x128xf32>, tile<128x8xf8E4M3FN>, tile<8x128xf8E4M3FN>
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E4M3FN>, !cuda_tile.tile<8x128xf8E4M3FN>
  }

  // mmaf_scaled with 64x64 size (mxfp4)
  // CHECK-LABEL: test_mmaf_scaled_mxfp4_64x64
  testing$func @test_mmaf_scaled_mxfp4_64x64(%lhs: !cuda_tile.tile<64x64xf4E2M1FN>, %rhs: !cuda_tile.tile<64x64xf4E2M1FN>, %acc: !cuda_tile.tile<64x64xf32>, %lhs_scale: !cuda_tile.tile<64x2xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<2x64xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<64x64xf4E2M1FN>, !cuda_tile.tile<64x64xf4E2M1FN>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<64x2xf8E8M0FNU>, !cuda_tile.tile<2x64xf8E8M0FNU>
  }

  // mmaf_scaled with non-square shapes (mxfp4)
  // CHECK-LABEL: test_mmaf_scaled_mxfp4_nonsquare
  testing$func @test_mmaf_scaled_mxfp4_nonsquare(%lhs: !cuda_tile.tile<128x64xf4E2M1FN>, %rhs: !cuda_tile.tile<64x256xf4E2M1FN>, %acc: !cuda_tile.tile<128x256xf32>, %lhs_scale: !cuda_tile.tile<128x2xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<2x256xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x64xf4E2M1FN>, !cuda_tile.tile<64x256xf4E2M1FN>, !cuda_tile.tile<128x256xf32>, !cuda_tile.tile<128x2xf8E8M0FNU>, !cuda_tile.tile<2x256xf8E8M0FNU>
  }

  // mmaf_scaled with batched (3D) tensors (mxfp4)
  // CHECK-LABEL: test_mmaf_scaled_mxfp4_batched
  testing$func @test_mmaf_scaled_mxfp4_batched(%lhs: !cuda_tile.tile<2x64x64xf4E2M1FN>, %rhs: !cuda_tile.tile<2x64x64xf4E2M1FN>, %acc: !cuda_tile.tile<2x64x64xf32>, %lhs_scale: !cuda_tile.tile<2x64x2xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<2x2x64xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<2x64x64xf4E2M1FN>, !cuda_tile.tile<2x64x64xf4E2M1FN>, !cuda_tile.tile<2x64x64xf32>, !cuda_tile.tile<2x64x2xf8E8M0FNU>, !cuda_tile.tile<2x2x64xf8E8M0FNU>
  }

  // mmaf_scaled f8E5M2 with 64x64 size
  // CHECK-LABEL: test_mmaf_scaled_fp8e5m2_64x64
  testing$func @test_mmaf_scaled_fp8e5m2_64x64(%lhs: !cuda_tile.tile<64x64xf8E5M2>, %rhs: !cuda_tile.tile<64x64xf8E5M2>, %acc: !cuda_tile.tile<64x64xf32>, %lhs_scale: !cuda_tile.tile<64x2xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<2x64xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<64x64xf8E5M2>, !cuda_tile.tile<64x64xf8E5M2>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<64x2xf8E8M0FNU>, !cuda_tile.tile<2x64xf8E8M0FNU>
  }

  // mmaf_scaled f8E4M3FN with 64x64 size
  // CHECK-LABEL: test_mmaf_scaled_fp8e4m3_64x64
  testing$func @test_mmaf_scaled_fp8e4m3_64x64(%lhs: !cuda_tile.tile<64x64xf8E4M3FN>, %rhs: !cuda_tile.tile<64x64xf8E4M3FN>, %acc: !cuda_tile.tile<64x64xf32>, %lhs_scale: !cuda_tile.tile<64x2xf8E8M0FNU>, %rhs_scale: !cuda_tile.tile<2x64xf8E8M0FNU>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<64x64xf8E4M3FN>, !cuda_tile.tile<64x64xf8E4M3FN>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<64x2xf8E8M0FNU>, !cuda_tile.tile<2x64xf8E8M0FNU>
  }

  // mmaf_scaled nvfp4 with 64x64 size (f4E2M1FN + f8E4M3FN scale)
  // CHECK-LABEL: test_mmaf_scaled_nvfp4_64x64
  testing$func @test_mmaf_scaled_nvfp4_64x64(%lhs: !cuda_tile.tile<64x64xf4E2M1FN>, %rhs: !cuda_tile.tile<64x64xf4E2M1FN>, %acc: !cuda_tile.tile<64x64xf32>, %lhs_scale: !cuda_tile.tile<64x4xf8E4M3FN>, %rhs_scale: !cuda_tile.tile<4x64xf8E4M3FN>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<64x64xf4E2M1FN>, !cuda_tile.tile<64x64xf4E2M1FN>, !cuda_tile.tile<64x64xf32>, !cuda_tile.tile<64x4xf8E4M3FN>, !cuda_tile.tile<4x64xf8E4M3FN>
  }

  // mmaf_scaled nvfp4 batched (3D) (f4E2M1FN + f8E4M3FN scale)
  // CHECK-LABEL: test_mmaf_scaled_nvfp4_batched
  testing$func @test_mmaf_scaled_nvfp4_batched(%lhs: !cuda_tile.tile<2x64x64xf4E2M1FN>, %rhs: !cuda_tile.tile<2x64x64xf4E2M1FN>, %acc: !cuda_tile.tile<2x64x64xf32>, %lhs_scale: !cuda_tile.tile<2x64x4xf8E4M3FN>, %rhs_scale: !cuda_tile.tile<2x4x64xf8E4M3FN>
  ) {
    // CHECK: %{{.+}} = mmaf_scaled
    %0 = mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<2x64x64xf4E2M1FN>, !cuda_tile.tile<2x64x64xf4E2M1FN>, !cuda_tile.tile<2x64x64xf32>, !cuda_tile.tile<2x64x4xf8E4M3FN>, !cuda_tile.tile<2x4x64xf8E4M3FN>
  }

  // CHECK-LABEL: concat
  testing$func @concat(%arg0: !cuda_tile.tile<1x2xf32>) {
    // CHECK: cat %{{.+}}, %{{.+}} dim = 0 : tile<1x2xf32>, tile<1x2xf32>
    // CHECK-SAME:  -> tile<2x2xf32>
    %0 = cat %arg0, %arg0 dim = 0 
      : tile<1x2xf32>, tile<1x2xf32> -> tile<2x2xf32>
  }

  // CHECK-LABEL: reduce_operation
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // CHECK: %{{.+}} = reduce %{{.+}} dim=0 identities=[0.000000e+00 : f32]
    // CHECK-SAME:  : tile<8xf32> -> tile<f32>
    // CHECK-NEXT: (%{{.+}}: tile<f32>, %{{.+}}: tile<f32>) {
    // CHECK-NEXT: %{{.+}} = addf %{{.+}}, %{{.+}} : tile<f32>
    // CHECK-NEXT: yield %{{.+}} : tile<f32>
    // CHECK: }
    %0 = reduce %arg0 dim=0 identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<f32>
    (%arg0_in: tile<f32>, %arg0_identity: tile<f32>) {
      %add = addf %arg0_in, %arg0_identity : tile<f32>
      yield %add : tile<f32>
    }
  }

  // CHECK-LABEL: reduce_operation_2d_dim1
  testing$func @reduce_operation_2d_dim1(%arg0: !cuda_tile.tile<8x64xf32>) {
    // CHECK: %{{.+}} = reduce %{{.+}} dim=1 identities=[0.000000e+00 : f32]
    // CHECK-SAME:  : tile<8x64xf32> -> tile<8xf32>
    // CHECK-NEXT: (%{{.+}}: tile<f32>, %{{.+}}: tile<f32>) {
    // CHECK-NEXT: %{{.+}} = addf %{{.+}}, %{{.+}} : tile<f32>
    // CHECK-NEXT: yield %{{.+}} : tile<f32>
    // CHECK-NEXT: }
    %0 = reduce %arg0 dim=1 identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8xf32>
    (%arg0_in: tile<f32>, %arg0_identity: tile<f32>) {
      %add = addf %arg0_in, %arg0_identity : tile<f32>
      yield %add : tile<f32>
    }
  }

  // CHECK-LABEL: reduce_operation_2d_dim0
  testing$func @reduce_operation_2d_dim0(%arg0: !cuda_tile.tile<8x64xf32>) {
    // CHECK: %{{.+}} = reduce %{{.+}} dim=0 identities=[0.000000e+00 : f32]
    // CHECK-SAME:  : tile<8x64xf32> -> tile<64xf32>
    // CHECK-NEXT: (%{{.+}}: tile<f32>, %{{.+}}: tile<f32>) {
    // CHECK-NEXT: %{{.+}} = addf %{{.+}}, %{{.+}} : tile<f32>
    // CHECK-NEXT: yield %{{.+}} : tile<f32>
    // CHECK-NEXT: }
    %0 = reduce %arg0 dim=0 identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<64xf32>
    (%arg0_in: tile<f32>, %arg0_identity: tile<f32>) {
      %add = addf %arg0_in, %arg0_identity : tile<f32>
      yield %add : tile<f32>
    }
  }

  // CHECK-LABEL: scan_operation
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // CHECK: %{{.+}} = scan %{{.+}} dim=0 reverse=false identities=[0.000000e+00 : f32]
    // CHECK-SAME:  : tile<8xf32> -> tile<8xf32>
    // CHECK-NEXT: (%{{.+}}: tile<f32>, %{{.+}}: tile<f32>) {
    // CHECK-NEXT: %{{.+}} = addf %{{.+}}, %{{.+}} : tile<f32>
    // CHECK-NEXT: yield %{{.+}} : tile<f32>
    // CHECK: }
    %0 = scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<8xf32>
    (%arg0_in: tile<f32>, %arg0_identity: tile<f32>) {
      %add = addf %arg0_in, %arg0_identity : tile<f32>
      yield %add : tile<f32>
    }
  }

  // CHECK-LABEL: scan_operation_reverse
  testing$func @scan_operation_reverse(%arg0: !cuda_tile.tile<8xf32>) {
    // CHECK: %{{.+}} = scan %{{.+}} dim=0 reverse=true identities=[0.000000e+00 : f32]
    // CHECK-SAME:  : tile<8xf32> -> tile<8xf32>
    // CHECK-NEXT: (%{{.+}}: tile<f32>, %{{.+}}: tile<f32>) {
    // CHECK-NEXT: %{{.+}} = addf %{{.+}}, %{{.+}} : tile<f32>
    // CHECK-NEXT: yield %{{.+}} : tile<f32>
    // CHECK: }
    %0 = scan %arg0 dim=0 reverse=true identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<8xf32>
    (%arg0_in: !cuda_tile.tile<f32>, %arg0_identity: !cuda_tile.tile<f32>) {
      %add = addf %arg0_in, %arg0_identity : tile<f32>
      yield %add : tile<f32>
    }
  }

  // CHECK-LABEL: scan_operation_2d_dim1
  testing$func @scan_operation_2d_dim1(%arg0: !cuda_tile.tile<8x64xf32>) {
    // CHECK: %{{.+}} = scan %{{.+}} dim=1 reverse=false identities=[0.000000e+00 : f32]
    // CHECK-SAME:  : tile<8x64xf32> -> tile<8x64xf32>
    // CHECK-NEXT: (%{{.+}}: tile<f32>, %{{.+}}: tile<f32>) {
    // CHECK-NEXT: %{{.+}} = addf %{{.+}}, %{{.+}} : tile<f32>
    // CHECK-NEXT: yield %{{.+}} : tile<f32>
    // CHECK-NEXT: }
    %0 = scan %arg0 dim=1 reverse=false identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8x64xf32>
    (%arg0_in: !cuda_tile.tile<f32>, %arg0_identity: !cuda_tile.tile<f32>) {
      %add = addf %arg0_in, %arg0_identity : tile<f32>
      yield %add : tile<f32>
    }
  }

  // CHECK-LABEL: scan_operation_2d_dim0
  testing$func @scan_operation_2d_dim0(%arg0: !cuda_tile.tile<8x64xf32>) {
    // CHECK: %{{.+}} = scan %{{.+}} dim=0 reverse=false identities=[0.000000e+00 : f32]
    // CHECK-SAME:  : tile<8x64xf32> -> tile<8x64xf32>
    // CHECK-NEXT: (%{{.+}}: tile<f32>, %{{.+}}: tile<f32>) {
    // CHECK-NEXT: %{{.+}} = addf %{{.+}}, %{{.+}} : tile<f32>
    // CHECK-NEXT: yield %{{.+}} : tile<f32>
    // CHECK-NEXT: }
    %0 = scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8x64xf32>
    (%arg0_in: !cuda_tile.tile<f32>, %arg0_identity: !cuda_tile.tile<f32>) {
      %add = addf %arg0_in, %arg0_identity : tile<f32>
      yield %add : tile<f32>
    }
  }

  // Regression test: divi/remi are NoMemoryEffect (not Pure) and must be
  // accepted inside reduce/scan bodies.
  // CHECK-LABEL: reduce_divi
  testing$func @reduce_divi(%arg0: !cuda_tile.tile<8xi32>) {
    // CHECK: %{{.+}} = reduce %{{.+}} dim=0 identities=[1 : i32]
    // CHECK-SAME:  : tile<8xi32> -> tile<i32>
    // CHECK-NEXT: (%{{.+}}: tile<i32>, %{{.+}}: tile<i32>) {
    // CHECK-NEXT: %{{.+}} = divi %{{.+}}, %{{.+}} signed : tile<i32>
    // CHECK-NEXT: yield %{{.+}} : tile<i32>
    // CHECK-NEXT: }
    %0 = reduce %arg0 dim=0 identities=[1 : i32] : tile<8xi32> -> tile<i32>
    (%arg0_in: tile<i32>, %arg0_acc: tile<i32>) {
      %q = divi %arg0_in, %arg0_acc signed : tile<i32>
      yield %q : tile<i32>
    }
  }

  // CHECK-LABEL: reduce_remi
  testing$func @reduce_remi(%arg0: !cuda_tile.tile<8xi32>) {
    // CHECK: %{{.+}} = reduce %{{.+}} dim=0 identities=[1 : i32]
    // CHECK-SAME:  : tile<8xi32> -> tile<i32>
    // CHECK-NEXT: (%{{.+}}: tile<i32>, %{{.+}}: tile<i32>) {
    // CHECK-NEXT: %{{.+}} = remi %{{.+}}, %{{.+}} signed : tile<i32>
    // CHECK-NEXT: yield %{{.+}} : tile<i32>
    // CHECK-NEXT: }
    %0 = reduce %arg0 dim=0 identities=[1 : i32] : tile<8xi32> -> tile<i32>
    (%arg0_in: tile<i32>, %arg0_acc: tile<i32>) {
      %r = remi %arg0_in, %arg0_acc signed : tile<i32>
      yield %r : tile<i32>
    }
  }

  // CHECK-LABEL: scan_divi
  testing$func @scan_divi(%arg0: !cuda_tile.tile<8xi32>) {
    // CHECK: %{{.+}} = scan %{{.+}} dim=0 reverse=false identities=[1 : i32]
    // CHECK-SAME:  : tile<8xi32> -> tile<8xi32>
    // CHECK-NEXT: (%{{.+}}: tile<i32>, %{{.+}}: tile<i32>) {
    // CHECK-NEXT: %{{.+}} = divi %{{.+}}, %{{.+}} signed : tile<i32>
    // CHECK-NEXT: yield %{{.+}} : tile<i32>
    // CHECK-NEXT: }
    %0 = scan %arg0 dim=0 reverse=false identities=[1 : i32] : tile<8xi32> -> tile<8xi32>
    (%arg0_in: tile<i32>, %arg0_acc: tile<i32>) {
      %q = divi %arg0_in, %arg0_acc signed : tile<i32>
      yield %q : tile<i32>
    }
  }

  // CHECK-LABEL: entry @tile_id()
  entry @tile_id() {
    // CHECK: get_tile_block_id : tile<i32>
    %0, %1, %2 = get_tile_block_id : tile<i32>
    // CHECK: get_num_tile_blocks : tile<i32>
    %3, %4, %5 = get_num_tile_blocks : tile<i32>
  }

  entry @cmp_operations() {
      // CHECK: %[[s0:.*]] = constant <f16: 4.200000e+01> : tile<f16>
      // CHECK: cmpf equal ordered %[[s0]], %[[s0]] : tile<f16>
      // CHECK: cmpf equal ordered %[[s0]], %[[s0]] : tile<f16>
      %s0 = constant <f16: 42.0> : tile<f16>
      %cmpf_scalar_asm = cmpf equal ordered %s0, %s0 : tile<f16> -> tile<i1> 
      %cmpf_scalar_generic = "cuda_tile.cmpf"(%s0, %s0) {comparison_predicate = #cuda_tile.comparison_predicate<equal>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<f16>, !cuda_tile.tile<f16>) -> !cuda_tile.tile<i1>

      // CHECK: %[[v0:.*]] = constant <f32: {{\[.*\]}}> : tile<4xf32>
      // CHECK: cmpf not_equal ordered %[[v0]], %[[v0]] : tile<4xf32>
      // CHECK: cmpf not_equal ordered %[[v0]], %[[v0]] : tile<4xf32>
      %v0 = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
      %cmpf_vector_asm = cmpf not_equal ordered %v0, %v0 : tile<4xf32> -> tile<4xi1>
      %cmpf_vector_generic = "cuda_tile.cmpf"(%v0, %v0) {comparison_predicate = #cuda_tile.comparison_predicate<not_equal>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<4xf32>, !cuda_tile.tile<4xf32>) -> !cuda_tile.tile<4xi1>

      // CHECK: %[[t0:.*]] = constant <f64: {{\[.*\]}}> : tile<2x2xf64>
      // CHECK: cmpf less_than unordered %[[t0]], %[[t0]] : tile<2x2xf64>
      // CHECK: cmpf less_than unordered %[[t0]], %[[t0]] : tile<2x2xf64>
      %t0 = constant <f64: [[1.0, 2.0], [3.0, 4.0]]> : tile<2x2xf64>
      %cmpf_tensor_asm = cmpf less_than unordered %t0, %t0 : tile<2x2xf64> -> tile<2x2xi1>
      %cmpf_tensor_generic = "cuda_tile.cmpf"(%t0, %t0) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, comparison_ordering = #cuda_tile.comparison_ordering<unordered>} : (!cuda_tile.tile<2x2xf64>, !cuda_tile.tile<2x2xf64>) -> !cuda_tile.tile<2x2xi1>

      // CHECK: %[[s1:.*]] = constant <i16: 42> : tile<i16>
      // CHECK: cmpi equal %[[s1]], %[[s1]], signed : tile<i16>
      // CHECK: cmpi equal %[[s1]], %[[s1]], signed : tile<i16>
      %s1 = constant <i16: 42> : tile<i16>
      %cmpi_scalar_asm = cmpi equal %s1, %s1, signed : tile<i16> -> tile<i1>
      %cmpi_scalar_generic = "cuda_tile.cmpi"(%s1, %s1) {comparison_predicate = #cuda_tile.comparison_predicate<equal>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i16>, !cuda_tile.tile<i16>) -> !cuda_tile.tile<i1>

      // CHECK: %[[v1:.*]] = constant <i32: {{\[.*\]}}> : tile<4xi32>
      // CHECK: cmpi not_equal %[[v1]], %[[v1]], signed : tile<4xi32>
      // CHECK: cmpi not_equal %[[v1]], %[[v1]], signed : tile<4xi32>
      %v1 = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
      %cmpi_vector_asm = cmpi not_equal %v1, %v1, signed : tile<4xi32> -> tile<4xi1>
      %cmpi_vector_generic = "cuda_tile.cmpi"(%v1, %v1) {comparison_predicate = #cuda_tile.comparison_predicate<not_equal>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<4xi32>, !cuda_tile.tile<4xi32>) -> !cuda_tile.tile<4xi1>

      // CHECK: %[[t1:.*]] = constant <i64: {{\[.*\]}}> : tile<2x2xi64>
      // CHECK: cmpi less_than %[[t1]], %[[t1]], unsigned : tile<2x2xi64>
      // CHECK: cmpi less_than %[[t1]], %[[t1]], unsigned : tile<2x2xi64>
      %t1 = constant <i64: [[1, 2], [3, 4]]> : tile<2x2xi64>
      %cmpi_tensor_asm = cmpi less_than %t1, %t1, unsigned : tile<2x2xi64> -> tile<2x2xi1>
      %cmpi_tensor_generic = "cuda_tile.cmpi"(%t1, %t1) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<unsigned>} : (!cuda_tile.tile<2x2xi64>, !cuda_tile.tile<2x2xi64>) -> !cuda_tile.tile<2x2xi1>
  }

  testing$func @math_func_exp(
                                %arg0: !cuda_tile.tile<2xf16>,
                                %arg1: !cuda_tile.tile<2xf32>,
                                %arg2: !cuda_tile.tile<2xf64>,
                                %arg3: !cuda_tile.tile<2xbf16>) {
    // CHECK: exp %{{.+}} : tile<2xf16>
    %0 = exp %arg0 : tile<2xf16>
    // CHECK: exp %{{.+}} : tile<2xf32>
    %1 = exp %arg1 : tile<2xf32>
    // CHECK: exp %{{.+}} rounding<approx> : tile<2xf32>
    %approx1 = exp %arg1 rounding<approx> : tile<2xf32>
    // CHECK: exp %{{.+}} : tile<2xf64>
    %2 = exp %arg2 : tile<2xf64>
    // CHECK: exp %{{.+}} : tile<2xbf16>
    %3 = exp %arg3 : tile<2xbf16>
  }


  testing$func @math_func_exp2(
                                %arg0: !cuda_tile.tile<2xf16>,
                                %arg1: !cuda_tile.tile<2xf32>,
                                %arg2: !cuda_tile.tile<2xf64>,
                                %arg3: !cuda_tile.tile<2xbf16>) {
    // CHECK: exp2 %{{.+}} : tile<2xf16>
    %0 = exp2 %arg0 : tile<2xf16>
    // CHECK: exp2 %{{.+}} : tile<2xf32>
    %1 = exp2 %arg1 : tile<2xf32>
    // CHECK: exp2 %{{.+}} : tile<2xf64>
    %2 = exp2 %arg2 : tile<2xf64>
    // CHECK: exp2 %{{.+}} : tile<2xbf16>
    %3 = exp2 %arg3 : tile<2xbf16>
  }

  testing$func @kernel2(%arg0: !cuda_tile.tile<2xi16>,
                        %arg1: !cuda_tile.tile<1x8x8xptr<f32>>,
                        %arg2: !cuda_tile.tile<4xi1>,
                        %arg3: !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>,
                        %arg4: !cuda_tile.tile<i16>,
                        %arg5: !cuda_tile.tile<1x8x8xi64>) {
    // Note: A divisibility of 4611686018427387904 for an i16 integer implies a
    // value of 0.
    // CHECK: assume div_by<4611686018427387904>, %{{.*}} : tile<2xi16>
    %0 = cuda_tile.assume #cuda_tile.div_by<4611686018427387904>, %arg0 : tile<2xi16>
    // CHECK: assume div_by<32>, %{{.*}} : tile<1x8x8xptr<f32>>
    %1 = cuda_tile.assume #cuda_tile.div_by<32>, %arg1 : tile<1x8x8xptr<f32>>
    // CHECK: assume div_by<32>, %{{.*}} : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
    %3 = cuda_tile.assume #cuda_tile.div_by<32>, %arg3 : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
    // CHECK: assume div_by<1, every 4 along 1>, %{{.*}} : tile<1x8x8xptr<f32>>
    %4 = cuda_tile.assume #cuda_tile.div_by<1, every 4 along 1>, %arg1 : tile<1x8x8xptr<f32>>
    // CHECK: assume div_by<1>, %{{.*}} : tile<i16>
    %5 = cuda_tile.assume #cuda_tile.div_by<1>, %arg4 : tile<i16>
    // CHECK: assume div_by<1, every 4 along 1>, %{{.*}} : tile<1x8x8xi64>
    %6 = cuda_tile.assume #cuda_tile.div_by<1, every 4 along 1>, %arg5 : tile<1x8x8xi64>

    // CHECK: assume same_elements<[1, 4, 2]>, %{{.*}} : tile<1x8x8xptr<f32>>
    %7 = cuda_tile.assume #cuda_tile.same_elements<[1, 4, 2]>, %arg1 : tile<1x8x8xptr<f32>>
    // CHECK: assume same_elements<[]>, %{{.*}} : tile<i16>
    %8 = cuda_tile.assume #cuda_tile.same_elements<[]>, %arg4 : tile<i16>

    // CHECK: assume bounded<0, 42>, %{{.*}} : tile<i16>
    %9 = cuda_tile.assume #cuda_tile.bounded<0, 42>, %arg4 : tile<i16>
    // CHECK: assume bounded<?, 42>, %{{.*}} : tile<i16>
    %10 = cuda_tile.assume #cuda_tile.bounded<?, 42>, %arg4 : tile<i16>
    // CHECK: assume bounded<-4, ?>, %{{.*}} : tile<i16>
    %11 = cuda_tile.assume #cuda_tile.bounded<-4, ?>, %arg4 : tile<i16>
    // CHECK: assume bounded<?, ?>, %{{.*}} : tile<i16>
    %12 = cuda_tile.assume #cuda_tile.bounded<?, ?>, %arg4 : tile<i16>
    // CHECK: assume bounded<-9223372036854775808, 9223372036854775807>, %{{.*}} : tile<1x8x8xi64>
    %13 = cuda_tile.assume #cuda_tile.bounded<-9223372036854775808, 9223372036854775807>, %arg5 : tile<1x8x8xi64>
  }

  testing$func @kernel3(%arg0: !cuda_tile.tile<2xi1>) {
    // CHECK: assert %{{.*}}, "foo" : tile<2xi1>
    cuda_tile.assert %arg0, "foo" : tile<2xi1>
  }

  testing$func @kernel4(%arg0: !cuda_tile.tile<2xf32>,
              %arg1: !cuda_tile.tile<2xf64>,
              %arg2: !cuda_tile.tile<2xf16>,
              %arg3: !cuda_tile.tile<2xbf16>) {
    // f32 operations
    // CHECK: cos %{{.*}} : tile<2xf32>
    %0 = cos %arg0 : tile<2xf32>
    // CHECK: cosh %{{.*}} : tile<2xf32>
    %1 = cosh %arg0 : tile<2xf32>
    // CHECK: sin %{{.*}} : tile<2xf32>
    %2 = sin %arg0 : tile<2xf32>
    // CHECK: sinh %{{.*}} : tile<2xf32>
    %3 = sinh %arg0 : tile<2xf32>
    // CHECK: tan %{{.*}} : tile<2xf32>
    %4 = tan %arg0 : tile<2xf32>
    // CHECK: tanh %{{.*}} : tile<2xf32>
    %5 = tanh %arg0 : tile<2xf32>
    
    // f64 operations
    // CHECK: cos %{{.*}} : tile<2xf64>
    %6 = cos %arg1 : tile<2xf64>
    // CHECK: cosh %{{.*}} : tile<2xf64>
    %7 = cosh %arg1 : tile<2xf64>
    // CHECK: sin %{{.*}} : tile<2xf64>
    %8 = sin %arg1 : tile<2xf64>
    // CHECK: sinh %{{.*}} : tile<2xf64>
    %9 = sinh %arg1 : tile<2xf64>
    // CHECK: tan %{{.*}} : tile<2xf64>
    %10 = tan %arg1 : tile<2xf64>
    // CHECK: tanh %{{.*}} : tile<2xf64>
    %11 = tanh %arg1 : tile<2xf64>

    // f16 operations
    // CHECK: tanh %{{.*}} : tile<2xf16>
    %12 = tanh %arg2 : tile<2xf16>

    // bf16 operations
    // CHECK: tanh %{{.*}} : tile<2xbf16>
    %13 = tanh %arg3 : tile<2xbf16>
  }

  // CHECK: entry @entry_with_kernel_scope_global
  entry @entry_with_kernel_scope_global() {}

  testing$func @kernel6(%arg0: !cuda_tile.tile<2xptr<i32>>,
                        %arg1: !cuda_tile.tile<2xi32>,
                        %arg2: !cuda_tile.tile<2xptr<f32>>,
                        %arg3: !cuda_tile.tile<2xf32>,
                        %arg4: !cuda_tile.tile<2xi1>) {
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, and
    %0, %t = atomic_rmw_tko relaxed device %arg0, and, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, or
    %1, %t1 = atomic_rmw_tko relaxed device %arg0, or, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, xor
    %2, %t2 = atomic_rmw_tko relaxed device %arg0, xor, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, add
    %3, %t3 = atomic_rmw_tko relaxed device %arg0, add, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, max
    %5, %t5 = atomic_rmw_tko relaxed device %arg0, max, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, min
    %6, %t6 = atomic_rmw_tko relaxed device %arg0, min, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, umax
    %7, %t7 = atomic_rmw_tko relaxed device %arg0, umax, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, umin
    %8, %t8 = atomic_rmw_tko relaxed device %arg0, umin, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, xchg
    %9, %t9 = atomic_rmw_tko relaxed device %arg0, xchg, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
    // CHECK: atomic_rmw_tko relaxed device {{.*}}, xchg
    %10, %t10 = atomic_rmw_tko relaxed device %arg0, xchg, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token

    // CHECK: atomic_rmw_tko relaxed device {{.*}}, xchg
    // CHECK-SAME: %{{.+}}, %{{.+}} : tile<2xptr<i32>>, tile<2xi32>, tile<2xi1> -> tile<2xi32>, token
    %11, %t11 = atomic_rmw_tko relaxed device %arg0, xchg, %arg1, %arg4
        : tile<2xptr<i32>>, tile<2xi32>, tile<2xi1> -> tile<2xi32>, token
  }

  testing$func @kernel7(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                        %arg1: !cuda_tile.tile<2xi32>,
                        %arg2: !cuda_tile.tile<2xi32>) {
    // CHECK: atomic_cas_tko relaxed device %{{.*}}, %{{.*}}, %{{.*}} :
    // CHECK-SAME: tile<2xptr<i32>>, tile<2xi32>
    %0, %t = atomic_cas_tko relaxed device %arg0, %arg1, %arg2
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
  }

  testing$func @kernel17(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<f32>>,
                        %arg1: !cuda_tile.tile<2xf32>,
                        %arg2: !cuda_tile.tile<2xf32>) {
    // CHECK: atomic_cas_tko relaxed device %{{.*}}, %{{.*}}, %{{.*}} :
    // CHECK-SAME: tile<2xptr<f32>>, tile<2xf32>
    %0, %t = atomic_cas_tko relaxed device %arg0, %arg1, %arg2
        : tile<2xptr<f32>>, tile<2xf32> -> tile<2xf32>, token
  }

  // CHECK: entry
  cuda_tile.entry @entry_with_two_args(%arg0: !cuda_tile.tile<f32>,
                                            %arg1: !cuda_tile.tile<ptr<f32>>) {}

  testing$func @kernel9( %arg0: !cuda_tile.tile<2xf32>,
                          %arg1: !cuda_tile.tile<2xf64>,
                          %arg2: !cuda_tile.tile<2xf16>,
                          %arg3: !cuda_tile.tile<2xbf16>) {
    // CHECK: %{{.+}} = negf %{{.+}} : tile<2xf32>
    %0 = negf %arg0 : tile<2xf32>
    // CHECK-NEXT: %{{.+}} = negf %{{.+}}  : tile<2xf64>
    %1 = negf %arg1 : tile<2xf64>
    // CHECK-NEXT: %{{.+}} = negf %{{.+}}  : tile<2xf16>
    %2 = negf %arg2 : tile<2xf16>
    // CHECK-NEXT: negf %{{.+}}  : tile<2xbf16>
    %3 = negf %arg3 : tile<2xbf16>
  }

  testing$func @kernel10( %arg0: !cuda_tile.tile<2xf32>,
                %arg1: !cuda_tile.tile<2xf64>,
                %arg2: !cuda_tile.tile<2xi8>,
                %arg3: !cuda_tile.tile<2xi16>,
                %arg4: !cuda_tile.tile<2xi32>,
                %arg5: !cuda_tile.tile<2xi1>) {
    // CHECK: %{{.+}} = fpowf %{{.+}}, %{{.+}} : tile<2xf32>
    %0 = fpowf %arg0, %arg0 : tile<2xf32>
    // CHECK-NEXT: %{{.+}} = fpowf %{{.+}}, %{{.+}} : tile<2xf64>
    %1 = fpowf %arg1, %arg1 : tile<2xf64>
    // CHECK-NEXT: %{{.+}} = fpowi %{{.+}}, %{{.+}} : tile<2xf32>, tile<2xi1>
    %5 = fpowi %arg0, %arg5 : tile<2xf32>, tile<2xi1>
    // CHECK-NEXT: %{{.+}} = fpowi %{{.+}}, %{{.+}} : tile<2xf32>, tile<2xi8>
    %2 = fpowi %arg0, %arg2 : tile<2xf32>, tile<2xi8>
    // CHECK-NEXT: %{{.+}} = fpowi %{{.+}}, %{{.+}} : tile<2xf32>, tile<2xi16>
    %3 = fpowi %arg0, %arg3 : tile<2xf32>, tile<2xi16>
    // CHECK-NEXT: %{{.+}} = fpowi %{{.+}}, %{{.+}} : tile<2xf32>, tile<2xi32>
    %4 = fpowi %arg0, %arg4 : tile<2xf32>, tile<2xi32>
  }


  testing$func @kernel11( %arg0: !cuda_tile.tile<2xf32>,
                %arg1: !cuda_tile.tile<2xf64>) {
    // CHECK: %{{.+}} = floor %{{.+}} : tile<2xf32>
    %0 = floor %arg0 : tile<2xf32>
    // CHECK-NEXT: %{{.+}} = floor %{{.+}}  : tile<2xf64>
    %1 = floor %arg1 : tile<2xf64>
  }

  testing$func @kernel14(%arg0: !cuda_tile.tile<512xf32>,
              %arg1: !cuda_tile.tile<512xf32>,
              %arg2: !cuda_tile.tile<512xf32> ) {
    // CHECK: fma %{{.+}}, %{{.+}}, %{{.+}} rounding<zero> : tile<512xf32>
    %1 = fma %arg0, %arg1, %arg2 rounding<zero> : tile<512xf32>
  }


  testing$func @kernel15(%arg0: !cuda_tile.tile<512xf32>,
              %arg1: !cuda_tile.tile<512xf32>,
              %arg2: !cuda_tile.tile<512xf32> ) {
    // CHECK: fma %{{.+}}, %{{.+}}, %{{.+}} rounding<zero> flush_to_zero : tile<512xf32>
    %1 = fma %arg0, %arg1, %arg2 rounding<zero> flush_to_zero : tile<512xf32>
  }


  testing$func @kernel16(%arg0: !cuda_tile.tile<512xf32>,
              %arg1: !cuda_tile.tile<512xf32>,
              %arg2: !cuda_tile.tile<512xf32> ) {
    // CHECK: fma %{{.+}}, %{{.+}}, %{{.+}} rounding<zero> flush_to_zero : tile<512xf32>
    %1 = fma %arg0, %arg1, %arg2 rounding<zero> flush_to_zero  : tile<512xf32>
  }

  testing$func @test_atomic_rmw_valid_sem_relaxed(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                          %arg1: !cuda_tile.tile<2xi32>) {
    // CHECK: atomic_rmw_tko relaxed device
    atomic_rmw_tko relaxed device %arg0, add, %arg1
          : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
  }

  testing$func @test_atomic_rmw_valid_sem_acquire(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                          %arg1: !cuda_tile.tile<2xi32>) {
    // CHECK: atomic_rmw_tko acquire device
    atomic_rmw_tko acquire device %arg0, add, %arg1
          : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
  }

  testing$func @test_atomic_rmw_valid_sem_release(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                          %arg1: !cuda_tile.tile<2xi32>) {
    // CHECK: atomic_rmw_tko release device
    atomic_rmw_tko release device %arg0, add, %arg1
          : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
  }

  testing$func @test_atomic_rmw_valid_sem_acq_rel(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                          %arg1: !cuda_tile.tile<2xi32>) {
    // CHECK: atomic_rmw_tko acq_rel device
    atomic_rmw_tko acq_rel device %arg0, add, %arg1
          : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
  }

  testing$func @test_atomic_rmw_f16(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<f16>>,
                        %arg1: !cuda_tile.tile<2xf16>) {
      // CHECK: atomic_rmw_tko relaxed device %{{.+}}, addf, %{{.+}}
      atomic_rmw_tko relaxed device %arg0, addf, %arg1
          : tile<2xptr<f16>>, tile<2xf16> -> tile<2xf16>, token
  }

  testing$func @test_atomic_rmw_bf16(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<bf16>>,
                        %arg1: !cuda_tile.tile<2xbf16>) {
      // CHECK: atomic_rmw_tko relaxed device %{{.+}}, addf, %{{.+}}
      atomic_rmw_tko relaxed device %arg0, addf, %arg1
          : tile<2xptr<bf16>>, tile<2xbf16> -> tile<2xbf16>, token
  }

  testing$func @kernel_atan2(%x32: !cuda_tile.tile<2xf32>,
                             %y32: !cuda_tile.tile<2xf32>,
                             %x64: !cuda_tile.tile<2xf64>,
                             %y64: !cuda_tile.tile<2xf64>,
                             %x16: !cuda_tile.tile<2xf16>,
                             %y16: !cuda_tile.tile<2xf16>,
                             %xbf16: !cuda_tile.tile<2xbf16>,
                             %ybf16: !cuda_tile.tile<2xbf16>) {
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<2xf32>
    %r0 = atan2 %x32, %y32 : tile<2xf32>
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<2xf64>
    %r1 = atan2 %x64, %y64 : tile<2xf64>
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<2xf16>
    %r2 = atan2 %x16, %y16 : tile<2xf16>
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<2xbf16>
    %r3 = atan2 %xbf16, %ybf16 : tile<2xbf16>
  }

  // Test atan2 with scalar tiles (rank 0)
  testing$func @kernel_atan2_scalar(%x32: !cuda_tile.tile<f32>,
                                    %y32: !cuda_tile.tile<f32>,
                                    %x64: !cuda_tile.tile<f64>,
                                    %y64: !cuda_tile.tile<f64>,
                                    %x16: !cuda_tile.tile<f16>,
                                    %y16: !cuda_tile.tile<f16>,
                                    %xbf16: !cuda_tile.tile<bf16>,
                                    %ybf16: !cuda_tile.tile<bf16>) {
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<f32>
    %r0 = atan2 %x32, %y32 : tile<f32>
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<f64>
    %r1 = atan2 %x64, %y64 : tile<f64>
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<f16>
    %r2 = atan2 %x16, %y16 : tile<f16>
    // CHECK: %{{.+}} = atan2 %{{.+}}, %{{.+}} : tile<bf16>
    %r3 = atan2 %xbf16, %ybf16 : tile<bf16>
  }

  testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
    // CHECK: unpack %{{.+}} : tile<64xi8> -> tile<512xi1>
    %i1 = unpack %arg0 : tile<64xi8> -> tile<512xi1>
    // CHECK: unpack %{{.+}} : tile<64xi8> -> tile<128xi4>
    %i4 = unpack %arg0 : tile<64xi8> -> tile<128xi4>
    // CHECK: unpack %{{.+}} : tile<64xi8> -> tile<128xf4E2M1FN>
    %0 = unpack %arg0 : tile<64xi8> -> tile<128xf4E2M1FN>
  }

  testing$func @test_pack_op(%arg_i1: !cuda_tile.tile<16xi1>, %arg_i4: !cuda_tile.tile<64xi4>, %arg_i16: !cuda_tile.tile<64xi16>, %arg_f32: !cuda_tile.tile<64xf32>) {
    // CHECK: pack %{{.+}} : tile<16xi1> -> tile<2xi8>
    %i1 = pack %arg_i1 : tile<16xi1> -> tile<2xi8>
    // CHECK: pack %{{.+}} : tile<64xi4> -> tile<32xi8>
    %i4 = pack %arg_i4 : tile<64xi4> -> tile<32xi8>
    // CHECK: pack %{{.+}} : tile<64xi16> -> tile<128xi8>
    %0 = pack %arg_i16 : tile<64xi16> -> tile<128xi8>
    // CHECK: pack %{{.+}} : tile<64xf32> -> tile<256xi8>
    %1 = pack %arg_f32 : tile<64xf32> -> tile<256xi8>
  }

  // CHECK-LABEL: test_alloca_private
  entry @test_alloca_private() {
    // CHECK: alloca num_elem = 64, alignment = 16 : tile<ptr<f32>>
    %0 = alloca num_elem = 64, alignment = 16 : tile<ptr<f32>>
  }

  // CHECK-LABEL: test_alloca_global
  entry @test_alloca_global() {
    // CHECK: alloca num_elem = 64, alignment = 16 global : tile<ptr<f32>>
    %0 = alloca num_elem = 64, alignment = 16 global : tile<ptr<f32>>
  }

  // ===== atomic_red_view_tko tests =====
  // Note: atomic_red_view_tko requires explicit memory ordering semantics (only RELAXED supported)
  // to enable TMA REDG optimization on Hopper+ GPUs.
  
  testing$func @test_atomic_red_view_tko_basic(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_all_modes(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], and, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t0 = atomic_red_view_tko relaxed device %view[%c0, %c0], and, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], or, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t1 = atomic_red_view_tko relaxed device %view[%c0, %c0], or, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], xor, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t2 = atomic_red_view_tko relaxed device %view[%c0, %c0], xor, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t3 = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], max, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t4 = atomic_red_view_tko relaxed device %view[%c0, %c0], max, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], min, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t5 = atomic_red_view_tko relaxed device %view[%c0, %c0], min, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], umax, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t6 = atomic_red_view_tko relaxed device %view[%c0, %c0], umax, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], umin, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t7 = atomic_red_view_tko relaxed device %view[%c0, %c0], umin, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_float(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xf32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], addf, %{{.+}} : tile<2x2xf32>, partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], addf, %value
        : tile<2x2xf32>, partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_with_token(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>,
      %in_token: !cuda_tile.token) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} token = %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value token = %in_token
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_nonzero_index(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<4x4xi32, strides=[4, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c1 = constant <i32: 1> : !cuda_tile.tile<i32>
    %c2 = constant <i32: 2> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<4x4xi32, strides=[4,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed device %view[%c1, %c2], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<4x4xi32, strides=[4, 1]>>, tile<i32> -> token
  }

  // ===== atomic_red_view_tko with tl_blk scope =====

  testing$func @test_atomic_red_view_tko_tl_blk_basic(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_tl_blk_all_modes(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], and, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t0 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], and, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], or, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t1 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], or, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], xor, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t2 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], xor, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t3 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], max, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t4 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], max, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], min, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t5 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], min, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], umax, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t6 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], umax, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], umin, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t7 = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], umin, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_tl_blk_float(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xf32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], addf, %{{.+}} : tile<2x2xf32>, partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], addf, %value
        : tile<2x2xf32>, partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_tl_blk_with_token(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>,
      %in_token: !cuda_tile.token) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed tl_blk %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} token = %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed tl_blk %view[%c0, %c0], add, %value token = %in_token
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_padded_partition_view(
      %view: !cuda_tile.partition_view<tile=(2x2), padding_value = zero, !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} : tile<2x2xi32>, partition_view<tile=(2x2), padding_value = zero, tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), padding_value = zero, tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  testing$func @test_atomic_red_view_tko_padded_strided_view(
      %view: !cuda_tile.strided_view<tile=(2x2), traversal_strides=[2, 2], padding_value = zero, !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = atomic_red_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}], add, %{{.+}} : tile<2x2xi32>, strided_view<tile=(2x2), traversal_strides=[2,2], padding_value = zero, tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
    %t = atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : tile<2x2xi32>, strided_view<tile=(2x2), traversal_strides=[2, 2], padding_value = zero, tensor_view<2x2xi32, strides=[2, 1]>>, tile<i32> -> token
  }

  // --- Loop terminator operand type check ---

  // Tests that yield, break, and continue accept operands of types that are
  // not defined by the cuda_tile dialect (e.g. builtin `index`, `i32`, `f32`).
  // This supports frontends that embed non-CudaTile ops and values inside
  // CudaTile control flow.


  // CHECK-LABEL: testing$func @continue_non_cuda_tile_type_in_loop_i32
  // CHECK-SAME: (%[[VAL:.*]]: i32)
  testing$func @continue_non_cuda_tile_type_in_loop_i32(%val: i32) {
    // CHECK: loop iter_values(%{{.*}} = %[[VAL]]) : i32
    loop iter_values(%arg = %val) : i32 {
      // CHECK: continue %{{.*}} : i32
      continue %arg : i32
    }
  }

  // CHECK-LABEL: testing$func @continue_non_cuda_tile_type_in_loop_f32
  // CHECK-SAME: (%[[VAL:.*]]: f32)
  testing$func @continue_non_cuda_tile_type_in_loop_f32(%val: f32) {
    // CHECK: loop iter_values(%{{.*}} = %[[VAL]]) : f32
    loop iter_values(%arg = %val) : f32 {
      // CHECK: continue %{{.*}} : f32
      continue %arg : f32
    }
  }

  // ============================================================================
  // ftof: f8E5M3FNU (UE5M3) parser/printer round-trip coverage.
  //
  // Each function exercises one allowed (src, dst, rounding) combination per
  // the FToFOp::verify rule table in tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp.
  // Disallowed combinations are covered in conversion_invalid.mlir.
  // ============================================================================

  // --- A -> ue5m3: rule 2 forces nearest_even for every source ---

  // CHECK-LABEL: ftof_to_ue5m3_from_f16
  testing$func @ftof_to_ue5m3_from_f16(%a: !cuda_tile.tile<2xf16>) {
    // CHECK: ftof %{{.+}} : tile<2xf16> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xf16> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_bf16
  testing$func @ftof_to_ue5m3_from_bf16(%a: !cuda_tile.tile<2xbf16>) {
    // CHECK: ftof %{{.+}} : tile<2xbf16> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xbf16> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_f32
  testing$func @ftof_to_ue5m3_from_f32(%a: !cuda_tile.tile<2xf32>) {
    // CHECK: ftof %{{.+}} : tile<2xf32> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xf32> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_f64
  testing$func @ftof_to_ue5m3_from_f64(%a: !cuda_tile.tile<2xf64>) {
    // CHECK: ftof %{{.+}} : tile<2xf64> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xf64> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_tf32
  testing$func @ftof_to_ue5m3_from_tf32(%a: !cuda_tile.tile<2xtf32>) {
    // CHECK: ftof %{{.+}} : tile<2xtf32> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xtf32> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_f8E4M3FN
  testing$func @ftof_to_ue5m3_from_f8E4M3FN(%a: !cuda_tile.tile<2xf8E4M3FN>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E4M3FN> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xf8E4M3FN> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_f8E5M2
  testing$func @ftof_to_ue5m3_from_f8E5M2(%a: !cuda_tile.tile<2xf8E5M2>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M2> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M2> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_f8E8M0FNU
  testing$func @ftof_to_ue5m3_from_f8E8M0FNU(%a: !cuda_tile.tile<2xf8E8M0FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E8M0FNU> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xf8E8M0FNU> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // CHECK-LABEL: ftof_to_ue5m3_from_f4E2M1FN
  testing$func @ftof_to_ue5m3_from_f4E2M1FN(%a: !cuda_tile.tile<2xf4E2M1FN>) {
    // CHECK: ftof %{{.+}} : tile<2xf4E2M1FN> -> tile<2xf8E5M3FNU>
    %0 = ftof %a : !cuda_tile.tile<2xf4E2M1FN> -> !cuda_tile.tile<2xf8E5M3FNU>
  }

  // --- ue5m3 -> f8E8M0FNU: rule 1 allows zero and positive_inf ---

  // CHECK-LABEL: ftof_from_ue5m3_to_f8E8M0FNU_zero
  testing$func @ftof_from_ue5m3_to_f8E8M0FNU_zero(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<zero> : tile<2xf8E5M3FNU> -> tile<2xf8E8M0FNU>
    %0 = ftof %a rounding<zero> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf8E8M0FNU>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f8E8M0FNU_positive_inf
  testing$func @ftof_from_ue5m3_to_f8E8M0FNU_positive_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<positive_inf> : tile<2xf8E5M3FNU> -> tile<2xf8E8M0FNU>
    %0 = ftof %a rounding<positive_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf8E8M0FNU>
  }

  // --- ue5m3 -> {f8E4M3FN, f8E5M2, f4E2M1FN}: rule 2 forces nearest_even ---

  // CHECK-LABEL: ftof_from_ue5m3_to_f8E4M3FN
  testing$func @ftof_from_ue5m3_to_f8E4M3FN(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xf8E4M3FN>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf8E4M3FN>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f8E5M2
  testing$func @ftof_from_ue5m3_to_f8E5M2(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xf8E5M2>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf8E5M2>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f4E2M1FN
  testing$func @ftof_from_ue5m3_to_f4E2M1FN(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xf4E2M1FN>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf4E2M1FN>
  }

  // --- ue5m3 -> f16: rule 6 allows nearest_even and zero ---

  // CHECK-LABEL: ftof_from_ue5m3_to_f16_nearest_even
  testing$func @ftof_from_ue5m3_to_f16_nearest_even(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xf16>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf16>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f16_zero
  testing$func @ftof_from_ue5m3_to_f16_zero(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<zero> : tile<2xf8E5M3FNU> -> tile<2xf16>
    %0 = ftof %a rounding<zero> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf16>
  }

  // --- ue5m3 -> {bf16, f32, f64, tf32}: rule 3 (strict widening) allows all
  //     five IEEE rounding modes. ---

  // CHECK-LABEL: ftof_from_ue5m3_to_bf16_nearest_even
  testing$func @ftof_from_ue5m3_to_bf16_nearest_even(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xbf16>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xbf16>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_bf16_zero
  testing$func @ftof_from_ue5m3_to_bf16_zero(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<zero> : tile<2xf8E5M3FNU> -> tile<2xbf16>
    %0 = ftof %a rounding<zero> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xbf16>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_bf16_negative_inf
  testing$func @ftof_from_ue5m3_to_bf16_negative_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<negative_inf> : tile<2xf8E5M3FNU> -> tile<2xbf16>
    %0 = ftof %a rounding<negative_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xbf16>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_bf16_positive_inf
  testing$func @ftof_from_ue5m3_to_bf16_positive_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<positive_inf> : tile<2xf8E5M3FNU> -> tile<2xbf16>
    %0 = ftof %a rounding<positive_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xbf16>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_bf16_nearest_away
  testing$func @ftof_from_ue5m3_to_bf16_nearest_away(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<nearest_away> : tile<2xf8E5M3FNU> -> tile<2xbf16>
    %0 = ftof %a rounding<nearest_away> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xbf16>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f32_nearest_even
  testing$func @ftof_from_ue5m3_to_f32_nearest_even(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xf32>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f32_zero
  testing$func @ftof_from_ue5m3_to_f32_zero(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<zero> : tile<2xf8E5M3FNU> -> tile<2xf32>
    %0 = ftof %a rounding<zero> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f32_negative_inf
  testing$func @ftof_from_ue5m3_to_f32_negative_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<negative_inf> : tile<2xf8E5M3FNU> -> tile<2xf32>
    %0 = ftof %a rounding<negative_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f32_positive_inf
  testing$func @ftof_from_ue5m3_to_f32_positive_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<positive_inf> : tile<2xf8E5M3FNU> -> tile<2xf32>
    %0 = ftof %a rounding<positive_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f32_nearest_away
  testing$func @ftof_from_ue5m3_to_f32_nearest_away(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<nearest_away> : tile<2xf8E5M3FNU> -> tile<2xf32>
    %0 = ftof %a rounding<nearest_away> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f64_nearest_even
  testing$func @ftof_from_ue5m3_to_f64_nearest_even(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xf64>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf64>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f64_zero
  testing$func @ftof_from_ue5m3_to_f64_zero(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<zero> : tile<2xf8E5M3FNU> -> tile<2xf64>
    %0 = ftof %a rounding<zero> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf64>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f64_negative_inf
  testing$func @ftof_from_ue5m3_to_f64_negative_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<negative_inf> : tile<2xf8E5M3FNU> -> tile<2xf64>
    %0 = ftof %a rounding<negative_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf64>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f64_positive_inf
  testing$func @ftof_from_ue5m3_to_f64_positive_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<positive_inf> : tile<2xf8E5M3FNU> -> tile<2xf64>
    %0 = ftof %a rounding<positive_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf64>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_f64_nearest_away
  testing$func @ftof_from_ue5m3_to_f64_nearest_away(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<nearest_away> : tile<2xf8E5M3FNU> -> tile<2xf64>
    %0 = ftof %a rounding<nearest_away> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xf64>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_tf32_nearest_even
  testing$func @ftof_from_ue5m3_to_tf32_nearest_even(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} : tile<2xf8E5M3FNU> -> tile<2xtf32>
    %0 = ftof %a : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xtf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_tf32_zero
  testing$func @ftof_from_ue5m3_to_tf32_zero(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<zero> : tile<2xf8E5M3FNU> -> tile<2xtf32>
    %0 = ftof %a rounding<zero> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xtf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_tf32_negative_inf
  testing$func @ftof_from_ue5m3_to_tf32_negative_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<negative_inf> : tile<2xf8E5M3FNU> -> tile<2xtf32>
    %0 = ftof %a rounding<negative_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xtf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_tf32_positive_inf
  testing$func @ftof_from_ue5m3_to_tf32_positive_inf(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<positive_inf> : tile<2xf8E5M3FNU> -> tile<2xtf32>
    %0 = ftof %a rounding<positive_inf> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xtf32>
  }

  // CHECK-LABEL: ftof_from_ue5m3_to_tf32_nearest_away
  testing$func @ftof_from_ue5m3_to_tf32_nearest_away(%a: !cuda_tile.tile<2xf8E5M3FNU>) {
    // CHECK: ftof %{{.+}} rounding<nearest_away> : tile<2xf8E5M3FNU> -> tile<2xtf32>
    %0 = ftof %a rounding<nearest_away> : !cuda_tile.tile<2xf8E5M3FNU> -> !cuda_tile.tile<2xtf32>
  }

} // end module
