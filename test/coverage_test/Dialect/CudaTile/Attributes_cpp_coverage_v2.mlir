// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// Attributes.cpp coverage: verifyWithAssumeOp error paths for DivBy, SameElements, Bounded

// DivByAttr::verifyWithAssumeOp: !tileType (non-tile, non-tensor_view type)
cuda_tile.module @divby_non_tile_type {
  testing$func @test(%arg0: i32) {
    // expected-error @below{{'cuda_tile.div_by' is valid only for tile of integer/pointer or tensor_view values}}
    cuda_tile.assume #cuda_tile.div_by<16>, %arg0 : i32
    return
  }
}

// -----

// DivByAttr::verifyWithAssumeOp: divisor == 0 (not a power of 2)
cuda_tile.module @divby_zero_divisor {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{'cuda_tile.div_by' divisor must be a power of 2}}
    cuda_tile.assume #cuda_tile.div_by<0>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// DivByAttr::verifyWithAssumeOp: along < 0
cuda_tile.module @divby_along_negative {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{'cuda_tile.div_by' every_dim (-1) must be >= 0 and < tile rank (1)}}
    cuda_tile.assume #cuda_tile.div_by<1, every 2 along -1>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// DivByAttr::verifyWithAssumeOp: every < 0
cuda_tile.module @divby_every_negative {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected 'cuda_tile.div_by' every_dim to be within 0 and the size of the respective dimension (16)}}
    cuda_tile.assume #cuda_tile.div_by<1, every -1 along 0>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// SameElementsAttr::verifyWithAssumeOp: !tileType (non-tile type)
cuda_tile.module @same_elements_non_tile {
  testing$func @test(%arg0: i32) {
    // expected-error @below{{'cuda_tile.same_elements' is valid only for tile of integer/pointer values}}
    cuda_tile.assume #cuda_tile.same_elements<[]>, %arg0 : i32
    return
  }
}

// -----

// SameElementsAttr::verifyWithAssumeOp: value negative
cuda_tile.module @same_elements_negative_value {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected 'cuda_tile.same_elements' value 0 to be within 0 and the size of the respective dimension (16)}}
    cuda_tile.assume #cuda_tile.same_elements<[-1]>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// BoundedAttr::verifyWithAssumeOp: !tileType (non-tile type)
cuda_tile.module @bounded_non_tile {
  testing$func @test(%arg0: i32) {
    // expected-error @below{{'cuda_tile.bounded' is valid only for tile of integer values}}
    cuda_tile.assume #cuda_tile.bounded<0, 0>, %arg0 : i32
    return
  }
}

// -----

// BoundedAttr::verifyWithAssumeOp: lb < minVal
cuda_tile.module @bounded_lb_below_min {
  testing$func @test(%arg0: !cuda_tile.tile<16xi8>) {
    // expected-error @below{{'cuda_tile.bounded' expects lower bound to be within [-128, 127]}}
    cuda_tile.assume #cuda_tile.bounded<-200, 0>, %arg0 : !cuda_tile.tile<16xi8>
    return
  }
}

// -----

// BoundedAttr::verifyWithAssumeOp: ub < minVal
cuda_tile.module @bounded_ub_below_min {
  testing$func @test(%arg0: !cuda_tile.tile<16xi8>) {
    // expected-error @below{{'cuda_tile.bounded' expects upper bound to be within [-128, 127]}}
    cuda_tile.assume #cuda_tile.bounded<?, -200>, %arg0 : !cuda_tile.tile<16xi8>
    return
  }
}

// -----

// Tests DivByAttr parsing error paths for Attributes.cpp coverage

// Test parseLess failure
cuda_tile.module @divby_parse_no_less {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected '<'}}
    cuda_tile.assume #cuda_tile.div_by 16>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// Test parseInteger divisor failure
cuda_tile.module @divby_parse_invalid_divisor {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected integer value}}
    // expected-error @below{{failed to parse parameter 'divisor' which is expected to be an integer}}
    cuda_tile.assume #cuda_tile.div_by<"abc">, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// Test parseKeyword("every") failure
cuda_tile.module @divby_parse_no_every_keyword {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected 'every'}}
    cuda_tile.assume #cuda_tile.div_by<16, 8 along 0>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// Test parseKeyword("along") failure
cuda_tile.module @divby_parse_no_along_keyword {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected 'along'}}
    cuda_tile.assume #cuda_tile.div_by<16, every 8 beyond 0>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// Test parseInteger everyVal failure
cuda_tile.module @divby_parse_invalid_every_val {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected integer value}}
    cuda_tile.assume #cuda_tile.div_by<16, every "x" along 0>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// Test parseInteger alongVal failure
cuda_tile.module @divby_parse_invalid_along_val {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected integer value}}
    cuda_tile.assume #cuda_tile.div_by<16, every 8 along "x">, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

// -----

// Attributes.cpp coverage: OptimizationHintsAttr verify and verifyWithOp error paths

// OptimizationHintsAttr::verify: non-dict entry value (true)
cuda_tile.module @opt_hints_verify_non_dict {
  // expected-error @below{{custom op 'cuda_tile.entry' expected dictionary attribute for optimization_hints entry `sm_100` got value=42 : i64}}
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100=42> {
    return
  }
}

// -----

// OptimizationHintsAttr::verifyWithOp: inner non-dict value (true)
// This path is checked separately in verifyWithOp after verify passes.
// The verify() already catches non-dict entries, so this error comes first.
cuda_tile.module @opt_hints_with_op_non_dict {
  // expected-error @below{{custom op 'cuda_tile.entry' expected dictionary attribute for optimization_hints entry `default` got value="abc"}}
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<default="abc"> {
    return
  }
}

// -----

// Attributes.cpp coverage: parser error paths for OptimizationHintsAttr and DivByAttr

// OptimizationHintsAttr::parse: parseGreater failure after comma-separated list
cuda_tile.module @opt_hints_missing_gt {
  // expected-error @below{{expected '>'}}
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=2} {
    return
  }
}

// -----

// DivByAttr::parse: parseGreater failure after every/along
cuda_tile.module @divby_parse_extra_after_along {
  testing$func @test(%arg0: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{expected '>'}}
    cuda_tile.assume #cuda_tile.div_by<16, every 4 along 0 extra>, %arg0 : !cuda_tile.tile<16xi32>
    return
  }
}

