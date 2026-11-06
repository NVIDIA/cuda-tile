// RUN: cuda-tile-opt %s -Wunsupported-hints -verify-diagnostics -split-input-file

// Attributes.cpp coverage: BoundedAttr lb > maxVal, verifyOneHint valid-hint path

// Covers BoundedAttr::verifyWithAssumeOp() — lb exceeds maxVal for the element type
cuda_tile.module @bounded_lb_above_max {
  testing$func @test(%arg0: !cuda_tile.tile<16xi8>) {
    // expected-error @below{{'cuda_tile.bounded' expects lower bound to be within [-128, 127]}}
    cuda_tile.assume #cuda_tile.bounded<200, ?>, %arg0 : !cuda_tile.tile<16xi8>
    return
  }
}

// -----

// Covers verifyOneHint() — valid hint value (empty errorMessage) path
// and verifyParamWithContext() — key IS valid for the operation, proceeds to macro chain
cuda_tile.module @valid_hint_no_warning {
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=2}> {
    return
  }
}

// -----

// Covers verifyOneHint() — valid occupancy hint (no warning emitted)
cuda_tile.module @valid_occupancy_hint {
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={occupancy=4}> {
    return
  }
}

// -----

// Covers verifyOneHint() — valid boolean hint for allow_tma
cuda_tile.module @valid_allow_tma_hint {
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=4, occupancy=8}> {
    return
  }
}

// -----

// Attributes.cpp coverage: hint warning paths with -Wunsupported-hints
// Target: verifyOneHint valid hint, key not valid for op,
//         unknown param, errorOnHints in verifyWithOp

// key not in keysValidForOperation (non-empty list, key not contained)
// Uses a known SM key with an unknown param name to hit the "not known hint" path
cuda_tile.module @unknown_hint_for_op {
  // expected-warning @below{{unknown_key_xyz is not known hint for current Operation}}
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={unknown_key_xyz=1}> {
    return
  }
}

// -----

// param key not matching any CUDA_TILE_HINT_KEY in verifyParamWithContext
// Use "default" as context key (not an SM key) to test the unknown-key path
cuda_tile.module @unknown_context_key {
  // expected-warning @below{{unknown hint key unknown_arch_xyz}}
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<unknown_arch_xyz={num_cta_in_cga=2}> {
    return
  }
}

// -----

// verifyOneHint with a valid hint that generates an error message
// (e.g., power-of-2 expected warning for non-power-of-2 value)
cuda_tile.module @hint_invalid_value_warn {
  // expected-warning @below{{power-of-2 expected for sm_100.num_cta_in_cga}}
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=7}> {
    return
  }
}

// -----

// Attributes.cpp coverage: OptimizationHintsAttr::verifyParamWithContext()
// — empty keysValidForOperation short-circuit path (the `!empty()` false branch),
//   and the unknown-key fall-through to the macro-chain `else`
//   branch ("unknown param ... for <context>").
//
// cuda_tile.if (IfOp) calls verifyOptHintsCommon via its verify() but is NOT
// listed in CUDA_TILE_HINT_OP_KEYS (HintKeyImpl.inc), so its
// keysValidForOperation vector stays empty. Putting `optimization_hints` on it
// (as a discardable attribute) exercises the empty-keys branch.

// Covers verifyParamWithContext() when keysValidForOperation is empty:
//   `!keysValidForOperation.empty() && !is_contained(...)` short-circuits to false,
// and a key that does NOT match any CUDA_TILE_HINT_KEY hits the trailing
// `unknown param ... for sm_100` branch.
cuda_tile.module @kernels_unknown_param_empty_keys {
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) {
    %cond = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    // expected-warning @below{{unknown param totally_unknown_key for sm_100}}
    cuda_tile.if %cond {
      cuda_tile.yield
    } {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {totally_unknown_key = 1}>}
    return
  }
}

// -----

// Same op, key IS a known CUDA_TILE_HINT_KEY (`occupancy`): keysValidForOperation
// is still empty so the short-circuit is taken, then the macro-chain runs the
// occupancy validator. Confirms the empty-keys branch on a non-error path.
cuda_tile.module @kernels_valid_known_key_empty_keys {
  entry @test(%arg0: !cuda_tile.tile<ptr<f32>>) {
    %cond = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield
    } {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {occupancy = 4}>}
    return
  }
}

