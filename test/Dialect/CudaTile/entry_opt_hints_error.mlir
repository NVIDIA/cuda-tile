// Test that invalid/unsupported hints produce errors with -Wunsupported-hints -Werr-hints
// RUN: cuda-tile-opt %s -Wunsupported-hints -Werr-hints -verify-diagnostics -split-input-file

cuda_tile.module @unknown_sm {
  // expected-warning @below{{unknown hint key sm_100a}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100a={num_cta_in_cga=2}> {
    return
  }
}

// -----

cuda_tile.module @sm_unknown_param {
  // expected-warning @below{{num_qqq is not known hint for current Operation}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_qqq=1}> {
    return
  }
}

// -----

cuda_tile.module @sm_not_int_param {
  // expected-warning @below{{integer value expected for sm_100.num_cta_in_cga}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga="a"}> {
    return
  }
}

// -----

cuda_tile.module @sm_not_power_of_2 {
  // expected-warning @below{{power-of-2 expected for sm_100.num_cta_in_cga}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=7}> {
    return
  }
}

// -----

cuda_tile.module @occupancy_invalid {
  // expected-warning @below{{value 64 clamped to 32 for sm_100.occupancy}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={occupancy=64}> {
    return
  }
}

// -----

cuda_tile.module @warps_not_int_param {
  // expected-warning @below{{integer value expected for sm_100.num_worker_warps_per_cta}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_worker_warps_per_cta="a"}> {
    return
  }
}

// -----

cuda_tile.module @warps_not_power_of_2 {
  // expected-warning @below{{power-of-2 expected for sm_100.num_worker_warps_per_cta}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_worker_warps_per_cta=9}> {
    return
  }
}

// -----

cuda_tile.module @warps_out_of_range {
  // expected-warning @below{{value 64 clamped to 32 for sm_100.num_worker_warps_per_cta}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_worker_warps_per_cta=64}> {
    return
  }
}

// -----

// A value below the [1, 32] range (< 1) is non-positive, so it is rejected by
// the power-of-2 check.
cuda_tile.module @warps_below_range {
  // expected-warning @below{{power-of-2 expected for sm_100.num_worker_warps_per_cta}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_worker_warps_per_cta=0}> {
    return
  }
}

