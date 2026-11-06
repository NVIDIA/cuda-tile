// RUN: cuda-tile-opt %s -verify-diagnostics  -split-input-file

cuda_tile.module @sm_not_dict {
  // expected-error @below{{custom op 'cuda_tile.entry' expected dictionary attribute for optimization_hints entry `sm_100` got value=2 : i64}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100=2> {
    return
  }
}

// -----

cuda_tile.module @hint_not_dict {
  // expected-error @below{{expected valid keyword or string}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<2> {
    return
  }
}

// -----

cuda_tile.module @duplicate_key_error {
  // expected-error @below{{duplicate optimization_hints key `sm_100`}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=2}, sm_100={num_cta_in_cga=4}> {
    return
  }
}

// -----

cuda_tile.module @parseLess_failure {
  // expected-error @below{{expected '<'}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints={sm_100={num_cta_in_cga=2}} {
    return
  }
}

// -----

cuda_tile.module @parseEqual_failure {
  // expected-error @below{{expected '='}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100 {num_cta_in_cga=2}> {
    return
  }
}

// -----

cuda_tile.module @parseAttribute_failure {
  // expected-error @below{{expected attribute value}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100=> {
    return
  }
}
