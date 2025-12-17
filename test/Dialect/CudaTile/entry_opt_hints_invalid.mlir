// RUN: cuda-tile-opt %s -verify-diagnostics  -split-input-file

cuda_tile.module @unknown_sm {
  // expected-error @below{{custom op 'cuda_tile.entry' unallowed key sm_100a}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100a={num_cta_in_cga=2}> {
    return
  }
}

// -----

cuda_tile.module @sm_not_dict {
  // expected-error @below{{custom op 'cuda_tile.entry' expected dictionary attribute for optimization_hints entry `sm_100` got value=2 : i64}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100=2> {
    return
  }
}

// -----

cuda_tile.module @sm_unknown_param {
  // expected-error @below{{custom op 'cuda_tile.entry' unknown param num_qqq for sm_100}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_qqq=1}> {
    return
  }
}

// -----

cuda_tile.module @sm_not_int_param {
  // expected-error @below{{custom op 'cuda_tile.entry' integer value expected for sm_100.num_cta_in_cga}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga="a"}> {
    return
  }
}

// -----

cuda_tile.module @sm_not_power_of_2 {
  // expected-error @below{{custom op 'cuda_tile.entry' expected power-of-two ≤ 16 for sm_100.num_cta_in_cga}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=7}> {
    return
  }
}

// -----

cuda_tile.module @occupancy_invalid {
  // expected-error @below{{custom op 'cuda_tile.entry' integer value in the range [1, 32] is expected for sm_100.occupancy}}
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={occupancy=64}> {
    return
  }
}
