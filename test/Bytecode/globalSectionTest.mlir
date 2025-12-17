// RUN: %round_trip_test %s %t


cuda_tile.module @kernels {
    cuda_tile.global @val <f64: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf64>
    cuda_tile.global @val2 alignment = 256 <i32: 42> : !cuda_tile.tile<1xi32>
  

  cuda_tile.entry @add_entry() {
    cuda_tile.return
  }
}
