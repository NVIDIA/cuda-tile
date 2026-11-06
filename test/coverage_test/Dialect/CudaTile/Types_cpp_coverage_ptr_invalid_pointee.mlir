// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// Types.cpp: PointerType::verify — invalid pointee type triggers the
// isValidPointee error path.

// i128 is not in the list of valid pointee types.
cuda_tile.module @test {
  testing$func @test_ptr_invalid_pointee(
      // expected-error @below {{failed to verify 'pointeeType'}}
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i128>>) {
    return
  }
}
