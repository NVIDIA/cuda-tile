// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// Public-release regression coverage for multicast pointer restrictions.
// CudaTile entry points require rank-0 tiles, so each test uses a scalar
// pointer tile while exercising the same verifier as vector-shaped programs.

//===----------------------------------------------------------------------===//
// LoadPtrTkoOp — multicast pointer rejected
//===----------------------------------------------------------------------===//

cuda_tile.module @load_multicast_rejected_module {
  cuda_tile.entry @load_multicast_rejected(
      %ptr: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
    // expected-error @below{{multicast pointer is not allowed}}
    %val, %tok = cuda_tile.load_ptr_tko weak %ptr
        : tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>> -> tile<f32>, token
    cuda_tile.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// StorePtrTkoOp — multicast pointer rejected
//===----------------------------------------------------------------------===//

cuda_tile.module @store_multicast_rejected_module {
  cuda_tile.entry @store_multicast_rejected(
      %ptr: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>,
      %val: !cuda_tile.tile<f32>) {
    // expected-error @below{{multicast pointer is not allowed}}
    %tok = cuda_tile.store_ptr_tko weak %ptr, %val
        : tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>, tile<f32> -> token
    cuda_tile.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// AtomicRMWTkoOp — multicast pointer rejected
//===----------------------------------------------------------------------===//

cuda_tile.module @atomic_rmw_multicast_rejected_module {
  cuda_tile.entry @atomic_rmw_multicast_rejected(
      %ptr: !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<multicast>>>,
      %arg: !cuda_tile.tile<i32>) {
    // expected-error @below{{multicast pointer is not supported for atomic_rmw}}
    %val, %tok = cuda_tile.atomic_rmw_tko relaxed device %ptr, add, %arg
        : tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>, tile<i32>
        -> tile<i32>, token
    cuda_tile.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// AtomicCASTkoOp — multicast pointer rejected
//===----------------------------------------------------------------------===//

cuda_tile.module @atomic_cas_multicast_rejected_module {
  cuda_tile.entry @atomic_cas_multicast_rejected(
      %ptr: !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<multicast>>>,
      %cmp: !cuda_tile.tile<i32>,
      %val: !cuda_tile.tile<i32>) {
    // expected-error @below{{multicast pointer is not supported for atomic_cas}}
    %res, %tok = cuda_tile.atomic_cas_tko relaxed device %ptr, %cmp, %val
        : tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>, tile<i32>
        -> tile<i32>, token
    cuda_tile.return
  }
}
