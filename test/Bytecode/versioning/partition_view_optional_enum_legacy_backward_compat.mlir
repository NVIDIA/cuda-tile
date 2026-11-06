// This legacy v13.2 fixture uses `cuda_tile.partition_view` directly as an
// entry argument. Entry arguments must now be rank-0 tiles, so reject the
// fixture before attempting to round-trip its legacy OptionalEnum encoding.
//
// The committed `partition_view_optional_enum_legacy-13.2.tileirbc`
// remains unchanged from the binary baked by release/tileir-13.2.

// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.2/partition_view_optional_enum_legacy-13.2.tileirbc \
// RUN:   2>&1 | FileCheck %s

// CHECK: unsupported type in bytecode for entry argument #0:
// CHECK-SAME: '!cuda_tile.partition_view<tile=(2), tensor_view<16xf32, strides=[1]>>'
// CHECK-SAME: expected a rank-0 !cuda_tile.tile type
