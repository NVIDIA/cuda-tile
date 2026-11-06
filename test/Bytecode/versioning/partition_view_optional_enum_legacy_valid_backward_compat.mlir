// Backward-compat regression: v13.2 bytecode writes a 1-byte presence flag for
// the `padding_value` OptionalEnum on `cuda_tile.partition_view`. Check both
// the absent and present cases, including the `byte(0)` for no padding.
//
// The fixture uses rank-0 tile entry arguments and creates the partition views
// inside the entries, so it satisfies the current bytecode invariants.
//
// Source used to bake `partition_view_optional_enum_legacy_valid-13.2.tileirbc`:
//
// COM: cuda_tile.module @kernels {
// COM:   cuda_tile.entry @partition_view_no_padding(
// COM:       %ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
// COM:     %tv = cuda_tile.make_tensor_view %ptr, shape = [16], strides = [1]
// COM:         : !cuda_tile.tensor_view<16xf32, strides=[1]>
// COM:     %view = cuda_tile.make_partition_view %tv
// COM:         : !cuda_tile.partition_view<tile=(2),
// COM:               tensor_view<16xf32, strides=[1]>>
// COM:     cuda_tile.return
// COM:   }
// COM:   cuda_tile.entry @partition_view_with_padding(
// COM:       %ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
// COM:     %tv = cuda_tile.make_tensor_view %ptr, shape = [16], strides = [1]
// COM:         : !cuda_tile.tensor_view<16xf32, strides=[1]>
// COM:     %view = cuda_tile.make_partition_view %tv
// COM:         : !cuda_tile.partition_view<tile=(2), padding_value = zero,
// COM:               tensor_view<16xf32, strides=[1]>>
// COM:     cuda_tile.return
// COM:   }
// COM: }

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.2/partition_view_optional_enum_legacy_valid-13.2.tileirbc \
// RUN:   | FileCheck %s

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.2/partition_view_optional_enum_legacy_valid-13.2.tileirbc \
// RUN:   -o %t.mlir
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.2 %t.mlir -o %t.bc
// RUN: cmp %t.bc \
// RUN:   %S/Inputs/13.2/partition_view_optional_enum_legacy_valid-13.2.tileirbc

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK:       entry @partition_view_no_padding(%{{.*}}: tile<ptr<f32>>)
// CHECK:         make_partition_view
// CHECK-SAME:      partition_view<tile=(2), tensor_view<16xf32, strides=[1]>>
// CHECK-NOT:       padding_value
// CHECK:       entry @partition_view_with_padding(%{{.*}}: tile<ptr<f32>>)
// CHECK:         make_partition_view
// CHECK-SAME:      partition_view<tile=(2), padding_value = zero,
// CHECK-SAME:        tensor_view<16xf32, strides=[1]>>
