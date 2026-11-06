// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.1 %s | FileCheck %s

// Reproducer for the BytecodeAttr evolved-parameter lifetime bug:
// `values` is an evolved ArrayRef<Attribute> parameter. Its generated
// deserializer used to back the ArrayRef with a SmallVector local to the
// version-gated block, leading to a use-after-free. This should be detected
// by UB sanitization, but to help detection a later ArrayRef parameter of
// the same size pressures allocator reuse.

cuda_tile.module @evolved_arrayref_attr_lifetime {
  entry @k() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>)
      arrayref = #cuda_tile.bytecode_test_evolved_arrayref_attr<7,
        ["value-00", "value-01", "value-02", "value-03",
         "value-04", "value-05", "value-06", "value-07",
         "value-08", "value-09", "value-10", "value-11"],
        ["clobber-00", "clobber-01", "clobber-02", "clobber-03",
         "clobber-04", "clobber-05", "clobber-06", "clobber-07",
         "clobber-08", "clobber-09", "clobber-10", "clobber-11"]>
      -> !cuda_tile.token
    // CHECK-LABEL: @k
    // CHECK: arrayref = <7, ["value-00"
    // CHECK-SAME: "value-11"
    // CHECK-SAME: ["clobber-00"
    // CHECK-SAME: "clobber-11"
    return
  }
}
