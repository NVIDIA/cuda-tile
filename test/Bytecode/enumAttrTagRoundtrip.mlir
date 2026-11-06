// Round-trip test for a CudaTileEnumAttr (BytecodeTestPriorityAttr) that has
// been assigned a BytecodeAttrTag. Exercises the EnumParameter classification
// fix and the CudaTileEnumAttr scan added to BytecodeAttrAnalysis.cpp.
//
// The attr is embedded inside an ArrayRef<Attribute> parameter so it is
// serialized self-contained (with its tag byte), exercising the
// serializeBytecodeTestPriorityData / parseBytecodeTestPriorityData code path
// generated from the BytecodeAttrTag assignment.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.1 %s | FileCheck %s

cuda_tile.module @enum_attr_tag_roundtrip {
  entry @priority_in_attr_array() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>)
      arrayref = #cuda_tile.bytecode_test_evolved_arrayref_attr<0,
        [#cuda_tile.bytecode_test_priority<low>,
         #cuda_tile.bytecode_test_priority<normal>,
         #cuda_tile.bytecode_test_priority<high>],
        [#cuda_tile.bytecode_test_priority<normal>]>
      -> !cuda_tile.token
    // CHECK-LABEL: @priority_in_attr_array
    // CHECK: arrayref = <0, [#cuda_tile.bytecode_test_priority<low>,
    // CHECK-SAME: #cuda_tile.bytecode_test_priority<normal>,
    // CHECK-SAME: #cuda_tile.bytecode_test_priority<high>],
    // CHECK-SAME: [#cuda_tile.bytecode_test_priority<normal>]>
    return
  }
}
