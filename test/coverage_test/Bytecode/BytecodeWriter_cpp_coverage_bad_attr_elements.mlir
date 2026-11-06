// Coverage test: failure paths in writeSelfContainedAttribute for ArrayAttr and
// DictionaryAttr elements when an unsupported attribute (UnitAttr) is nested
// inside optimization_hints. Exercises the "failed to write ArrayAttr element"
// and "failed to write DictionaryAttr element" emitErr branches, plus the
// Default "unsupported attribute type" path in the TypeSwitch.
//
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -split-input-file %s 2>&1 | FileCheck %s

// UnitAttr nested in an ArrayAttr inside the optimization_hints DictionaryAttr
// is unsupported by writeSelfContainedAttribute. This trips the Default case
// in the TypeSwitch, which surfaces back through the ArrayAttr element loop
// and the DictionaryAttr element loop as cascading emitErr() diagnostics.
// CHECK: unsupported attribute type encountered during serialization of attribute 'arrayElement'
// CHECK: failed to write ArrayAttr element
// CHECK: failed to write DictionaryAttr element
cuda_tile.module @kernels {
  cuda_tile.entry @bad_array_element(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={items=[unit]}> {
    cuda_tile.return
  }
}

// -----

// UnitAttr as a direct DictionaryAttr value (rather than wrapped in an
// ArrayAttr) trips writeSelfContainedAttribute for the dict value and
// produces the DictionaryAttr element failure path.
// CHECK: unsupported attribute type encountered during serialization of attribute 'unit_val'
// CHECK: failed to write DictionaryAttr element
cuda_tile.module @kernels {
  cuda_tile.entry @bad_dict_element(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={unit_val=unit}> {
    cuda_tile.return
  }
}
