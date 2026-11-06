// Covers BytecodeWriter::FunctionTableWriter::writeSingleAttribute()
// failure paths that no other existing test reaches:
//
//   * The Default-arm of the TypeSwitch in writeSingleAttribute (the
//     "unsupported attribute type encountered during serialization"
//     diagnostic) — fired when an attribute kind is not a TypeAttr /
//     StringAttr / IntegerAttr / FloatAttr / BoolAttr /
//     DenseElementsAttr / ArrayAttr / DictionaryAttr and is also not
//     handled by the generated writeCudaTileAttr.
//   * The DictionaryAttr-arm's `failed(writeSelfContainedAttribute)`
//     guard that wraps each inner NamedAttribute, surfacing as
//     "failed to write DictionaryAttr element". This is the
//     propagated-failure arm of the outer DictionaryAttr Case that
//     happy-path round-trips never visit.
//   * The OptimizationHintsAttr top-level writeSingleAttribute call
//     similarly propagates the inner failure back up through the
//     outer DictionaryAttr's `failed(...)` guard, so both nesting
//     levels of the DictionaryAttr Case's failure arm are exercised.
//
// We deliberately use a SymbolRefAttr (`@somefunc`) as the value
// because the verifier accepts it inside the optimization_hints dict
// (the parser does not constrain the value type), but the bytecode
// writer's TypeSwitch has no Case for SymbolRefAttr — so the writer
// is forced to take the Default arm at serialization time.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:     -bytecode-version=13.4 -verify-diagnostics %s

cuda_tile.module @kernels {
  // expected-error @below {{unsupported attribute type encountered during serialization of attribute 'ref': @somefunc}}
  // expected-error @below {{failed to write DictionaryAttr element: @somefunc}}
  // expected-error @below {{failed to write DictionaryAttr element: {ref = @somefunc}}}
  // expected-error @below {{unsupported attribute type encountered during serialization of attribute 'optimization_hints': #cuda_tile.optimization_hints<sm_100 = {ref = @somefunc}>}}
  cuda_tile.entry @entry_unsupported_attr(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100 = {ref = @somefunc}> {
    return
  }
}
