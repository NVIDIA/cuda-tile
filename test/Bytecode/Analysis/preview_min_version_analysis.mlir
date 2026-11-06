// Tests for MinVersionAnalyzer utility on public-preview content.
// RUN: cuda-tile-translate --preview=simt -analyze-min-version -split-input-file %s | FileCheck %s
//
// Same section shape as min_version_analysis.mlir: one printed version per
// section, chained empty-line/next-line checks so a wrong version is reported
// against its own section.

// A symbolic shape differs from TileType's public fallback and is a preview
// extension, so it requires the current version. The only op that can produce
// one is preview too, so this does not isolate the shape from the op.
// CHECK: 13.5
cuda_tile.module @symbolic_tile_current {
  entry @kernel() {
    %value = testing$bytecode_test_preview_type : !cuda_tile.tile<[M]x4xf32>
    return
  }
}

// -----

// `index` is public-preview, so a serialized `index` requires the current
// version. Only a preview-tagged testing op can produce one, so like the
// symbolic shape above this does not isolate the type from the op.
// CHECK-EMPTY:
// CHECK-NEXT: 13.5
cuda_tile.module @preview_index_type_current {
  entry @kernel() {
    %value = testing$bytecode_test_preview_type : index
    return
  }
}

// -----

// The same `index` in a discardable attribute is not serialized, so neither the
// walk nor the preview-feature classifier scores it.
// CHECK-EMPTY:
// CHECK-NEXT: 13.1
cuda_tile.module @preview_index_discardable_attr_ignored {
  entry @kernel() {
    return
  } {cuda_tile.probe = 0 : index}
}
