// Test bytecode writer with f8 FloatAttr in optimization hints to exercise
// writeAPFloatRepresentation -> writeAPInt with bitWidth <= 8.
// Covers the writeAPInt() bitWidth <= 8 true branch via f8E4M3FN/f8E5M2
// FloatAttr inside a self-contained DictionaryAttr in optimization_hints.
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s -o %t.tilebc
// RUN: cuda-tile-translate -cudatilebc-to-mlir %t.tilebc -o %t.roundtrip.mlir
// RUN: cuda-tile-opt %s -no-implicit-module -o %t.ref.mlir
// RUN: %PYTHON %S/../../diff_helper.py %t.ref.mlir %t.roundtrip.mlir

cuda_tile.module @kernels {
  // f8E4M3FN FloatAttr in hints exercises writeAPInt bitWidth == 8 path
  cuda_tile.entry @f8e4m3fn_hint(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={scale_f8=2.5 : f8E4M3FN}> {
    cuda_tile.return
  }

  // f8E5M2 FloatAttr in hints also exercises writeAPInt bitWidth == 8
  cuda_tile.entry @f8e5m2_hint(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={threshold_f8=1.0 : f8E5M2}> {
    cuda_tile.return
  }

  // bf16 FloatAttr in hints exercises writeAPInt bitWidth == 16
  cuda_tile.entry @bf16_hint(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={weight_bf16=3.14 : bf16}> {
    cuda_tile.return
  }

  // Combine f8 hint with other hint types in one entry to stress
  // the self-contained attribute serialization paths
  cuda_tile.entry @mixed_f8_hints(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={f8_val=2.0 : f8E4M3FN, i_val=42, label="test", flag=true, f32_val=1.5 : f32}> {
    cuda_tile.return
  }
}
