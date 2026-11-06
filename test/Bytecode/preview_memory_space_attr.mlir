// Tests that the explicit MemorySpaceAttr has public-preview runtime identity.

// RUN: split-file %s %t
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %t/valid.mlir | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t/valid.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt %t/valid.mlir -o %t.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.cutile 2>&1 | FileCheck %s --check-prefix=READ-NOFLAG
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t.cutile | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %t/invalid.mlir 2>&1 | FileCheck %s --check-prefix=INVALID

//--- valid.mlir

cuda_tile.module @preview_memory_space_attribute {
  entry @f() {
    "cuda_tile.testing$bytecode_test_memory_space_attr"() {
      memory_space = #cuda_tile.preview$memory_space<smem>
    } : () -> ()
    return
  }
}

// ROUNDTRIP: "cuda_tile.testing$bytecode_test_memory_space_attr"()
// ROUNDTRIP-SAME: memory_space = #cuda_tile<preview$memory_space<smem>>
// WRITE-NOFLAG: uses preview feature 'simt', which is not enabled
// READ-NOFLAG: public-preview bytecode requires at least one enabled preview feature

//--- invalid.mlir

cuda_tile.module @invalid_memory_space_attribute {
  entry @f() {
    "cuda_tile.testing$bytecode_test_memory_space_attr"() {
      memory_space = #cuda_tile.preview$memory_space<invalid>
    } : () -> ()
    return
  }
}

// INVALID: expected ::mlir::cuda_tile::MemorySpace to be one of: local, smem
