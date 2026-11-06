// Test optimization pipeline levels and verbose output in CudaTileOptimizer.cpp
// Covers: buildDefaultCudaTilePipeline (opt levels 0-3, fuseFMA),
//         buildCudaTileOptimizationPipeline, optimizeTileIRModule,
//         optimizeTileIR full flow, parseTextInto (empty text), loadInputModule,
//         emitOutputs (MlirStdout)

// --- opt-level 0: no passes added ---
// RUN: cuda-tile-optimize %s --opt-level 0 | FileCheck %s --check-prefix=CHECK-OPT0

// --- opt-level 1: canonicalize + CSE ---
// RUN: cuda-tile-optimize %s --opt-level 1 --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-OPT1

// --- opt-level 2: + LICM ---
// RUN: cuda-tile-optimize %s --opt-level 2 --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-OPT2

// --- opt-level 3 (default): + loop-split + canonicalize ---
// RUN: cuda-tile-optimize %s --opt-level 3 --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-OPT3

// --- fuse-fma at opt-level 0 ---
// RUN: cuda-tile-optimize %s --opt-level 0 --fuse-fma --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-FMA

// --- fuse-fma at opt-level 3 ---
// RUN: cuda-tile-optimize %s --opt-level 3 --fuse-fma --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-FMA3

cuda_tile.module @test_module {
  entry @test_entry() {
    return
  }
}

// CHECK-OPT0: cuda_tile.module @test_module
// CHECK-OPT0:   entry @test_entry()
// CHECK-OPT0:     return

// CHECK-OPT1: Pipeline:
// CHECK-OPT1: canonicalize
// CHECK-OPT1: cse
// CHECK-OPT1: cuda_tile.module @test_module

// CHECK-OPT2: Pipeline:
// CHECK-OPT2: canonicalize
// CHECK-OPT2: cse
// CHECK-OPT2: cuda-tile-licm
// CHECK-OPT2: cuda_tile.module @test_module

// CHECK-OPT3: Pipeline:
// CHECK-OPT3: canonicalize
// CHECK-OPT3: cse
// CHECK-OPT3: cuda-tile-licm
// CHECK-OPT3: loop-split
// CHECK-OPT3: canonicalize
// CHECK-OPT3: cuda_tile.module @test_module

// CHECK-FMA: Pipeline:
// CHECK-FMA: fuse-fma
// CHECK-FMA: cuda_tile.module @test_module

// CHECK-FMA3: Pipeline:
// CHECK-FMA3: fuse-fma
// CHECK-FMA3: canonicalize
// CHECK-FMA3: cse
// CHECK-FMA3: cuda-tile-licm
// CHECK-FMA3: loop-split
// CHECK-FMA3: canonicalize
// CHECK-FMA3: cuda_tile.module @test_module
