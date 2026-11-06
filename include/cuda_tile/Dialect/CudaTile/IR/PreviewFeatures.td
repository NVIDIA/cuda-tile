//===- PreviewFeatures.td - Preview feature registry -------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_PREVIEW_FEATURES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_PREVIEW_FEATURES_TD

/// A named public-preview feature.
///
/// Feature records and their command-line spellings are permanent. When a
/// feature graduates or is removed, mark it retired instead of deleting it so
/// old command lines receive a specific diagnostic and the numeric value is
/// never reused.
class CudaTilePreviewFeature<string cppNameArg, int valueArg,
                             string spellingArg, bit retiredArg = 0> {
  string cppName = cppNameArg;
  int value = valueArg;
  string spelling = spellingArg;
  bit retired = retiredArg;
}

#ifdef TILE_IR_INCLUDE_TESTS
def CudaTile_TestOnlyPreviewFeature
    : CudaTilePreviewFeature<"TestOnly", 0, "test_only">;
#endif // TILE_IR_INCLUDE_TESTS

def CudaTile_SIMTPreviewFeature
    : CudaTilePreviewFeature<"SIMT", 1, "simt">;

#endif // CUDATILE_DIALECT_CUDATILE_IR_PREVIEW_FEATURES_TD
