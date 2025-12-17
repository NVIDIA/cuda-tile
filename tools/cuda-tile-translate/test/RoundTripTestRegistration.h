//===- RoundTripTestRegistration.h - Round-trip Testing ---------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_TEST_BYTECODE_TESTREGISTRATION_H
#define CUDA_TILE_TEST_BYTECODE_TESTREGISTRATION_H

namespace mlir {
namespace cuda_tile {

void registerTileIRTestTranslations();

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_TEST_BYTECODE_TESTREGISTRATION_H
