//===- SiteInitializer.cpp - CUDA Tile Pybind11 Registration ----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "cuda_tile-c/Registration.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_site_initialize_1, m) {
  m.doc() = "All CUDA Tile IR related dialects (cuda_tile) and passes.";

  // NB: This is a special API hook that will be automatically called during
  // library initialization.
  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirCudaTileRegisterAllDialects(registry);
  });

  // NB: This is not a special API hook and must be invoked manually by a user
  // in Python to register the passes.
  m.def("register_passes", []() { mlirCudaTileRegisterAllPasses(); });
}
