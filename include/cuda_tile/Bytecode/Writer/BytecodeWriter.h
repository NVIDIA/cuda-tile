//===- BytecodeWriter.h - CUDA Tile Bytecode Writer -------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_BYTECODE_WRITER_H
#define CUDA_TILE_BYTECODE_WRITER_H

#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::cuda_tile {

/// Configuration controlling how a module is serialized to bytecode.
///
/// The target version and whether it was explicitly requested are coupled: they
/// are only ever set together through a constructor and exposed read-only, so
/// they cannot drift out of sync after construction.
class BytecodeWriterConfig {
public:
  BytecodeWriterConfig() = default;

  /// Implicit so a bare version can be passed where a config is expected (e.g.
  /// `writeBytecode(os, module, version)`); such a version counts as explicit.
  BytecodeWriterConfig(BytecodeVersion version)
      : bytecodeVersion(version), versionExplicitlySet(true) {}

  /// Version plus whether it was explicitly requested; used by callers that get
  /// the version from a command-line default that may not have been set.
  BytecodeWriterConfig(BytecodeVersion version, bool versionExplicitlySet)
      : bytecodeVersion(version), versionExplicitlySet(versionExplicitlySet) {}

  BytecodeVersion getBytecodeVersion() const { return bytecodeVersion; }
  bool isVersionExplicitlySet() const { return versionExplicitlySet; }

private:
  /// The bytecode version to target.
  BytecodeVersion bytecodeVersion =
      BytecodeVersion::kCurrentCompatibilityVersion;

  /// Whether `bytecodeVersion` was explicitly requested (vs. a default). Only
  /// affects preview: an explicit non-current version is a conflict (rejected),
  /// while a default version is silently promoted to current.
  bool versionExplicitlySet = false;
};

/// Writes a cuda_tile module to the provided output stream in bytecode format.
/// Public-preview features must be enabled on the module's CUDA Tile dialect
/// before calling this function.
LogicalResult writeBytecode(raw_ostream &os, cuda_tile::ModuleOp module,
                            const BytecodeWriterConfig &config = {});

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_BYTECODE_WRITER_H
