//===- MinVersionAnalyzer.cpp -----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimum Bytecode Version Analysis
//
// This file implements the MinVersionAnalyzer which determines the minimum
// bytecode version required for serializing a given CUDA Tile IR module.
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Analysis/MinVersionAnalyzer.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.h"

// Include generated attribute, enum, and parameter version lookup. Must precede
// the type version map below, which calls detail::getAttrMinVersionRecursive
// (emitted here) to score attributes embedded in a type parameter.
#define GEN_ATTR_VERSION_MAP
#define GEN_ENUM_VALUE_VERSION_MAP
#define GEN_ATTR_PARAM_VERSION_MAP
#include "AttrBytecode.inc"

// Include generated type version lookup. Its getTypeMinVersionRecursive walker
// calls detail::getAttrMinVersionRecursive for attribute-valued type params.
#define GEN_TYPE_VERSION_MAP
#include "TypeBytecode.inc"

// Include generated per-operation version checkers.
#define GEN_OP_VERSION_CHECKERS
#include "Bytecode.inc"

using namespace mlir;
using namespace mlir::cuda_tile;

FailureOr<BytecodeVersion>
mlir::cuda_tile::getMinBytecodeVersion(ModuleOp module) {
  BytecodeVersion maxVersion = BytecodeVersion::kMinSupportedVersion;
  DenseSet<Type> processedTypes;

  // Callback for type version checking (with deduplication).
  // If we've already processed this type, its version contribution has been
  // recorded in maxVersion, so return minimum to avoid redundant work.
  auto getTypeVersion = [&](Type type) -> BytecodeVersion {
    if (!processedTypes.insert(type).second) {
      return BytecodeVersion::kMinSupportedVersion;
    }
    auto version = detail::getTypeMinVersionRecursive(type);
    return version.value_or(BytecodeVersion::kMinSupportedVersion);
  };

  // Callback for attribute version checking (attr version, enum case, params,
  // and any nested attributes/types).
  auto getAttrVersion = [&](Attribute attr) -> BytecodeVersion {
    BytecodeVersion version = BytecodeVersion::kMinSupportedVersion;
    AttrTypeWalker walker;
    walker.addWalk([&](Attribute nestedAttr) {
      version = std::max(
          version,
          detail::getAttrMinVersionRecursive(nestedAttr).value_or(
              BytecodeVersion::kMinSupportedVersion));
      // Debug info is hand-serialized, so it has no generated version entry
      // and each gated field has to be mirrored here. `flags` is the only one
      // today: below 13.5 the writer refuses the module rather than drop a bit
      // that changes what the subprogram means.
      if (auto subprogram = dyn_cast<DISubprogramAttr>(nestedAttr)) {
        if (subprogram.getFlags() != DISubprogramFlags::None) {
          version =
              std::max(version, BytecodeVersion::kMinDISubprogramFlagsVersion);
        }
      }
    });
    walker.addWalk([&](Type nestedType) -> WalkResult {
      version = std::max(version, getTypeVersion(nestedType));
      // getTypeVersion already recursively handles serialized type parameters,
      // including their version-dependent fallback rules. Do not generically
      // descend into the type's raw storage a second time.
      return WalkResult::skip();
    });
    (void)walker.walk<WalkOrder::PreOrder>(attr);
    return version;
  };

  // Process each operation using generated per-op version checkers. The
  // dispatch reports kCurrentVersion for public-preview ops (they are
  // version-locked to current).
  //
  // An operation's attributes are deliberately not inspected here. An inherent
  // attribute is scored by that operation's generated checker, which knows
  // which values the writer elides; scoring it again would report the
  // attribute's own version even for a value that never reaches the wire. A
  // discardable attribute is not serialized. Function `sym_visibility` is
  // handled separately by the function table at kMinSupportedVersion and is
  // excluded from the generated operation checker.
  module->walk([&](Operation *op) {
#define GEN_OP_VERSION_CHECK_DISPATCH
#include "Bytecode.inc"
    // The writer serializes every operation's location, so its debug info
    // counts as well.
    opVersion = std::max(opVersion, getAttrVersion(op->getLoc()));
    maxVersion = std::max(maxVersion, opVersion);
  });

  // Public-preview content is version-locked to the current version, and the
  // walk above cannot see all of it: a preview builtin type reached only
  // through an attribute's type (`0 : index`) contributes nothing. Defer to the
  // same classifier the writer gates on so the two cannot disagree.
  if (!getRequiredPreviewFeatures(module.getOperation()).empty())
    maxVersion = std::max(maxVersion, BytecodeVersion::kCurrentVersion);

  return maxVersion;
}
