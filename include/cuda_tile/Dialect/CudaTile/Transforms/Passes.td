//===- Passes.td - CUDA Tile Dialect Passes ----------------*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDATILE_DIALECT_CUDATILE_TRANSFORMS_PASSES_TD
#define CUDATILE_DIALECT_CUDATILE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SynthesizeDebugInfoScopes
//===----------------------------------------------------------------------===//

def SynthesizeDebugInfoScopesPass : Pass<
  "synthesize-debug-info-scopes", "::mlir::cuda_tile::ModuleOp"
> {
  let summary = "Synthesize debug info scope information for a module";
  let description = [{
    To generate debug information of any kind, cuda_tile requires that the
    necessary debug information metadata is attached to operations within the
    module (this is in addition to the simple file location information). For
    frontends that are not yet equipped to properly emit debug information,
    this pass can be used to synthesize the necessary information to at least
    produce line table information. This pass is not intended to be a
    replacement for proper debug information emission from a frontend, but
    can provide a convienient stop-gap.
  }];
}

//===----------------------------------------------------------------------===//
// FuseFMA
//===----------------------------------------------------------------------===//

def FuseFMAPass : InterfacePass<
  "fuse-fma", "mlir::FunctionOpInterface"
> {
  let summary = "Fuse multiply-add and multiply-subtract operations into FMA operations (non-numeric-preserving)";
  let description = [{
    Fuses multiply-add and multiply-subtract operations into FMA operations.
    
    NON-NUMERIC-PRESERVING: Changes rounding behavior from double-round 
    to single-round FMA, affecting exact bit patterns.
    
    Patterns:
    1. MulAddPattern: (a * b) + c → FMA(a, b, c)
    2. MulSubPattern: (a * b) - c → FMA(a, b, -c)
    
    Additional optimizations:
    - Applies canonicalization patterns for AddFOp to enable more fusion opportunities
    
    Constraints: Preserves rounding modes/FTZ modifiers, requires single-use multiply.
    Targets: Any FunctionOpInterface operation.
  }];
}

//===----------------------------------------------------------------------===//
// LoopSplit
//===----------------------------------------------------------------------===//

def LoopSplitPass : InterfacePass<"loop-split", "mlir::FunctionOpInterface"> {
  let summary = "Split loops when predicate in if-condition compares iv with loop invariant";
  let description = [{
    Perform loop splitting like in the following example:
    Before:
        %4 = for %arg1 in (%1 to %0, step %2) : tile<i32> iter_values(%7 = %1) -> (tile<i32>) {
        %5 = cmpi greater_than %arg1, %3, signed : tile<i32>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %arg1, %0 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %arg1 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %3 : tile<i32>

    After:
      %0 = addi %cst_32_i32, %cst_1_i32 : tile<i32>
      %for = for %loopIdx in (%cst_0_i32 to %0, step %cst_1_i32) : tile<i32> iter_values(%iterArg0 = %cst_0_i32) -> (tile<i32>) {
        %2 = addi %iterArg0, %loopIdx : tile<i32>
        continue %2 : tile<i32>
      }
      %for_0 = for %loopIdx in (%0 to %cst_128_i32, step %cst_1_i32) : tile<i32> iter_values(%iterArg0 = %for) -> (tile<i32>) {
        %2 = muli %loopIdx, %cst_128_i32 : tile<i32>
        %3 = addi %iterArg0, %2 : tile<i32>
        continue %3 : tile<i32>
      }
      %1 = addi %for_0, %cst_32_i32 : tile<i32>
  }];
  let options = [
    Option<"splitThreshold","split-threshold",
      "int", /*default=*/"1",
      "Threshold to split loop only if-block contaings not less than given number of operations"
      >
  ];
}

#endif // CUDATILE_DIALECT_CUDATILE_TRANSFORMS_PASSES_TD
