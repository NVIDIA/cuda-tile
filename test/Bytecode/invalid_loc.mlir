// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -split-input-file %s 2>&1 | FileCheck %s

#loc1 = loc("/tmp/foo.py":1:1)
#loc2 = loc("/tmp/foo.py":1:2)
#loc3 = loc(fused[#loc1, #loc2])
cuda_tile.module @invalid_fusedloc {
  entry @kernel() {
    // CHECK: unsupported location, got FusedLoc, expected DILocAttr or CallSiteLoc
    %a = constant <i32: 1> : tile<i32> loc(#loc3)
    return
  }
}

// -----

#loc1 = loc("/tmp/foo.py":1:1)
#loc2 = loc("name"(#loc1))
cuda_tile.module @invalid_nameloc {
  entry @kernel() {
    // CHECK: unsupported location, got NameLoc, expected DILocAttr or CallSiteLoc
    %a = constant <i32: 1> : tile<i32> loc(#loc2)
    return
  }
}

// -----

#loc1 = loc("/tmp/foo.py":1:1)
#loc2 = loc("/tmp/foo.py":1:2)
#loc_fused = loc(fused[#loc1, #loc2])
#loc3 = loc(callsite(#loc_fused at #loc1))
#loc4 = loc(callsite(#loc3 at #loc3))
cuda_tile.module @invalid_callsite_fused {
  entry @kernel() {
    // CHECK: unsupported location, got FusedLoc, expected DILocAttr or CallSiteLoc
    %a = constant <i32: 1> : tile<i32> loc(#loc4)
  }
}
