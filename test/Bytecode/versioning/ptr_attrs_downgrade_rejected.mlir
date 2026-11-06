// Pointer classifications were introduced in 13.4 and cannot be silently
// removed when targeting 13.3. The `unicast` and `multicast` cases were added
// in 13.5, so writing either case as 13.4 must also be rejected.
//
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.3 %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: type parameter 'ptrAttr' requires bytecode version 13.4+ to differ from fallback or default value, but targeting 13.3
//
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.4 %s -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=ENUM-CASE %s
// ENUM-CASE: enum value for type parameter 'ptrAttr' is not available in bytecode version 13.4

cuda_tile.module @ptr_attr_downgrade {
  entry @kernel(%ptr: tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>) {
    return
  }
}
