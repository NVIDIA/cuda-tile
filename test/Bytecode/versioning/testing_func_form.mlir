// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.4 %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OLD-VERSION
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=250.0 %s \
// RUN:   | FileCheck %s --check-prefix=ROUNDTRIP

// OLD-VERSION: function table entry requires bytecode version 13.5
// ROUNDTRIP: testing$func @f()

cuda_tile.module @m {
  testing$func @f() {
    return
  }
}
