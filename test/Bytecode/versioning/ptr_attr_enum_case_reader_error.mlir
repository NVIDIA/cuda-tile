// Test reader rejection of an enum case that postdates the file's version.
//
// This bytecode file claims to be version 13.4 but carries the pointer
// classification `unicast`, a case added in 13.5. A reader must reject this
// unsupported case for a 13.4 file rather than accept an invalid encoding.
//
// COM: The file was produced by writing the module below at 13.4 with
// COM: `ptr_attr<none>` and flipping the classification byte from 0 to 1, which
// COM: is the buffer a writer without the version gate would have emitted.
// COM:
// COM: cuda_tile.module @kernels {
// COM:   entry @kernel(%ptr: tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>) {
// COM:     return
// COM:   }
// COM: }

// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.4/unicast-ptr-attr-in-13.4.tileirbc 2>&1 | FileCheck %s

// CHECK: enum value 1 is not available in bytecode version 13.4
