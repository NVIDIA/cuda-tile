// Tests that public-preview types independently trigger preview bytecode
// policy, including when nested inside an ordinary stable type.

// RUN: split-file %s %t
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt,test_only %t/direct.mlir | FileCheck %s --check-prefix=DIRECT
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=test_only %t/nested.mlir | FileCheck %s --check-prefix=NESTED
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %t/symbolic.mlir | FileCheck %s --check-prefix=SYMBOLIC
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %t/fragment.mlir | FileCheck %s --check-prefix=FRAGMENT
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %t/param-placeholders.mlir | FileCheck %s --check-prefix=PARAM-PLACEHOLDERS
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %t/none.mlir | FileCheck %s --check-prefix=NONE
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t/direct.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t/nested.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t/symbolic.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t/fragment.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t/param-placeholders.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 --preview=simt,test_only %t/direct.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-VERSION-CONFLICT
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 --preview=simt %t/symbolic.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-VERSION-CONFLICT
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 --preview=simt %t/fragment.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-VERSION-CONFLICT
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 --preview=simt %t/param-placeholders.mlir 2>&1 | FileCheck %s --check-prefix=WRITE-VERSION-CONFLICT
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt %t/symbolic.mlir -o %t.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.cutile 2>&1 | FileCheck %s --check-prefix=READ-NOFLAG
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t.cutile | FileCheck %s --check-prefix=SYMBOLIC
// A preview type tag the header does not advertise is malformed, and only
// reachable by clearing kContainsPreview (bit 0 of byte 10) in the buffer above.
// RUN: %PYTHON -c "import sys; d = bytearray(open(sys.argv[1], 'rb').read()); d[10] &= 0xFE; open(sys.argv[2], 'wb').write(d)" %t.cutile %t.notag.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t.notag.cutile 2>&1 | FileCheck %s --check-prefix=READ-TYPE-NOTAG
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt %t/fragment.mlir -o %t-fragment.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t-fragment.cutile 2>&1 | FileCheck %s --check-prefix=READ-NOFLAG
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t-fragment.cutile | FileCheck %s --check-prefix=FRAGMENT
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt %t/param-placeholders.mlir -o %t-param-placeholders.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t-param-placeholders.cutile 2>&1 | FileCheck %s --check-prefix=READ-NOFLAG
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t-param-placeholders.cutile | FileCheck %s --check-prefix=PARAM-PLACEHOLDERS

//--- direct.mlir

cuda_tile.module @direct_preview_type_test {
  entry @f() {
    %direct = testing$bytecode_test_preview_type
      : !cuda_tile.preview$bytecode_test_type<i32>
    %string = testing$bytecode_test_preview_type
      : !cuda_tile.preview$string
    return
  }
}

// DIRECT: %{{.*}} = testing$bytecode_test_preview_type : !cuda_tile<preview$bytecode_test_type<i32>>
// DIRECT: %{{.*}} = testing$bytecode_test_preview_type : !cuda_tile<preview$string>

//--- nested.mlir

cuda_tile.module @nested_preview_type_test {
  entry @f() {
    %nested = testing$bytecode_test_preview_type
      : !cuda_tile.bytecode_test_type_container<!cuda_tile.preview$bytecode_test_type<f32>>
    return
  }
}

// NESTED: %{{.*}} = testing$bytecode_test_preview_type
// NESTED-SAME: <!cuda_tile<preview$bytecode_test_type<f32>>>

//--- symbolic.mlir

cuda_tile.module @symbolic_type_test {
  entry @f() {
    %direct = testing$bytecode_test_preview_type
      : !cuda_tile.preview$sym_t<T>
    %symbolic_element = testing$bytecode_test_preview_type
      : !cuda_tile.tile<4x!cuda_tile.preview$sym_t<U>>
    %symbolic_shape = testing$bytecode_test_preview_type
      : !cuda_tile.tile<[M]x8xf32>
    return
  }
}

// SYMBOLIC-LABEL: entry @f()
// SYMBOLIC: %[[VALUE0:.+]] = testing$bytecode_test_preview_type
// SYMBOLIC-SAME: !cuda_tile<preview$sym_t<T>>
// SYMBOLIC: %[[VALUE1:.+]] = testing$bytecode_test_preview_type
// SYMBOLIC-SAME: !cuda_tile.tile<4xpreview$sym_t<U>>
// SYMBOLIC: %[[VALUE2:.+]] = testing$bytecode_test_preview_type
// SYMBOLIC-SAME: !cuda_tile.tile<[M]x8xf32>

//--- fragment.mlir

cuda_tile.module @fragment_type_test {
  entry @f() {
    %fragment = testing$bytecode_test_preview_type
      : !cuda_tile.preview$fragment<2x2xf32, local,
          mem_layout = <(2, 2):(2, 1)>,
          fragmap = <(1, 1, (2, 2)):(0, 0, (1, 2))>>
    return
  }
}

// FRAGMENT-LABEL: entry @f()
// FRAGMENT: %[[VALUE:.+]] = testing$bytecode_test_preview_type
// FRAGMENT-SAME: !cuda_tile<preview$fragment<2x2xf32, local,
// FRAGMENT-SAME: mem_layout = <(2, 2):(2, 1)>,
// FRAGMENT-SAME: fragmap = <(1, 1, (2, 2)):(0, 0, (1, 2))>>

//--- param-placeholders.mlir

cuda_tile.module @param_placeholder_type_test {
  entry @f() {
    %dtype = testing$bytecode_test_preview_type
      : !cuda_tile.preview$Datatype
    %shape = testing$bytecode_test_preview_type
      : !cuda_tile.preview$shape
    return
  }
}

// PARAM-PLACEHOLDERS-LABEL: entry @f()
// PARAM-PLACEHOLDERS: %[[DTYPE:.+]] = testing$bytecode_test_preview_type
// PARAM-PLACEHOLDERS-SAME: !cuda_tile<preview$Datatype>
// PARAM-PLACEHOLDERS: %[[SHAPE:.+]] = testing$bytecode_test_preview_type
// PARAM-PLACEHOLDERS-SAME: !cuda_tile<preview$shape>

// WRITE-NOFLAG: uses preview feature
// WRITE-VERSION-CONFLICT: only valid at the current bytecode version 13.5, but version 13.4 was explicitly requested
// READ-NOFLAG: public-preview bytecode requires at least one enabled preview feature
// READ-TYPE-NOTAG: encountered public-preview type tag {{[0-9]+}} but the header does not advertise preview content

//--- none.mlir
// The builtin `none` type is what an extern_impl constant-parameter wildcard
// (`?`) carries, so it must survive the bytecode round-trip.

cuda_tile.module @wildcard_const_param_test {
  cuda_tile.preview$func extern @gemm<[M: index, N: index]>(
      %x: !cuda_tile.tile<[M]xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>})

  cuda_tile.preview$extern_impl @gemm_impl<[M: index, ?]>
    implements @gemm<[M, 4]>
    (%x: !cuda_tile.preview$fragment<1xf32, local,
         mem_layout = #cuda_tile.preview$layout<(1):(1)>,
         fragmap = #cuda_tile.preview$layout<(M, 1, 1):(1, 0, 0)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
}

// NONE: preview$extern_impl @gemm_impl<[M: index, ?]>
