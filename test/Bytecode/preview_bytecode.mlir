// Tests for public-preview bytecode gating via --preview=<feature>.
//
// Preview ops/types/attrs are intentionally non-portable: they are gated behind
// an explicit opt-in on both the writer and the reader, are version-locked to
// the current bytecode version, and are advertised in the header's version tag.

// Round-trip succeeds when preview is enabled on both ends.
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt %s | FileCheck %s --check-prefix=ROUNDTRIP

// Writing preview content without the opt-in is a hard error.
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG

// Explicitly requesting a non-current version for preview content is a conflict.
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 --preview=simt %s 2>&1 | FileCheck %s --check-prefix=WRITE-VERSION-CONFLICT

// Reading preview bytecode requires the opt-in: emit with the flag, then read
// back both without (error) and with (success) the flag.
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt %s -o %t.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.cutile 2>&1 | FileCheck %s --check-prefix=READ-NOFLAG
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t.cutile | FileCheck %s --check-prefix=READ-OK
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=test_only %t.cutile 2>&1 | FileCheck %s --check-prefix=READ-WRONG-FEATURE

// The header must agree with the payload. The writer cannot disagree, so these
// patch the buffer above: byte 9 is the minor version, byte 10 holds
// kContainsPreview in bit 0.
// RUN: %PYTHON -c "import sys; d = bytearray(open(sys.argv[1], 'rb').read()); d[10] &= 0xFE; open(sys.argv[2], 'wb').write(d)" %t.cutile %t.notag.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t.notag.cutile 2>&1 | FileCheck %s --check-prefix=READ-OPCODE-NOTAG

// Preview is version-locked, so an older minor is refused even with the tag set.
// RUN: %PYTHON -c "import sys; d = bytearray(open(sys.argv[1], 'rb').read()); d[9] = 4; open(sys.argv[2], 'wb').write(d)" %t.cutile %t.oldversion.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt %t.oldversion.cutile 2>&1 | FileCheck %s --check-prefix=READ-OLD-VERSION

// Textual optimizer entry points apply the same gate.
// RUN: not cuda-tile-opt -no-implicit-module %s -o %t.mlir 2>&1 | FileCheck %s --check-prefix=TEXT-NOFLAG
// RUN: cuda-tile-opt -no-implicit-module --preview=simt %s -o %t.mlir
// RUN: not cuda-tile-optimize --opt-level=0 --quiet %s 2>&1 | FileCheck %s --check-prefix=TEXT-NOFLAG
// RUN: cuda-tile-optimize --opt-level=0 --quiet --preview=simt %s

// Unknown feature names are rejected at option parsing time.
// RUN: not cuda-tile-translate --preview=NotAFeature -mlir-to-cudatilebc %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN-FEATURE

cuda_tile.module @preview_test {
  entry @f() {
    preview$bytecode_test_op req = 7 opt = 9
    return
  }
}

// ROUNDTRIP: preview$bytecode_test_op req = 7 opt = 9

// WRITE-NOFLAG: uses preview feature

// WRITE-VERSION-CONFLICT: only valid at the current bytecode version 13.5, but version 13.4 was explicitly requested

// READ-NOFLAG: public-preview bytecode requires at least one enabled preview feature
// READ-WRONG-FEATURE: uses preview feature
// TEXT-NOFLAG: uses preview feature

// READ-OPCODE-NOTAG: encountered public-preview opcode {{[0-9]+}} but the header does not advertise preview content
// READ-OLD-VERSION: public-preview bytecode at version 13.4{{.*}} is not readable

// READ-OK: preview$bytecode_test_op req = 7 opt = 9

// UNKNOWN-FEATURE: unknown preview feature 'NotAFeature'
