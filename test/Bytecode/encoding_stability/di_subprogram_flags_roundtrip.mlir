// Encoding-stability / version-gate guard for `DISubprogramAttr`'s `flags`.
//
// `flags` is serialized only from bytecode version 13.5 onward
// (kMinDISubprogramFlagsVersion); the writer and reader both gate on that
// version so older bytecode stays byte-compatible. This test pins that
// contract:
//   * default + 13.5 -- a set flag round-trips verbatim
//   * 13.4 (below the gate) -- a set flag is an error, never dropped. This is
//     also how nonempty arg_attrs behave: the writer rejects both below 13.5
//     instead of silently dropping information.
//   * 13.4 with no flags -- still serializes, and the encoding is unchanged

// RUN: split-file %s %t

// Default round-trip (writes at the current version, which is >= 13.5).
// RUN: %round_trip_test %t/declaration.mlir %t/declaration --mlir-print-debuginfo

// Pin the introduction version so the flag is exercised at the exact gate
// boundary, independent of whatever the default bytecode version happens to be.
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --mlir-print-debuginfo -bytecode-version=13.5 %t/declaration.mlir -o %t.13_5.mlir
// RUN: cuda-tile-opt --no-implicit-module --mlir-print-debuginfo %t/declaration.mlir -o %t.decl_ref.mlir
// RUN: diff %t.13_5.mlir %t.decl_ref.mlir

// Below the gate (13.4) a set flag must be rejected outright.
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %t/declaration.mlir 2>&1 | FileCheck %s --check-prefix=VERSION-ERR
// VERSION-ERR: debug info subprogram 'declared_fn' has flags 'declaration' which require bytecode version 13.5 or newer, but the requested version is 13.4

// Minimum-version analysis reports the same gate, so a caller that asks before
// writing does not run into the error above.
// RUN: cuda-tile-translate -analyze-min-version -no-implicit-module %t/declaration.mlir | FileCheck %s --check-prefix=MIN-DECL
// MIN-DECL: 13.5
// RUN: cuda-tile-translate -analyze-min-version -no-implicit-module %t/no_flags.mlir | FileCheck %s --check-prefix=MIN-PLAIN
// MIN-PLAIN: 13.1

// A subprogram with no flags is unaffected by the gate and still round-trips
// below it, which is what keeps 13.4 bytecode byte-compatible.
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --mlir-print-debuginfo -bytecode-version=13.4 %t/no_flags.mlir -o %t.13_4.mlir
// RUN: cuda-tile-opt --no-implicit-module --mlir-print-debuginfo %t/no_flags.mlir -o %t.plain_ref.mlir
// RUN: diff %t.13_4.mlir %t.plain_ref.mlir

//--- declaration.mlir

cuda_tile.module @kernels {
  entry @declared_fn() {
    return loc(#loc_unknown)
  } loc(#di_loc_decl)
} loc(#loc_unknown)

#di_file = #cuda_tile.di_file<"di_subprogram_flags.mlir" in "foo">
#loc_unknown = loc(unknown)
#loc_fn = loc("di_subprogram_flags.mlir":1:1)
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file>
#di_decl = #cuda_tile.di_subprogram<file = #di_file, line = 1, name = "declared_fn", linkageName = "declared_fn", compileUnit = #di_compile_unit, scopeLine = 1, flags = declaration>
#di_loc_decl = #cuda_tile.di_loc<#loc_fn in #di_decl>

//--- no_flags.mlir

cuda_tile.module @kernels {
  entry @plain_fn() {
    return loc(#loc_unknown)
  } loc(#di_loc_plain)
} loc(#loc_unknown)

#di_file = #cuda_tile.di_file<"di_subprogram_flags.mlir" in "foo">
#loc_unknown = loc(unknown)
#loc_fn = loc("di_subprogram_flags.mlir":1:1)
#di_compile_unit = #cuda_tile.di_compile_unit<file = #di_file>
#di_plain = #cuda_tile.di_subprogram<file = #di_file, line = 1, name = "plain_fn", linkageName = "plain_fn", compileUnit = #di_compile_unit, scopeLine = 1>
#di_loc_plain = #cuda_tile.di_loc<#loc_fn in #di_plain>
