// RUN: FileCheck %s --check-prefix=OPCODE \
// RUN:   --input-file=%S/../../include/cuda_tile/Dialect/CudaTile/IR/BytecodeOpcodes.td
// RUN: FileCheck %s --check-prefix=TYPE \
// RUN:   --input-file=%S/../../include/cuda_tile/Dialect/CudaTile/IR/BytecodeTypeOpcodes.td
// RUN: FileCheck %s --check-prefix=ATTR \
// RUN:   --input-file=%S/../../include/cuda_tile/Dialect/CudaTile/IR/BytecodeAttrOpcodes.td
// RUN: FileCheck %s --check-prefix=FEATURE \
// RUN:   --input-file=%S/../../include/cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.td
// RUN: FileCheck %s --check-prefix=FLAG \
// RUN:   --input-file=%S/../../lib/Bytecode/BytecodeEnums.h

// Pin the canonical numeric preview ABI assignments.

// OPCODE-DAG: def : PreviewOpcode<CudaTile_CallOp, 0x4002>;
// OPCODE-DAG: def : PreviewOpcode<CudaTile_FuncOp, 0x4003>;
// OPCODE-DAG: def : PreviewOpcode<CudaTile_ExternImplOp, 0x4004>;

// TYPE-DAG: def : PreviewCudaTileTypeTag<"SymbolicType", 0xD1>;
// TYPE-DAG: def : PreviewCudaTileTypeTag<"FragmentType", 0xD2>;
// TYPE-DAG: def : PreviewCudaTileTypeTag<"DTypeParamType", 0xD3>;
// TYPE-DAG: def : PreviewCudaTileTypeTag<"ShapeParamType", 0xD4>;
// TYPE-DAG: def : PreviewCudaTileTypeTag<"StringParamType", 0xD5>;
// TYPE-DAG: "IndexType", "::mlir::IndexType", 0xD6,

// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprInt", 0xD1>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprVar", 0xD2>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprMul", 0xD3>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprDiv", 0xD4>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprAdd", 0xD5>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprSub", 0xD6>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprFloorDiv", 0xD7>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicTuple", 0xD8>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"Layout", 0xD9>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprCeilDiv", 0xDA>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"ConcreteType", 0xDB>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprNeg", 0xDC>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"MemorySpace", 0xDD>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"Synchronization", 0xDE>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"Access", 0xDF>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"ScratchRequest", 0xE0>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"ConstParam", 0xE2>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprBuiltin", 0xE3>;
// ATTR-DAG: def : PreviewBytecodeAttrTag<"SymbolicExprConstStr", 0xE4>;

// FEATURE-DAG: : CudaTilePreviewFeature<"SIMT", 1, "simt">;

// FLAG-DAG: FormMask = 0x0C,
// FLAG-DAG: FormNormal = 0x00,
// FLAG-DAG: FormExternImpl = 0x04,
