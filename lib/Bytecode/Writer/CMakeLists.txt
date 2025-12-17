# tile_ir/lib/Bytecode/Writer/CMakeLists.txt

add_mlir_library(CudaTileBytecodeWriter
  BytecodeWriter.cpp

  DEPENDS
    CudaTileBytecodeIncGen
    CudaTileIncGen

  LINK_LIBS PUBLIC
    CudaTileBytecodeCommon
    CudaTileDialect
    MLIRIR
    MLIRSupport

  LINK_COMPONENTS
    Support
)

if(NOT DEFINED CUDA_TILE_TABLEGEN_EXE)
    message(FATAL_ERROR "CUDA_TILE_TABLEGEN_EXE is not defined")
endif()

# Generate bytecode writer functions from operations
set(LLVM_TARGET_DEFINITIONS ../../../include/cuda_tile/Dialect/CudaTile/IR/Ops.td)
tablegen(CUDA_TILE Bytecode.inc -gen-cuda-tile-bytecode)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
             ${CMAKE_CURRENT_BINARY_DIR}/Bytecode.inc)

# Generate opcode definitions from bytecode opcodes
set(LLVM_TARGET_DEFINITIONS ../../../include/cuda_tile/Dialect/CudaTile/IR/BytecodeOpcodes.td)
tablegen(CUDA_TILE StaticOpcodes.inc -gen-cuda-tile-opcodes)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
             ${CMAKE_CURRENT_BINARY_DIR}/StaticOpcodes.inc)

# Ensure the bytecode include is generated with other targets
add_public_tablegen_target(CudaTileBytecodeIncGen)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
