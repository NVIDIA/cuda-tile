# tile_ir/lib/Bytecode/Reader/CMakeLists.txt

add_mlir_library(CudaTileBytecodeReader
  BytecodeReader.cpp

  DEPENDS
    CudaTileBytecodeIncGen
    CudaTileBytecodeReaderIncGen
    CudaTileIncGen

  LINK_LIBS PRIVATE
    CudaTileBytecodeCommon
    CudaTileDialect
    MLIRIR
    MLIRParser
    MLIRSupport

  LINK_COMPONENTS
    Core
    Support
)

if(NOT DEFINED CUDA_TILE_TABLEGEN_EXE)
    message(FATAL_ERROR "CUDA_TILE_TABLEGEN_EXE is not defined")
endif()

# Add invocation for bytecode reader generation
set(LLVM_TARGET_DEFINITIONS ../../../include/cuda_tile/Dialect/CudaTile/IR/Ops.td)

# Use the custom tablegen command with our generator
tablegen(CUDA_TILE BytecodeReader.inc -gen-cuda-tile-bytecode-reader)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
             ${CMAKE_CURRENT_BINARY_DIR}/BytecodeReader.inc)

# Ensure the bytecode include is generated with other targets
add_public_tablegen_target(CudaTileBytecodeReaderIncGen)
# Include local build directory and Writer's directory for StaticOpcodes.inc
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Writer)
