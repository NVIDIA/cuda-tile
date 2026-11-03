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

# Helper macro to generate bytecode includes from TableGen definitions.
macro(cuda_tile_tablegen output_file input_td gen_flag)
  set(LLVM_TARGET_DEFINITIONS ${input_td})
  tablegen_internal(CUDA_TILE ${output_file} ${gen_flag})
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
               ${CMAKE_CURRENT_BINARY_DIR}/${output_file})
endmacro()

set(TD_DIR ${CUDA_TILE_SOURCE_DIR}/include/cuda_tile/Dialect/CudaTile/IR)
cuda_tile_tablegen(Bytecode.inc ${TD_DIR}/Ops.td -gen-cuda-tile-bytecode)
cuda_tile_tablegen(StaticOpcodes.inc ${TD_DIR}/BytecodeOpcodes.td -gen-cuda-tile-opcodes)
cuda_tile_tablegen(TypeBytecode.inc ${TD_DIR}/BytecodeTypeOpcodes.td -gen-cuda-tile-type-bytecode)
cuda_tile_tablegen(AttrBytecode.inc ${TD_DIR}/BytecodeAttrOpcodes.td -gen-cuda-tile-attr-bytecode)

# Ensure the bytecode include is generated with other targets
add_public_tablegen_target(CudaTileBytecodeIncGen)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
