
add_mlir_library(CudaTileBytecodeCommon
  Version.cpp

  DEPENDS
    CudaTileBytecodeIncGen
    CudaTileIncGen

  LINK_LIBS PUBLIC
    CudaTileDialect
    MLIRIR
    MLIRSupport

  LINK_COMPONENTS
    Support
)

# Ensure testing flag is available for version support.
if(CUDA_TILE_ENABLE_TESTING)
  target_compile_definitions(CudaTileBytecodeCommon PRIVATE TILE_IR_INCLUDE_TESTS=1)
endif()

# Include Writer's build directory to access shared StaticOpcodes.inc
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Writer)
