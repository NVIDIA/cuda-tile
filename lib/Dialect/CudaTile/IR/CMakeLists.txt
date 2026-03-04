set(LLVM_OPTIONAL_SOURCES
  CudaTileTesting.cpp
)

set(CudaTileDialect_SOURCES
  Attributes.cpp
  CudaTile.cpp
  Interfaces.cpp
  Types.cpp
  Traits.cpp
)

set(LLVM_TARGET_DEFINITIONS OpsCanonicalization.td)
mlir_tablegen(OpsCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(CudaTileOpsCanonicalizationIncGen)

if(CUDA_TILE_ENABLE_TESTING)
  list(APPEND CudaTileDialect_SOURCES CudaTileTesting.cpp)
endif()

add_mlir_library(CudaTileDialect
  ${CudaTileDialect_SOURCES}

  DEPENDS
  CudaTileIncGen
  CudaTileOpsCanonicalizationIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
)

target_include_directories(CudaTileDialect
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
