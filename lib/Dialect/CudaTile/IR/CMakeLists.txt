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

if(CUDA_TILE_ENABLE_TESTING)
  list(APPEND CudaTileDialect_SOURCES CudaTileTesting.cpp)
endif()

add_mlir_library(CudaTileDialect
  ${CudaTileDialect_SOURCES}

  DEPENDS
  CudaTileIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
