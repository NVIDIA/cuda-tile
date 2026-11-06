set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

set(LIBS
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRIR
  CudaTileBytecodeCommon
  CudaTileDialect
  CudaTileTransforms
)

# Append test passes if we're building the tests.
if(CUDA_TILE_ENABLE_TESTING)
  list(APPEND LIBS CudaTileTestPasses)
endif()

add_mlir_tool(cuda-tile-opt
  cuda-tile-opt.cpp

  DEPENDS
  ${LIBS}
)

if(CUDA_TILE_ENABLE_TESTING)
  target_compile_definitions(cuda-tile-opt PUBLIC CUDA_TILE_ENABLE_TESTING)
endif()

target_link_libraries(cuda-tile-opt PRIVATE ${LIBS})
llvm_update_compile_flags(cuda-tile-opt)

