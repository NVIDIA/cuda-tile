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
  CudaTileDialect
  CudaTileTransforms
  )

add_mlir_tool(cuda-tile-opt
  cuda-tile-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(cuda-tile-opt PRIVATE ${LIBS})
llvm_update_compile_flags(cuda-tile-opt)
