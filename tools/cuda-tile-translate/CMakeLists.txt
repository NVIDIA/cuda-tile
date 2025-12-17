set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

# Collect dependencies
set(LIBS
  MLIRDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTranslateLib
  CudaTileDialect
  CudaTileBytecodeTranslation
)

add_mlir_tool(cuda-tile-translate
  cuda-tile-translate.cpp
  test/RoundTripTestRegistration.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cuda-tile-translate PRIVATE
  ${LIBS}
  LLVMSupport
)
llvm_update_compile_flags(cuda-tile-translate) 
