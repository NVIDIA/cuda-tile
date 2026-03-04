add_mlir_library(CudaTileOptimizer
  CudaTileOptimizer.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRTransforms
  CudaTileDialect
  CudaTileTransforms
  CudaTileBytecodeCommon
  CudaTileBytecodeReader
  CudaTileBytecodeWriter

  LINK_COMPONENTS
  Support
)
