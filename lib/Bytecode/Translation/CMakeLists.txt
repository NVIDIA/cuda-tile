add_mlir_library(CudaTileBytecodeTranslation
  BytecodeTranslation.cpp

  DEPENDS
    CudaTileDialect
    CudaTileBytecodeWriter
    CudaTileBytecodeReader

  LINK_LIBS PUBLIC
    CudaTileBytecodeWriter
    CudaTileBytecodeReader
    CudaTileDialect
    MLIRIR
    MLIRPass
)
