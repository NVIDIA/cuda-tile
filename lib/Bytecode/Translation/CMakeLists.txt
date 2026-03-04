add_mlir_library(CudaTileBytecodeTranslation
  BytecodeTranslation.cpp

  DEPENDS
    CudaTileDialect
    CudaTileBytecodeCommon
    CudaTileBytecodeWriter
    CudaTileBytecodeReader

  LINK_LIBS PUBLIC
    CudaTileBytecodeCommon
    CudaTileBytecodeWriter
    CudaTileBytecodeReader
    CudaTileDialect
    MLIRIR
    MLIRPass
)
