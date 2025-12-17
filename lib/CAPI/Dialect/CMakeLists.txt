add_mlir_public_c_api_library(CudaTileCAPIDialects
  CudaTileDialect.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR  
  CudaTileDialect
  CudaTileTransforms
  CudaTileBytecodeWriter
  CudaTileBytecodeCommon
)
