add_mlir_public_c_api_library(CudaTileCAPIRegistration
  Registration.cpp

  LINK_LIBS PUBLIC
  CudaTileDialect
  CudaTileTransforms
  MLIRCAPIIR
)

add_subdirectory(Dialect)
