add_mlir_library(CudaTileTransforms
  FuseFMA.cpp
  LoopSplit.cpp
  SynthesizeDebugInfoScopes.cpp

  DEPENDS
  CudaTileTransformsIncGen

  LINK_LIBS PUBLIC
  CudaTileDialect
  MLIRTransforms

  LINK_COMPONENTS
  Support
)
