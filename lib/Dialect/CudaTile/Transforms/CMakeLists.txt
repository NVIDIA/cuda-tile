add_mlir_library(CudaTileTransforms
  FuseFMA.cpp
  LICM.cpp
  LoopSplit.cpp
  SynthesizeDebugInfoScopes.cpp
  TransformsUtils.cpp
  DEPENDS
  CudaTileTransformsIncGen

  LINK_LIBS PUBLIC
  CudaTileDialect
  MLIRAnalysis
  MLIRPass
  MLIRTransforms

  LINK_COMPONENTS
  Core
  Support
)
