#include "cuda_tile-c/Registration.h"
#include <stdio.h>

// RUN: test-cuda-tile-capi-register

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();

  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirCudaTileRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);

  MlirDialect cudaTile = mlirContextGetOrLoadDialect(
      ctx, mlirStringRefCreateFromCString("cuda_tile"));

  if (mlirDialectIsNull(cudaTile)) {
    fprintf(stderr, "failed to load cuda_tile dialect!");
    return -1;
  }

  return 0;
}
