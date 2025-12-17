include(AddMLIRPython)

# For sanitizer enabled clang build, need to enable build pybind DSOs with shared library of sanitizers
# Read more at https://github.com/google/sanitizers/issues/1086
if(CUDA_TILE_ENABLE_SANITIZER AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options("-shared-libasan")
endif()

# Specifies that all MLIR packages are co-located under the `cuda_tile`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=cuda_tile._mlir.")

set(MLIR_PYBIND_PKG_ROOT ${LLVM_ROOT}/python_packages/mlir_core)


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CudaTilePythonSources)

declare_mlir_python_sources(CudaTilePythonSources.Dialects
  ADD_TO_PARENT CudaTilePythonSources
)

################################################################################
#  CudaTile Dialect Python Bindings
################################################################################

# Build a PyBind11 module with special name for auto registration during python site initialization.
# Read more at https://github.com/llvm/llvm-project/blob/main/mlir/python/mlir/_mlir_libs/__init__.py#L41

declare_mlir_python_extension(CudaTilePythonSources.SiteInitializer.PyBind
  # Note: It appears this module name is not important but we happen to use the same name for all other Python
  # bindings. Name is unchanged in order to minimize the disruption to cuTile and other users.
  MODULE_NAME   _site_initialize_1
  ADD_TO_PARENT CudaTilePythonSources
  PYTHON_BINDINGS_LIBRARY pybind11
  SOURCES
    SiteInitializer.cpp
  EMBED_CAPI_LINK_LIBS
    CudaTileCAPIRegistration
  PRIVATE_LINK_LIBS
    LLVMSupport
)
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CudaTilePythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda_tile"
  TD_FILE dialects/CudaTileOps.td
  SOURCES
    dialects/cuda_tile_ops.py
    dialects/cuda_tile_utils.py
  DIALECT_NAME cuda_tile
  GEN_ENUM_BINDINGS_TD_FILE
    "../../include/cuda_tile/Dialect/CudaTile/IR/Types.td"
)

declare_mlir_python_extension(CudaTilePythonSources.Dialects.cuda_tile.Pybind
  MODULE_NAME _cuda_tile
  ADD_TO_PARENT CudaTilePythonSources.Dialects.cuda_tile
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY pybind11
  SOURCES
    Dialect/DialectCudaTile.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    CudaTileCAPIDialects # CAPI functions via embed (controlled exports)
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(CudaTilePythonCAPI
  INSTALL_COMPONENT     CudaTilePythonModules
  INSTALL_DESTINATION   python_packages/cuda_tile/_mlir/_mlir_libs
  OUTPUT_DIRECTORY      "${CUDA_TILE_BINARY_DIR}/python_packages/cuda_tile/_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    MLIRPythonExtension.Core
    CudaTilePythonSources
  EMBED_LIBS
    CudaTileCAPIDialects
)

################################################################################
# Instantiation of all Python modules
################################################################################

## NOTE: we need to either change the above RELATIVE_INSTALL_ROOT or the INSTALL_PREFIX or this path to
## make things work, before the RPATH was being set to $ORIGIN which means that it would only search paths relative
## to the parent directory of the shared library. We map them to this path so that they are in the same directory
## as the shared library and thus found by the dynamic loader.
add_mlir_python_modules(CudaTilePythonModules
  ROOT_PREFIX    "${CUDA_TILE_BINARY_DIR}/python_packages/cuda_tile/_mlir"
  INSTALL_PREFIX "python_packages/cuda_tile/_mlir"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CudaTilePythonSources
  COMMON_CAPI_LINK_LIBS
    CudaTilePythonCAPI
    # FIXME: This is a WAR for PDL rewrite CAPIs
    MLIRCAPITransforms
  )

# Create debug links for Windows Debug builds (only on Windows Debug)
if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  include(../cmake/WindowsPythonDebugUtils.cmake)

  add_windows_debug_links_installation(
    CudaTilePythonModules
    "${CUDA_TILE_BINARY_DIR}/python_packages/cuda_tile/_mlir/_mlir_libs"
    "${CMAKE_INSTALL_PREFIX}/python_packages/cuda_tile/_mlir/_mlir_libs"
    CudaTilePythonModules
  )
endif()

