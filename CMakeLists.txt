# ------------------------------------------------------------------------------
# Project Configuration and Requirements
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20.0)

project(CUDA_TILE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/IncludeCompilerChecks.cmake)
include(cmake/IncludeCudaTileUtils.cmake)

set_cuda_tile_build_type()

set(CUDA_TILE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH
   "Path to CUDA Tile source directory")
set(CUDA_TILE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH
   "Path to CUDA Tile binary directory")
set(CUDA_TILE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} CACHE PATH
   "Path to CUDA Tile install directory")
option(CUDA_TILE_ENABLE_TESTING "Enable CUDA Tile testing" OFF)
option(CUDA_TILE_ENABLE_CCACHE "Enable ccache for faster rebuilds" OFF)

message(STATUS "CUDA Tile source directory: ${CUDA_TILE_SOURCE_DIR}")
message(STATUS "CUDA Tile binary directory: ${CUDA_TILE_BINARY_DIR}")
message(STATUS "CUDA Tile install directory: ${CUDA_TILE_INSTALL_DIR}")
message(STATUS "CUDA Tile testing: ${CUDA_TILE_ENABLE_TESTING}")
message(STATUS "CUDA Tile ccache: ${CUDA_TILE_ENABLE_CCACHE}")

# Configure ccache if enabled.
if(CUDA_TILE_ENABLE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(WARNING "ccache enabled but not found in PATH")
  endif()
endif()

# Project-level compilation flags.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# TEMPORARY HACKS!
# ------------------------------------------------------------------------------
add_definitions(-DUSE_12_9_COMPATIBLE_LLVM=0)

# ------------------------------------------------------------------------------
# LLVM/MLIR Configuration
# ------------------------------------------------------------------------------
include(cmake/IncludeLLVM.cmake)

# Check that none or either CUDA_TILE_USE_LLVM_INSTALL_DIR or
# CUDA_TILE_USE_LLVM_SOURCE_DIR is set.
if (DEFINED CUDA_TILE_USE_LLVM_INSTALL_DIR AND
    DEFINED CUDA_TILE_USE_LLVM_SOURCE_DIR)
  message(FATAL_ERROR "Either CUDA_TILE_USE_LLVM_INSTALL_DIR or "
          "CUDA_TILE_USE_LLVM_SOURCE_DIR may be set, but not both")
endif()

# Configure build to use a pre-installed version of LLVM or build it from
# source by downloading them or using the ones provided by the user.
if (CUDA_TILE_USE_LLVM_INSTALL_DIR)
  configure_pre_installed_llvm()
else()
  configure_llvm_from_sources()
endif()

# Search for the MLIR package based on the previous cmake configuration.
message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_CMAKE_DIR} NO_DEFAULT_PATH)

# Configure CUDA Tile and MLIR python bindings. These checks need to happen after
# `find_package(MLIR)` as some MLIR CMake variables can be overridden.
option(CUDA_TILE_ENABLE_BINDINGS_PYTHON "Enable CUDA Tile Python bindings" OFF)
message(STATUS "CUDA Tile Python bindings: ${CUDA_TILE_ENABLE_BINDINGS_PYTHON}")
if (CUDA_TILE_ENABLE_BINDINGS_PYTHON)
  message(STATUS "MLIR python bindings: ${MLIR_ENABLE_BINDINGS_PYTHON}")
  # If this check fails and `MLIR_ENABLE_BINDINGS_PYTHON` was set to `ON` before the
  # `find_package(MLIR)` above, make sure that `find_package(MLIR)` is picking up
  # the correct MLIR project and not the one from an unexpected location.
  if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
    message(FATAL_ERROR "CUDA Tile IR Python bindings require MLIR Python bindings enabled")
  endif()

  # Configure MLIR python dev packages.
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

# Components needed from LLVM and MLIR.

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Include directories for LLVM and MLIR.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Create a cross-compilation target for CUDA Tile. This is mostly required by
# cuda-tile-tblgen as we need to use the host's tablegen executable.
if(CMAKE_CROSSCOMPILING)
  include(CrossCompile)
  set(LLVM_USE_HOST_TOOLS ON)

  if (NOT DEFINED CUDA_TILE_USE_NATIVE_LLVM_INSTALL_DIR)
    message(FATAL_ERROR "CUDA_TILE_USE_NATIVE_LLVM_INSTALL_DIR is not defined")
  endif()

  llvm_create_cross_target(CUDA_TILE NATIVE "" Release
    -DCUDA_TILE_USE_LLVM_INSTALL_DIR=${CUDA_TILE_USE_NATIVE_LLVM_INSTALL_DIR}
    -DCUDA_TILE_ENABLE_TESTING=OFF
    -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=OFF
    -DCMAKE_C_COMPILER=${NATIVE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${NATIVE_CXX_COMPILER}
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
  )
endif()

print_llvm_config()

#-------------------------------------------------------------------------------
# CUDA Tile Configuration
#-------------------------------------------------------------------------------

# Include directories for CUDA Tile IR.
include_directories(${CUDA_TILE_SOURCE_DIR}/include)
include_directories(${CUDA_TILE_BINARY_DIR}/include)

# Place generated executables (and libraries) in bin (or lib), instead of
# tools/<tool_name>/.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CUDA_TILE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CUDA_TILE_BINARY_DIR}/lib)

# Used by lit test configuration
set(CUDA_TILE_TOOL_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
set(CUDA_TILE_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})

#-------------------------------------------------------------------------------
# CUDA Tile Subdirectories
#-------------------------------------------------------------------------------

# 'cuda-tile-tblgen' is part of the build process and must be built before
# anything else.
add_subdirectory(tools/cuda-tile-tblgen)

# The CMake in 'tools/cuda-tile-tblgen' defines CUDA_TILE_TABLEGEN_EXE to a
# default 'cuda-tile-tblgen'. We need to include the full path of the host
# executable.
if(CMAKE_CROSSCOMPILING)
  get_native_tool_path(cuda-tile-tblgen CUDA_TILE_TABLEGEN_EXE)
  set(MLIR_TABLEGEN_EXE "${CUDA_TILE_USE_NATIVE_LLVM_INSTALL_DIR}/bin/${MLIR_TABLEGEN_EXE}")
else()
  set(CUDA_TILE_TABLEGEN_EXE ${CUDA_TILE_BINARY_DIR}/bin/cuda-tile-tblgen)
endif()

message(STATUS "CUDA_TILE_TABLEGEN_EXE: ${CUDA_TILE_TABLEGEN_EXE}")

# Pass testing flag to both TableGen and C++ compilation for conditional operation generation.
if (CUDA_TILE_ENABLE_TESTING)
  list(APPEND LLVM_TABLEGEN_FLAGS -DTILE_IR_INCLUDE_TESTS)
  add_compile_definitions(TILE_IR_INCLUDE_TESTS=1)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
if (CUDA_TILE_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if(CUDA_TILE_ENABLE_TESTING)
  add_subdirectory(test)
endif()

# ------------------------------------------------------------------------------
# Install CUDA Tile headers
#-------------------------------------------------------------------------------

# TODO: Use custom targets to install the headers instead of installing the
#       whole directory?
install(DIRECTORY ${CUDA_TILE_SOURCE_DIR}/include
        DESTINATION ${CUDA_TILE_INSTALL_DIR}/include
        FILES_MATCHING
          PATTERN "*.def"
          PATTERN "*.h"
          PATTERN "*.inc"
          PATTERN "*.td")

