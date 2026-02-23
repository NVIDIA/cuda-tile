find_package(Python3 REQUIRED)

set(LLVM_TOOLS_TO_INSTALL FileCheck;not)

macro(print_llvm_config)
  message(STATUS "Summary of the LLVM/MLIR CMake environment:")

  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  message(STATUS "LLVM_ENABLE_ASSERTIONS: ${LLVM_ENABLE_ASSERTIONS}")
  message(STATUS "LLVM_ENABLE_RTTI: ${LLVM_ENABLE_RTTI}")
  message(STATUS "LLVM_CONFIG_HAS_RTTI: ${LLVM_CONFIG_HAS_RTTI}")
  message(STATUS "LLVM_ENABLE_EH: ${LLVM_ENABLE_EH}")
  message(STATUS "LLVM_SOURCE_DIR: ${LLVM_SOURCE_DIR}")
  message(STATUS "LLVM_BINARY_DIR: ${LLVM_BINARY_DIR}")
  message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
  message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
  message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
  message(STATUS "MLIR_ENABLE_BINDINGS_PYTHON: ${MLIR_ENABLE_BINDINGS_PYTHON}")
  message(STATUS "MLIR_ENABLE_EXECUTION_ENGINE: ${MLIR_ENABLE_EXECUTION_ENGINE}")
  message(STATUS "LLVM_LIT: ${LLVM_LIT}")
  message(STATUS "LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endmacro()

macro(download_llvm_sources)
  include(FetchContent)

  set(LLVM_GIT_REPO "https://github.com/llvm/llvm-project.git")
  set(LLVM_BUILD_COMMIT_HASH b82c7fc65229c8b2b6a964f023f6ec59b3cf9210)
  message(STATUS "Downloading LLVM sources from ${LLVM_GIT_REPO}@${LLVM_BUILD_COMMIT_HASH} to ${LLVM_SOURCE_DIR}")

  # Set FetchContent directories. SOURCE_DIR and BINARY_DIR and SUBBUILD_DIR
  # are relative to FETCHCONTENT_BASE_DIR and it looks like they can't be
  # nested.
  set(FETCHCONTENT_BASE_DIR ${CUDA_TILE_BINARY_DIR})
  set(FETCHCONTENT_SOURCE_DIR ${LLVM_PROJECT_NAME})
  set(FETCHCONTENT_BINARY_DIR ${LLVM_PROJECT_BUILD_FOLDER_NAME})
  set(FETCHCONTENT_SUBBUILD_DIR ${LLVM_PROJECT_NAME}-subbuild)
  set(FETCHCONTENT_QUIET FALSE)

  fetchContent_Declare(
    ${LLVM_PROJECT_NAME}
    GIT_REPOSITORY ${LLVM_GIT_REPO}
    GIT_TAG ${LLVM_BUILD_COMMIT_HASH}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${FETCHCONTENT_SOURCE_DIR}
    BINARY_DIR ${FETCHCONTENT_BINARY_DIR}
    SUBBUILD_DIR ${FETCHCONTENT_SUBBUILD_DIR}
  )

  fetchContent_MakeAvailable(${LLVM_PROJECT_NAME})
endmacro()

# -----------------------------------------------------------------------------
# Configure build to download and build LLVM sources.
# -----------------------------------------------------------------------------
macro(configure_llvm_from_sources)
  if (CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Cross-compilation is not supported when building LLVM from sources")
  endif()

  # Set up LLVM sources.
  set(LLVM_PROJECT_NAME "llvm-project")
  set(LLVM_PROJECT_BUILD_FOLDER_NAME "${LLVM_PROJECT_NAME}-build")
  set(LLVM_BINARY_DIR ${CUDA_TILE_BINARY_DIR}/${LLVM_PROJECT_BUILD_FOLDER_NAME})

  if (CUDA_TILE_USE_LLVM_SOURCE_DIR)
    message(STATUS "Building LLVM from sources provided at ${CUDA_TILE_USE_LLVM_SOURCE_DIR}")
    set(LLVM_SOURCE_DIR ${CUDA_TILE_USE_LLVM_SOURCE_DIR})
  else()
    message(STATUS "Building LLVM from sources")
    download_llvm_sources()
    set(LLVM_SOURCE_DIR ${CUDA_TILE_BINARY_DIR}/${FETCHCONTENT_SOURCE_DIR})
  endif()

  # Set LLVM cmake options.
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
  set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
  set(LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "")
  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
  set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "")
  # Disable PCH to avoid flag mismatches between LLVM and CUDA Tile targets
  # (e.g. NDEBUG differences).
  set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)
  set(LLVM_BUILD_UTILS ON CACHE BOOL "")
  set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

  # Propagate ccache setting to LLVM build.
  if(CUDA_TILE_ENABLE_CCACHE)
    set(LLVM_CCACHE_BUILD ON CACHE BOOL "")
  endif()

  # Set MLIR cmake options.
  set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "")
  set(MLIR_ENABLE_BINDINGS_PYTHON ${CUDA_TILE_ENABLE_BINDINGS_PYTHON} CACHE BOOL "")

  # Trigger the CMake configuration of LLVM and MLIR.
  list(APPEND CMAKE_MESSAGE_INDENT "[LLVM] -- ")
  add_subdirectory(${LLVM_SOURCE_DIR}/llvm ${LLVM_BINARY_DIR} EXCLUDE_FROM_ALL)
  list(POP_BACK CMAKE_MESSAGE_INDENT)

  if (CUDA_TILE_ENABLE_TESTING)
    # Ensure FileCheck and not are always built even with EXCLUDE_FROM_ALL.
    # These tools are required for testing.
    foreach(_TOOL_NAME ${LLVM_TOOLS_TO_INSTALL})
      add_custom_target(llvm-test-tool-${_TOOL_NAME} ALL DEPENDS ${_TOOL_NAME})

      # Install LLVM tools to third_party/llvm/bin.
      # Use install(TARGETS) since these are CMake targets built via add_subdirectory.
      # This correctly resolves output paths across all platforms and generators.
      install(TARGETS ${_TOOL_NAME}
        RUNTIME DESTINATION third_party/llvm/bin
      )
    endforeach()
  endif()

  set(LLVM_CMAKE_DIR "${LLVM_BINARY_DIR}/lib/cmake/llvm")
  set(LLVM_DIR "${LLVM_CMAKE_DIR}")
  # It looks like MLIR picks up the cmake directory from the main project's
  # build directory and not from the same directory LLVM does so we need to
  # set it differently here. We may want to fix that upstream.
  set(MLIR_CMAKE_DIR "${CUDA_TILE_BINARY_DIR}/lib/cmake/mlir")
  set(MLIR_DIR "${MLIR_CMAKE_DIR}")

endmacro()

# --------------------------------------------------------------
# Configure build to use pre-installed LLVM and sub-projects.
# `CUDA_TILE_USE_LLVM_INSTALL_DIR` must be set.
# --------------------------------------------------------------
macro(configure_pre_installed_llvm)
  message(STATUS "Using pre-installed version of LLVM at ${CUDA_TILE_USE_LLVM_INSTALL_DIR}")

  if (CUDA_TILE_ENABLE_TESTING)
    message(STATUS "Using external lit tool at '${LLVM_EXTERNAL_LIT}'")
    if (NOT DEFINED LLVM_EXTERNAL_LIT)
      message(FATAL_ERROR "LLVM_EXTERNAL_LIT must be set when build CUDA Tile with"
              " a pre-built version of LLVM and CUDA_TILE_ENABLE_TESTING is enabled")
    endif()
  endif()


  # Install LLVM tools to third_party/llvm/bin.
  if (CUDA_TILE_ENABLE_TESTING)
    foreach(_TOOL_NAME ${LLVM_TOOLS_TO_INSTALL})
      install(
        PROGRAMS ${CUDA_TILE_USE_LLVM_INSTALL_DIR}/bin/${_TOOL_NAME}${CMAKE_EXECUTABLE_SUFFIX}
          DESTINATION third_party/llvm/bin
        )
    endforeach()
  endif()

  set(LLVM_CMAKE_DIR ${CUDA_TILE_USE_LLVM_INSTALL_DIR}/lib/cmake/llvm)
  set(LLVM_DIR "${LLVM_CMAKE_DIR}")
  set(MLIR_CMAKE_DIR ${CUDA_TILE_USE_LLVM_INSTALL_DIR}/lib/cmake/mlir)
  set(MLIR_DIR "${MLIR_CMAKE_DIR}")

  link_directories( ${LLVM_LIBRARY_DIRS} )
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions( ${LLVM_DEFINITIONS_LIST} )
endmacro()
