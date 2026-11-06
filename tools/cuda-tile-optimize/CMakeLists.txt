set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

# Collect dependencies
set(LIBS
  MLIRSupport
  CudaTileOptimizer
)

add_mlir_tool(cuda-tile-optimize
  cuda-tile-optimize.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cuda-tile-optimize PRIVATE
  ${LIBS}
  LLVMSupport
)
llvm_update_compile_flags(cuda-tile-optimize)

# Set the date for the version information.
# Cache generated timestamps so Ninja's CMake regeneration check reaches a
# fixed point instead of rewriting build.ninja on every pass.
if (NOT DEFINED STD_DATE OR STD_DATE STREQUAL "")
  string(TIMESTAMP STD_DATE "%a_%b_%d_%H:%M:%S_UTC_%Y" UTC)
  set(STD_DATE "${STD_DATE}" CACHE STRING
      "Build timestamp embedded in tool version information" FORCE)
endif()
target_compile_definitions(cuda-tile-optimize PRIVATE -DSTD_DATE="${STD_DATE}")

# Add the tool versions for the version information if present.
if (TOOLS_VERSION)
  target_compile_definitions(cuda-tile-optimize PRIVATE -DTOOLS_VERSION="${TOOLS_VERSION}")
endif()
if (TOOLS_VERSION_EXTENDED)
  target_compile_definitions(cuda-tile-optimize PRIVATE -DTOOLS_VERSION_EXTENDED="${TOOLS_VERSION_EXTENDED}")
endif()
