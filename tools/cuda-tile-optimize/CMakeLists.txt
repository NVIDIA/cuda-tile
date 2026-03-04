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
if (NOT STD_DATE)
  string(TIMESTAMP STD_DATE "%a_%b_%d_%H:%M:%S_UTC_%Y" UTC)
endif()
target_compile_definitions(cuda-tile-optimize PRIVATE -DSTD_DATE="${STD_DATE}")

# Add the tool versions for the version information if present.
if (TOOLS_VERSION)
  target_compile_definitions(cuda-tile-optimize PRIVATE -DTOOLS_VERSION="${TOOLS_VERSION}")
endif()
if (TOOLS_VERSION_EXTENDED)
  target_compile_definitions(cuda-tile-optimize PRIVATE -DTOOLS_VERSION_EXTENDED="${TOOLS_VERSION_EXTENDED}")
endif()

