# Override setting to build tile-opt by default with `add_mlir_tool`
set(LLVM_BUILD_TOOLS ON)
add_subdirectory(cuda-tile-opt)
add_subdirectory(cuda-tile-translate)
add_subdirectory(cuda-tile-optimize)

install(TARGETS cuda-tile-opt cuda-tile-optimize cuda-tile-translate cuda-tile-tblgen RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
