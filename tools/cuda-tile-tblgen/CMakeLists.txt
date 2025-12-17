set(LLVM_LINK_COMPONENTS
  TableGen
)

add_tablegen(cuda-tile-tblgen CUDA_TILE
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT CUDA_TILE
  cuda-tile-tblgen.cpp
  BytecodeGen.cpp
  SpecGen.cpp
  Emitter.cpp
  CudaTileOp.cpp
  CudaTileType.cpp
  CudaTileAttr.cpp
  BytecodeReaderGen.cpp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Using GCC
    set(CUDA_TILE_TBLGEN_COMPILE_OPTS "-Wno-global-constructors")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Using Clang
    set(CUDA_TILE_TBLGEN_COMPILE_OPTS "-Wno-global-constructors")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Using Microsoft Visual Studio Compiler
    set(CUDA_TILE_TBLGEN_COMPILE_OPTS "")
endif()

target_compile_options(cuda-tile-tblgen PRIVATE ${CUDA_TILE_TBLGEN_COMPILE_OPTS})

target_link_libraries(cuda-tile-tblgen
  PRIVATE
  MLIRIR
  MLIRSupport
  MLIRTblgenLib
)

mlir_check_all_link_libraries(cuda-tile-tblgen)
