### TableGen generation of the .h, and .cpp files.
set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=cuda_tile)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=cuda_tile)

set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=cuda_tile)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=cuda_tile)
mlir_tablegen(TypeConstraints.h.inc -gen-type-constraint-decls -dialect=cuda_tile)
mlir_tablegen(TypeConstraints.cpp.inc -gen-type-constraint-defs -dialect=cuda_tile)

set(LLVM_TARGET_DEFINITIONS AttrDefs.td)
mlir_tablegen(AttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=cuda_tile)
mlir_tablegen(AttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=cuda_tile)
mlir_tablegen(Enums.h.inc -gen-enum-decls -dialect=cuda_tile)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs -dialect=cuda_tile)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_public_tablegen_target(CudaTileIncGen)
add_dependencies(mlir-headers CudaTileIncGen)

add_dependencies(mlir-headers cuda-tile-tblgen)
# TableGen Markdown documentation generation.
#####################################################################
# Note: MLIR generated documentation use Hugo, so we enable Hugo specific features
# here.
set(LLVM_TARGET_DEFINITIONS Dialect.td)
tablegen(MLIR CudaTile.md -gen-dialect-doc -allow-hugo-specific-features)
set(LLVM_TARGET_DEFINITIONS Ops.td)
tablegen(MLIR CudaTileOps.md -gen-op-doc -allow-hugo-specific-features)
set(LLVM_TARGET_DEFINITIONS Types.td)
tablegen(MLIR CudaTileTypes.md -gen-typedef-doc -allow-hugo-specific-features)

set(GEN_DOC_DIR ${CMAKE_SOURCE_DIR}/media/docs/Dialects/CudaTile)
file(MAKE_DIRECTORY ${GEN_DOC_DIR})

add_custom_command(
	OUTPUT ${GEN_DOC_DIR}/CudaTile.md
        COMMAND ${CMAKE_COMMAND} -E cat
        ${CMAKE_CURRENT_BINARY_DIR}/CudaTile.md
        ${CMAKE_CURRENT_BINARY_DIR}/CudaTileOps.md
        ${CMAKE_CURRENT_BINARY_DIR}/CudaTileTypes.md
        > ${GEN_DOC_DIR}/CudaTile.md
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/CudaTile.md
        ${CMAKE_CURRENT_BINARY_DIR}/CudaTileOps.md
        ${CMAKE_CURRENT_BINARY_DIR}/CudaTileTypes.md)

# TableGen generation of the CUDA Tile IR specification.

if(NOT DEFINED CUDA_TILE_TABLEGEN_EXE)
    message(FATAL_ERROR "CUDA_TILE_TABLEGEN_EXE is not defined and is required to generate the specification.")
endif()


set(GEN_SPEC_DIR ${CMAKE_BINARY_DIR}/docs/spec)
file(MAKE_DIRECTORY ${GEN_SPEC_DIR})
file(MAKE_DIRECTORY ${GEN_SPEC_DIR}/examples)

# Generate the specification for it.
set(LLVM_TARGET_DEFINITIONS Dialect.td)
set(LLVM_TARGET_DEFINITIONS Ops.td)
tablegen(CUDA_TILE ops.rst -gen-op-spec -examples-directory ${GEN_SPEC_DIR}/examples)

add_custom_command(
       OUTPUT ${GEN_SPEC_DIR}/ops.rst
        COMMAND ${CMAKE_COMMAND} -E cat
        ${CMAKE_CURRENT_BINARY_DIR}/ops.rst
        > ${GEN_SPEC_DIR}/ops.rst
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ops.rst)

# Targets for the specification.
add_custom_target(CudaTileSpecGen DEPENDS ${GEN_SPEC_DIR}/ops.rst)

# Declare custom targets for the MLIR Markdown.
add_custom_target(CudaTileDocGen DEPENDS ${GEN_DOC_DIR}/CudaTile.md)

# Now add the targets to the mlir-doc target so they will fire for that target.
add_dependencies(mlir-doc CudaTileDocGen CudaTileSpecGen)
