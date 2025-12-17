# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_subdirectory(CAPI)

set(MLIR_TEST_DEPENDS
  FileCheck
  cuda-tile-opt
  cuda-tile-translate
  ${CAPI_TEST_TARGETS}
)

# This target can be used to just build the dependencies
# for the check-cuda-tile target without executing the tests.
# This is useful for bots when splitting the build step from
# the testing step.
add_custom_target(check-cuda-tile-build-only
  DEPENDS ${MLIR_TEST_DEPENDS}
)

add_lit_testsuite(check-cuda-tile "Running the cuda_tile regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
  )
set_target_properties(check-cuda-tile PROPERTIES FOLDER "Tests")

add_lit_testsuites(cuda_tile ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
)

# Install the CAPI test binaries.
install(TARGETS ${CAPI_TEST_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
