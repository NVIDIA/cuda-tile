function(add_capi_test_executable target_name)
  cmake_parse_arguments(ARG "" "" "LINK_LIBS" ${ARGN})

  add_llvm_executable(${target_name} ${ARG_UNPARSED_ARGUMENTS})
  target_link_libraries(${target_name} ${ARG_LINK_LIBS})

  set(CAPI_TEST_TARGETS ${CAPI_TEST_TARGETS} ${target_name} PARENT_SCOPE)
endfunction()

add_capi_test_executable(test-cuda-tile-capi-register
  register.c
  LINK_LIBS PRIVATE
    CudaTileCAPIRegistration
)

# Export the list of CAPI tests to the parent directory.
set(CAPI_TEST_TARGETS ${CAPI_TEST_TARGETS} PARENT_SCOPE)
