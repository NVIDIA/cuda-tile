# -----------------------------------------------------------------------------
# Set and verify build type for CUDA Tile. If no CMAKE_BUILD_TYPE or
# CMAKE_CONFIGURATION_TYPES is set, default to `Release` build. If
# CMAKE_BUILD_TYPE is set to an unsupported value, print an error message
# and exit.
# -----------------------------------------------------------------------------
macro(set_cuda_tile_build_type)
  set(CMAKE_BUILD_TYPE_OPTIONS Release Debug RelWithDebInfo MinSizeRel)
  set(DEFAULT_BUILD_TYPE "Release")

  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to ${DEFAULT_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_BUILD_TYPE})" FORCE)
  else()
    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

    if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_BUILD_TYPE_OPTIONS)
      message(FATAL_ERROR "
      Unsupported build type selected. Use -DCMAKE_BUILD_TYPE=<type> to specify a valid build type for CUDA Tile.
      Available options are:
        * -DCMAKE_BUILD_TYPE=Release - For an optimized build with no assertions or debug info.
        * -DCMAKE_BUILD_TYPE=Debug - For an unoptimized build with assertions and debug info.
        * -DCMAKE_BUILD_TYPE=RelWithDebInfo - For an optimized build with no assertions but with debug info.
        * -DCMAKE_BUILD_TYPE=MinSizeRel - For a build optimized for size instead of speed.
      ")
    endif()
  endif()
endmacro(set_cuda_tile_build_type)
