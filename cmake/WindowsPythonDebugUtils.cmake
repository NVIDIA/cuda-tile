# Utilities for handling Windows Python extension debug symbol linking.
# In Debug builds on Windows, CMake/nanobind appends a "_d" suffix to .pyd files (e.g., foo_d.pyd),
# but the Python interpreter expects the standard name (e.g., foo.pyd). This leads to import errors.
# This module provides functions to create hardlinks (or copies) from debug-named extensions to standard names,
# ensuring seamless imports in Debug mode. Commonly used for Python C++ extension development on Windows.
#
# add_windows_debug_links_installation(MODULE_TARGET BUILD_DIR INSTALL_DIR INSTALL_COMPONENT)
#   - MODULE_TARGET: CMake target name for the Python modules
#   - BUILD_DIR: Build directory containing the Python extensions
#   - INSTALL_DIR: Installation directory for Python extensions
#   - INSTALL_COMPONENT: CMake install component name
function(add_windows_debug_links_installation MODULE_TARGET BUILD_DIR INSTALL_DIR INSTALL_COMPONENT)
  if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Create debug links during build
    create_windows_debug_links(${MODULE_TARGET} ${BUILD_DIR})
    
    # Also install the hardlinked files (without _d suffix) during installation
    install(CODE "
      message(STATUS \"Installing debug links for ${MODULE_TARGET}...\")
      set(build_dir \"${BUILD_DIR}\")
      set(install_dir \"${INSTALL_DIR}\")
      
      # Find all debug Python extension files in build directory
      file(GLOB debug_files \"\${build_dir}/*_d.cp312-win_amd64.pyd\")
      
      foreach(debug_file \${debug_files})
        # Get just the filename
        get_filename_component(file_name \"\${debug_file}\" NAME)
        
        # Create the clean filename by removing \"_d\" suffix
        string(REPLACE \"_d.cp312\" \".cp312\" clean_name \"\${file_name}\")
        set(build_clean_file \"\${build_dir}/\${clean_name}\")
        set(install_clean_file \"\${install_dir}/\${clean_name}\")
        
        # Copy the hardlinked file from build to install directory
        if(EXISTS \"\${build_clean_file}\")
          file(COPY \"\${build_clean_file}\" DESTINATION \"\${install_dir}\")
          message(STATUS \"Installed: \${clean_name}\")
        else()
          message(WARNING \"Hardlinked file not found: \${build_clean_file}\")
        endif()
      endforeach()
    " COMPONENT ${INSTALL_COMPONENT})
  endif()
endfunction()

# Function to create hardlinks for debug Python extensions
# Parameters:
#   TARGET_NAME - The CMake target name
#   BUILD_DIR - The build directory containing the extensions
function(create_windows_debug_links TARGET_NAME BUILD_DIR)
  if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Creating non-debug links for ${TARGET_NAME}"
      COMMAND ${CMAKE_COMMAND} -E echo "Creating debug links in: ${BUILD_DIR}"
      
      # Generate and execute inline script to create hardlinks
      COMMAND ${CMAKE_COMMAND} 
        -DBUILD_DIR="${BUILD_DIR}"
        -P "${CMAKE_CURRENT_BINARY_DIR}/DebugLinksInlineScript.cmake"
      COMMENT "Creating clean Python extension links"
    )
    
    # Generate inline script at configure time
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/DebugLinksInlineScript.cmake" "
if(NOT BUILD_DIR)
    message(FATAL_ERROR \"BUILD_DIR must be specified\")
endif()

if(EXISTS \"\${BUILD_DIR}\")
    file(GLOB debug_files \"\${BUILD_DIR}/*_d.cp312-win_amd64.pyd\")
    
    foreach(debug_file \${debug_files})
        get_filename_component(file_name \"\${debug_file}\" NAME)
        string(REPLACE \"_d.cp312\" \".cp312\" clean_name \"\${file_name}\")
        set(clean_file \"\${BUILD_DIR}/\${clean_name}\")
        
        if(EXISTS \"\${clean_file}\")
            file(REMOVE \"\${clean_file}\")
        endif()
        
        execute_process(
            COMMAND \${CMAKE_COMMAND} -E create_hardlink \"\${debug_file}\" \"\${clean_file}\"
            RESULT_VARIABLE link_result
            ERROR_VARIABLE link_error
        )
        
        if(link_result EQUAL 0)
            message(STATUS \"Created link: \${clean_name} -> \${file_name}\")
        else()
            message(WARNING \"Failed to create hardlink for \${file_name}: \${link_error}\")
        endif()
    endforeach()
else()
    message(WARNING \"Build directory does not exist: \${BUILD_DIR}\")
endif()
")
  endif()
endfunction()
