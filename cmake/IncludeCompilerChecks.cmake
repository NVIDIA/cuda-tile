# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(GCC_MIN_VER 7.4)
set(CLANG_MIN_VER 5.0)
set(PREBUILT_LLVM_CLANG_VERSION 17.0.6)
set(MSVC_MIN_VER 19.29)

function(check_compiler_version NAME NICE_NAME MINIMUM_VERSION)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL NAME)
    return()
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_VERSION)
    message(FATAL_ERROR "Host ${NICE_NAME} version must be at least ${MINIMUM_VERSION}, your version is ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()
endfunction(check_compiler_version)

check_compiler_version("GNU" "GCC" ${GCC_MIN_VER})
check_compiler_version("Clang" "Clang" ${CLANG_MIN_VER})
check_compiler_version("MSVC" "MSVC" ${MSVC_MIN_VER})

# More Clang specific checks
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if((NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL ${PREBUILT_LLVM_CLANG_VERSION}) AND TILE_IR_ENABLE_SANITIZER)
    if(NOT CUDA_TILE_USE_LLVM_INSTALL_DIR)
      message(FATAL_ERROR "To use prebuilt LLVM package with sanitizer enabled, the exact same compiler version is expected! Please use Clang ${PREBUILT_LLVM_CLANG_VERSION}")
    else()
      message(WARNING "You are building with sanitizer ON and your customized LLVM, make sure the exact same compiler version is used to match the compiler version of your specified LLVM!")
    endif()
  endif()
endif()
