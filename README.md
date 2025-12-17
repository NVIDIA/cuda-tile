# CUDA Tile IR

CUDA Tile IR is an MLIR-based intermediate representation and compiler
infrastructure for CUDA kernel optimization, focusing on tile-based computation
patterns and optimizations targeting NVIDIA tensor core units. The project
provides a comprehensive ecosystem for expressing and optimizing tiled
computations for NVIDIA GPUs, simplifying the development of high-performance
CUDA kernels through abstractions for common tiling patterns, memory hierarchy
management, and GPU-specific optimizations.

This open-source release is aligned with the **CUDA Toolkit 13.1** release. For
more information about CUDA Tile, visit https://developer.nvidia.com/cuda/tile.

## Core Components

CUDA Tile is composed of:

- **CUDA Tile Dialect**: A domain-specific MLIR dialect that provides
  first-class operations and types for tile-based computations
- **Python Bindings**: Complete Python API for programmatic IR construction,
  manipulation, and transformation
- **Bytecode:**: Efficient binary representation with support for serialization
  and de-serialization between the CUDA Tile dialect and binary format.
- **Conformance Test Suite**: Comprehensive tests ensuring compliance with the
  CUDA Tile specification and validation of dialect semantics

## CUDA Tile Specification

CUDA Tile development is driven by the CUDA Tile IR specification, which defines
the formal semantics, operations, and type system for tile-based computations on
NVIDIA GPUs. For detailed information about the CUDA Tile IR specification,
including dialect operations, type system, and transformation passes, please
refer to the [CUDA Tile Specification](https://docs.nvidia.com/cuda/tile-ir/13.1/index.html).

## Building CUDA Tile

### Prerequisites

- CMake 3.20.0 or higher
- C++17 compatible compiler
- Python 3.6+ (for Python bindings)
- MLIR/LLVM sources or pre-built libraries at a compatible commit (see
  [cmake/IncludeLLVM.cmake](cmake/IncludeLLVM.cmake#L29) for the exact version)
- Ninja build system (optional)

### Quick Start

For a quick start, use the following commands from the top of the repository to
configure and build a release version of CUDA Tile with Python bindings enabled.
MLIR/LLVM sources will be automatically downloaded from
https://github.com/llvm/llvm-project:

```bash
# Configure
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON

# Build
cmake --build build

# Run tests
cmake --build build --target check-cuda-tile
```

### Build Configuration Options

#### MLIR/LLVM Build Configuration

CUDA Tile requires MLIR/LLVM at a specific compatible commit. The exact commit
hash is specified in [cmake/IncludeLLVM.cmake](cmake/IncludeLLVM.cmake#L29).
CUDA Tile can be built with MLIR/LLVM in three different ways:

1. **Automatic Download from GitHub** (Default): CMake automatically downloads
   MLIR/LLVM sources from the official GitHub repository and builds them at the
   compatible commit. This is the slowest option but requires no manual LLVM
   setup.

   ```bash
   cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
   ```

2. **Use Local LLVM Sources**: CMake builds MLIR/LLVM from existing sources on
   your system. The commit hash of the source must be compatible with commit
   specified in [cmake/IncludeLLVM.cmake](cmake/IncludeLLVM.cmake#L29).

   ```bash
   cmake -G Ninja -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDA_TILE_USE_LLVM_SOURCE_DIR=/path/to/llvm/sources
   ```

3. **Use Pre-built LLVM Libraries**: CMake links against pre-compiled LLVM
   libraries. The commit hash of the source must be compatible with commit
   specified in [cmake/IncludeLLVM.cmake](cmake/IncludeLLVM.cmake#L29).

   ```bash
   cmake -G Ninja -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDA_TILE_USE_LLVM_INSTALL_DIR=/path/to/llvm/install
   ```

#### Python Bindings

CUDA Tile provides Python bindings for programmatic IR manipulation (disabled by
default). To enable them, add the `-DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON` flag
to your cmake configuration:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON
```

When building MLIR/LLVM from sources, MLIR Python bindings will be automatically
enabled. However, when using pre-built LLVM libraries, you must ensure they were
built with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`.

#### Ccache

To build with `ccache` enabled, add `-DCUDA_TILE_ENABLE_CCACHE=ON` to
your cmake configuration:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TILE_ENABLE_CCACHE=ON
```

When building LLVM from sources, this setting is automatically propagated to
the LLVM build.

## Testing

CUDA Tile uses LLVM's lit testing infrastructure for comprehensive testing.
Testing is enabled by default (`-DCUDA_TILE_ENABLE_TESTING=ON`). To run the test
suite:

```bash
cmake --build build --target check-cuda-tile
```

## Integrating CUDA Tile Into Your Project

CUDA Tile can be integrated into your project in two ways, depending on your
build system and requirements.

### Option 1: Using Pre-built CUDA Tile Libraries

To use pre-built CUDA Tile libraries in your project, include the necessary
headers and link against the required libraries based on your use case. For
example:

```cmake
include_directories(${CUDA_TILE_INSTALL_DIR}/include)

# CUDA Tile dialect
target_link_libraries(your_target PRIVATE
  CudaTileDialect           # CUDA Tile dialect operations and types
)

# Bytecode support.
target_link_libraries(your_target PRIVATE
  CudaTileBytecodeReader    # Read bytecode format
  CudaTileBytecodeWriter    # Write bytecode format
)
```

### Option 2: Integrating CUDA Tile Sources

To build CUDA Tile from source as part of your project:

1. Integrate CUDA Tile sources into your project with CMake's FetchContent, Git
   submodules, or any other integration method. Example using FetchContent:

```cmake
include(FetchContent)

# Define CUDA Tile directories
set(CUDA_TILE_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/cuda_tile-src)
set(CUDA_TILE_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/cuda_tile-build)

FetchContent_Declare(
  cuda_tile
  GIT_REPOSITORY https://github.com/NVIDIA/cuda-tile.git
  GIT_TAG        main
  SOURCE_DIR     ${CUDA_TILE_SOURCE_DIR}
  BINARY_DIR     ${CUDA_TILE_BINARY_DIR}
)
```

2. Configure CUDA Tile build options (before calling
   `FetchContent_MakeAvailable`, if using FetchContent):

```cmake
set(CUDA_TILE_USE_LLVM_INSTALL_DIR ${YOUR_LLVM_INSTALL_DIR} CACHE PATH "")
set(CUDA_TILE_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")
set(CUDA_TILE_ENABLE_TESTING OFF CACHE BOOL "")

FetchContent_MakeAvailable(cuda_tile)
```

3. Include headers from source and build directories, then link libraries as in
   Option 1:

```cmake
include_directories(${CUDA_TILE_SOURCE_DIR}/include)
include_directories(${CUDA_TILE_BINARY_DIR}/include)
```

## Example: Writing and Running a CUDA Tile IR Program

The following shows how to compile and run a simple Tile IR kernel that prints data from a pointer.

Tile IR bytecode can be produced from an MLIR program using the `cuda-tile-translate` tool.
This can be loaded directly using the CUDA driver API, which will JIT compile the program automatically.
To compile ahead of time, you can use the `tileiras` tool from the CUDA Toolkit to compile the bytecode
into a cubin for a particular GPU target. This example shows the latter to illustrate the extra step, but the
driver launch API is the same in either case (just substitute the path to the bytecode file).

### Prequisites

This example assumes you have built the CUDA Tile IR dialect tools according to the instructions above.

You will need a supported CUDA device, CUDA Toolkit 13.1+, and a compatible driver.

### CUDA Tile IR Program

Save the following into a file `example.mlir`.

```
cuda_tile.module @example_module {
    entry @example_kernel(%data_pr : tile<ptr<f32>>) {
        print "Running example module\n"
        %offsets = iota : tile<128xi32>
        %data_ptr_reshaped = reshape %data_pr : tile<ptr<f32>> -> tile<1xptr<f32>>
        %data_ptr_broadcasted = broadcast %data_ptr_reshaped : tile<1xptr<f32>> -> tile<128xptr<f32>>
        %data_ptr_tensor = offset %data_ptr_broadcasted, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
        %data, %token = load_ptr_tko weak %data_ptr_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
        print "Data: %f\n", %data : tile<128xf32>
        return
    }
}
```

### C++ Host Program

Save the following into a file `example_host.cpp`.

```
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

// Macro to check for errors from CUDA driver API calls.
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              errStr);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Data tile to be passed to the kernel.
float data[] = {0,   5,   10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,
                65,  70,  75,  80,  85,  90,  95,  100, 105, 110, 115, 120, 125,
                130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190,
                195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255,
                260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320,
                325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385,
                390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450,
                455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515,
                520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
                585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635};

int main() {
  // Declare and initialize CUDA driver API handles.
  CUdevice cuDevice;
  CUcontext cuContext;
  CUmodule cuModule;
  CUfunction example_kernel;
  CUstream stream;

  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Load the compiled cubin file and get the entry CUDA Tile IR function.
  // CUDA Tile IR bytecode can also be directly loaded (JIT compilation).
  CUDA_CHECK(cuModuleLoad(&cuModule, "example.cubin"));
  CUDA_CHECK(cuModuleGetFunction(&example_kernel, cuModule, "example_kernel"));

  // Allocate memory on the device and copy the input data to it.
  CUdeviceptr data_ptr;
  CUDA_CHECK(cuMemAlloc(&data_ptr, sizeof(data)));
  CUDA_CHECK(cuMemcpyHtoD(data_ptr, data, sizeof(data)));

  // Launch the kernel.
  void *kernel_args[] = {&data_ptr};
  CUDA_CHECK(cuLaunchKernel(example_kernel, // function
                            1, 1, 1,        // grid dims: must be (1,1,1)
                            1, 1, 1,        // block dims
                            0,              // shared memory bytes: must be 0
                            stream,         // cuda stream
                            kernel_args,    // kernel arguments
                            NULL            // extra parameters
                            ));
  CUDA_CHECK(cuCtxSynchronize());

  // Clean up.
  CUDA_CHECK(cuModuleUnload(cuModule));
  CUDA_CHECK(cuCtxDestroy(cuContext));

  return 0;
}
```

### Instructions

1. Compile the textual mlir program to CUDA Tile IR bytecode: `cuda-tile-translate example.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o example.tilebc`.
2. For AoT compilation, compile the bytecode file to a cubin: `tileiras --gpu-name sm_100 example.tilebc -o example.cubin`.
    1. Substitute `sm_100` with your supported target architecture.
    2. To JIT compile the bytecode at launch time, skip this step and replace `example.cubin` with `example.tilebc` in `host_example.cpp`.
3. Compile the host program: `g++ example_host.cpp -o example -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda`.
    1. Substitute `g++` with your C++ compiler, and the paths with the correct paths to your CUDA headers and libraries.
4. Execute: `./example`.

You should see the following terminal output:
```
Running example module
Data: [0.000000, 5.000000, 10.000000, 15.000000, 20.000000, 25.000000, 30.000000, 35.000000, 40.000000, 45.000000, 50.000000, 55.000000, 60.000000, 65.000000, 70.000000, 75.000000, 80.000000, 85.000000, 90.000000, 95.000000, 100.000000, 105.000000, 110.000000, 115.000000, 120.000000, 125.000000, 130.000000, 135.000000, 140.000000, 145.000000, 150.000000, 155.000000, 160.000000, 165.000000, 170.000000, 175.000000, 180.000000, 185.000000, 190.000000, 195.000000, 200.000000, 205.000000, 210.000000, 215.000000, 220.000000, 225.000000, 230.000000, 235.000000, 240.000000, 245.000000, 250.000000, 255.000000, 260.000000, 265.000000, 270.000000, 275.000000, 280.000000, 285.000000, 290.000000, 295.000000, 300.000000, 305.000000, 310.000000, 315.000000, 320.000000, 325.000000, 330.000000, 335.000000, 340.000000, 345.000000, 350.000000, 355.000000, 360.000000, 365.000000, 370.000000, 375.000000, 380.000000, 385.000000, 390.000000, 395.000000, 400.000000, 405.000000, 410.000000, 415.000000, 420.000000, 425.000000, 430.000000, 435.000000, 440.000000, 445.000000, 450.000000, 455.000000, 460.000000, 465.000000, 470.000000, 475.000000, 480.000000, 485.000000, 490.000000, 495.000000, 500.000000, 505.000000, 510.000000, 515.000000, 520.000000, 525.000000, 530.000000, 535.000000, 540.000000, 545.000000, 550.000000, 555.000000, 560.000000, 565.000000, 570.000000, 575.000000, 580.000000, 585.000000, 590.000000, 595.000000, 600.000000, 605.000000, 610.000000, 615.000000, 620.000000, 625.000000, 630.000000, 635.000000]
```

## Contributions and Support

**Note: We are currently not accepting external contributions.**

While CUDA Tile is an open-source project, we are not accepting external
contributions at this time. The project is under active development with a
focused roadmap. We encourage you to use GitHub Issues to report bugs, provide
feedback, and share your experiences with CUDA Tile. Your input helps us improve
the project and prioritize future development.

## License

CUDA Tile IR is licensed under the
[Apache License v2.0 with LLVM Exceptions](https://llvm.org/LICENSE.txt).
