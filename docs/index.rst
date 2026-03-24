cutile-basic
============

A BASIC to CUDA Tile IR transpiler. Write GPU kernels in BASIC, compile them to
CUDA Tile IR (MLIR), and launch them on NVIDIA GPUs.

.. grid:: 1 2 2 3
   :gutter: 4
   :padding: 2 2 0 0
   :class-container: sd-text-center

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Installation, dependencies, and your first compilation.

   .. grid-item-card:: Language Reference
      :link: language_reference
      :link-type: doc

      Complete BASIC syntax reference including tile/GPU extensions.

   .. grid-item-card:: Architecture
      :link: architecture
      :link-type: doc

      Compiler pipeline from source to GPU execution.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Python API documentation for all modules.

   .. grid-item-card:: Examples
      :link: examples
      :link-type: doc

      Walkthrough of example programs from hello world to GEMM.

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc

      Command-line interface flags and usage.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   language_reference
   architecture
   api
   examples
   cli
