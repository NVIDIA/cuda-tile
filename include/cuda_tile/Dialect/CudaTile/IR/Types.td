//===- Types.td - CUDA Tile Type Definitions ---------------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td"
include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// Integer Types
//===----------------------------------------------------------------------===//

// i1 values are interpreted based on operation semantics:
// - Unsigned interpretation: 0, 1 (i.e., 0b00000000, 0b00000001)
// - Signed interpretation: 0, -1 (two's complement for 1-bit, i.e., 0b00000000, 0b11111111)
// Operations on i1 values must preserve the LSB-only semantics. i1 values are
// canonicalized to 0x00 (false) or 0x01 (true) before storage and after loading
// from memory.
def CudaTile_Int1  : TypeAlias<I1, "i1">;
def CudaTile_Int8  : TypeAlias<I8, "i8">;
def CudaTile_Int16 : TypeAlias<I16, "i16">;
def CudaTile_Int32 : TypeAlias<I32, "i32">;
def CudaTile_Int64 : TypeAlias<I64, "i64">;

def CudaTile_AnyInt : AnyTypeOf<[CudaTile_Int1,
                                 CudaTile_Int8,
                                 CudaTile_Int16,
                                 CudaTile_Int32,
                                 CudaTile_Int64]> {
  let cppFunctionName = "isAnyInt";
}

//===----------------------------------------------------------------------===//
// Floating-point Types
//===----------------------------------------------------------------------===//

def CudaTile_Float16  : TypeAlias<F16, "f16">;
def CudaTile_BFloat16 : TypeAlias<BF16, "bf16">;
def CudaTile_Float32  : TypeAlias<F32, "f32">;
def CudaTile_TFloat32 : TypeAlias<TF32, "tf32">;
def CudaTile_Float64  : TypeAlias<F64, "f64">;

def CudaTile_Float8E4M3FN : TypeAlias<F8E4M3FN, "f8E4M3FN">;
def CudaTile_Float8E5M2   : TypeAlias<F8E5M2, "f8E5M2">;
def CudaTile_Float8E8M0FNU : TypeAlias<F8E8M0FNU, "f8E8M0FNU">;
def CudaTile_AnyFloat : AnyTypeOf<[CudaTile_Float16,
                                   CudaTile_BFloat16,
                                   CudaTile_Float32,
                                   CudaTile_TFloat32,
                                   CudaTile_Float64,
                                   CudaTile_Float8E4M3FN,
                                   CudaTile_Float8E5M2,
                                   CudaTile_Float8E8M0FNU,
                                  ]> {
  let cppFunctionName = "isAnyFloat";
}

def CudaTile_NumberType : AnyTypeOf<[CudaTile_AnyFloat,
                                     CudaTile_AnyInt]> {
  string cppType = "::mlir::Type";
}

//===----------------------------------------------------------------------===//
// Pointer Type
//===----------------------------------------------------------------------===//

def CudaTile_PointerType : CudaTileTypeDef<"Pointer", "ptr", "pointerType"> {
  let summary = "Pointer type";

  let description = [{
    An elemental pointer type $pointerType represents a single location in
    global device memory. Pointer types are typed, i.e., they carry the
    type they point to. Any `CudaTile_NumberType` can be used as pointee type.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType), [{
      return $_get(pointeeType.getContext(), pointeeType);
    }]>
  ];

  let parameters = (ins CudaTile_NumberType:$pointeeType);

  let assemblyFormat = "`<` custom<CudaTileType>($pointeeType) `>`";
}

//===----------------------------------------------------------------------===//
// Tile Type
//===----------------------------------------------------------------------===//

def CudaTile_TileElementType : AnyTypeOf<[CudaTile_NumberType,
                                          CudaTile_PointerType
                                         ]> {
  string cppType = "::mlir::Type";
}

def CudaTile_TileType : CudaTileTypeDef<"Tile", "tile", "tileType",
    [ShapedTypeInterface]> {
  let summary = "Tile type";

  let description = [{
    A tile type has a shape and and element type. The shape of the tile
    must be fully static. All elements of the tile have the same element
    type. Any `CudaTile_NumberType` or `CudaTile_PointerType` can be used as
    element type.

    Only power-of-two shape dimensions are supported.

    Examples:
    ```
    !cuda_tile.tile<5x4xf32>

    !cuda_tile.tile<4x!cuda_tile.ptr<i8>>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        CudaTile_TileElementType:$elementType);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType)>
  ];

  let extraClassDeclaration = [{
    // All interface methods of ShapedTypeInterface must be implemented.

    /// Return "true" if the type has a rank.
    bool hasRank() const { return true; }

    /// Return a new type with the given shape and element type.
    TileType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;
  }];
}

// Checks if a type is an instance of cuda_tile::TileType
def CudaTile_IsTileTypePred
  : CPred<"::llvm::isa<::mlir::cuda_tile::TileType>($_self)">;

class CudaTile_TileOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "tile">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([CudaTile_IsTileTypePred], preds)>,
      summary, "::mlir::cuda_tile::TileType"> {
        list<Type> allowedElementTypes = allowedTypes;
      }

// Ranked Tile
class CudaTile_RankedTileOf<list<Type> allowedTypes, list<int> ranks>
  : CudaTile_TileOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tile">;

// Rank-0 (Scalar) Tile
class CudaTile_ScalarTileOf<Type elementType>
  : CudaTile_RankedTileOf<[elementType], [0]>,
    BuildableType<!if(!eq(elementType.builderCall, ""), "",
      "::mlir::cuda_tile::TileType::get(ArrayRef<int64_t>(), " #
      elementType.builderCall #  ")")
    >;

//===----------------------------------------------------------------------===//
// TensorView Type
//===----------------------------------------------------------------------===//

def CudaTile_TensorViewType : CudaTileTypeDef<
    "TensorView",
    "tensor_view",
    "tensor_viewType"
> {
  let summary = "tensor view type";

  let description = [{
    :code:`!cuda_tile.tensor_view` represents a reference to a tensor in global
    memory.

    It consists of:
    * :code:`elementType`: the type of the elements in the :code:`tensor_view`.
    * :code`shape`: an integer array that specifies the size of each dimension.
      Sizes must be strictly positive.
    * :code:`strides`: an integer array that describes the stride of each
      dimension. The stride is the number of elements to offset in memory when
      increasing the corresponding index by one. Strides must be strictly
      positive.

    The shape and the stride can be dynamic on a per-dimension basis. In those
    cases, their values are printed as :code:`?`.

    .. note::

      Only power-of-two tile dimensions are supported.

    Examples:

    ```
    // A 512x1024 global memory tensor in row-major (lexicographic) order.
    !cuda_tile.tensor_view<512x1024xf16, strides=[1024, 1]>

    // A 512x1024 global memory tensor in column-major (colexicographic) order.
    !cuda_tile.tensor_view<512x1024xf16, strides=[1, 512]>

    // A 512x1024 global memory tensor that enumerates the same memory location
    // multiple times.
    !cuda_tile.tensor_view<512x1024xf16, strides=[1, 1]>

    // A 32x16x32 global memory tensor that is neither row-major nor
    // column-major.
    !cuda_tile.tensor_view<32x16x32xf16, strides=[512, 1, 16]>

    // A ?x? global memory tensor with a unit stride at the last dimension.
    !cuda_tile.tensor_view<?x?xf16, strides=[?, 1]>

    // A ?x16 global memory tensor with a unit stride at the first dimension.
    !cuda_tile.tensor_view<?x16xf32, strides=[1, ?]>
    ```
  }];

  let parameters = (ins
    CudaTile_NumberType:$elementType,
    ArrayRefParameter<"int64_t">:$shape,
    ArrayRefParameter<"int64_t">:$strides
  );

  let extraClassDeclaration = [{
    /// Value used to represent dynamic shape and stride dimensions.
    static constexpr int64_t kDynamic = ::mlir::ShapedType::kDynamic;

    /// Return how many shape dimensions are dynamic.
    size_t dynamicShapeAmount();
    /// Return how many stride dimensions are dynamic.
    size_t dynamicStrideAmount();
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// PartitionView Type
//===----------------------------------------------------------------------===//

def CudaTile_PartitionViewType : CudaTileTypeDef<
      "PartitionView",
      "partition_view",
      "partitionView",
      [DeclareTypeInterfaceMethods<CudaTile_TileView>]
> {
  let summary = "partition view type";

  let description = [{
    :code:`!cuda_tile.partition_view` represents a view into a
    :code:`tensor_view` where tiles are laid out in a grid pattern across the
    original :code:`tensor_view`.

    :code:`!cuda_tile.partition_view` is a :code:`TileView` with the following
    specification:
    * Index space rank: as many dimensions as the underlying
      :code:`tensor_view`.
    * Tile sizes: as specified by :code:`tile_shape`.

    It consists of:
    * :code:`tile_shape`: a dense integer array that describes the shape of the
      tiles in the view.
    * :code:`tensor_view`: the type of the :code:`tensor_view` into which the
      view is looking.
    * :code:`dim_map`: an integer array that specifies for each tile dimension
      the corresponding dimension in the underlying :code:`tensor_view`.
    * :code:`padding_value`: an optional enum, specifying the value that should
      be used for out-of-bounds accesses (loads) into the :code:`tensor_view`.

    Supported padding values include:
    * :code:`zero`: zero
    * :code:`neg_zero`: negative zero
    * :code:`nan`: NaN
    * :code:`pos_inf`: positive infinity
    * :code:`neg_inf`: negative infinity

    .. note::

      Only power-of-two tile dimensions are supported.

    Examples:

    ```
    // (1) A view into a 16xf32 tensor_view with a tile size of 2. The table
    // below visualizes for each element of the tensor_view the corresponding
    // tile, as indicated by its index.
    //
    //                               16
    // ←─────────────────────────────────────────────────────────────→
    // (0) (0) (1) (1) (2) (2) (3) (3) (4) (4) (5) (5) (6) (6) (7) (7)
    //
    !pv_1d = !cuda_tile.partition_view<
      tile=(2),
      tensor_view=!cuda_tile.tensor_view<16xf32, strides=[1]>
    >

    // (2) A view into a 32x16xf32 tensor_view with a tile size of 4x2. By
    // convention, in the below table, the Y axis corresponds to the first
    // tensor_view dimension and the X axis corresponds to the second one.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //     │ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //     │ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //     │ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //  64 │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (2,0) (2,0) (2,1) (2,1) (2,2) (2,2) (2,3) (2,3) (2,4) (2,4) ...
    //    ...
    //
    !pv_2d = !cuda_tile.partition_view<
      tile=(4x2),
      tensor_view=!cuda_tile.tensor_view<64x16xf32, strides=[16, 1]>
    >

    // (3) A view into a 32x16xf32 tensor_view with a tile size of 4x2. The
    // first tile dimension is mapped to the second tensor_view dimension. The
    // second tile dimension is mapped to the first tensor_view dimension.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     │ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //  64 │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //    ...
    //
    !pv_2d_transposed = !cuda_tile.partition_view<
      tile=(4x2),
      tensor_view=!cuda_tile.tensor_view<64x16xf32, strides=[16, 1]>,
      dim_map=[1, 0]
    >

    // Note: A load from partition_view with non-default dim_map is
    // semantically identical to a load with default dim_map followed by a
    // permutation.
    //
    // %0 = load_view_tko ... %view[%a, %b]
    //     : partition_view<tile=(4x2), ..., dim_map=[1, 0]> -> tile<4x2xf32>
    //
    // Is identical to:
    //
    // %0 = load_view_tko ... %view[%b, %a]
    //     : partition_view<tile=(2x4), ..., dim_map=[0, 1]> -> tile<2x4xf32>
    // %1 = permute %0 [1, 0] : tile<2x4xf32> -> tile<4x2xf32>
    ```

    The partition view index space is determined by the :code:`tile_shape`, the
    :code:`tensor_view` shape and :code:`dim_map`. In the above examples,
    :code:`!pv_2d` has an index space shape of :code:`16x8`, whereas
    :code:`!pv_2d_transposed` has an index space shape of :code:`4x32`.

    Indices into the partition view must lie within the index space of the
    partition view. Otherwise, the behavior is undefined. For example, loading
    the tile at index :code:`(0, 8)` from a partition view of type :`!pv_2d` is
    invalid.

    While partition view indices must be in-bounds, the tile itself may run
    out-of-bounds. I.e., it may fully or partially overlap with the underlying
    :code:`tensor_view`. Tiles cannot be fully outside of the underlying
    :code:`tensor_view` because that would require the partition view indices
    to lie outside of the the partition view index space.
    * **Load operations**: If :code:`padding_value` is set, out-of-bounds tile
      elements yield the padding value. If not set, out-of-bounds elements yield
      unspecified values.
    * **Store operations**: Out-of-bounds tile elements are masked during stores.

    Example:

    ```
    // (4) A view into a 8x2xf32 tensor_view with a tile size of 1x4 and NaN
    // padding. The right half of the below table consists of padded NaN
    // values.
    //
    //            2
    //       ←─────────→
    //     ↑ (0,0) (0,0) (0,0) (0,0)
    //     │ (1,0) (1,0) (1,0) (1,0)
    //   8 │ (2,0) (2,0) (2,0) (2,0)
    //     │ (3,0) (3,0) (3,0) (3,0)
    //     │ (4,0) (4,0) (4,0) (4,0)
    //    ...
    //
    !pv_2d_padded = !cuda_tile.partition_view<
      tile=(1x4),
      padding_value = nan,
      tensor_view=!cuda_tile.tensor_view<8x2xf32, strides=[2,1]>,
    >
    ```
  }];

  let parameters = (ins "::mlir::DenseI32ArrayAttr":$tile_shape,
                        CudaTile_TensorViewType:$tensor_view,
                        ArrayRefParameter<"int32_t">:$dim_map,
                        OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr">:$padding_value);

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//

def CudaTile_TokenType : CudaTileTypeDef<"Token", "token", "token"> {
  let summary = "cuda tile token type";
  let description = [{
    Tokens are not runtime values. Their purpose is to explicitly represent
    ordering constraints between token-ordered operations executed within a tile.
  }];
}

//===----------------------------------------------------------------------===//
// Any Type
//===----------------------------------------------------------------------===//

def CudaTile_AnyType : AnyTypeOf<[
  CudaTile_NumberType,
  Type<CPred<"::llvm::isa<::mlir::cuda_tile::CudaTileType>($_self)">>
]>;

//===----------------------------------------------------------------------===//
// Numerical Tile Types
//===----------------------------------------------------------------------===//

def CudaTile_IntTileType : CudaTile_TileOf<[
  CudaTile_Int1, CudaTile_Int8, CudaTile_Int16, CudaTile_Int32, CudaTile_Int64
]>;

def CudaTile_IntTileInt64Type : CudaTile_TileOf<[CudaTile_Int64]>;

def CudaTile_BaseFloatTileType : CudaTile_TileOf<[
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64
]>;

def CudaTile_FloatTileType : CudaTile_TileOf<[
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64,
  CudaTile_TFloat32, CudaTile_Float8E4M3FN, CudaTile_Float8E5M2,
  CudaTile_Float8E8M0FNU,
]>;

def CudaTile_NumberTileType : CudaTile_TileOf<[
  CudaTile_Int1, CudaTile_Int8, CudaTile_Int16, CudaTile_Int32, CudaTile_Int64,
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64,
  CudaTile_TFloat32, CudaTile_Float8E4M3FN, CudaTile_Float8E5M2,
  CudaTile_Float8E8M0FNU,
]>;

def CudaTile_PointerTileType : CudaTile_TileOf<[CudaTile_PointerType]>;

#endif  // CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD
