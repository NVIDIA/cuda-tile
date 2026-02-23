//===- Types.td - CUDA Tile Type Definitions ---------------*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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
def CudaTile_AnyFloat : AnyTypeOf<[CudaTile_Float16,
                                   CudaTile_BFloat16,
                                   CudaTile_Float32,
                                   CudaTile_TFloat32,
                                   CudaTile_Float64,
                                   CudaTile_Float8E4M3FN,
                                   CudaTile_Float8E5M2
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
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType)>
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
    `!cuda_tile.tensor_view` represents a reference to a tensor in global
    memory.

    It consists of
      * elementType, the type of the elements in the tensor_view,
      * shape, an int64_t array, describes the size of each dimension,
      * strides, an int64_t array, describes the stride of each dimension,
          corresponding to how many elements are offset in memory when
          increasing the index on each dimension by one.

    The shape and the stride can be dynamic on a per-dimension basis. In those
    cases, their values are printed as `?`.

    Only power-of-two shape dimensions are supported.

    Examples:

    ```
      !cuda_tile.tensor_view<1024x1024xf16, strides=[1024,1]>
      !cuda_tile.tensor_view<32x16x32xf16, strides=[512,1,16]>
      !cuda_tile.tensor_view<?x?xf16, strides=[?,1]>
      !cuda_tile.tensor_view<?x16xf32, strides=[1,?]>
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
    `!cuda_tile.partition_view` represents a view into a tensor_view where
    tiles are laid out in a grid pattern over the original layout of the
    tensor_view.

    It consists of
      * tileShape, a dense integer array, describes the shape of tiles in the
          view,
      * tensor_view, the type of the tensor_view in which the view is looking,
      * dimMap, an int32_t array, a mapping from the dimensions of the tiles to
          the dimensions of the underlying tensor_view.
      * padding_value, an optional enum, specifying the value that should be
        loaded for accesses outside of tensor_view bounds.

    The shape of tiles and associated dimMap are encoded at the type level,
    while instances of the type carry information about the shape of the
    associated tensor_view.

    Loading values outside the bounds of the underlying tensor_view is defined
    and returns the value specified by :code:`padding_value`, or an unspecified
    value if :code:`padding_value` is not present.

    Supported padding values include:
    - :code:`zero`: Zero for out-of-bounds elements
    - :code:`neg_zero`: Negative zero
    - :code:`nan`: NaN
    - :code:`pos_inf`: Positive infinity
    - :code:`neg_inf`: Negative infinity

    Only power-of-two tile dimensions are supported.

    This type is a TileView, with the following specification:
    - Index space rank: as many dimensions as the attached tensor_view.
    - Tile size: as specified by tileShape.

    Examples:

    ```
    !cuda_tile.partition_view<
      tile=(2),
      tensor_view=!cuda_tile.tensor_view<16xf32, strides=[1]>
    >

    !cuda_tile.partition_view<
      tile=(2x2),
      tensor_view=!cuda_tile.tensor_view<16x16xf32, strides=[16,1]>
    >

    !cuda_tile.partition_view<
      tile=(2x2),
      tensor_view=!cuda_tile.tensor_view<16x16xf32, strides=[16,1]>,
      dim_map=[1, 0]
    >

    !cuda_tile.partition_view<
      tile=(2x2),
      padding_value = nan,
      tensor_view=!cuda_tile.tensor_view<16x16xf32, strides=[16,1]>,
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
  CudaTile_TFloat32, CudaTile_Float8E4M3FN, CudaTile_Float8E5M2
]>;

def CudaTile_NumberTileType : CudaTile_TileOf<[
  CudaTile_Int1, CudaTile_Int8, CudaTile_Int16, CudaTile_Int32, CudaTile_Int64,
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64,
  CudaTile_TFloat32, CudaTile_Float8E4M3FN, CudaTile_Float8E5M2
]>;

def CudaTile_PointerTileType : CudaTile_TileOf<[CudaTile_PointerType]>;

#endif  // CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD
