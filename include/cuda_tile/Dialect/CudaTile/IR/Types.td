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
// Versioned Type Alias
//===----------------------------------------------------------------------===//

/// CudaTile type alias with version information.
class CudaTileTypeAlias<Type t, string ver, string summary = t.summary>
  : TypeAlias<t, summary> {
  string sinceVersion = ver;
}

//===----------------------------------------------------------------------===//
// Integer Types
//===----------------------------------------------------------------------===//

// i1 values are interpreted based on operation semantics:
// - Unsigned interpretation: 0, 1 (i.e., 0b00000000, 0b00000001)
// - Signed interpretation: 0, -1 (two's complement for 1-bit, i.e., 0b00000000, 0b11111111)
// Operations on i1 values must preserve the LSB-only semantics. i1 values are
// canonicalized to 0x00 (false) or 0x01 (true) before storage and after loading
// from memory.
def CudaTile_Int1  : CudaTileTypeAlias<I1, "13.1", "i1">;
def CudaTile_Int4  : CudaTileTypeAlias<I<4>, "13.3", "i4">;
def CudaTile_Int8  : CudaTileTypeAlias<I8, "13.1", "i8">;
def CudaTile_Int16 : CudaTileTypeAlias<I16, "13.1", "i16">;
def CudaTile_Int32 : CudaTileTypeAlias<I32, "13.1", "i32">;
def CudaTile_Int64 : CudaTileTypeAlias<I64, "13.1", "i64">;

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

def CudaTile_Float16  : CudaTileTypeAlias<F16, "13.1", "f16">;
def CudaTile_BFloat16 : CudaTileTypeAlias<BF16, "13.1", "bf16">;
def CudaTile_Float32  : CudaTileTypeAlias<F32, "13.1", "f32">;
def CudaTile_TFloat32 : CudaTileTypeAlias<TF32, "13.1", "tf32">;
def CudaTile_Float64  : CudaTileTypeAlias<F64, "13.1", "f64">;

def CudaTile_Float8E4M3FN : CudaTileTypeAlias<F8E4M3FN, "13.1", "f8E4M3FN">;
def CudaTile_Float8E5M2   : CudaTileTypeAlias<F8E5M2, "13.1", "f8E5M2">;
def CudaTile_Float8E8M0FNU : CudaTileTypeAlias<F8E8M0FNU, "13.2", "f8E8M0FNU">;
def CudaTile_Float4E2M1FN : CudaTileTypeAlias<F4E2M1FN, "13.3", "f4E2M1FN">;
def CudaTile_Float8E5M3FNU : CudaTileTypeAlias<F8E5M3FNU, "13.4", "f8E5M3FNU">;
def CudaTile_AnyFloat : AnyTypeOf<[CudaTile_Float16,
                                   CudaTile_BFloat16,
                                   CudaTile_Float32,
                                   CudaTile_TFloat32,
                                   CudaTile_Float64,
                                   CudaTile_Float8E4M3FN,
                                   CudaTile_Float8E5M2,
                                   CudaTile_Float8E8M0FNU,
                                   CudaTile_Float4E2M1FN,
                                   CudaTile_Float8E5M3FNU,
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

def CudaTile_PointerType : CudaTileTypeDef<"Pointer", "ptr", "pointerType", "13.1"> {
  let summary = "Pointer type";

  let description = [{
    An element pointer type $pointerType represents a single location in
    global device memory. Pointer types are typed, i.e., they carry the
    type they point to. Any numeric type can be used as pointee type.

    The optional :code:`#cuda_tile.ptr_attr<...>` attribute can be
    :code:`#cuda_tile.ptr_attr<none>`, indicating local memory access. When
    absent, the pointer is local.

    An :code:`f4E2M1FN` pointer addresses the byte containing a pair of packed
    f4 elements. It is used as the scalar base pointer of a tensor view. Direct
    pointer-tile loads and stores do not support :code:`f4E2M1FN`.
  }]
  # [{

    Unicast and multicast pointers are also supported:
    :code:`#cuda_tile.ptr_attr<unicast>` and
    :code:`#cuda_tile.ptr_attr<multicast>`, indicating unicast and multicast
    memory access. Unicast memory access requires an NVLink fabric and
    multicast memory access requires an NVLink fabric with an NVLink
    SHARP-supported NVSwitch. Using either without a supported fabric results
    in undefined behavior.
  }]
  ;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType), [{
      return $_get(pointeeType.getContext(), pointeeType, ::mlir::cuda_tile::PtrAttrAttr{});
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType, "::mlir::cuda_tile::PtrAttrAttr":$ptrAttr), [{
      return $_get(pointeeType.getContext(), pointeeType, ptrAttr);
    }]>
  ];

  let parameters = (ins
    CudaTileConstrainedTypeParam<CudaTile_NumberType, "13.1">:$pointeeType,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PtrAttrAttr", "ptr_attr">, "13.4">:$ptrAttr
  );

  let assemblyFormat = "`<` custom<CudaTileType>($pointeeType) (`,` $ptrAttr^)? `>`";
}

//===----------------------------------------------------------------------===//
// Fragment Type (public-preview slice at the SIMT-interop ABI boundary)
//===----------------------------------------------------------------------===//

def CudaTile_FragmentType
    : CudaTilePreviewTypeDef<"Fragment", "fragment", "fragment",
                             CudaTile_SIMTPreviewFeature> {
  let summary = "Fragment type: one invocation's view of a tile at the ABI boundary";

  let description = [{
    A public-preview `fragment` describes the slice of a tile that a single
    external-function invocation sees, together with where that slice lives and
    how it is laid out in its buffer.

    `!cuda_tile.tile` describes a whole tile and has value semantics. I.e., a
    tile cannot be modified in-place. `!cuda_tile.preview$fragment` describes the
    per-invocation buffer that the external function receives. It has buffer
    semantics. I.e., it has a memory space and can be modified in-place.

    A fragment carries:

    - a fragment shape (e.g. `1` for a scalar, `8x8` for an 8x8 fragment).
    - an element type (e.g. `f32`).
    - a memory space, indicating where the fragment buffer lives, either
      `local` or `smem`.
    - `mem_layout`: the in-buffer element order within one fragment, expressed
      as a `#cuda_tile.preview$layout`. This layout must be a permutation of
      `[0, element count)`: its domain size and its storage span must both
      equal the fragment's element count, and its modes must use regularly
      tiled strides. Padding, overlap, and broadcast (stride-zero) modes are
      rejected. For an `smem` fragment the span must not exceed the signed
      32-bit integer range.
    - `fragmap`: the fragment map, expressed as a `#cuda_tile.preview$layout`:
      `(invocation, participant, footprint) -> tile element`. May contain
      symbolic expressions.
    - `alignment`: the alignment in number of elements, which must be a power
      of two. Defaults to 1.

    `mem_layout`, `fragmap`, and `alignment` may contain symbolic expressions.

    Reading a `fragmap` (a three-mode `#cuda_tile.preview$layout`):
    ```
    #cuda_tile.preview$layout<(N, 1, 1) : (1, 0, 0)>
                       │  │  │     │  │  │
                       │  │  │     │  │  └─ footprint stride
                       │  │  │     │  └──── participant stride
                       │  │  │     └─────── invocation stride
                       │  │  └──────────── footprint = per-invocation buffer size
                       │  └─────────────── physical threads per invocation
                       │                   (for both local and smem)
                       └────────────────── invocation count
    ```

    Participation describes the data mapping and does not imply a
    synchronization level. The implementation's explicit synchronization
    property independently states whether the call uses no cooperative
    synchronization or may synchronize at warp, warpgroup, or CTA scope. For
    an `smem` fragment, participants normally have stride zero because they
    share one CTA-local buffer.

    In the visualizations below each tile element is labelled with the
    invocation `i` that owns it.

    Example 1 — scalar fragment: one tile element per invocation.
    ```
    Tile:     !cuda_tile.tile<1024xf32>
    Fragment: !cuda_tile.preview$fragment<1xf32, local,
                  mem_layout = #cuda_tile.preview$layout<(1):(1)>,
                  fragmap = #cuda_tile.preview$layout<(1024, 1, 1):(1, 0, 0)>>

    I = 1024 invocations, P = 1, footprint F = 1.
    map: (i, p, f) -> i*1 = i, so invocation i owns tile element i.

      tile element:  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ …
      invocation:    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ …
    ```

    Example 2 — footprint 2, contiguous (each invocation owns an adjacent pair).
    ```
    Tile:     !cuda_tile.tile<1024xf32>
    Fragment: !cuda_tile.preview$fragment<2xf32, local,
                  mem_layout = #cuda_tile.preview$layout<(2):(1)>,
                  fragmap = #cuda_tile.preview$layout<(512, 1, 2):(2, 0, 1)>>

      tile element:  │ 0   1 │ 2   3 │ 4   5 │ 6   7 │ …
      invocation:    │   0   │   1   │   2   │   3   │ …
    ```

    Example 3 — footprint 2, strided (same shape as Example 2, different stride).
    ```
    Tile:     !cuda_tile.tile<1024xf32>
    Fragment: !cuda_tile.preview$fragment<2xf32, local,
                  mem_layout = #cuda_tile.preview$layout<(2):(1)>,
                  fragmap = #cuda_tile.preview$layout<(512, 1, 2):(1, 0, 512)>>

      tile element:  │ 0 │ 1 │ … │ 511 │ 512 │ 513 │ … │ 1023 │
      invocation:    │ 0 │ 1 │ … │ 511 │  0  │  1  │ … │  511 │
    ```

    The memory space must be `local` or `smem`. The parser rejects any other
    memory space.

    Example 4 — parametric footprint: symbolic dimension `F`. Symbolic shape
    dims are written in brackets within the `NxMx` list, e.g. `[F]xf32`.
    ```
    Fragment: !cuda_tile.preview$fragment<[F]xf32, local,
                  mem_layout = #cuda_tile.preview$layout<(F):(1)>,
                  fragmap = #cuda_tile.preview$layout<(N, 1, F):(F, 0, 1)>>
    ```
  }];

  let parameters = (ins
    // Authoritative symbolic shape: a SymbolicTupleAttr whose elements are
    // SymbolicExprIntAttr (concrete) or SymbolicExprVarAttr (parametric).
    CudaTileAttrInterfaceParam<
      "::mlir::cuda_tile::SymbolicExprAttr", "",
      "symbolic shape of the fragment",
      "::mlir::cuda_tile::FragmentType::getSymbolicShape($_param_concreteShapeCache, $_ctxt)">:$symbolicShapeAttr,
    // Concrete shape cache: kDynamic for symbolic dims, the integer value for
    // concrete dims. Access via getShape().
    CudaTilePreviewTypeParam<ArrayRefParameter<"int64_t", "concrete shape cache">>:$concreteShapeCache,
    CudaTilePreviewTypeParam<TypeParameter<"::mlir::Type", "element type">>:$element_type,
    CudaTilePreviewEnumParam<CudaTile_MemorySpace>:$memory_space,
    CudaTilePreviewTypeParam<TypeParameter<"::mlir::cuda_tile::LayoutAttr", "in-buffer layout">>:$mem_layout,
    CudaTilePreviewTypeParam<TypeParameter<"::mlir::cuda_tile::LayoutAttr", "fragment map">>:$fragmap,
    // Alignment in elements. Absent alignment defaults to a concrete 1.
    CudaTileAttrInterfaceParam<
      "::mlir::cuda_tile::SymbolicExprAttr", "",
      "alignment in elements",
      "::mlir::cuda_tile::SymbolicExprIntAttr::get($_ctxt, 1)">:$alignment
  );

  // Requires a custom C++ assembly implementation to handle the symbolic
  // dimensions and the optional (elided-when-1) alignment.
  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Build from a concrete (fully static) shape.
    TypeBuilderWithInferredContext<(ins
        "::llvm::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType,
        "::mlir::cuda_tile::MemorySpace":$memorySpace,
        "::mlir::cuda_tile::LayoutAttr":$memLayout,
        "::mlir::cuda_tile::LayoutAttr":$fragmap,
        CArg<"::mlir::cuda_tile::SymbolicExprAttr", "{}">:$alignment), [{
      auto *ctx = elementType.getContext();
      ::mlir::cuda_tile::SymbolicExprAttr effectiveAlignment =
          alignment ? alignment : SymbolicExprIntAttr::get(ctx, 1);
      return $_get(ctx, getSymbolicShape(shape, ctx), shape, elementType,
                   memorySpace, memLayout, fragmap, effectiveAlignment);
    }]>,
    // Build from a symbolic shape (may contain symbolic dims).
    TypeBuilderWithInferredContext<(ins
        "::mlir::cuda_tile::SymbolicExprAttr":$symbolicShape,
        "::mlir::Type":$elementType,
        "::mlir::cuda_tile::MemorySpace":$memorySpace,
        "::mlir::cuda_tile::LayoutAttr":$memLayout,
        "::mlir::cuda_tile::LayoutAttr":$fragmap,
        CArg<"::mlir::cuda_tile::SymbolicExprAttr", "{}">:$alignment), [{
      auto *ctx = elementType.getContext();
      ::mlir::cuda_tile::SymbolicExprAttr effectiveAlignment =
          alignment ? alignment : SymbolicExprIntAttr::get(ctx, 1);
      ::llvm::SmallVector<int64_t> shapeCache;
      if (auto tuple = ::llvm::dyn_cast<SymbolicTupleAttr>(symbolicShape)) {
        shapeCache.reserve(tuple.getValues().size());
        for (SymbolicExprAttr dim : tuple.getValues()) {
          if (auto intDim = ::llvm::dyn_cast<SymbolicExprIntAttr>(dim))
            shapeCache.push_back(intDim.getValue());
          else
            shapeCache.push_back(::mlir::ShapedType::kDynamic);
        }
      }
      return $_get(ctx, symbolicShape, shapeCache, elementType, memorySpace,
                   memLayout, fragmap, effectiveAlignment);
    }]>
  ];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Return a symbolic tuple representing a concrete (all-integer) fragment
    /// shape. Used by the builder and the bytecode fallback.
    static SymbolicTupleAttr getSymbolicShape(::llvm::ArrayRef<int64_t> shape,
                                              ::mlir::MLIRContext *context) {
      ::llvm::SmallVector<SymbolicExprAttr> dims;
      dims.reserve(shape.size());
      for (int64_t dim : shape)
        dims.push_back(SymbolicExprIntAttr::get(context, dim));
      return SymbolicTupleAttr::get(context, dims);
    }

    /// Return the concrete shape (kDynamic for symbolic dims).
    ::llvm::ArrayRef<int64_t> getShape() const;

    /// Return the total number of elements (product of concrete dims). Only
    /// valid when `isConcrete()` is true.
    int64_t getNumElements() const {
      assert(isConcrete());
      int64_t n = 1;
      for (int64_t d : getShape())
        n *= d;
      return n;
    }

    /// Return true when all shape dims are concrete (no kDynamic) and the
    /// layouts and alignment contain no symbolic variables.
    bool isConcrete() const;
  }];
}

//===----------------------------------------------------------------------===//
// Symbolic Type
//===----------------------------------------------------------------------===//

def CudaTile_SymbolicType
    : CudaTilePreviewTypeDef<"Symbolic", "sym_t", "symbolicType",
                             CudaTile_SIMTPreviewFeature> {
  let summary =
      "A not-yet-resolved (parametric) dtype usable as an element type";
  let description = [{
    Represents a symbolic dtype parameter (e.g. `T`) as a proper MLIR type so
    it can stand in the element-type position of a tile, alongside concrete
    element types such as `f32` or `!cuda_tile.ptr<i8>`. The expression is
    resolved to a concrete type at instantiation time.

    The expression carries the parameter identifier and is printed bare
    (without quotes) for the currently supported SymbolicExprVarAttr case,
    mirroring how dtype parameters are spelled at use sites.

    Example:
      !cuda_tile.preview$sym_t<T>
  }];

  let parameters = (ins
    CudaTileAttrInterfaceParam<"::mlir::cuda_tile::SymbolicExprAttr", "">:$expr
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// DType Parameter Placeholder Type
//===----------------------------------------------------------------------===//
//
// A simple marker type used as the `type` field in ConstParamAttr to declare
// a DType parameter:
//   `#cuda_tile.preview$const_param<name = "T", type = !cuda_tile.preview$Datatype>`
//
// In TileType, dtype parameters are represented by SymbolicType, not by
// DTypeParamType. This parallels how SymbolicExprVarAttr("M") is used for
// symbolic shape dimensions.
def CudaTile_DTypeParamType
    : CudaTilePreviewTypeDef<"DTypeParam", "Datatype", "Datatype",
                             CudaTile_SIMTPreviewFeature> {
  let summary = "Placeholder for a compile-time dtype parameter";
  let description = [{
    Marker type used in ConstParamAttr to declare that a const_param is a
    DType parameter. Has no parameters — analogous to ShapeParamType.

    Example: `#cuda_tile.preview$const_param<name = "T", type = !cuda_tile.preview$Datatype>`
  }];
}

//===----------------------------------------------------------------------===//
// Shape Parameter Placeholder Type
//===----------------------------------------------------------------------===//

def CudaTile_ShapeParamType
    : CudaTilePreviewTypeDef<"ShapeParam", "shape", "Shape",
                             CudaTile_SIMTPreviewFeature> {
  let summary = "Placeholder for a compile-time shape parameter";
  let description = [{
    Marker type used in `ConstParamAttr` to declare a compile-time shape
    parameter. The type itself has no rank or dimensions. At instantiation, its
    value is supplied as a tuple of integer dimensions, such as `(128, 64)`.

    This type only describes the kind of a constant parameter; it is not a tile
    shape and cannot be used as a tile dimension. Symbolic tile dimensions are
    represented by `SymbolicExprVarAttr` instead. The placeholder must be
    resolved before lowering.

    Example: `#cuda_tile.preview$const_param<name = "S", type = !cuda_tile.preview$shape>`
  }];
}

//===----------------------------------------------------------------------===//
// String Parameter Placeholder Type
//===----------------------------------------------------------------------===//

def CudaTile_StringParamType
    : CudaTilePreviewTypeDef<"StringParam", "string", "String",
                             CudaTile_SIMTPreviewFeature> {
  let summary = "Placeholder for a compile-time string parameter";
  let description = [{
    Marker type used in ConstParamAttr to declare a string parameter. Concrete
    string arguments use SymbolicExprConstStrAttr; SymbolicExprVarAttr remains
    available for forwarding an in-scope string parameter.

    Example: `#cuda_tile.const_param<name = "mode", type = !cuda_tile.preview$string>`
  }];
}

//===----------------------------------------------------------------------===//
// Tile Type
//===----------------------------------------------------------------------===//

def CudaTile_TileElementType : AnyTypeOf<[CudaTile_NumberType,
                                          CudaTile_PointerType,
                                          CudaTile_Int4,
                                          CudaTile_SymbolicType
                                         ]> {
  string cppType = "::mlir::Type";
}

def CudaTile_TileType : CudaTileTypeDef<"Tile", "tile", "tileType", "13.1",
    [ShapedTypeInterface]> {
  let summary = "Tile type";

  let description = [{
    A tile is a value type that has a shape and an element type. The shape of the tile
    must be fully static. All elements of the tile have the same element
    type. Any numeric type or pointer type can be used as an
    element type.

    Only power-of-two shape dimensions are supported.

    A tile whose element type is :code:`ptr<f4E2M1FN>` must have rank 0. This
    scalar pointer form is used to construct tensor views. Executing
    :code:`cuda_tile.offset` on it with an odd f4 element offset is undefined
    behavior. General ranked f4 pointer tiles are not supported; use an f4 tile
    view for memory accesses.

    Examples:
    ```
    !cuda_tile.tile<8x4xf32>

    !cuda_tile.tile<4x!cuda_tile.ptr<i8>>
    ```
  }];

  let parameters = (ins
    CudaTilePreviewTypeExtensionParam<
      "::mlir::cuda_tile::SymbolicExprAttr",
      "symbolic shape of the tile",
      CudaTile_SIMTPreviewFeature,
      "::mlir::cuda_tile::TileType::getSymbolicShape($_param_concreteShapeCache, $_ctxt)">:$symbolicShapeAttr,
    CudaTileTypeParam<ArrayRefParameter<"int64_t", "concrete shape cache">, "13.1">:$concreteShapeCache,
    CudaTileConstrainedTypeParam<CudaTile_TileElementType, "13.1">:$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType), [{
        return $_get(elementType.getContext(),
                     getSymbolicShape(shape, elementType.getContext()), shape,
                     elementType);
      }]>,
    TypeBuilderWithInferredContext<(ins
      "::mlir::cuda_tile::SymbolicExprAttr":$symbolicShape,
      "Type":$elementType), [{
        SmallVector<int64_t> shapeCache;
        if (auto tuple = ::llvm::dyn_cast<SymbolicTupleAttr>(symbolicShape)) {
          shapeCache.reserve(tuple.getValues().size());
          for (SymbolicExprAttr dim : tuple.getValues()) {
            if (auto intDim = ::llvm::dyn_cast<SymbolicExprIntAttr>(dim))
              shapeCache.push_back(intDim.getValue());
            else
              shapeCache.push_back(::mlir::ShapedType::kDynamic);
          }
        }
        return $_get(elementType.getContext(), symbolicShape, shapeCache,
                     elementType);
      }]>
  ];

  let extraClassDeclaration = [{
    // All interface methods of ShapedTypeInterface must be implemented.

    /// Return a symbolic tuple representing a concrete tile shape.
    static SymbolicTupleAttr getSymbolicShape(ArrayRef<int64_t> shape,
                                              MLIRContext *context) {
      SmallVector<SymbolicExprAttr> shapeAttrs;
      shapeAttrs.reserve(shape.size());
      for (int64_t dim : shape)
        shapeAttrs.push_back(SymbolicExprIntAttr::get(context, dim));
      return SymbolicTupleAttr::get(context, shapeAttrs);
    }

    /// Return "true" if the type has a rank.
    bool hasRank() const { return true; }

    /// Return the shape cache. Symbolic dimensions are represented as
    /// ShapedType::kDynamic.
    ArrayRef<int64_t> getShape() const;

    /// Return a new type with the given shape and element type.
    TileType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;

    /// Return true if this tile has no symbolic dimensions or element type.
    bool isConcreteTile() const {
      return !llvm::isa<SymbolicType>(getElementType()) &&
             llvm::none_of(getConcreteShapeCache(), [](int64_t dim) {
               return ShapedType::isDynamic(dim);
             });
    }
  }];
}

// Checks if a type is an instance of concrete cuda_tile::TileType
def CudaTile_IsTileTypePred
  : CPred<"::llvm::isa<::mlir::cuda_tile::TileType>($_self) && "
          "::llvm::cast<::mlir::cuda_tile::TileType>($_self).isConcreteTile()">;

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
    "tensor_viewType",
    "13.1"
> {
  let summary = "tensor view type";

  let description = [{
    :code:`!cuda_tile.tensor_view` represents a reference to a tensor in global
    memory.

    It consists of:
    * :code:`elementType`: the type of the elements in the :code:`tensor_view`.
    * :code:`shape`: an integer array that specifies the size of each dimension.
      Sizes must be strictly positive.
    * :code:`strides`: an integer array that describes the stride of each
      dimension. The stride is the number of elements to offset in memory when
      increasing the corresponding index by one. Strides must be strictly
      positive.
    * :code:`ptr_attr`: an optional attribute inferred from the
      :code:`base` pointer. The attribute can be
      :code:`#cuda_tile.ptr_attr<none>`,
      :code:`#cuda_tile.ptr_attr<unicast>`, or
      :code:`#cuda_tile.ptr_attr<multicast>`. It indicates whether the view
      access is local, unicast, or multicast.

    The shape and the stride can be dynamic on a per-dimension basis. In those
    cases, their values are printed as :code:`?`.

    .. note::

      Only power-of-two tile dimensions are supported.

    .. note::

      For an :code:`f4E2M1FN` view access, or a multicast view access with an
      :code:`f16` or :code:`bf16` element type, elements must be densely packed
      in global memory. The tensor view must have exactly one statically encoded
      element stride equal to 1. That stride identifies the **packing
      dimension**. A dynamic stride does not identify a packing dimension, and
      multiple static unit strides are ambiguous; accesses in either case
      cannot be compiled.

      For such an access, every non-packing element stride must be even. An
      even non-packing stride ensures that stepping in another dimension
      preserves the required alignment of the packing-dimension origin. For an
      f16/bf16 multicast access, the tensor-view extent in the packing dimension
      must also be even. For an :code:`f4E2M1FN` access, that extent may be
      odd. The tile extent in the packing dimension must still be even. If the
      tensor-view extent in that dimension is odd, its final element has no
      in-bounds partner. Any access whose tile includes that element has
      undefined behavior. Values required to be even may be statically encoded
      in the tensor-view type or supplied at runtime; the compiler does not
      require proof that they are even. Violating any of these preconditions
      when the access executes is undefined behavior. The tensor-view extent is
      distinct from the statically encoded
      :code:`tile_shape` of a concrete tile-view type. Individual tile-view
      types impose the additional evenness preconditions documented on those
      types.

      Executing an f16/bf16 multicast memory access through a tensor view
      constructed from a multicast base requires the base address to be at
      least 4-byte aligned; otherwise, the access has undefined behavior.

      For :code:`f4E2M1FN`, elements :code:`i` and :code:`i+1` (where
      :code:`i` is a multiple of 2) are packed into a byte as follows: element
      :code:`i` is stored in bits :code:`3...0` and element :code:`i+1` is
      stored in bits :code:`7...4`. This corresponds to a little-endian nibble
      order.

      Example: A :code:`tensor_view<2xf4E2M1FN, strides=[1]>` with values
      :code:`[0.5, 1.5]` is stored as:
      ```
      Bit position:    7     6     5     4     3     2     1     0
                    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
                    │  0  │  0  │  1  │  1  │  0  │  0  │  0  │  1  │
                    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                    ├──── upper nibble ────┤ ├──── lower nibble ────┤
                    │          1.5         │ │         0.5          │
                    └──────────────────────┘ └──────────────────────┘
      Bit pattern of 0.5 in f4E2M1FN: 0001
      Bit pattern of 1.5 in f4E2M1FN: 0011
      ```

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
    CudaTileConstrainedTypeParam<CudaTile_NumberType, "13.1">:$elementType,
    CudaTileTypeParam<ArrayRefParameter<"int64_t", "shape of the tensor view">, "13.1">:$shape,
    CudaTileTypeParam<ArrayRefParameter<"int64_t", "strides of the tensor view">, "13.1">:$strides,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PtrAttrAttr">, "13.4">:$ptrAttr
  );

  let builders = [
    TypeBuilder<(ins "Type":$elementType,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<int64_t>":$strides), [{
      return $_get($_ctxt, elementType, shape, strides,
                   ::mlir::cuda_tile::PtrAttrAttr{});
    }]>
  ];

  let extraClassDeclaration = [{
    /// Value used to represent dynamic shape and stride dimensions.
    static constexpr int64_t kDynamic = ::mlir::ShapedType::kDynamic;

    /// Return how many shape dimensions are dynamic.
    size_t dynamicShapeAmount();
    /// Return how many stride dimensions are dynamic.
    size_t dynamicStrideAmount();

    /// Return the unique statically encoded unit-stride dimension for an f4
    /// tensor view. Return std::nullopt for a non-f4 view or when the packing
    /// dimension is missing or ambiguous.
    std::optional<unsigned> getF4PackingDimension() const;
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
      "13.1",
      [DeclareTypeInterfaceMethods<CudaTile_TileView>]
> {
  let summary = "partition view type";

  let description = [{
    :code:`!cuda_tile.partition_view` represents a view into a
    :code:`tensor_view` where tiles are laid out in a grid pattern across the
    original :code:`tensor_view`. The grid is aligned with the start of each
    dimension and there are no gaps or overlaps between tiles.

    :code:`!cuda_tile.partition_view` has the following specification:
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
      tensor_view<16xf32, strides=[1]>
    >

    // (2) A view into a 64x16xf32 tensor_view with a tile size of 4x2. By
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
      tensor_view<64x16xf32, strides=[16, 1]>
    >

    // (3) A view into a 64x16xf32 tensor_view with a tile size of 4x2. The
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
      tensor_view<64x16xf32, strides=[16, 1]>,
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
    the tile at index :code:`(0, 8)` from a partition view of type :code:`!pv_2d` above
    is invalid, because the maximum index in dimension 1 is 7.

    While partition view indices must be in-bounds, the accessed tile itself may run
    out-of-bounds. I.e., it may only partially overlap with the underlying
    :code:`tensor_view`. Tiles cannot be fully outside of the underlying
    :code:`tensor_view` because that would require the partition view indices
    to lie outside the partition view index space.
    When tiles are partially out of bounds, operations follow the following semantics:
    * **Load operations**: If :code:`padding_value` is set, out-of-bounds tile
      elements yield the padding value. If not set, out-of-bounds elements yield
      unspecified values.
    * **Store operations**: Out-of-bounds tile elements are masked during stores.

    .. note::

      For an :code:`f4E2M1FN` view access, and for a multicast view access with
      an :code:`f16` or :code:`bf16` element type, the statically encoded tile
      extent mapped to the underlying tensor view's packing dimension must be
      even. Violating this precondition is undefined behavior.

      For :code:`f4E2M1FN`, this requirement also ensures that every tile origin
      is byte-aligned, because partition traversal equals the tile extent.

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
      tensor_view<8x2xf32, strides=[2,1]>
    >
    ```
  }];

  let parameters = (ins
    CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "tile shape", "13.1">:$tile_shape,
    CudaTileStrTypeParam<"::mlir::cuda_tile::TensorViewType", "tensor view", "13.1">:$tensor_view,
    CudaTileTypeParam<ArrayRefParameter<"int32_t", "dimension mapping">, "13.1">:$dim_map,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "padding value">, "13.1">:$padding_value
  );

  let extraClassDeclaration = [{
    /// Returns the distance (in elements) between the start of successive tiles
    /// along each dimension. On PartitionView, this is an alias for
    /// getTileShape.
    ::mlir::DenseI32ArrayAttr getTraversalStrides() {
      return getTileShape();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StridedView Type
//===----------------------------------------------------------------------===//

def CudaTile_StridedViewType : CudaTileTypeDef<
      "StridedView",
      "strided_view",
      "stridedView",
      "13.3",
      [DeclareTypeInterfaceMethods<CudaTile_TileView>]
> {
  let summary = "strided view type";

  let description = [{
    :code:`!cuda_tile.strided_view` represents a view into a
    :code:`tensor_view` where tiles are laid out in a grid pattern across the
    original :code:`tensor_view`. The grid is aligned with the start of each
    dimension, but the grid pattern's striding factor is
    parametric, allowing interleaved or overlapping tiles.

    :code:`!cuda_tile.strided_view` has the following specification:
    * Index space rank: as many dimensions as the underlying
      :code:`tensor_view`.
    * Tile sizes: as specified by :code:`tile_shape`.

    It consists of:
    * :code:`tile_shape`: an integer array that describes the shape of the
      tiles in the view.
    * :code:`traversal_strides`: an integer array that describes the traversal
      strides when traversing the underlying :code:`tensor_view`. For example, if the shape of
      tiles is 2x2 and the traversal strides are [3,3], the :code:`tensor_view` will
      be traversed with a gap of one element between each tile. Alternatively,
      if the traversal strides are [1,1], the :code:`tensor_view` will be traversed
      with an overlapping sliding window, advancing by one element for each
      tile.
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

      Only power-of-two tile dimensions are supported. In contrast, traversal
      strides can be any strictly positive value.

    Examples:

    ```
    // (1) A view into a 16xf32 tensor_view with a tile size of 2 and a
    // traversal stride of 2. This behavior is identical to PartitionView. The
    // table below visualizes for each element of the tensor_view the
    // corresponding tile, as indicated by its index.
    //
    //                               16
    // ←─────────────────────────────────────────────────────────────→
    // (0) (0) (1) (1) (2) (2) (3) (3) (4) (4) (5) (5) (6) (6) (7) (7)
    //
    !sv_1d_tra2 = !cuda_tile.strided_view<
      tile=(2),
      traversal_strides=[2],
      tensor_view<16xf32, strides=[1]>
    >

    // (2) A view into a 16xf32 tensor_view with a tile size of 2 and a
    // traversal stride of 3. This time, one out of three elements are
    // skipped, as the stride moves three elements while the tile only
    // covers two. Notice that tile 5 is partially out of bounds.
    //
    //                               16
    // ←─────────────────────────────────────────────────────────────→
    // (0) (0) ( ) (1) (1) ( ) (2) (2) ( ) (3) (3) ( ) (4) (4) ( ) (5) (5)
    //
    !sv_1d_tra3 = !cuda_tile.strided_view<
      tile=(2),
      traversal_strides=[3],
      tensor_view<16xf32, strides=[1]>
    >

    // (3) A view into a 8xf32 tensor_view with a tile size of 2 and a
    // traversal stride of 1. This time, tiles are overlapping, and the
    // same element will be present in multiple tiles. For example,
    // element 1 of the tensor view is present both as the second element of
    // tile 0, and as the first element of tile 1. Tile 7 is partially out of bounds.
    // (Note the space is still 1-dimensional; the second row indicates overlap,
    // not a second dimension.)
    //
    //               8
    // ←─────────────────────────────→
    // (0) (0) (2) (2) (4) (4) (6) (6)
    //     (1) (1) (3) (3) (5) (5) (7) (7)
    //
    !sv_1d_tra1 = !cuda_tile.strided_view<
      tile=(2),
      traversal_strides=[1],
      tensor_view<8xf32, strides=[1]>
    >

    // (4) A view into a 64x16xf32 tensor_view with a tile size of 4x2 and
    // traversal strides of [4,3]. Tiles are adjacent in the first dimension,
    // but have a gap of one element in the second dimension. By convention,
    // in the below table, the Y axis corresponds to the first tensor_view
    // dimension and the X axis corresponds to the second one.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //     │ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //     │ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //     │ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //  64 │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (2,0) (2,0) (   ) (2,1) (2,1) (   ) (2,2) (2,2) (   ) (2,3) ...
    //    ...
    //
    !sv_2d = !cuda_tile.strided_view<
      tile=(4x2),
      traversal_strides=[4,3],
      tensor_view<64x16xf32, strides=[16, 1]>
    >

    // (5) A view into a 64x16xf32 tensor_view with a tile size of 4x2 and
    // traversal strides of [4,3], as above, but with dim_map=[1, 0].
    // (The first tile dimension is mapped to the second tensor_view dimension
    // and the second tile dimension is mapped to the first tensor_view dimension.)
    // Tiles are adjacent in the second dimension, but have a gap of one element in
    // the first dimension.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     │ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     | (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) ...
    //     │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //  64 │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //     | (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //     | (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) ...
    //    ...
    //
    !sv_2d_transposed = !cuda_tile.strided_view<
      tile=(4x2),
      traversal_strides=[4,3],
      tensor_view<64x16xf32, strides=[16, 1]>,
      dim_map=[1, 0]
    >

    // Note: A load from a strided_view with non-default dim_map is
    // semantically identical to a load with default dim_map followed by a
    // permutation.
    //
    // %0 = load_view_tko ... %view[%a, %b]
    //     : strided_view<tile=(4x2), traversal_strides=[4,2], ...,
    //                    dim_map=[1, 0]> -> tile<4x2xf32>
    //
    // Is identical to:
    //
    // %0 = load_view_tko ... %view[%b, %a]
    //     : strided_view<tile=(2x4), traversal_strides=[2,4], ...,
    //                    dim_map=[0, 1]> -> tile<2x4xf32>
    // %1 = permute %0 [1, 0] : tile<2x4xf32> -> tile<4x2xf32>
    ```

    The strided view index space is determined by the
    :code:`traversal_strides`, the :code:`tensor_view` shape, and
    :code:`dim_map`. Partial tiles at the edges of the :code:`tensor_view`
    are included in the index space. In the above examples, :code:`!sv_1d_tra2`
    has an index space shape of :code:`8`, whereas :code:`!sv_1d_tra3` has an
    index space shape of :code:`6`. Similarly, :code:`!sv_2d` has an index
    space shape of :code:`16x6`, and :code:`!sv_2d_transposed` has an index
    space shape of :code:`4x22`.

    Indices accessing the strided view must lie within the index space of the
    strided view. Otherwise, the behavior is undefined. For example, loading
    the tile at index :code:`(0, 6)` from a strided view of type :code:`!sv_2d`
    above is invalid, as the maximum index in dimension 1 is 5.

    While strided view indices must be in-bounds, the accessed tile itself may
    run out-of-bounds. I.e., it may only partially overlap with the underlying
    :code:`tensor_view`. Tiles cannot be fully outside of the underlying
    :code:`tensor_view` because that would require the strided view indices
    to lie outside of the strided view index space.
    When tiles are partially out of bounds, operations follow the following semantics:
    * **Load operations**: If :code:`padding_value` is set, out-of-bounds tile
      elements yield the padding value. If not set, out-of-bounds elements yield
      unspecified values.
    * **Store operations**: Out-of-bounds tile elements are masked during stores.

    .. note::

      For an :code:`f4E2M1FN` view access, both the statically encoded tile
      extent and traversal stride mapped to the underlying tensor view's
      packing dimension must be even. Violating either precondition is
      undefined behavior.

    Example:

    ```
    // (6) A view into a 8x2xf32 tensor_view with a tile size of 1x4 and NaN
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
    !sv_2d_padded = !cuda_tile.strided_view<
      tile=(1x4),
      traversal_strides=[1,4],
      padding_value = nan,
      tensor_view<8x2xf32, strides=[2,1]>
    >
    ```
  }];

  let parameters = (ins CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "tile shape", "13.3">:$tile_shape,
                        CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "traversal strides", "13.3">:$traversal_strides,
                        CudaTileStrTypeParam<"::mlir::cuda_tile::TensorViewType", "tensor view", "13.3">:$tensor_view,
                        CudaTileTypeParam<ArrayRefParameter<"int32_t", "dimension mapping">, "13.3">:$dim_map,
                        CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "padding value">, "13.3">:$padding_value);

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// GatherScatterView Type
//===----------------------------------------------------------------------===//

def CudaTile_GatherScatterViewType : CudaTileTypeDef<
      "GatherScatterView",
      "gather_scatter_view",
      "gatherScatterView",
      "13.3",
      [DeclareTypeInterfaceMethods<CudaTile_TileView>]
> {
  let summary = "gather/scatter view type";

  let description = [{
    :code:`!cuda_tile.gather_scatter_view` represents a view into a
    :code:`tensor_view` where one dimension is accessed using a sparse
    gather/scatter pattern while the remaining dimensions are accessed
    contiguously.

    :code:`!cuda_tile.gather_scatter_view` has the following specification:
    * Index space rank: as many dimensions as the underlying
      :code:`tensor_view`.
    * Tile sizes: as specified by :code:`tile_shape`.

    It consists of:
    * :code:`tile_shape`: an integer array that describes the shape of the
      tiles in the view.
    * :code:`tensor_view`: the type of the :code:`tensor_view` into which the
      view is looking.
    * :code:`sparse_dim`: a non-negative integer that specifies which dimension
      to gather/scatter over. Must be strictly less than the rank of the
      :code:`tensor_view`.
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

    .. note::

      An :code:`f4E2M1FN` gather/scatter view access is rejected if the
      underlying tensor view's packing dimension is the sparse dimension. The
      statically encoded tile extent in the packing dimension and the scalar
      index value selecting its origin must both be even. Executing an access
      with either value odd is undefined behavior.

    When loading or storing via a gather_scatter_view, the index at position
    :code:`sparse_dim` must be a **1D tile** whose size equals
    :code:`tile_shape[sparse_dim]`. Each element of this tile is an
    independent index into the underlying :code:`tensor_view` along that
    dimension. The loaded or stored tile will be the concatenation of the
    elements in the rows selected by the sparse dimension index. All other
    indices must be **scalar tiles** of the same element type, each choosing
    the offset in the underlying :code:`tensor_view` along that dimension
    and selecting a contiguous block of :code:`tile_shape[dim]` elements.

    Examples:

    ```
    // (1) A 1D gather/scatter view over an 8xf32 tensor_view with a tile
    // size of 4 and sparse_dim=0. Since the view is 1D, the only index is
    // the gather index: a size-4 1D tile of element indices. The table below
    // visualizes a load with gather_indices=[6, 1, 4, 3]. Gathered elements
    // are indicated with their index in the single loaded tile. ( ) marks
    // elements that are not gathered.
    //
    //               8
    // ←─────────────────────────────→
    // ( ) (1) ( ) (3) (2) ( ) (0) ( )
    //
    // In pseudocode, we can say that loading with gather_indices=[6, 1, 4, 3]
    // is equivalent to the following:
    //   result = [tensor_view[6], tensor_view[1], tensor_view[4], tensor_view[3]]
    //
    !gsv_1d = !cuda_tile.gather_scatter_view<
      tile=(4),
      tensor_view<8xf32, strides=[1]>,
      sparse_dim=0
    >

    // (2) A 2D gather/scatter view over an 8x8xf32 tensor_view with tile
    // size 4x4 and sparse_dim=0. The first index (at sparse_dim) is a size-4
    // 1-D tile of row indices; the second index is a scalar selecting a
    // block of 4 contiguous columns. The table below visualizes a load
    // with gather_indices=[5, 1, 7, 3] and col_idx=0. Each number
    // indicates the result row, and ( ) marks elements not gathered.
    //
    //                      8
    //       ←─────────────────────────────→
    //     ↑ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 0: not gathered
    //     │ (1) (1) (1) (1) ( ) ( ) ( ) ( )    row 1: → result row 1
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 2: not gathered
    //   8 │ (3) (3) (3) (3) ( ) ( ) ( ) ( )    row 3: → result row 3
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 4: not gathered
    //     │ (0) (0) (0) (0) ( ) ( ) ( ) ( )    row 5: → result row 0
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 6: not gathered
    //     ↓ (2) (2) (2) (2) ( ) ( ) ( ) ( )    row 7: → result row 2
    //
    !gsv_2d = !cuda_tile.gather_scatter_view<
      tile=(4x4),
      tensor_view<8x8xf32, strides=[8, 1]>,
      sparse_dim=0
    >

    // (3) A larger gather/scatter view with zero-padding for out-of-bounds
    // accesses. If the index for dimension 1 is in the range [241, 255],
    // the out-of-bounds elements will be filled with zero. If no padding value
    // were set, the values would be unspecified. Out of bounds sparse dimension
    // indices, in dimension 0, will yield a row of zero (the padding value).
    // Likewise, the values of the entire row are unspecified if no padding
    // value is set.
    //
    !gsv_2d_padded = !cuda_tile.gather_scatter_view<
      tile=(8x16),
      padding_value = zero,
      tensor_view<128x256xf32, strides=[256, 1]>,
      sparse_dim=0
    >

    // (4) Gather along the second dimension (sparse_dim=1). Here the
    // first index is a scalar selecting which block of 8 rows, and
    // the second index is a 1D tile of 16 column indices.
    //
    !gsv_2d_col = !cuda_tile.gather_scatter_view<
      tile=(8x16),
      tensor_view<128x256xf32, strides=[256, 1]>,
      sparse_dim=1
    >

    // (5) Scatter (store) using the same view type as example (2).
    // Given a 4x4 tile to store and scatter_indices=[5, 1, 7, 3]
    // with col_idx=0, row 0 of the tile is written to row 5 of the
    // tensor_view, row 1 to row 1, row 2 to row 7, and row 3 to
    // row 3. Columns 0-3 are written contiguously. The table below
    // shows which tensor_view rows are written [*] and which are
    // untouched ( ).
    //
    //                      8
    //       ←─────────────────────────────→
    //     ↑ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 0: untouched
    //     │ [1] [1] [1] [1] ( ) ( ) ( ) ( )    row 1: ← tile row 1
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 2: untouched
    //   8 │ [3] [3] [3] [3] ( ) ( ) ( ) ( )    row 3: ← tile row 3
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 4: untouched
    //     │ [0] [0] [0] [0] ( ) ( ) ( ) ( )    row 5: ← tile row 0
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 6: untouched
    //     ↓ [2] [2] [2] [2] ( ) ( ) ( ) ( )    row 7: ← tile row 2
    //
    // In pseudocode:
    //   tensor_view[5, 0:4] = tile[0, :]
    //   tensor_view[1, 0:4] = tile[1, :]
    //   tensor_view[7, 0:4] = tile[2, :]
    //   tensor_view[3, 0:4] = tile[3, :]
    //
    !gsv_2d_scatter = !cuda_tile.gather_scatter_view<
      tile=(4x4),
      tensor_view<8x8xf32, strides=[8, 1]>,
      sparse_dim=0
    >
    ```

    The gather/scatter view index space shape equals the shape of the
    underlying :code:`tensor_view`. In the above examples, :code:`!gsv_1d`
    has an index space shape of :code:`8`, :code:`!gsv_2d` has an index
    space shape of :code:`8x8`, and :code:`!gsv_2d_padded` has an index
    space shape of :code:`128x256`.

    Indices accessing the :code:`gather_scatter_view` on non-sparse dimensions
    must be in-bounds, though the accessed tile itself may run out-of-bounds.
    Individual gather/scatter indices (at :code:`sparse_dim`) may reference
    rows that are out-of-bounds of the underlying :code:`tensor_view`.
    When tiles are partially out of bounds, operations follow the following
    semantics:
    * **Load operations**: If :code:`padding_value` is set, out-of-bounds
      elements yield the padding value. If not set, out-of-bounds elements
      yield unspecified values.
    * **Store operations**: Out-of-bounds elements are masked during stores.

    .. note::

      Gather/Scatter sparse dimension indices do not need to be unique. Repeated
      sparse index values in a load operation result in the same row being loaded
      multiple times. Repeated sparse index values in a store operation result in
      undefined values in the repeated row.
  }];

  let parameters = (ins
    CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "tile shape", "13.3">:$tile_shape,
    CudaTileStrTypeParam<"::mlir::cuda_tile::TensorViewType", "tensor view", "13.3">:$tensor_view,
    CudaTileStrTypeParam<"uint32_t", "sparse dimension", "13.3">:$sparse_dim,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "padding value">, "13.3">:$padding_value
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//

def CudaTile_TokenType : CudaTileTypeDef<"Token", "token", "token", "13.1"> {
  let summary = "cuda tile token type";
  let description = [{
    Tokens are not runtime values. Their purpose is to explicitly represent
    ordering constraints between token-ordered operations executed within a
    tile block.

    Within a tile block, if two operations :code:`A -> B` are connected by a
    token chain, the compiler makes sure that it is as-if all effects of
    :code:`A` are visible to :code:`B`. If two operations are not connected
    by a token chain, the compiler may arbitrarily reorder them as long as
    SSA dominance is not violated.

    Example:

    ```
    %val = ... : tile<64xf32>
    %tok0 = store_ptr_tko weak %ptr, %val -> token
    %data, %tok1 = load_ptr_tko weak %ptr -> tile<64xf32>, token
    ```

    The above example has a race condition. The store and load operations
    operate on the same memory location and the absence of a token dependency
    means that the effects of the store operation may or may not be visible to
    the load operation. (Or they may be partly visible.) To remove the race
    condition, the store and load operation could be connected by a token.

    If two operations :code:`A -> B` are connected by a token chain, but you
    can prove that the effects of :code:`A` are independent from the effects of
    :code:`B`, then you can remove the apparent token chain because it will
    still be as-if the effects of :code:`A` are visible before the effects of
    :code:`B`.

    Example:

    ```
    // Assume that %ptr0 and %ptr1 are non-aliasing (non-overlapping) pointers.
    %val = ... : tile<64xf32>
    %tok1 = store_ptr_tko weak %ptr0, %val token=%tok0 -> token
    %data, %tok2 = load_ptr_tko weak %ptr1 token=%tok1 -> tile<64xf32>, token
    use(%tok2)
    ```

    In the above example, :code:`%ptr0` and :code:`%ptr1` are non-aliasing.
    Therefore, the effect of the store operation is irrelevant to the load
    operation. Rewriting the IR as follows preserves the semantics:

    ```
    %val = ... : tile<64xf32>
    %tok1 = store_ptr_tko weak %ptr0, %val token=%tok0 -> token
    %data, %tok2 = load_ptr_tko weak %ptr1 token=%tok0 -> tile<64xf32>, token
    %tok3 = join_tokens %tok1, %tok2 : token
    use(%tok3)
    ```

    Note: This is only possible because of the non-overlapping pointers, and
    because the memory operations are both :code:`weak`, so no external thread
    could observe the reordering either.
  }];
}

//===----------------------------------------------------------------------===//
// Any Type
//===----------------------------------------------------------------------===//

def CudaTile_AnyType : AnyTypeOf<[
  CudaTile_NumberType,
  Type<CPred<"::llvm::isa<::mlir::cuda_tile::CudaTileDialectType>($_self)">>
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
  CudaTile_Float8E8M0FNU, CudaTile_Float4E2M1FN,
  CudaTile_Float8E5M3FNU
]>;

def CudaTile_NumberTileType : CudaTile_TileOf<[
  CudaTile_Int1, CudaTile_Int8, CudaTile_Int16, CudaTile_Int32, CudaTile_Int64,
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64,
  CudaTile_TFloat32, CudaTile_Float8E4M3FN, CudaTile_Float8E5M2,
  CudaTile_Float8E8M0FNU, CudaTile_Float4E2M1FN,
  CudaTile_Float8E5M3FNU
]>;

def CudaTile_PointerTileType : CudaTile_TileOf<[CudaTile_PointerType]>;

#ifdef TILE_IR_INCLUDE_TESTS
include "cuda_tile/Dialect/CudaTile/IR/TestingTypes.td"
#endif // TILE_IR_INCLUDE_TESTS

#endif  // CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD
