# MLIR General Imports
from ._ods_common import _cext as _ods_cext
from ._ods_common import (
    equally_sized_accessor as _ods_equally_sized_accessor,
    get_default_loc_context as _ods_get_default_loc_context,
    get_op_result_or_op_results as _get_op_result_or_op_results,
    get_op_result_or_value as _get_op_result_or_value,
    get_op_results_or_values as _get_op_results_or_values,
    segmented_accessor as _ods_segmented_accessor,
)
from ..ir import Context, Type

_ods_ir = _ods_cext.ir

# Cuda Tile imports
from ._cuda_tile_ops_gen import _Dialect
from . import _cuda_tile_enum_gen as _cuda_tile_enum
from . import _cuda_tile_ops_gen as _cuda_tile
from .._mlir_libs import _cuda_tile as _cuda_tile_capi


# =============================================================================
# Minimal Element Type Wrappers (for MmaDescriptor and make_tile_type)
# =============================================================================
# These provide simple wrappers with .mlir_type property for user-facing APIs.
# CUDA Tile code should use MLIR types directly where possible.


class _ElementTypeMeta(type):
    """Metaclass providing mlir_type as a class property."""

    _mlir_type_fn = None

    @property
    def mlir_type(cls):
        if cls._mlir_type_fn is None:
            raise NotImplementedError
        return cls._mlir_type_fn()


class _ElementType(metaclass=_ElementTypeMeta):
    """Base class for element type wrappers."""

    pass


class Boolean(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.IntegerType.get_signless(1))


class Int4(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.IntegerType.get_signless(4))


class Int8(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.IntegerType.get_signless(8))


class Int16(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.IntegerType.get_signless(16))


class Int32(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.IntegerType.get_signless(32))


class Int64(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.IntegerType.get_signless(64))


class Float16(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.F16Type.get())


class BFloat16(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.BF16Type.get())


class TFloat32(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.FloatTF32Type.get())


class Float32(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.F32Type.get())


class Float64(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.F64Type.get())


class Float8E5M2(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.Float8E5M2Type.get())


class Float8E4M3FN(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.Float8E4M3FNType.get())


class Float8E4M3FNUZ(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.Float8E4M3FNUZType.get())


class Float8E8M0FNU(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.Float8E8M0FNUType.get())


class Float4E2M1FN(_ElementType):
    _mlir_type_fn = staticmethod(lambda: _ods_ir.Float4E2M1FNType.get())


try:

    class Float8E5M3FNU(_ElementType):
        _mlir_type_fn = staticmethod(lambda: _ods_ir.Float8E5M3FNUType.get())

except AttributeError:
    pass


def _get_mlir_type(el_type):
    """Extract MLIR type from element type wrapper or return as-is if already MLIR type."""
    if hasattr(el_type, "mlir_type"):
        return el_type.mlir_type
    return el_type


def _infer_mlir_type_from_python(value):
    """Infer MLIR type from a Python value (int, float, bool)."""
    if isinstance(value, bool):
        return _ods_ir.IntegerType.get_signless(1)
    elif isinstance(value, int):
        return _ods_ir.IntegerType.get_signless(32)
    elif isinstance(value, float):
        return _ods_ir.F32Type.get()
    raise ValueError(f"Cannot infer MLIR type from {type(value).__name__}")


def _get_last_operation_name(block):
    ops = block.operations
    if len(ops) == 0:
        return None

    op = ops[len(ops) - 1]
    if hasattr(op, "operation"):
        op = op.operation
    return op.name


def _is_cuda_tile_if_terminator(operation_name):
    return operation_name in (
        "cuda_tile.yield",
        "cuda_tile.break",
        "cuda_tile.continue",
        "cuda_tile.return",
    )


def _ensure_if_region_terminator(block, *, loc=None, ip=None):
    if not _is_cuda_tile_if_terminator(_get_last_operation_name(block)):
        _cuda_tile.YieldOp(operands_=[], loc=loc, ip=ip)


def _resolve_load_store_hint(
    result_tile_type,
    arch,
    latency,
    *,
    allow_tma=None,
    apply_tile_size_gate=True,
):
    """Preserve explicit optimization hints for pointer and view operations."""
    if arch is not None or latency is not None or allow_tma is not None:
        # User-specified path. Preserve user's allow_tma value (None / True /
        # False) so the emitted attr shape matches what they asked for.
        return arch, latency, allow_tma

    return None, None, None


# =============================================================================
# End Element Type Wrappers
# =============================================================================


# Global imports
from itertools import chain
from functools import wraps as _wraps, partialmethod
import inspect as _inspect
import os as _os
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from numbers import Number
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# Types
# =============================================================================

from .._mlir_libs._cuda_tile import (
    FragmentType,
    PointerType,
    TileType,
    TensorViewType,
    PartitionViewType,
    StridedViewType,
    GatherScatterViewType,
    TokenType,
)

# =============================================================================
# Attributes
# =============================================================================

from .._mlir_libs._cuda_tile import (
    AtomicRMWModeAttr,
    IntegerOverflowAttr,
    MemoryOrderingSemanticsAttr,
    MemoryScopeAttr,
    PaddingValueAttr,
    OptimizationHintsAttr,
    RoundingModeAttr,
    SignednessAttr,
    ComparisonOrderingAttr,
    ComparisonPredicateAttr,
    SymbolVisibilityAttr,
)



# =============================================================================
# Enums and helpers
# =============================================================================


class AtomicRMWMode(Enum):
    """
    Enum for atomic read-modify-write operations.

    """

    AND = "and"
    OR = "or"
    XOR = "xor"
    ADD = "add"
    ADDF = "addf"
    MAX = "max"
    MIN = "min"
    UMAX = "umax"
    UMIN = "umin"
    XCHG = "xchg"


class MultimemReductionKind(Enum):
    """Reduction kinds supported by multimem view operations."""

    ADD = "add"
    MIN = "min"
    MAX = "max"
    UMIN = "umin"
    UMAX = "umax"
    AND = "and"
    OR = "or"
    XOR = "xor"


class MemoryScope(Enum):
    """
    Enum for operations that require memory scope
    """

    TL_BLK = "tl_blk"
    DEVICE = "device"
    SYS = "sys"


class PtrAttr(Enum):
    """Pointer communication classification."""

    NONE = "none"
    UNICAST = "unicast"
    MULTICAST = "multicast"


class PaddingValue(Enum):
    """
    Enum for operations that support padding values.
    """

    ZERO = "zero"
    NEG_ZERO = "neg_zero"
    NAN = "nan"
    POS_INF = "pos_inf"
    NEG_INF = "neg_inf"


class MemoryOrderingSemantics(Enum):
    """
    Enum for operations that require memory ordering semantics
    """

    WEAK = "weak"
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"


class RoundingMode(Enum):
    """
    Enum for operations that support rounding mode.
    """

    NEAREST_EVEN = "nearest_even"
    ZERO = "zero"
    NEGATIVE_INF = "negative_inf"
    POSITIVE_INF = "positive_inf"
    APPROX = "approx"
    FULL = "full"
    NEAREST_INT_TO_ZERO = "nearest_int_to_zero"
    NEAREST_AWAY = "nearest_away"


class IntegerOverflow(Enum):
    """
    Enum for operations that support overflow flags.
    """

    NONE = "none"
    NSW = "no_signed_wrap"
    NUW = "no_unsigned_wrap"
    NW = "no_wrap"


class Signedness(Enum):
    """
    Enum for operations that support signedness.
    """

    SIGNED = "signed"
    UNSIGNED = "unsigned"


class ComparisonPredicates(Enum):
    """
    Enum for comparison predicates.
    """

    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"


class ComparisonOrdering(Enum):
    """
    Enum for operations that support comparison ordering.
    """

    ORDERED = "ordered"
    UNORDERED = "unordered"


class SymbolVisibility(Enum):
    """
    Enum for global symbol visibility.

    Controls the visibility and optimization behavior of global variables:
    - PUBLIC: Accessible from host code, never eliminated
    - PRIVATE: Device-only access, enables aggressive optimization
    """

    PUBLIC = "public"
    PRIVATE = "private"


def get_atomic_rmw_mode_attr(
    mode: AtomicRMWMode, context: Optional[Context] = None
) -> AtomicRMWModeAttr:
    """
    Convert an enum value to the corresponding AtomicRMWModeAttr.

    Args:
        mode: AtomicRMWMode enum value
        context: Optional MLIR context

    Returns:
        AtomicRMWModeAttr with the given mode
    """
    assert mode is not None, "AtomicRMWMode cannot be None"

    return AtomicRMWModeAttr.get(mode.value, context)


def get_memory_scope_attr(
    scope: MemoryScope, context: Optional[Context] = None
) -> MemoryScopeAttr:
    """
    Convert an enum value to the corresponding MemoryScopeAttr.

    Args:
        scope: MemoryScope enum value
        context: Optional MLIR context

    Returns:
        MemoryScopeAttr with the given scope
    """
    assert scope is not None, "MemoryScope cannot be None"

    return MemoryScopeAttr.get(scope.value, context)


def get_padding_value_attr(
    padding_value: PaddingValue, context: Optional[Context] = None
) -> PaddingValueAttr:
    """
    Convert an enum value to the corresponding PaddingValueAttr.
    """
    return PaddingValueAttr.get(padding_value.value, context)


def get_symbol_visibility_attr(
    visibility: SymbolVisibility, context: Optional[Context] = None
) -> SymbolVisibilityAttr:
    """
    Convert an enum value to SymbolVisibilityAttr.
    """
    if context is None:
        context = _ods_ir.Context.current
    return SymbolVisibilityAttr.get(visibility.value, context)


def get_memory_ordering_semantics_attr(
    semantics: MemoryOrderingSemantics, context: Optional[Context] = None
) -> MemoryOrderingSemanticsAttr:
    """
    Convert an enum value to the corresponding MemoryOrderingSemanticsAttr.

    Args:
        semantics: MemoryOrderingSemantics enum value
        context: Optional MLIR context

    Returns:
        MemoryOrderingSemanticsAttr with the given semantics
    """
    assert semantics is not None, "MemoryOrderingSemantics cannot be None"

    return MemoryOrderingSemanticsAttr.get(semantics.value, context)


def get_rounding_mode_attr(
    mode: RoundingMode, context: Optional[Context] = None
) -> RoundingModeAttr:
    """
    Convert an enum value to the corresponding RoundingModeAttr.

    Args:
        mode: RoundingMode enum value
        context: Optional MLIR context

    Returns:
        RoundingModeAttr with the given mode
    """

    assert mode is not None, "mode must not be None"

    return RoundingModeAttr.get(mode.value, context)


def get_integer_overflow_attr(
    overflow: IntegerOverflow, context: Optional[Context] = None
) -> IntegerOverflowAttr:
    """
    Convert an enum value to the corresponding IntegerOverflowAttr.
    """

    return IntegerOverflowAttr.get(overflow.value, context)


def get_comparison_predicate_attr(
    predicate: ComparisonPredicates, context: Optional[Context] = None
) -> ComparisonPredicateAttr:
    """
    Convert an enum value to the corresponding ComparisonPredicateAttr.
    """

    assert predicate is not None, "ComparisonPredicate cannot be None"

    return ComparisonPredicateAttr.get(predicate.value, context)


def get_signedness_attr(
    signedness: Signedness, context: Optional[Context] = None
) -> SignednessAttr:
    """
    Convert an enum value to the corresponding SignednessAttr.
    """

    assert signedness is not None, "Signedness cannot be None"

    return SignednessAttr.get(signedness.value, context)


def get_comparison_ordering_attr(
    ordering: ComparisonOrdering, context: Optional[Context] = None
) -> ComparisonOrderingAttr:
    """
    Convert an enum value to the corresponding ComparisonOrderingAttr.
    """

    assert ordering is not None, "ComparisonOrdering cannot be None"

    return ComparisonOrderingAttr.get(ordering.value, context)


# =============================================================================
# Supported MMA Configurations
# =============================================================================


class MMAConfig:
    """Base class for MMA configuration."""

    def __init__(
        self,
        name: str,
        lhs_dtype,
        rhs_dtype,
        acc_dtype,
        lhs_signed: Signedness = Signedness.SIGNED,
        rhs_signed: Signedness = Signedness.SIGNED,
    ):
        self.name = name
        self.lhs_dtype = lhs_dtype
        self.rhs_dtype = rhs_dtype
        self.acc_dtype = acc_dtype
        self.lhs_signed = lhs_signed
        self.rhs_signed = rhs_signed

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"MMAConfig({self.name})"

    def matches_types(self, lhs_mlir_type, rhs_mlir_type, acc_mlir_type):
        """Check if the given MLIR types match this configuration"""
        lhs_mlir_type_expected = _get_mlir_type(self.lhs_dtype)
        rhs_mlir_type_expected = _get_mlir_type(self.rhs_dtype)
        acc_mlir_type_expected = _get_mlir_type(self.acc_dtype)
        return (
            lhs_mlir_type_expected == lhs_mlir_type
            and rhs_mlir_type_expected == rhs_mlir_type
            and acc_mlir_type_expected == acc_mlir_type
        )


# Concrete MMA Configuration Classes
class MMAConfig_U8_U8_S32(MMAConfig):
    """u8 x u8 -> s32"""

    def __init__(self):
        super().__init__(
            name="u8xu8->s32",
            lhs_dtype=Int8,
            rhs_dtype=Int8,
            acc_dtype=Int32,
            lhs_signed=Signedness.UNSIGNED,
            rhs_signed=Signedness.UNSIGNED,
        )


class MMAConfig_S8_S8_S32(MMAConfig):
    """s8 x s8 -> s32"""

    def __init__(self):
        super().__init__(
            name="s8xs8->s32",
            lhs_dtype=Int8,
            rhs_dtype=Int8,
            acc_dtype=Int32,
            lhs_signed=Signedness.SIGNED,
            rhs_signed=Signedness.SIGNED,
        )


class MMAConfig_E4M3_E4M3_F32(MMAConfig):
    """e4m3 x e4m3 -> f32"""

    def __init__(self):
        super().__init__(
            name="e4m3xe4m3->f32",
            lhs_dtype=Float8E4M3FN,
            rhs_dtype=Float8E4M3FN,
            acc_dtype=Float32,
        )


class MMAConfig_E4M3_E4M3_F16(MMAConfig):
    """e4m3 x e4m3 -> f16"""

    def __init__(self):
        super().__init__(
            name="e4m3xe4m3->f16",
            lhs_dtype=Float8E4M3FN,
            rhs_dtype=Float8E4M3FN,
            acc_dtype=Float16,
        )


class MMAConfig_E5M2_E5M2_F32(MMAConfig):
    """e5m2 x e5m2 -> f32"""

    def __init__(self):
        super().__init__(
            name="e5m2xe5m2->f32",
            lhs_dtype=Float8E5M2,
            rhs_dtype=Float8E5M2,
            acc_dtype=Float32,
        )


class MMAConfig_E5M2_E5M2_F16(MMAConfig):
    """e5m2 x e5m2 -> f16"""

    def __init__(self):
        super().__init__(
            name="e5m2xe5m2->f16",
            lhs_dtype=Float8E5M2,
            rhs_dtype=Float8E5M2,
            acc_dtype=Float16,
        )


class MMAConfig_F16_F16_F32(MMAConfig):
    """f16 x f16 -> f32"""

    def __init__(self):
        super().__init__(
            name="f16xf16->f32",
            lhs_dtype=Float16,
            rhs_dtype=Float16,
            acc_dtype=Float32,
        )


class MMAConfig_F16_F16_F16(MMAConfig):
    """f16 x f16 -> f16"""

    def __init__(self):
        super().__init__(
            name="f16xf16->f16",
            lhs_dtype=Float16,
            rhs_dtype=Float16,
            acc_dtype=Float16,
        )


class MMAConfig_BF16_BF16_F32(MMAConfig):
    """bf16 x bf16 -> f32"""

    def __init__(self):
        super().__init__(
            name="bf16xbf16->f32",
            lhs_dtype=BFloat16,
            rhs_dtype=BFloat16,
            acc_dtype=Float32,
        )


class MMAConfig_F32_F32_F32(MMAConfig):
    """f32 x f32 -> f32"""

    def __init__(self):
        super().__init__(
            name="f32xf32->f32",
            lhs_dtype=Float32,
            rhs_dtype=Float32,
            acc_dtype=Float32,
        )


class MMAConfig_TF32_TF32_F32(MMAConfig):
    """tf32 x tf32 -> f32"""

    def __init__(self):
        super().__init__(
            name="tf32xtf32->f32",
            lhs_dtype=TFloat32,
            rhs_dtype=TFloat32,
            acc_dtype=Float32,
        )


class MMAConfig_F64_F64_F64(MMAConfig):
    """f64 x f64 -> f64"""

    def __init__(self):
        super().__init__(
            name="f64xf64->f64",
            lhs_dtype=Float64,
            rhs_dtype=Float64,
            acc_dtype=Float64,
        )


# Registry of supported MMA configurations for caching
_SUPPORTED_MMA_CONFIGS = None


def _initialize_mma_configs():
    """Initialize MMA configurations using automatic subclass discovery"""
    global _SUPPORTED_MMA_CONFIGS
    if _SUPPORTED_MMA_CONFIGS is not None:
        return _SUPPORTED_MMA_CONFIGS

    configs = []

    try:
        # Automatically discover all MMAConfig subclasses
        for config_class in MMAConfig.__subclasses__():
            try:
                config = config_class()
                configs.append(config)
            except Exception:
                continue

    except Exception:
        configs = []

    _SUPPORTED_MMA_CONFIGS = configs
    return _SUPPORTED_MMA_CONFIGS


def find_mma_config(lhs_mlir_type, rhs_mlir_type, acc_mlir_type):
    """Find a matching MMA configuration for the given MLIR types"""
    configs = _initialize_mma_configs()

    for config in configs:
        if config.matches_types(lhs_mlir_type, rhs_mlir_type, acc_mlir_type):
            return config
    return None


def get_supported_mma_configs():
    """Get all supported MMA configurations"""
    return _initialize_mma_configs()


# =============================================================================
# End MMA Configuration System
# =============================================================================


# =============================================================================
# MMA Scaled Configuration System
# =============================================================================


class MMAScaledConfig:
    """Base class for scaled MMA configuration."""

    def __init__(
        self,
        name: str,
        operand_dtype,
        scale_dtype,
        acc_dtype=Float32,
        scale_factor=None,
    ):
        self.name = name
        self.operand_dtype = operand_dtype
        self.scale_dtype = scale_dtype
        self.acc_dtype = acc_dtype
        # Default: 16 for e4m3 scale (4X / mxf4nvf4), 32 otherwise (2X / mxf4, mxf8f6f4)
        if scale_factor is None:
            scale_factor = 16 if scale_dtype == Float8E4M3FN else 32
        self.scale_factor = scale_factor

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"MMAScaledConfig({self.name})"

    def matches_types(self, operand_mlir_type, scale_mlir_type):
        """Check if the given MLIR types match this configuration"""
        operand_mlir_type_expected = _get_mlir_type(self.operand_dtype)
        scale_mlir_type_expected = _get_mlir_type(self.scale_dtype)
        return (
            operand_mlir_type_expected == operand_mlir_type
            and scale_mlir_type_expected == scale_mlir_type
        )


# Concrete MMA Scaled Configuration Classes
# MXFP8 configurations: f8E8M0FNU scale with f8E5M2/f8E4M3FN operands
class MMAScaledConfig_E5M2_E8M0(MMAScaledConfig):
    """e5m2 operands with e8m0 scale -> f32 (mxfp8)"""

    def __init__(self):
        super().__init__(
            name="e5m2+e8m0->f32",
            operand_dtype=Float8E5M2,
            scale_dtype=Float8E8M0FNU,
        )


class MMAScaledConfig_E4M3_E8M0(MMAScaledConfig):
    """e4m3 operands with e8m0 scale -> f32 (mxfp8)"""

    def __init__(self):
        super().__init__(
            name="e4m3+e8m0->f32",
            operand_dtype=Float8E4M3FN,
            scale_dtype=Float8E8M0FNU,
        )


# MXFP4 configuration: f8E8M0FNU scale with f4E2M1FN operands
class MMAScaledConfig_E2M1_E8M0(MMAScaledConfig):
    """e2m1 operands with e8m0 scale -> f32 (mxfp4)"""

    def __init__(self):
        super().__init__(
            name="e2m1+e8m0->f32",
            operand_dtype=Float4E2M1FN,
            scale_dtype=Float8E8M0FNU,
        )


# NVFP4 configuration: f8E4M3FN scale with f4E2M1FN operands
class MMAScaledConfig_E2M1_E4M3(MMAScaledConfig):
    """e2m1 operands with e4m3 scale -> f32 (nvfp4)"""

    def __init__(self):
        super().__init__(
            name="e2m1+e4m3->f32",
            operand_dtype=Float4E2M1FN,
            scale_dtype=Float8E4M3FN,
        )


# MXFP4 4X configuration: f8E8M0FNU scale with f4E2M1FN operands, vecSize=16 (mxf4nvf4)
class MMAScaledConfig_E2M1_E8M0_4X(MMAScaledConfig):
    """e2m1 operands with e8m0 scale, 4X mode -> f32 (mxf4nvf4)"""

    def __init__(self):
        super().__init__(
            name="e2m1+e8m0(4X)->f32",
            operand_dtype=Float4E2M1FN,
            scale_dtype=Float8E8M0FNU,
            scale_factor=16,
        )


# Registry of supported MMA Scaled configurations for caching
_SUPPORTED_MMA_SCALED_CONFIGS = None


def _initialize_mma_scaled_configs():
    """Initialize MMA Scaled configurations using automatic subclass discovery"""
    global _SUPPORTED_MMA_SCALED_CONFIGS
    if _SUPPORTED_MMA_SCALED_CONFIGS is not None:
        return _SUPPORTED_MMA_SCALED_CONFIGS

    configs = []

    try:
        # Automatically discover all MMAScaledConfig subclasses
        for config_class in MMAScaledConfig.__subclasses__():
            try:
                config = config_class()
                configs.append(config)
            except Exception:
                continue

    except Exception:
        configs = []

    _SUPPORTED_MMA_SCALED_CONFIGS = configs
    return _SUPPORTED_MMA_SCALED_CONFIGS


def find_mma_scaled_config(operand_mlir_type, scale_mlir_type):
    """Find a matching MMA Scaled configuration for the given MLIR types"""
    configs = _initialize_mma_scaled_configs()

    for config in configs:
        if config.matches_types(operand_mlir_type, scale_mlir_type):
            return config
    return None


def get_supported_mma_scaled_configs():
    """Get all supported MMA Scaled configurations"""
    return _initialize_mma_scaled_configs()


# =============================================================================
# End MMA Scaled Configuration System
# =============================================================================


def _binary_op(lhs, rhs, op: str, predAtt="", is_reversed=False) -> "Tile":
    """Generate arithmatic binary operations."""

    rhs = _check_is_rhs_tile(lhs, rhs)

    op = getattr(_cuda_tile, f"{op}Op")

    return return_results(op(lhs, rhs))


def _comparison_op(
    lhs,
    rhs,
    comparison_predicate: ComparisonPredicates,
    signedness: Signedness = Signedness.SIGNED,
    ordering: ComparisonOrdering = ComparisonOrdering.ORDERED,
):
    """Generate comparison operations."""

    rhs = _check_is_rhs_tile(lhs, rhs)

    if isinstance(lhs.element_type, _ods_ir.IntegerType) and isinstance(
        rhs.element_type, _ods_ir.IntegerType
    ):
        return cmpi(comparison_predicate, lhs, rhs, signedness)
    elif isinstance(lhs.element_type, _ods_ir.FloatType) and isinstance(
        rhs.element_type, _ods_ir.FloatType
    ):
        return cmpf(comparison_predicate, ordering, lhs, rhs)
    else:
        raise TypeError(
            f"Unsupported element types: {lhs.element_type}, {rhs.element_type}"
        )


class Tile(_ods_ir.Value):
    """
    A class representing a Tile object with an associated type and value.
    Inherits from _ods_ir.Value, and acts as a wrapper around an IR value with
    a specified tile type.
    """

    def __init__(self, value: _ods_ir.Value, type: _ods_ir.Type):
        tile_type = TileType(type)
        if isinstance(value, _ods_ir.Value) is False:
            raise Exception("Tile value is not IR Value")
        super().__init__(value)
        self.tile_type = tile_type
        self.value = value

    @property
    def element_type(self):
        return self.tile_type.element_type

    @property
    def shape(self):
        return self.tile_type.shape

    @property
    def num_elements(self):
        res = 1
        for s in self.shape:
            res *= s
        return res

    def __call__(self, *args, **kwargs):
        return self.value

    def __repr__(self):
        shape_str = "x".join(
            map(str, chain(self.tile_type.shape, (self.tile_type.element_type,)))
        )
        return f"cuda_tile.Tile({shape_str})"

    def __abs__(self):
        if isinstance(self.element_type, _ods_ir.IntegerType):
            return absi(self)
        if isinstance(self.element_type, _ods_ir.FloatType):
            return absf(self)
        raise TypeError(
            "Cannot perform absolute value on non-numeric element type ",
            self.element_type,
        )

    def __add__(self, rhs):
        return add(self, rhs)

    def __pow__(self, rhs):
        return pow(self, rhs)

    def __rpow__(self, rhs):
        return pow(rhs, self)

    def __neg__(self):
        if isinstance(self.element_type, _ods_ir.IntegerType):
            # TODO: after sign is tracked, make invalid to use on unsigned int
            return negi(self)
        if isinstance(self.element_type, _ods_ir.FloatType):
            return negf(self)
        raise TypeError(
            "Cannot perform neg on non-numeric element type ", self.element_type
        )

    def __radd__(self, rhs):
        return add(self, rhs)

    def __mod__(self, rhs):
        return rem(self, rhs)

    def __rmod__(self, rhs):
        return rem(self, rhs)

    def __sub__(self, rhs):
        return sub(self, rhs)

    def __rsub__(self, rhs):
        return sub(self, rhs)

    def __mul__(self, rhs):
        return mul(self, rhs)

    def __rmul__(self, rhs):
        return mul(self, rhs)

    def __floordiv__(self, rhs):
        if not isinstance(self.element_type, _ods_ir.IntegerType):
            raise TypeError("// is only supported on integer tiles")
        return floordivi(self, rhs)

    def __rfloordiv__(self, rhs):
        if not isinstance(self.element_type, _ods_ir.IntegerType):
            raise TypeError("// is only supported on integer tiles")
        return floordivi(self, rhs)

    def __and__(self, rhs):
        return andi(self, rhs)

    def __rand__(self, rhs):
        return andi(self, rhs)

    def __or__(self, rhs):
        return ori(self, rhs)

    def __ror__(self, rhs):
        return ori(self, rhs)

    def __rshift__(self, rhs):
        return shri(self, rhs)

    def __lshift__(self, rhs):
        return shli(self, rhs)

    def __truediv__(self, rhs):
        return div(self, rhs)

    __ne__ = partialmethod(
        _comparison_op,
        comparison_predicate=ComparisonPredicates.NOT_EQUAL,
        signedness=Signedness.SIGNED,
        ordering=ComparisonOrdering.ORDERED,
    )
    __lt__ = partialmethod(
        _comparison_op,
        comparison_predicate=ComparisonPredicates.LESS_THAN,
        signedness=Signedness.SIGNED,
        ordering=ComparisonOrdering.ORDERED,
    )
    __le__ = partialmethod(
        _comparison_op,
        comparison_predicate=ComparisonPredicates.LESS_THAN_OR_EQUAL,
        signedness=Signedness.SIGNED,
        ordering=ComparisonOrdering.ORDERED,
    )
    __gt__ = partialmethod(
        _comparison_op,
        comparison_predicate=ComparisonPredicates.GREATER_THAN,
        signedness=Signedness.SIGNED,
        ordering=ComparisonOrdering.ORDERED,
    )
    __ge__ = partialmethod(
        _comparison_op,
        comparison_predicate=ComparisonPredicates.GREATER_THAN_OR_EQUAL,
        signedness=Signedness.SIGNED,
        ordering=ComparisonOrdering.ORDERED,
    )
    __eq__ = partialmethod(
        _comparison_op,
        comparison_predicate=ComparisonPredicates.EQUAL,
        signedness=Signedness.SIGNED,
        ordering=ComparisonOrdering.ORDERED,
    )

    # TODO implement them once we are ready
    # __truediv__ = partialmethod(_binary_op, op="Div")
    # __xor__ = partialmethod(_binary_op, op="XOr")
    # __and__ = partialmethod(_binary_op, op="And")
    # __or__ = partialmethod(_binary_op, op="Or")


class Pointer(Tile):
    """
    Represents a pointer to memory as a scalar tile type.
    This is an annotation class: not all pointer tiles are of the Pointer class,
    but tiles of the Pointer class are definitely pointer tiles.
    """

    def __init__(self, value: _ods_ir.Value, typ: _ods_ir.Type):
        super().__init__(value, typ)
        if not self.shape == [] or not PointerType.isinstance(self.element_type):
            raise TypeError("Pointer must be a scalar tile of pointer element type")


class TileView(_ods_ir.Value):
    """
    Represents a view that can be used to access tiles in global memory.
    """

    @property
    def view_tile_type(self) -> TileType:
        raise NotImplementedError()

    @property
    def view_index_rank(self) -> int:
        raise NotImplementedError()


class TensorView(TileView):
    """
    A class representing a TensorView object with an associated type and value.
    Inherits from _ods_ir.Value, and acts as a wrapper around an IR value with
    a specified tensor view type.
    """

    tensor_view_type: TensorViewType
    value: _ods_ir.Value

    def __init__(self, value: _ods_ir.Value, type: _ods_ir.Type):
        tensor_view_type = TensorViewType(type)
        if isinstance(value, _ods_ir.Value) is False:
            raise Exception("TensorView value is not IR Value")
        super().__init__(value)
        self.tensor_view_type = tensor_view_type
        self.value = value

    @property
    def element_type(self):
        return self.tensor_view_type.element_type

    @property
    def shape(self):
        return self.tensor_view_type.shape

    @property
    def strides(self):
        return self.tensor_view_type.strides

    @property
    def index_type(self):
        """Returns the MLIR index type for this tensor view."""
        return self.tensor_view_type.index_type

    @property
    def view_tile_type(self) -> TileType:
        return TileType(self.tensor_view_type.view_tile_type)

    @property
    def view_index_rank(self) -> int:
        return self.tensor_view_type.view_index_rank


class PartitionView(TileView):
    """
    A class representing a PartitionView object with an associated type and
    value. Inherits from _ods_ir.Value, and acts as a wrapper around an IR
    value with a specified tile partition view type.
    """

    view_type: PartitionViewType
    value: _ods_ir.Value

    def __init__(self, value: _ods_ir.Value, type: _ods_ir.Type):
        view_type = PartitionViewType(type)
        if isinstance(value, _ods_ir.Value) is False:
            raise Exception("PartitionView value is not IR Value")
        super().__init__(value)
        self.view_type = view_type
        self.value = value

    @property
    def tile_shape(self):
        return self.view_type.tile_shape

    @property
    def tensor_view_type(self):
        return self.view_type.tensor_view_type

    @property
    def dim_map(self):
        return self.view_type.dim_map

    @property
    def view_tile_type(self) -> TileType:
        return TileType(self.view_type.view_tile_type)

    @property
    def view_index_rank(self) -> int:
        return self.view_type.view_index_rank


class StridedView(TileView):
    """
    A class representing a StridedView object with an associated type and
    value. Inherits from _ods_ir.Value, and acts as a wrapper around an IR
    value with a specified tile strided view type.
    """

    view_type: StridedViewType
    value: _ods_ir.Value

    def __init__(self, value: _ods_ir.Value, type: _ods_ir.Type):
        view_type = StridedViewType(type)
        if isinstance(value, _ods_ir.Value) is False:
            raise Exception("StridedView value is not IR Value")
        super().__init__(value)
        self.view_type = view_type
        self.value = value

    @property
    def tile_shape(self):
        return self.view_type.tile_shape

    @property
    def tensor_view_type(self):
        return self.view_type.tensor_view_type

    @property
    def dim_map(self):
        return self.view_type.dim_map

    @property
    def view_tile_type(self) -> TileType:
        return TileType(self.view_type.view_tile_type)

    @property
    def view_index_rank(self) -> int:
        return self.view_type.view_index_rank


class GatherScatterView(TileView):
    """
    A class representing a GatherScatterView object with an associated type and
    value. Inherits from _ods_ir.Value, and acts as a wrapper around an IR value
    with a specified tile gather/scatter view type.
    """

    view_type: GatherScatterViewType
    value: _ods_ir.Value

    def __init__(self, value: _ods_ir.Value, type: _ods_ir.Type):
        view_type = GatherScatterViewType(type)
        if isinstance(value, _ods_ir.Value) is False:
            raise Exception("GatherScatterView value is not IR Value")
        super().__init__(value)
        self.view_type = view_type
        self.value = value

    @property
    def tile_shape(self):
        return self.view_type.tile_shape

    @property
    def tensor_view_type(self):
        return TensorViewType(self.view_type.tensor_view_type)

    @property
    def sparse_dim(self):
        return self.view_type.sparse_dim

    @property
    def padding_value(self):
        return self.view_type.padding_value

    @property
    def view_tile_type(self) -> TileType:
        return TileType(self.view_type.view_tile_type)

    @property
    def view_index_rank(self) -> int:
        return self.view_type.view_index_rank


class Token(_ods_ir.Value):
    """
    A class representing a Token object.
    """

    value: _ods_ir.Value

    def __init__(self, value: _ods_ir.Value):
        if isinstance(value, _ods_ir.Value) is False:
            raise Exception("Token value is not IR Value")
        if not TokenType.isinstance(value.type):
            raise Exception("Token must have TokenType type")
        super().__init__(value)
        self.value = value


# =============================================================================
# Utils
# =============================================================================


def cuda_tile_op(opFunc):
    """
    This is a decorator that needs to be used in each cuda_tile OP to
    manage pre-generation things. Currently, it only generate source
    location.
    """

    @_wraps(opFunc)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            frame = _inspect.currentframe().f_back
            file_loc = _ods_ir.Location.file(
                frame.f_code.co_filename, frame.f_lineno, 0
            )
            loc = _ods_ir.Location.name(frame.f_code.co_name, childLoc=file_loc)
        res_or_list = opFunc(*args, **kwargs, loc=loc)
        return res_or_list

    return wrapper


def _index_list_to_tiles(index: List[Tile | int]) -> List[Tile]:
    """
    Ensures all tiles in index are scalar integer tiles of the same type,
    and converts constant indices to tiles of that type.
    """

    dynamic_indices = filter(lambda x: isinstance(x, Tile), index)
    index_type = next(
        map(lambda x: x.tile_type, dynamic_indices), make_tile_type(Int64, [])
    )

    if (
        not isinstance(index_type.element_type, _ods_ir.IntegerType)
        or len(index_type.shape) != 0
    ):
        raise ValueError(
            f"Expected index values to be scalar integer tiles, got {index_type}"
        )

    index_type_bitwidth = index_type.element_type.width

    index_tiles = []
    for i, v in enumerate(index):
        if isinstance(v, Tile):
            if v.tile_type != index_type:
                raise TypeError(
                    f"Expected indices of type {index_type}, "
                    f"got index {i} of type {v.tile_type}"
                )
            index_tiles.append(v)
        else:
            if v.bit_length() > index_type_bitwidth:
                raise ValueError(
                    f"Constant index value {v} is too large for index type {index_type}"
                )
            index_tiles.append(constant(v, tile_type=index_type))

    return index_tiles


def _inbounds_attr_or_default(
    inbounds: Optional[Sequence[bool]], rank: int
) -> _ods_ir.DenseBoolArrayAttr:
    if inbounds is None:
        return _ods_ir.DenseBoolArrayAttr.get([False] * rank)

    inbounds_values = list(inbounds)
    if len(inbounds_values) != rank:
        raise ValueError(f"Expected {rank} inbounds values, got {len(inbounds_values)}")

    for i, value in enumerate(inbounds_values):
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected inbounds value {i} to be bool, got {type(value).__name__}"
            )

    return _ods_ir.DenseBoolArrayAttr.get(inbounds_values)


def _gather_scatter_index_list_to_tiles(
    view: GatherScatterView, index: List[Tile | int]
) -> List[Tile]:
    """
    Convert gather/scatter view indices to tiles.

    The sparse dimension is a 1D integer tile with length equal to the sparse
    tile dimension. Other dimensions are scalar integer tiles or constants.
    """

    sparse_index = index[view.sparse_dim]
    if not isinstance(sparse_index, Tile):
        raise TypeError(
            "Expected gather/scatter sparse dimension index to be a Tile, "
            f"got {type(sparse_index).__name__}"
        )
    if sparse_index.shape != [view.tile_shape[view.sparse_dim]]:
        raise ValueError(
            "Expected gather/scatter sparse dimension index shape "
            f"{[view.tile_shape[view.sparse_dim]]}, got {sparse_index.shape}"
        )
    if not isinstance(sparse_index.element_type, _ods_ir.IntegerType):
        raise TypeError(
            "Expected gather/scatter sparse dimension index element type to be "
            f"integer, got {sparse_index.element_type}"
        )

    index_element_type = sparse_index.element_type
    index_type_bitwidth = index_element_type.width
    index_tiles = []
    for i, v in enumerate(index):
        expected_shape = [view.tile_shape[i]] if i == view.sparse_dim else []
        if isinstance(v, Tile):
            if v.shape != expected_shape:
                raise ValueError(
                    f"Expected index {i} shape {expected_shape}, got {v.shape}"
                )
            if v.element_type != index_element_type:
                raise TypeError(
                    f"Expected index {i} element type {index_element_type}, "
                    f"got {v.element_type}"
                )
            index_tiles.append(v)
        else:
            if i == view.sparse_dim:
                raise TypeError(
                    "Expected gather/scatter sparse dimension index to be a Tile"
                )
            if v.bit_length() > index_type_bitwidth:
                raise ValueError(
                    f"Constant index value {v} is too large for index type "
                    f"{index_element_type}"
                )
            index_tiles.append(
                constant(v, tile_type=make_tile_type(index_element_type, []))
            )

    return index_tiles


def _view_index_list_to_tiles(
    view: TileView, index: Sequence[Tile | int]
) -> List[Tile]:
    if isinstance(view, GatherScatterView):
        return _gather_scatter_index_list_to_tiles(view, list(index))
    return _index_list_to_tiles(list(index))


def _wrap_result_value(value):
    result_type = value.type
    if TokenType.isinstance(result_type):
        return Token(value)
    if TileType.isinstance(result_type):
        return Tile(value, result_type)
    raise ValueError(f"Unsupported result type {result_type}")


def return_results(op) -> Union[
    Tile,
    Token,
    Tuple[Union[Tile, Token], ...],
    None,
]:
    """
    Return op results as Tile, Token, or a tuple of those wrappers.

    - If the op has 1 result and it's a Token -> return Token
    - If the op has 1 result and it's a Tile -> return Tile
    - If the op has >1 results -> return a tuple of matching wrappers
    """

    results = op.results

    if len(results) == 1:
        try:
            return _wrap_result_value(results[0])
        except Exception as e:
            raise ValueError(f"Failed to create result from single op result: {e}")

    elif len(results) >= 2:
        try:
            converted: List[Union[Tile, Token]] = []
            for i, v in enumerate(results):
                try:
                    converted.append(_wrap_result_value(v))
                except ValueError as e:
                    raise ValueError(f"Unsupported result type for result {i}: {e}")
            return tuple(converted)
        except Exception as e:
            raise ValueError(f"Failed to process multiple results: {e}")

    else:
        # The operation has no results.
        return None


def return_tensor_view(op) -> TensorView:
    value = _get_op_result_or_op_results(op)
    return TensorView(_get_op_result_or_op_results(op), value.type)


def return_partition_view(op) -> PartitionView:
    value = _get_op_result_or_op_results(op)
    return PartitionView(_get_op_result_or_op_results(op), value.type)


def return_strided_view(op) -> StridedView:
    value = _get_op_result_or_op_results(op)
    return StridedView(_get_op_result_or_op_results(op), value.type)


def return_gather_scatter_view(op) -> GatherScatterView:
    value = _get_op_result_or_op_results(op)
    return GatherScatterView(value, value.type)


def _ensure_attr(value, type):
    """
    If the given value is an attribute, return it. Otherwise, turn it into a
    FloatAttr or IntegerAttr, depending on the given type.
    """

    if isinstance(value, _ods_ir.Attribute):
        return value
    else:
        if isinstance(type, _ods_ir.FloatType):
            return _ods_ir.FloatAttr.get(type, value)
        else:
            assert isinstance(
                type, _ods_ir.IntegerType
            ), "expected integer or float type"
            return _ods_ir.IntegerAttr.get(type, value)


@_ods_cext.register_operation(_Dialect, replace=True)
class _ConstantOp(_cuda_tile.ConstantOp):
    """Specialization for the constant op class."""

    def __init__(self, ty, values, *, loc=None, ip=None):
        assert isinstance(ty, TileType), "expected tile type"
        el_ty = ty.element_type
        assert isinstance(
            el_ty, (_ods_ir.FloatType, _ods_ir.IntegerType)
        ), "expected integer or float element type"
        attrs = [_ensure_attr(v, el_ty) for v in values]
        super().__init__(_ods_ir.DenseElementsAttr.get(attrs, ty), loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class _GlobalOp(_cuda_tile.GlobalOp):
    """Specialization for the global op class."""

    def __init__(
        self,
        ty,
        sym_name,
        values,
        *,
        alignment=None,
        symbol_visibility: SymbolVisibility | None = None,
        constant: bool = False,
        loc=None,
        ip=None,
    ):
        assert isinstance(ty, TileType), "expected tile type"
        el_ty = ty.element_type
        assert isinstance(
            el_ty, (_ods_ir.FloatType, _ods_ir.IntegerType)
        ), "expected integer or float element type"
        attrs = [_ensure_attr(v, el_ty) for v in values]

        # Build keyword arguments
        kwargs = {}
        if alignment is not None:
            kwargs["alignment"] = alignment
        if symbol_visibility is not None:
            # Convert enum to attribute using helper function
            kwargs["symbol_visibility"] = get_symbol_visibility_attr(symbol_visibility)
        if constant:
            # UnitAttr is used for constant (present = true, absent = false)
            kwargs["constant"] = _ods_ir.UnitAttr.get()

        super().__init__(
            sym_name, _ods_ir.DenseElementsAttr.get(attrs, ty), loc=loc, ip=ip, **kwargs
        )


def make_tile_type(
    el_type, shape: Union[int, str, Sequence[Union[int, str]]] = None
) -> TileType:
    """Create a TileType with a specified element type and shape.

    Args:
        el_type: A type wrapper, raw MLIR type, or symbolic dtype name.
        shape: An integer dimension or sequence of integer/symbolic dimensions.
    """
    shape = [shape] if isinstance(shape, (int, str)) else list(shape or [])
    for dim in shape:
        if isinstance(dim, int) and dim >= 0:
            continue
        if isinstance(dim, str) and dim:
            continue
        raise ValueError(f"Invalid tile dimension {dim!r} in shape {shape}")

    if isinstance(el_type, str) or any(isinstance(dim, str) for dim in shape):
        dims_asm = "x".join(
            str(dim) if isinstance(dim, int) else f"[{dim}]" for dim in shape
        )
        if isinstance(el_type, str):
            element_asm = f"!cuda_tile.preview$sym_t<{el_type}>"
        else:
            element_asm = str(_get_mlir_type(el_type))
        separator = "x" if dims_asm else ""
        return TileType(
            _ods_ir.Type.parse(
                f"!cuda_tile.tile<{dims_asm}{separator}{element_asm}>"
            )
        )

    mlir_type = _get_mlir_type(el_type)
    tile_type = TileType.get(shape, mlir_type)

    if tile_type is None:
        type_name = getattr(el_type, "__name__", type(el_type).__name__)
        raise RuntimeError(
            f"Error creating TileType with shape {shape} and element type {type_name}"
        )

    return tile_type


def make_pointer_type(el_type, *, ptr_attr: PtrAttr | None = None) -> PointerType:
    """Create a pointer type with an optional communication classification."""
    mlir_type = _get_mlir_type(el_type)
    if ptr_attr is None:
        return PointerType.get(mlir_type)
    if not isinstance(ptr_attr, PtrAttr):
        raise TypeError(f"Expected PtrAttr, got {type(ptr_attr).__name__}")
    return PointerType(
        _ods_ir.Type.parse(
            f"!cuda_tile.ptr<{mlir_type}, #cuda_tile.ptr_attr<{ptr_attr.value}>>"
        )
    )


def make_tensor_view_type(
    el_type,
    shape: List[int | None] | None = None,
    strides: List[int | None] | None = None,
    *,
    ptr_attr: PtrAttr | None = None,
) -> TensorViewType:
    """Creates a TensorViewType from an element, a shape and strides.

    Args:
        el_type: Element type - can be a type wrapper (Int32, Float32, etc.) or raw MLIR type
        shape: Shape as list of ints or None values
        strides: Strides as list of ints or None values
    """
    shape = shape if shape is not None else []
    strides = strides if strides is not None else []

    if not all(dim is None or (isinstance(dim, int) and dim >= 0) for dim in shape):
        raise ValueError(
            f"Shape must be a list of non-negative ints or None values, got {shape}"
        )
    if not all(dim is None or (isinstance(dim, int) and dim >= 0) for dim in strides):
        raise ValueError(
            f"Strides must be a list of non-negative ints or None values, got {strides}"
        )
    if ptr_attr is not None and not isinstance(ptr_attr, PtrAttr):
        raise TypeError(f"Expected PtrAttr, got {type(ptr_attr).__name__}")

    elem_mlir_type = _get_mlir_type(el_type)
    if ptr_attr is None:
        tensor_view_type = TensorViewType.get(elem_mlir_type, shape, strides)
    else:
        shape_text = "x".join("?" if dim is None else str(dim) for dim in shape)
        shape_prefix = f"{shape_text}x" if shape_text else ""
        strides_text = ",".join("?" if dim is None else str(dim) for dim in strides)
        tensor_view_type = TensorViewType(
            _ods_ir.Type.parse(
                f"!cuda_tile.tensor_view<{shape_prefix}{elem_mlir_type}, "
                f"strides=[{strides_text}], #cuda_tile.ptr_attr<{ptr_attr.value}>>"
            )
        )

    if tensor_view_type is None:
        type_name = getattr(el_type, "__name__", type(el_type).__name__)
        raise RuntimeError(
            f"Error creating TensorViewType with element type {type_name}, "
            f"shape {shape}, strides {strides}"
        )

    return tensor_view_type


def _check_partition_view_like_type(
    tensor_view_type, tile_shape: List[int], dim_map: List[int]
):
    if not isinstance(tensor_view_type, TensorViewType):
        raise TypeError(f"Expected tensor view type, got {tensor_view_type}")

    if not isinstance(tile_shape, list) and not isinstance(tile_shape, tuple):
        raise TypeError(
            f"Expected tile_shape to be a list or tuple, got {type(tile_shape).__name__}"
        )

    if not all(isinstance(dim, int) for dim in tile_shape):
        raise TypeError(
            f"Expected tile_shape to be an array of integers, got {tile_shape}"
        )

    if not all(dim > 0 for dim in tile_shape):
        raise ValueError(
            f"Expected tile_shape dimensions to be positive, got {tile_shape}"
        )

    if len(tile_shape) != len(tensor_view_type.shape):
        raise ValueError(
            f"Expected tile shape of same rank as tensor view, got tile shape "
            f"{tile_shape} and tensor view shape {tensor_view_type.shape}"
        )

    if not isinstance(dim_map, list) and not isinstance(dim_map, tuple):
        raise TypeError(
            f"Expected dim_map to be a list or tuple, got {type(dim_map).__name__}"
        )

    if not all(isinstance(dim, int) for dim in dim_map):
        raise TypeError(f"Expected dim_map to be an array of integers, got {dim_map}")

    if len(dim_map) != len(tile_shape):
        raise ValueError(
            f"Expected dim_map length to match tile_shape length, got {len(dim_map)} vs {len(tile_shape)}"
        )

    if set(dim_map) != set(range(len(tensor_view_type.shape))):
        raise ValueError(
            f"Dim map should map exactly to the dimensions of the tensor view, got {dim_map}"
        )


def make_partition_view_type(
    tensor_view_type,
    tile_shape: List[int],
    dim_map: List[int] | None = None,
    padding_value: PaddingValue | None = None,
) -> PartitionViewType:
    """
    Creates a PartitionViewType from a tensor view MLIR type, a tile shape,
    the type of the indices to use within the view, a dimension mapping and
    whether out-of-bound accesses should be masked.
    """

    if not isinstance(tile_shape, list) and not isinstance(tile_shape, tuple):
        raise TypeError(
            f"Expected tile_shape to be a list or tuple, got {type(tile_shape).__name__}"
        )

    dim_map = dim_map or list(range(len(tile_shape)))

    _check_partition_view_like_type(tensor_view_type, tile_shape, dim_map)

    padding_value_attr = (
        get_padding_value_attr(padding_value) if padding_value else None
    )
    partition_view_type = PartitionViewType.get(
        tile_shape, tensor_view_type, dim_map, padding_value_attr
    )

    if partition_view_type is None:
        raise RuntimeError(
            f"Error creating PartitionViewType with element type {tensor_view_type}, "
            f"tile_shape {tile_shape}, dim_map {dim_map} "
            f"{'with padding value ' + str(padding_value) if padding_value else ''}"
        )

    return partition_view_type


def make_strided_view_type(
    tensor_view_type,
    tile_shape: List[int],
    traversal_strides: List[int],
    dim_map: List[int] | None = None,
    padding_value: PaddingValue | None = None,
) -> StridedViewType:
    """
    Creates a StridedViewType from a tensor view MLIR type, a tile shape,
    traversal strides, the type of the indices to use within the view, a
    dimension mapping and whether out-of-bound accesses should be masked.
    """

    if not isinstance(tile_shape, list) and not isinstance(tile_shape, tuple):
        raise TypeError(
            f"Expected tile_shape to be a list or tuple, got {type(tile_shape).__name__}"
        )

    dim_map = dim_map or list(range(len(tile_shape)))

    _check_partition_view_like_type(tensor_view_type, tile_shape, dim_map)

    if not isinstance(traversal_strides, list) and not isinstance(
        traversal_strides, tuple
    ):
        raise TypeError(
            f"Expected traversal_strides to be a list or tuple, got {type(traversal_strides).__name__}"
        )

    if len(tile_shape) != len(traversal_strides):
        raise ValueError(
            "Expected tile shape and traversal strides to have the same length, "
            f"got {len(tile_shape)} vs {len(traversal_strides)}"
        )

    if not all(isinstance(stride, int) and stride > 0 for stride in traversal_strides):
        raise ValueError(
            "Expected traversal strides to be a list of strictly positive integers, "
            f"got {traversal_strides}"
        )

    padding_value_attr = (
        get_padding_value_attr(padding_value) if padding_value else None
    )
    strided_view_type = StridedViewType.get(
        tile_shape, traversal_strides, tensor_view_type, dim_map, padding_value_attr
    )

    if strided_view_type is None:
        raise RuntimeError(
            f"Error creating StridedViewType with element type {tensor_view_type}, "
            f"tile_shape {tile_shape}, traversal_strides {traversal_strides}, dim_map {dim_map} "
            f"{'with padding value ' + str(padding_value) if padding_value else ''}"
        )

    return strided_view_type


def make_gather_scatter_view_type(
    tensor_view_type,
    tile_shape: List[int],
    sparse_dim: int,
    padding_value: PaddingValue | None = None,
) -> GatherScatterViewType:
    """
    Creates a GatherScatterViewType from a tensor view MLIR type, a tile shape,
    and the sparse dimension to gather/scatter over.
    """

    if not isinstance(tensor_view_type, TensorViewType):
        raise TypeError(f"Expected tensor view type, got {tensor_view_type}")

    if not isinstance(tile_shape, list) and not isinstance(tile_shape, tuple):
        raise TypeError(
            f"Expected tile_shape to be a list or tuple, got {type(tile_shape).__name__}"
        )

    if not all(isinstance(dim, int) for dim in tile_shape):
        raise TypeError(
            f"Expected tile_shape to be an array of integers, got {tile_shape}"
        )

    if not all(dim > 0 for dim in tile_shape):
        raise ValueError(
            f"Expected tile_shape dimensions to be positive, got {tile_shape}"
        )

    if not all((dim & (dim - 1)) == 0 for dim in tile_shape):
        raise ValueError(
            "Expected gather/scatter tile_shape dimensions to be powers of two, "
            f"got {tile_shape}"
        )

    if len(tile_shape) != len(tensor_view_type.shape):
        raise ValueError(
            f"Expected tile shape of same rank as tensor view, got tile shape "
            f"{tile_shape} and tensor view shape {tensor_view_type.shape}"
        )

    if not isinstance(sparse_dim, int):
        raise TypeError(
            f"Expected sparse_dim to be an integer, got {type(sparse_dim).__name__}"
        )

    if sparse_dim < 0 or sparse_dim >= len(tile_shape):
        raise ValueError(
            f"Expected sparse_dim in range [0, {len(tile_shape)}), got {sparse_dim}"
        )

    if padding_value is not None and not isinstance(padding_value, PaddingValue):
        raise TypeError(
            "Expected padding_value to be PaddingValue or None, got "
            f"{type(padding_value).__name__}"
        )

    padding_value_attr = (
        get_padding_value_attr(padding_value) if padding_value else None
    )
    gather_scatter_view_type = GatherScatterViewType.get(
        tile_shape, tensor_view_type, sparse_dim, padding_value_attr
    )

    if gather_scatter_view_type is None:
        raise RuntimeError(
            f"Error creating GatherScatterViewType with element type {tensor_view_type}, "
            f"tile_shape {tile_shape}, sparse_dim {sparse_dim} "
            f"{'with padding value ' + str(padding_value) if padding_value else ''}"
        )

    return gather_scatter_view_type


def check_same_type(func):
    """Decorator to check if lhs and rhs have the same tile type."""

    @_wraps(func)
    def wrapper(lhs, rhs, *args, **kwargs):
        if lhs.tile_type != rhs.tile_type:
            raise TypeError("expected matching lhs/rhs tile types")
        return func(lhs, rhs, *args, **kwargs)

    return wrapper


def check_data_type_binary(tile_name, expected_type):
    """Decorator to check if the specified tile has the expected data type."""

    def decorator(func):
        @_wraps(func)
        def wrapper(lhs, rhs, *args, **kwargs):
            tile = lhs if tile_name == "lhs" else rhs
            if not isinstance(tile.element_type, expected_type):
                raise TypeError(
                    f"expected {tile_name} to have element type {expected_type}"
                )
            return func(lhs, rhs, *args, **kwargs)

        return wrapper

    return decorator


def check_data_type_unary(tile_name, expected_type):
    """Decorator to check if the specified tile has the expected data type."""

    def decorator(func):
        @_wraps(func)
        def wrapper(source, *args, **kwargs):
            if not isinstance(source.element_type, expected_type):
                raise TypeError(
                    f"expected {tile_name} to have element type {expected_type}"
                )
            return func(source, *args, **kwargs)

        return wrapper

    return decorator


def promote_rhs_to_tile(func):
    """
    If rhs is a not tile, create a constant tile with the same element type and
    shape as lhs.

    Note: This decorator can be applied only to functions that a lhs and a rhs
    operand as the first two arguments.
    """

    @_wraps(func)
    def wrapper(lhs, rhs, *args, **kwargs):
        rhs = _check_is_rhs_tile(lhs, rhs)
        return func(lhs, rhs, *args, **kwargs)

    return wrapper


# =============================================================================
# OPs
# =============================================================================

# TODO: order ops alphabetically. It is really hard to navigate.


@cuda_tile_op
def broadcast(
    shape: List[int],
    source: Tile,
    *,
    loc=None,
    ip=None,
) -> Tile:
    """Broadcasts the source tile to the given shape."""
    result_type = TileType.get(shape, source.element_type)
    op = _cuda_tile.BroadcastOp(result_type, source, loc=loc, ip=ip)
    return return_results(op)


@cuda_tile_op
def print_tko(str, args: Iterable[Tile], *, input_token=None, loc=None, ip=None):
    """Prints the provided string and arguments to the output."""
    if not all(isinstance(arg, Tile) for arg in args):
        raise TypeError(
            "All elements in 'args' must be of type Tile. Constexpr cannot be printed direclty."
        )
    return return_results(
        _cuda_tile.PrintTkoOp(str, args, token=input_token, loc=loc, ip=ip)
    )


def _check_is_rhs_tile(lhs: Tile, rhs: Tile, preserve_rhs_type=False):
    """
    To allow mixing of Python values and SSA values, we generate an MLIR value
    using `constant` for the RHS, matching the shape of the LHS tile.

    By default the element type is copied from the LHS. When preserve_rhs_type
    is True, the element type is inferred from the Python value itself. This is
    needed for ops like FPowI where the RHS type intentionally differs from the
    LHS (e.g. float base, int exponent).

    Example:
        a = cuda_tile.tile
        c = a + 1  # Here, you can use 1 directly without needing `a + broadcast(constant(1))`
        d = a + [1, 2, 3]  # Here, you can use a list matching the tile shape without needing `a + constant([1, 2, 3])`

    Args:
        lhs (Tile): The left-hand side operand, which is a tile.
        rhs       : The right-hand side operand, which can be a Python value, list, or tile.
        preserve_rhs_type (bool): If True, infer element type from the RHS value instead of
                                  copying the LHS tile type.

    Returns:
        Tile: The right-hand side operand, converted to an MLIR tile if it was a Python value.

    Raises:
        ValueError: If rhs is a list with shape that doesn't match lhs tile shape.
    """
    if isinstance(rhs, Tile):
        return rhs

    lhs_shape = list(lhs.tile_type.shape)

    if isinstance(rhs, list):
        try:
            rhs_shape, _ = _flatten_constants(rhs)
        except AssertionError as e:
            raise AssertionError(
                f"can not promote irregular rhs list ({rhs}) to tile: {e}"
            )
        if rhs_shape != lhs_shape:
            raise ValueError(
                f"rhs list shape {rhs_shape} does not match lhs tile shape {lhs_shape}"
            )
        if preserve_rhs_type:
            return constant(rhs)
        return constant(rhs, tile_type=lhs.tile_type)

    if isinstance(rhs, Number):
        if preserve_rhs_type:
            return constant(
                [rhs],
                tile_type=make_tile_type(_infer_mlir_type_from_python(rhs), lhs_shape),
            )
        return constant([rhs], tile_type=lhs.tile_type)

    raise TypeError(
        f"rhs must be a cuda_tile.tile, list, or Python scalar, got {type(rhs).__name__}"
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.IntegerType)
def absi(source: Tile, *, loc=None, ip=None) -> Tile:
    """Performs element-wise absolute value on input integer tile."""
    return return_results(_cuda_tile.AbsIOp(source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def absf(source: Tile, *, loc=None, ip=None) -> Tile:
    """Performs element-wise absolute value on input float tile."""
    return return_results(_cuda_tile.AbsFOp(source, loc=loc, ip=ip))


@cuda_tile_op
def alloca(num_elem: int, alignment: int, *, loc=None, ip=None) -> Tile:
    """
    Create a dynamic memory allocation operation that returns a pointer.

    Allocates memory sufficient to hold `num_elem` elements and returns a pointer
    to the allocated memory, or traps if the request cannot be satisfied. The
    returned address is guaranteed to be aligned to `alignment` bytes, which must
    be a non-zero power of two. The lifetime of the allocation is limited to the
    block in which the alloca resides.

    Args:
        num_elem: Number of elements to allocate (must be positive)
        alignment: Memory alignment in bytes (must be a power of 2)

    Returns:
        A scalar tile containing a pointer to the allocated memory
    """
    if num_elem <= 0:
        raise ValueError(f"num_elem must be positive, got {num_elem}")
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")

    ptr_type = PointerType.get(_ods_ir.IntegerType.get_signless(64))
    result_type = TileType.get([], ptr_type)
    return return_results(
        _cuda_tile.AllocaOp(
            result_type,
            num_elem=num_elem,
            alignment=alignment,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def _addi(
    lhs: Tile,
    rhs: Tile,
    *,
    overflow: IntegerOverflow,
    loc=None,
    ip=None,
) -> Tile:
    op = _cuda_tile.AddIOp(
        lhs, rhs, overflow=get_integer_overflow_attr(overflow), loc=loc, ip=ip
    )
    return return_results(op)


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def _addf(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool,
    rounding_mode: RoundingMode,
    loc=None,
    ip=None,
) -> Tile:

    if rounding_mode not in [
        RoundingMode.NEAREST_EVEN,
        RoundingMode.ZERO,
        RoundingMode.NEGATIVE_INF,
        RoundingMode.POSITIVE_INF,
    ]:
        raise ValueError(
            f"Invalid rounding mode for addf: {rounding_mode}, expected one of NEAREST_EVEN, ZERO, NEGATIVE_INF, POSITIVE_INF"
        )

    op = _cuda_tile.AddFOp(
        lhs,
        rhs,
        flush_to_zero=flush_to_zero,
        rounding_mode=get_rounding_mode_attr(rounding_mode),
        loc=loc,
        ip=ip,
    )
    return return_results(op)


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_data_type_binary("acc", _ods_ir.FloatType)
@check_same_type
def fma(
    lhs: Tile,
    rhs: Tile,
    acc: Tile,
    *,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    flush_to_zero: bool = False,
    loc=None,
    ip=None,
) -> Tile:

    if rounding_mode not in [
        RoundingMode.NEAREST_EVEN,
        RoundingMode.ZERO,
        RoundingMode.NEGATIVE_INF,
        RoundingMode.POSITIVE_INF,
    ]:
        raise ValueError(
            f"Invalid rounding mode for fma: {rounding_mode}, expected one of NEAREST_EVEN, ZERO, NEGATIVE_INF, POSITIVE_INF"
        )

    return return_results(
        _cuda_tile.FmaOp(
            lhs,
            rhs,
            acc,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def _offset(
    lhs: Tile,
    rhs: Tile,
    *,
    loc=None,
    ip=None,
) -> Tile:
    if not isinstance(rhs, Tile):
        rhs = constant(rhs, el_type=Int32)
    if not isinstance(rhs.element_type, _ods_ir.IntegerType):
        raise TypeError("pointer lhs element type requires integer rhs element type")
    if lhs.tile_type.shape != rhs.tile_type.shape:
        raise TypeError("expected pointer and offset to have identical shapes")
    op = _cuda_tile.OffsetOp(lhs, rhs, loc=loc, ip=ip)
    return return_results(op)


@cuda_tile_op
def add(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool = False,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    overflow: IntegerOverflow = IntegerOverflow.NONE,
    loc=None,
    ip=None,
) -> Tile:
    # Performs element-wise addition of two tiles.
    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        if flush_to_zero:
            raise Exception(
                "flush_to_zero modifier can only be used with floating point tiles"
            )
        return _addi(
            lhs,
            rhs,
            overflow=overflow,
            loc=loc,
            ip=ip,
        )
    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        if overflow != IntegerOverflow.NONE:
            raise Exception("overflow modifier can only be used with integer tiles")
        return _addf(
            lhs,
            rhs,
            flush_to_zero=flush_to_zero,
            rounding_mode=rounding_mode,
            loc=loc,
            ip=ip,
        )
    elif PointerType.isinstance(lhs.element_type):
        if flush_to_zero:
            raise Exception(
                "flush_to_zero modifier can only be used with floating point tiles"
            )
        return _offset(
            lhs,
            rhs,
            loc=loc,
            ip=ip,
        )
    else:
        raise TypeError("expected integer, float or pointer element type")


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def andi(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.AndIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
def assert_(value: Tile, message, *, loc=None, ip=None):
    _cuda_tile.AssertOp(value, message=message, loc=loc, ip=ip)


@cuda_tile_op
def assume_div_by(
    value: Tile,
    divisor: int,
    every=None,
    along=None,
    loc=None,
    ip=None,
) -> Tile:
    def _build_div_by_attr(divisor: int):
        # TODO: There are no Python bindings for cuda_tile.div_by, so we parse
        # the textual representation as a workaround.
        attr = f"#cuda_tile.div_by<{divisor}"
        if every:
            assert along != None, "must provide 'along' together with 'every'"
            attr = attr + f", every {every} along {along}"
        attr = attr + ">"
        return _ods_ir.Attribute.parse(attr)

    el_ty = value.element_type
    if not isinstance(el_ty, _ods_ir.IntegerType) and not PointerType.isinstance(el_ty):
        raise TypeError("expected integer/pointer element type on value")
    predicate = _build_div_by_attr(divisor)
    return return_results(_cuda_tile.AssumeOp(value, predicate, loc=loc, ip=ip))


@cuda_tile_op
def assume_same_elements(value: Tile, group_size: List[int], loc=None, ip=None) -> Tile:
    def _build_same_elements_attr(group_size: List[int], rank: int):
        if len(group_size) != rank:
            raise TypeError("expected length of same_elements to match tile rank")
        # TODO: There are no Python bindings for cuda_tile.same_elements, so we
        # parse the textual representation as a workaround.
        return _ods_ir.Attribute.parse(
            f"#cuda_tile.same_elements<[{', '.join([str(x) for x in group_size])}]>"
        )

    el_ty = value.element_type
    if not isinstance(el_ty, _ods_ir.IntegerType) and not PointerType.isinstance(el_ty):
        raise TypeError("expected integer/pointer element type on value")
    predicate = _build_same_elements_attr(group_size, len(value.tile_type.shape))
    return return_results(_cuda_tile.AssumeOp(value, predicate, loc=loc, ip=ip))


@cuda_tile_op
def assume_bounded(value: Tile, lb=None, ub=None, *, loc=None, ip=None) -> Tile:
    el_ty = value.element_type
    if not isinstance(el_ty, _ods_ir.IntegerType):
        raise TypeError("expected integer element type on value")
    lb_str = "?" if lb is None else str(lb)
    ub_str = "?" if ub is None else str(ub)
    predicate = _ods_ir.Attribute.parse(f"#cuda_tile.bounded<{lb_str}, {ub_str}>")
    return return_results(_cuda_tile.AssumeOp(value, predicate, loc=loc, ip=ip))


@cuda_tile_op
def atomic_cas_tko(
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: MemoryScope,
    pointers: Tile,
    cmp: Tile,
    val: Tile,
    mask: Tile = None,
    *,
    input_token=None,
    return_token=False,
    loc=None,
    ip=None,
) -> Tile | Tuple[Tile, Token]:
    """
    Executes an atomic compare-and-swap (CAS) on the given memory pointers with
    specified memory ordering and scope. Compares the current memory contents with
    the provided compare tile, and swaps in the new value if equal.

    :param memory_ordering_semantics: Memory ordering guarantees ("relaxed", "strong", or "weak")
    :type memory_ordering_semantics: str
    :param memory_scope: Memory visibility scope ("device", "sys", "tl_blk", or None)
    :type memory_scope: Optional[str]
    :param pointers: Tile of pointers on which to perform the CAS
    :type pointers: Tile
    :param cmp: Tile containing the compare values
    :type cmp: Tile
    :param val: Tile containing the values to swap in
    :type val: Tile
    :param mask: Optional tile of boolean values indicating which elements to process
    :type mask: Optional[Tile]
    :param input_token: Optional synchronization token for ordering
    :type input_token: Optional[Token]
    :param return_token: If True, return both the result tile and a synchronization token
    :type return_token: bool
    :param loc: Source location for MLIR operation tracking
    :type loc: Optional[Location]
    :param ip: Insertion point for MLIR operation
    :type ip: Optional[InsertionPoint]

    :return: The result tile if return_token is False; otherwise a (Tile, Token) tuple
    :rtype: Tile | Tuple[Tile, Token]
    """
    sem_attr = get_memory_ordering_semantics_attr(memory_ordering_semantics)
    scope_attr = get_memory_scope_attr(memory_scope)

    if input_token is not None:
        if not isinstance(input_token, Token):
            raise ValueError("input_token must be a Token")

    # Create the operation with or without the mask parameter
    if mask is None:
        op = _cuda_tile.AtomicCASTkoOp(
            sem_attr, scope_attr, pointers, cmp, val, token=input_token, loc=loc, ip=ip
        )
    else:
        op = _cuda_tile.AtomicCASTkoOp(
            sem_attr,
            scope_attr,
            pointers,
            cmp,
            val,
            mask=mask,
            token=input_token,
            loc=loc,
            ip=ip,
        )

    # Return both tile and token if requested
    if return_token:
        return return_results(op)

    # Otherwise, return only the tile result
    return Tile(op.results[0], op.results[0].type)


@cuda_tile_op
def atomic_rmw_tko(
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: MemoryScope,
    pointers: Tile,
    mode: AtomicRMWMode,
    arg: Tile,
    *,
    mask=None,
    input_token=None,
    return_token: bool = False,
    loc=None,
    ip=None,
) -> Tile | Tuple[Tile, Token]:
    """Perform an atomic read-modify-write (RMW) operation.

    Executes an atomic read-modify-write on the given memory pointers using the specified
    operation mode and argument tile, with memory ordering and scope control.

    :param memory_ordering_semantics: Memory ordering guarantees ("relaxed", "strong", or "weak")
    :type memory_ordering_semantics: str
    :param memory_scope: Memory visibility scope ("device", "sys", "tl_blk", or None)
    :type memory_scope: Optional[str]
    :param pointers: Tile of pointers on which to perform the RMW
    :type pointers: Tile
    :param mode: Operation mode for the atomic RMW (e.g., "add", "max", "min")
    :type mode: str
    :param arg: Tile containing the values used in the RMW operation
    :type arg: Tile
    :param mask: Optional boolean mask matching the argument tile shape
    :type mask: Optional[Tile]
    :param input_token: Optional synchronization token for ordering
    :type input_token: Optional[Token]
    :param return_token: If True, return both the result tile and a synchronization token
    :type return_token: bool
    :param loc: Source location for MLIR operation tracking
    :type loc: Optional[Location]
    :param ip: Insertion point for MLIR operation
    :type ip: Optional[InsertionPoint]

    :return: The result tile if return_token is False; otherwise a (Tile, Token) tuple
    :rtype: Tile | Tuple[Tile, Token]
    """

    assert isinstance(mode, AtomicRMWMode), (
        "Expected mode to be an AtomicRMWMode enum value, got " + type(mode).__name__
    )

    mode_attr = get_atomic_rmw_mode_attr(mode)
    sem_attr = get_memory_ordering_semantics_attr(memory_ordering_semantics)
    scope_attr = get_memory_scope_attr(memory_scope)

    if mask is not None:
        if not isinstance(mask, Tile):
            raise ValueError("mask must be a Tile")
        if not (
            isinstance(mask.element_type, _ods_ir.IntegerType)
            and mask.element_type.width == 1
        ):
            raise ValueError("mask must have boolean element type")
        if mask.tile_type.shape != arg.tile_type.shape:
            raise ValueError("mask must have the same shape as the argument")

    if input_token is not None:
        if not isinstance(input_token, Token):
            raise ValueError("input_token must be a Token")

    op = _cuda_tile.AtomicRMWTkoOp(
        sem_attr,
        scope_attr,
        pointers=pointers,
        mode=mode_attr,
        arg=arg,
        mask=mask,
        token=input_token,
        loc=loc,
        ip=ip,
    )

    # Return both tile and token if requested
    if return_token:
        return return_results(op)

    # Otherwise, return only the tile result
    return Tile(op.results[0], op.results[0].type)


def _multimem_view_operands(
    view: TileView,
    indices: Sequence[Tile | int],
    input_token: Token | None,
    inbounds: Sequence[bool] | None,
) -> tuple[list[Tile], _ods_ir.DenseBoolArrayAttr]:
    if not isinstance(view, TileView):
        raise TypeError(f"Expected a tile view, got {type(view).__name__}")
    if view.view_index_rank != len(indices):
        raise ValueError(
            f"Expected {view.view_index_rank} index values, got {len(indices)}"
        )
    if input_token is not None and not isinstance(input_token, Token):
        raise TypeError("input_token must be a Token")
    if inbounds is None:
        inbounds = [False] * len(indices)
    elif len(inbounds) != len(indices) or not all(
        isinstance(value, bool) for value in inbounds
    ):
        raise ValueError("inbounds must contain one bool per view index")
    return (
        _view_index_list_to_tiles(view, indices),
        _ods_ir.DenseBoolArrayAttr.get(inbounds),
    )


def _multimem_memory_attrs(
    operation: str,
    semantics: MemoryOrderingSemantics,
    scope: MemoryScope | None,
    *,
    allowed_semantics: tuple[MemoryOrderingSemantics, ...],
    scope_required: bool,
) -> tuple[MemoryOrderingSemanticsAttr, MemoryScopeAttr | None]:
    if not isinstance(semantics, MemoryOrderingSemantics):
        raise TypeError("memory_ordering_semantics must be a MemoryOrderingSemantics")
    if semantics not in allowed_semantics:
        allowed = ", ".join(value.value for value in allowed_semantics)
        raise ValueError(f"{operation} requires one of: {allowed}")
    if scope_required and not isinstance(scope, MemoryScope):
        raise ValueError(f"{operation} requires a memory scope")
    if not scope_required and scope is not None:
        raise ValueError(f"{operation} with weak ordering must not have a memory scope")
    return (
        get_memory_ordering_semantics_attr(semantics),
        None if scope is None else get_memory_scope_attr(scope),
    )


def _multimem_reduction_value(
    reduction: MultimemReductionKind | None,
) -> str | None:
    if reduction is None:
        return None
    if not isinstance(reduction, MultimemReductionKind):
        raise TypeError("reduction must be a MultimemReductionKind or None")
    return reduction.value


@cuda_tile_op
def multimem_reduce_view_tko(
    view: TileView,
    tile: Tile,
    indices: Sequence[Tile | int],
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.RELEASE,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
    reduction: MultimemReductionKind | None = None,
    input_token: Token | None = None,
    inbounds: Sequence[bool] | None = None,
    loc=None,
    ip=None,
) -> Token:
    """Atomically reduce a tile through a multicast-backed tile view."""
    if not isinstance(tile, Tile):
        raise TypeError(f"Expected a Tile, got {type(tile).__name__}")
    index_tiles, inbounds_attr = _multimem_view_operands(
        view, indices, input_token, inbounds
    )
    if tile.tile_type != view.view_tile_type:
        raise TypeError(
            f"Expected tile type to be {view.view_tile_type}, got {tile.tile_type}"
        )
    semantics_attr, scope_attr = _multimem_memory_attrs(
        "multimem_reduce_view_tko",
        memory_ordering_semantics,
        memory_scope,
        allowed_semantics=(
            MemoryOrderingSemantics.RELAXED,
            MemoryOrderingSemantics.RELEASE,
        ),
        scope_required=True,
    )
    return return_results(
        _cuda_tile.MultimemReduceViewTkoOp(
            memory_ordering_semantics=semantics_attr,
            memory_scope=scope_attr,
            view=view,
            tile=tile,
            index=index_tiles,
            inbounds=inbounds_attr,
            token=input_token,
            reduction=_multimem_reduction_value(reduction),
            results=[TokenType.get()],
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def multimem_load_reduce_view_tko(
    view: TileView,
    indices: Sequence[Tile | int],
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.RELAXED,
    memory_scope: MemoryScope | None = MemoryScope.SYS,
    reduction: MultimemReductionKind | None = None,
    input_token: Token | None = None,
    inbounds: Sequence[bool] | None = None,
    loc=None,
    ip=None,
) -> Tuple[Tile, Token]:
    """Load and reduce through a multicast-backed tile view."""
    index_tiles, inbounds_attr = _multimem_view_operands(
        view, indices, input_token, inbounds
    )
    weak = memory_ordering_semantics == MemoryOrderingSemantics.WEAK
    semantics_attr, scope_attr = _multimem_memory_attrs(
        "multimem_load_reduce_view_tko",
        memory_ordering_semantics,
        memory_scope,
        allowed_semantics=(
            MemoryOrderingSemantics.WEAK,
            MemoryOrderingSemantics.RELAXED,
            MemoryOrderingSemantics.ACQUIRE,
        ),
        scope_required=not weak,
    )
    return return_results(
        _cuda_tile.MultimemLoadReduceViewTkoOp(
            reduced_values=view.view_tile_type,
            result_token=TokenType.get(),
            memory_ordering_semantics=semantics_attr,
            memory_scope=scope_attr,
            view=view,
            index=index_tiles,
            inbounds=inbounds_attr,
            token=input_token,
            reduction=_multimem_reduction_value(reduction),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def multimem_store_view_tko(
    view: TileView,
    tile: Tile,
    indices: Sequence[Tile | int],
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.RELAXED,
    memory_scope: MemoryScope | None = MemoryScope.SYS,
    input_token: Token | None = None,
    inbounds: Sequence[bool] | None = None,
    loc=None,
    ip=None,
) -> Token:
    """Store a tile through a multicast-backed tile view."""
    if not isinstance(tile, Tile):
        raise TypeError(f"Expected a Tile, got {type(tile).__name__}")
    index_tiles, inbounds_attr = _multimem_view_operands(
        view, indices, input_token, inbounds
    )
    if tile.tile_type != view.view_tile_type:
        raise TypeError(
            f"Expected tile type to be {view.view_tile_type}, got {tile.tile_type}"
        )
    weak = memory_ordering_semantics == MemoryOrderingSemantics.WEAK
    semantics_attr, scope_attr = _multimem_memory_attrs(
        "multimem_store_view_tko",
        memory_ordering_semantics,
        memory_scope,
        allowed_semantics=(
            MemoryOrderingSemantics.WEAK,
            MemoryOrderingSemantics.RELAXED,
            MemoryOrderingSemantics.RELEASE,
        ),
        scope_required=not weak,
    )
    return return_results(
        _cuda_tile.MultimemStoreViewTkoOp(
            memory_ordering_semantics=semantics_attr,
            memory_scope=scope_attr,
            view=view,
            tile=tile,
            index=index_tiles,
            inbounds=inbounds_attr,
            token=input_token,
            results=[TokenType.get()],
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def atomic_red_view_tko(
    view: TileView,
    index: Sequence[Union[Tile, int]],
    mode: AtomicRMWMode,
    value: Tile,
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.RELAXED,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
    input_token=None,
    inbounds: Optional[Sequence[bool]] = None,
    loc=None,
    ip=None,
) -> Token:
    """View-based atomic reduction on global memory.

    Performs element-wise atomic read-modify-write operations on global memory
    locations specified by ``view`` at tile position ``index``. Unlike
    ``atomic_rmw_tko``, this operation uses view-based addressing and does not
    return the original values from memory. The values written are determined by
    ``mode`` and ``value``.

    For each element at position [i, j, ...] in ``value``::

        atomic {
            addr = view[index][i, j, ...]
            x = *addr
            y = mode(x, value[i, j, ...])
            *addr = y
            // x is discarded (not returned)
        }

    This operation is suitable for accumulation patterns where multiple CTAs
    update the same memory locations and the original values are not needed,
    such as distributed gradient accumulation.

    Args:
        view: Target partition view specifying the global memory locations.
        index: N-dimensional index of the tile to reduce into. Must have the
            same rank as the view's index space. Each element is either a
            scalar integer Tile or a Python int (converted automatically).
        mode: Atomic RMW mode. Supported modes by data type:
            - ADD, AND, MAX, MIN, OR, UMAX, UMIN, XOR: i32, i64
            - ADDF: f16, bf16, f32, f64
        value: Value tile to apply atomically. Shape must match the view.
        memory_ordering_semantics: Memory ordering for the atomic reduction.
            Currently only RELAXED is supported.
        memory_scope: Memory visibility scope (tl_blk, device).
        input_token: Optional input token for ordering.
        inbounds: Optional per-index-dimension flags. A true value asserts
            that the corresponding tile index dimension is statically in-bounds.
            Defaults to all false.

    Returns:
        Token for operation ordering.
    """
    if not isinstance(view, TileView):
        raise TypeError(f"Expected TileView, got {type(view).__name__}")
    if not isinstance(value, Tile):
        raise TypeError(f"Expected Tile, got {type(value).__name__}")
    assert isinstance(
        mode, AtomicRMWMode
    ), f"Expected AtomicRMWMode, got {type(mode).__name__}"

    if value.tile_type != view.view_tile_type:
        raise TypeError(
            f"Expected value tile type to be {view.view_tile_type}, "
            f"got {value.tile_type}"
        )

    if mode == AtomicRMWMode.XCHG:
        raise ValueError("atomic_red_view_tko op cannot use xchg operation")

    _INT_MODES = {
        AtomicRMWMode.ADD,
        AtomicRMWMode.AND,
        AtomicRMWMode.OR,
        AtomicRMWMode.XOR,
        AtomicRMWMode.MAX,
        AtomicRMWMode.MIN,
        AtomicRMWMode.UMAX,
        AtomicRMWMode.UMIN,
    }
    if mode in _INT_MODES:
        if not (
            isinstance(value.element_type, _ods_ir.IntegerType)
            and value.element_type.width in (32, 64)
        ):
            raise ValueError(f"'{mode.value}' works only with integers i32 and i64")
    elif mode == AtomicRMWMode.ADDF:
        if not isinstance(
            value.element_type,
            (_ods_ir.F16Type, _ods_ir.BF16Type, _ods_ir.F32Type, _ods_ir.F64Type),
        ):
            raise ValueError(
                f"'{mode.value}' works only with floats f16, bf16, f32, and f64"
            )

    if view.view_index_rank != len(index):
        raise ValueError(
            f"Expected {view.view_index_rank} index values, got {len(index)}"
        )

    if input_token is not None and not isinstance(input_token, Token):
        raise ValueError("input_token must be a Token")

    if memory_ordering_semantics != MemoryOrderingSemantics.RELAXED:
        raise ValueError(
            "atomic_red_view_tko only supports RELAXED memory ordering semantics"
        )

    if memory_scope == MemoryScope.SYS:
        raise ValueError(
            "atomic_red_view_tko does not support SYS memory scope for TMA compatibility; "
            "use TL_BLK or DEVICE"
        )

    if isinstance(view, GatherScatterView):
        raise ValueError("atomic_red_view_tko does not support gather_scatter_view")

    index_tiles = _view_index_list_to_tiles(view, index)
    mode_attr = get_atomic_rmw_mode_attr(mode)
    sem_attr = get_memory_ordering_semantics_attr(memory_ordering_semantics)
    scope_attr = get_memory_scope_attr(memory_scope)

    inbounds_attr = _inbounds_attr_or_default(inbounds, len(index_tiles))

    op = _cuda_tile.AtomicRedViewTkoOp(
        memory_ordering_semantics=sem_attr,
        memory_scope=scope_attr,
        view=view,
        index=index_tiles,
        mode=mode_attr,
        value=value,
        inbounds=inbounds_attr,
        token=input_token,
        loc=loc,
        ip=ip,
    )

    return return_results(op)


@cuda_tile_op
def bitcast(el_type, src: Tile, *, loc=None, ip=None) -> Tile:
    el_type = _get_mlir_type(el_type)

    # Check that neither source nor destination types are pointer types
    if PointerType.isinstance(src.element_type):
        raise TypeError("bitcast source cannot be a pointer type")
    if PointerType.isinstance(el_type):
        raise TypeError("bitcast destination cannot be a pointer type")

    result_type = TileType.get(src.shape, el_type)
    from_width = src.element_type.width
    to_width = el_type.width
    if from_width != to_width:
        raise TypeError(
            f"mismatching bitwidth between {src.element_type} ({from_width}) "
            f"and {result_type.element_type} ({to_width})"
        )
    return return_results(
        _cuda_tile.BitcastOp(result=result_type, source=src, loc=loc, ip=ip)
    )


def _get_element_bit_width(elem_type) -> int:
    """Get element bit width matching C++ getTileSizeInBits logic."""
    if isinstance(elem_type, _ods_ir.IntegerType) and elem_type.width == 1:
        return 1
    if isinstance(elem_type, _ods_ir.FloatTF32Type):
        return 32
    return elem_type.width


@cuda_tile_op
def int_to_ptr(el_type, src: Tile, *, loc=None, ip=None) -> Tile:
    el_type = _get_mlir_type(el_type)

    # Ensure src is a tile with i64 element type
    if not (
        isinstance(src.element_type, _ods_ir.IntegerType)
        and src.element_type.width == 64
    ):
        raise TypeError("expected source to have i64 element type")

    to_is_ptr = PointerType.isinstance(el_type)
    result_type = TileType.get(src.shape, el_type)
    if not to_is_ptr:
        raise TypeError("expected destination type to be a tile of pointers")
    return return_results(
        _cuda_tile.IntToPtrOp(result=result_type, source=src, loc=loc, ip=ip)
    )


@cuda_tile_op
def ptr_to_int(src: Tile, *, loc=None, ip=None) -> Tile:
    from_is_ptr = PointerType.isinstance(src.element_type)
    i64 = _ods_ir.IntegerType.get_signless(64)
    result_type = TileType.get(src.shape, i64)
    if not from_is_ptr:
        raise TypeError("expected tile of pointer source type")
    return return_results(
        _cuda_tile.PtrToIntOp(result=result_type, source=src, loc=loc, ip=ip)
    )


@cuda_tile_op
def ptr_to_ptr(el_type, src: Tile, *, loc=None, ip=None) -> Tile:
    el_type = _get_mlir_type(el_type)
    result_type = TileType.get(src.shape, el_type)
    return return_results(
        _cuda_tile.PtrToPtrOp(result=result_type, source=src, loc=loc, ip=ip)
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def cos(source: Tile, *, loc=None, ip=None) -> Tile:
    """Computes the cosine of the source tile element-wise."""
    return return_results(_cuda_tile.CosOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.IntegerType)
def negi(
    source: Tile, *, overflow: IntegerOverflow = IntegerOverflow.NONE, loc=None, ip=None
) -> Tile:
    """Computes the arithmetic inverse of the source integer tile element-wise."""
    if overflow == IntegerOverflow.NUW:
        raise ValueError("'no_unsigned_wrap' overflow flag is not supported for negi")
    return return_results(
        _cuda_tile.NegIOp(
            source=source, overflow=get_integer_overflow_attr(overflow), loc=loc, ip=ip
        )
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def negf(source: Tile, *, loc=None, ip=None) -> Tile:
    """Computes the negative of the source tile element-wise."""
    return return_results(_cuda_tile.NegFOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def floor(source: Tile, *, loc=None, ip=None) -> Tile:
    """Computes the floor of the source tile element-wise."""
    return return_results(_cuda_tile.FloorOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def cosh(source: Tile, *, loc=None, ip=None) -> Tile:
    """Computes the hyperbolic cosine of the source tile."""
    return return_results(_cuda_tile.CosHOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def ori(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    """Performs element-wise, bit-wise "or" of two tiles."""
    return return_results(_cuda_tile.OrIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def fpowf(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    """Raises lhs to the power of rhs element-wise (float base, float exponent)."""
    return return_results(_cuda_tile.FPowFOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_binary("lhs", _ods_ir.FloatType)
def fpowi(lhs: Tile, rhs, *, loc=None, ip=None) -> Tile:
    """Raises lhs to the power of rhs element-wise (float base, signed integer exponent)."""
    if not isinstance(rhs, Tile):
        rhs = _check_is_rhs_tile(lhs, rhs, preserve_rhs_type=True)
    if not isinstance(rhs.element_type, _ods_ir.IntegerType):
        raise TypeError("expected rhs to have element type IntegerType")
    if lhs.shape != rhs.shape:
        raise TypeError("expected matching lhs/rhs tile shapes")
    if rhs.element_type.width == 64:
        raise TypeError("i64 exponent is not supported for fpowi; use i8, i16, or i32")
    return return_results(_cuda_tile.FPowIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_binary("lhs", _ods_ir.FloatType)
def pow(lhs: Tile, rhs, *, loc=None, ip=None) -> Tile:
    """Raises lhs to the power of rhs element-wise. Dispatches to fpowf or fpowi."""
    if not isinstance(rhs, Tile):
        rhs = _check_is_rhs_tile(lhs, rhs, preserve_rhs_type=True)
    if isinstance(rhs.element_type, _ods_ir.FloatType):
        return fpowf(lhs, rhs, loc=loc, ip=ip)
    elif isinstance(rhs.element_type, _ods_ir.IntegerType):
        return fpowi(lhs, rhs, loc=loc, ip=ip)
    else:
        raise TypeError("expected rhs to have element type FloatType or IntegerType")


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def atan2(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    """Compute element-wise arc tangent of lhs/rhs with correct quadrant."""
    return return_results(_cuda_tile.Atan2Op(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def exp2(
    source: Tile,
    *,
    flush_to_zero: bool = False,
    loc=None,
    ip=None,
) -> Tile:
    """Raises 2 to the power of source."""
    return return_results(
        _cuda_tile.Exp2Op(
            source=source,
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def exp(
    source: Tile,
    *,
    rounding_mode: RoundingMode = RoundingMode.FULL,
    loc=None,
    ip=None,
) -> Tile:
    """Raises e to the power of source."""

    if rounding_mode not in [RoundingMode.APPROX, RoundingMode.FULL]:
        raise ValueError(
            f"Invalid rounding mode for exp: {rounding_mode}, expected APPROX or FULL"
        )
    if rounding_mode == RoundingMode.APPROX and not isinstance(
        source.element_type, _ods_ir.F32Type
    ):
        raise ValueError(
            "approx rounding mode only supported for f32 data type, but got: "
            f"{source.element_type}"
        )

    return return_results(
        _cuda_tile.ExpOp(
            source=source,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def divi(
    lhs: Tile,
    rhs: Tile,
    *,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
    rounding_mode: RoundingMode = RoundingMode.ZERO,
) -> Tile:
    if rounding_mode not in [
        RoundingMode.ZERO,
        RoundingMode.NEGATIVE_INF,
        RoundingMode.POSITIVE_INF,
    ]:
        raise ValueError(
            f"Invalid rounding mode for divi: {rounding_mode}, expected one of ZERO, NEGATIVE_INF, or POSITIVE_INF"
        )
    return return_results(
        _cuda_tile.DivIOp(
            lhs,
            rhs,
            get_signedness_attr(signedness),
            loc=loc,
            ip=ip,
            rounding=get_rounding_mode_attr(rounding_mode),
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def divf(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool,
    rounding_mode: RoundingMode,
    loc=None,
    ip=None,
) -> Tile:

    if rounding_mode not in [
        RoundingMode.NEAREST_EVEN,
        RoundingMode.ZERO,
        RoundingMode.NEGATIVE_INF,
        RoundingMode.POSITIVE_INF,
        RoundingMode.APPROX,
        RoundingMode.FULL,
    ]:
        raise ValueError(
            f"Invalid rounding mode for divf: {rounding_mode}, expected one of NEAREST_EVEN, ZERO, NEGATIVE_INF, POSITIVE_INF, APPROX, FULL"
        )

    return return_results(
        _cuda_tile.DivFOp(
            lhs,
            rhs,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def div(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool = False,
    approx: bool = False,
    full: bool = False,
    signedness: Signedness = Signedness.SIGNED,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    loc=None,
    ip=None,
) -> Tile:
    """Performs element-wise division of two tiles."""
    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        if flush_to_zero:
            raise ValueError(
                "flush_to_zero is only valid for floating-point operations"
            )
        if approx:
            raise ValueError("approx is only valid for floating-point operations")
        if full:
            raise ValueError("full is only valid for floating-point operations")
        return divi(lhs, rhs, signedness=signedness, loc=loc, ip=ip)
    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        return divf(
            lhs,
            rhs,
            flush_to_zero=flush_to_zero,
            rounding_mode=rounding_mode,
            loc=loc,
            ip=ip,
        )
    else:
        raise TypeError("expected integer or float element type")


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def remi(
    lhs: Tile,
    rhs: Tile,
    *,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
) -> Tile:
    return return_results(
        _cuda_tile.RemIOp(
            lhs=lhs,
            rhs=rhs,
            signedness=get_signedness_attr(signedness),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def remf(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.RemFOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
def rem(
    lhs: Tile,
    rhs: Tile,
    *,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
) -> Tile:
    """Performs element-wise remainder of two tiles."""
    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        signedness = Signedness.SIGNED if not signedness else signedness
        return remi(lhs, rhs, signedness=signedness, loc=loc, ip=ip)
    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        return remf(lhs, rhs, loc=loc, ip=ip)
    else:
        raise TypeError("expected integer or float element type")


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def _subi(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.SubIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def _subf(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool,
    rounding_mode: RoundingMode,
    loc=None,
    ip=None,
) -> Tile:

    if rounding_mode not in [
        RoundingMode.NEAREST_EVEN,
        RoundingMode.ZERO,
        RoundingMode.NEGATIVE_INF,
        RoundingMode.POSITIVE_INF,
    ]:
        raise ValueError(
            f"Invalid rounding mode for subf: {rounding_mode}, expected one of NEAREST_EVEN, ZERO, NEGATIVE_INF, POSITIVE_INF"
        )

    return return_results(
        _cuda_tile.SubFOp(
            lhs,
            rhs,
            flush_to_zero=flush_to_zero,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def sub(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool = False,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    loc=None,
    ip=None,
) -> Tile:
    """Performs element-wise subtraction of two tiles."""
    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        if flush_to_zero:
            raise Exception(
                "flush_to_zero modifier can only be used with floating point tiles"
            )
        return _subi(lhs, rhs, loc=loc, ip=ip)
    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        return _subf(
            lhs,
            rhs,
            flush_to_zero=flush_to_zero,
            rounding_mode=rounding_mode,
            loc=loc,
            ip=ip,
        )
    else:
        raise TypeError("expected integer or float element type")


@cuda_tile_op
def cat(lhs: Tile, rhs: Tile, dim, *, loc=None, ip=None) -> Tile:
    """Concatenates lhs and rhs along the specified dimension."""

    # Verify that the dimension is valid
    rank = len(lhs.tile_type.shape)
    if dim < 0 or dim >= rank:
        raise ValueError(f"Expected dim to be in the range [0, {rank}), but got: {dim}")

    # Verify that lhs and rhs have the same element type
    if lhs.element_type != rhs.element_type:
        raise ValueError("Expected lhs and rhs to have the same element type.")

    # Verify that lhs and rhs have the same shape except for the concatenation dimension
    lhs_shape = lhs.tile_type.shape
    rhs_shape = rhs.tile_type.shape
    if len(lhs_shape) != len(rhs_shape):
        raise ValueError("Expected lhs and rhs to have the same rank.")

    for idx in range(rank):
        if idx != dim and lhs_shape[idx] != rhs_shape[idx]:
            raise ValueError(
                f"Expected lhs and rhs shapes to match at position {idx}, "
                f"but got: {lhs_shape[idx]} and {rhs_shape[idx]}"
            )

    # Compute result type.
    result_shape = lhs_shape
    result_shape[dim] = result_shape[dim] + rhs_shape[dim]
    result_type = TileType.get(result_shape, lhs.element_type)

    # Perform the concatenation operation
    return return_results(
        _cuda_tile.CatOp(result=result_type, lhs=lhs, rhs=rhs, dim=dim, loc=loc, ip=ip)
    )


@cuda_tile_op
def mma(
    lhs: Tile,
    rhs: Tile,
    acc: Tile,
    *,
    signedness_lhs: Signedness = Signedness.SIGNED,
    signedness_rhs: Signedness = Signedness.SIGNED,
    fast_acc: bool = False,
    loc=None,
    ip=None,
) -> Tile:
    """Computes the mma product of lhs and rhs."""
    # Check shapes.
    lhs_rank = len(lhs.tile_type.shape)
    rhs_rank = len(rhs.tile_type.shape)
    acc_rank = len(acc.tile_type.shape)

    if lhs_rank not in (2, 3):
        raise ValueError("lhs operand must be a 2D or 3D tile")
    if rhs_rank not in (2, 3):
        raise ValueError("rhs operand must be a 2D or 3D tile")
    if acc_rank not in (2, 3):
        raise ValueError("acc operand must be a 2D or 3D tile")
    if lhs_rank != rhs_rank or rhs_rank != acc_rank:
        raise ValueError("lhs, rhs, acc must have the same rank")

    batched = int(lhs_rank == 3)
    if batched:
        if lhs.tile_type.shape[0] != rhs.tile_type.shape[0]:
            raise ValueError(
                "dim 0 of lhs and dim 0 of rhs (batch dimension) must match"
            )
        if lhs.tile_type.shape[0] != acc.tile_type.shape[0]:
            raise ValueError(
                "dim 0 of lhs and dim 0 of acc (batch dimension) must match"
            )
    if lhs.tile_type.shape[batched + 1] != rhs.tile_type.shape[batched]:
        raise ValueError(
            f"dim {batched + 1} of lhs and dim {batched} of rhs must match"
        )
    if lhs.tile_type.shape[batched] != acc.tile_type.shape[batched]:
        raise ValueError(f"dim {batched} of lhs and dim {batched} of acc must match")
    if rhs.tile_type.shape[batched + 1] != acc.tile_type.shape[batched + 1]:
        raise ValueError(
            f"dim {batched + 1} of rhs and dim {batched + 1} of acc must match"
        )

    # Validate MMA element type combinations using registry
    lhs_element_type = lhs.element_type
    rhs_element_type = rhs.element_type
    acc_element_type = acc.element_type

    # Find matching MMA configuration
    mma_config = find_mma_config(lhs_element_type, rhs_element_type, acc_element_type)

    if mma_config is None:
        # Generate helpful error message by showing supported configurations
        supported_configs = get_supported_mma_configs()
        if supported_configs:
            config_descriptions = [config.name for config in supported_configs]
            raise TypeError(
                f"Unsupported MMA element type combination: "
                f"{lhs_element_type} x {rhs_element_type} -> {acc_element_type}. "
                f"Supported configurations: {', '.join(config_descriptions)}"
            )
        else:
            # Fallback error if configurations haven't been initialized yet
            raise TypeError(
                f"Unsupported MMA element type combination: "
                f"{lhs_element_type} x {rhs_element_type} -> {acc_element_type}"
            )

    if isinstance(acc.element_type, _ods_ir.IntegerType):
        if fast_acc:
            raise ValueError("fast_acc is only supported for floating-point MMA")
        op = _cuda_tile.MmaIOp(
            lhs=lhs,
            rhs=rhs,
            acc=acc,
            signedness_lhs=get_signedness_attr(signedness_lhs),
            signedness_rhs=get_signedness_attr(signedness_rhs),
            loc=loc,
            ip=ip,
        )
    else:
        op = _cuda_tile.MmaFOp(
            lhs=lhs,
            rhs=rhs,
            acc=acc,
            fast_acc=fast_acc,
            loc=loc,
            ip=ip,
        )
    return return_results(op)


@cuda_tile_op
def mmaf_scaled(
    lhs: Tile,
    rhs: Tile,
    acc: Tile,
    lhs_scale: Tile,
    rhs_scale: Tile,
    *,
    loc=None,
    ip=None,
) -> Tile:
    """Computes scaled matrix-multiply-accumulate for low-precision types."""
    # Check shapes - mirrors verifyMmaShapes in CudaTile.cpp
    lhs_rank = len(lhs.tile_type.shape)
    rhs_rank = len(rhs.tile_type.shape)
    acc_rank = len(acc.tile_type.shape)
    lhs_scale_rank = len(lhs_scale.tile_type.shape)
    rhs_scale_rank = len(rhs_scale.tile_type.shape)

    if lhs_rank not in (2, 3):
        raise ValueError("operands must be 2D or 3D tiles")
    if lhs_rank != rhs_rank or rhs_rank != acc_rank:
        raise ValueError("lhs, rhs, acc must have the same rank")
    if lhs_rank != lhs_scale_rank or lhs_rank != rhs_scale_rank:
        raise ValueError("lhs, rhs, acc, lhs_scale, rhs_scale must have the same rank")

    batched = int(lhs_rank == 3)
    row_dim = batched + 0
    col_dim = batched + 1

    # Check batch dimensions (mirrors verifyMmaShapes)
    if batched:
        if lhs.tile_type.shape[0] != rhs.tile_type.shape[0]:
            raise ValueError(
                "dim 0 of lhs and dim 0 of rhs (batch dimension) must match"
            )
        if lhs.tile_type.shape[0] != acc.tile_type.shape[0]:
            raise ValueError(
                "dim 0 of lhs and dim 0 of acc (batch dimension) must match"
            )

    # Check MMA shape constraints (mirrors verifyMmaShapes)
    if lhs.tile_type.shape[col_dim] != rhs.tile_type.shape[row_dim]:
        raise ValueError(f"dim {col_dim} of lhs and dim {row_dim} of rhs must match")
    if lhs.tile_type.shape[row_dim] != acc.tile_type.shape[row_dim]:
        raise ValueError(f"dim {row_dim} of lhs and dim {row_dim} of acc must match")
    if rhs.tile_type.shape[col_dim] != acc.tile_type.shape[col_dim]:
        raise ValueError(f"dim {col_dim} of rhs and dim {col_dim} of acc must match")

    # Check scale shape constraints (mirrors MmaFScaledOp::verify)
    if batched:
        if lhs.tile_type.shape[0] != lhs_scale.tile_type.shape[0]:
            raise ValueError(
                f"shape error: dim 0 of lhs ({lhs.tile_type.shape[0]}) and "
                f"dim 0 of lhs_scale ({lhs_scale.tile_type.shape[0]}) must match"
            )
        if rhs.tile_type.shape[0] != rhs_scale.tile_type.shape[0]:
            raise ValueError(
                f"shape error: dim 0 of rhs ({rhs.tile_type.shape[0]}) and "
                f"dim 0 of rhs_scale ({rhs_scale.tile_type.shape[0]}) must match"
            )

    # M and N dimensions must match, K can be different
    if lhs.tile_type.shape[row_dim] != lhs_scale.tile_type.shape[row_dim]:
        raise ValueError(
            f"shape error: dim {row_dim} of lhs ({lhs.tile_type.shape[row_dim]}) and "
            f"dim {row_dim} of lhs_scale ({lhs_scale.tile_type.shape[row_dim]}) must match"
        )
    if rhs.tile_type.shape[col_dim] != rhs_scale.tile_type.shape[col_dim]:
        raise ValueError(
            f"shape error: dim {col_dim} of rhs ({rhs.tile_type.shape[col_dim]}) and "
            f"dim {col_dim} of rhs_scale ({rhs_scale.tile_type.shape[col_dim]}) must match"
        )

    # Validate element type combinations using registry (mirrors MmaFScaledOp::verify)
    lhs_element_type = lhs.element_type
    lhs_scale_element_type = lhs_scale.element_type

    # Find matching MMA Scaled configuration
    mma_scaled_config = find_mma_scaled_config(lhs_element_type, lhs_scale_element_type)

    if mma_scaled_config is None:
        # Generate helpful error message by showing supported configurations
        supported_configs = get_supported_mma_scaled_configs()
        if supported_configs:
            config_descriptions = [config.name for config in supported_configs]
            raise TypeError(
                f"Unsupported MMA Scaled element type combination: "
                f"operand={lhs_element_type}, scale={lhs_scale_element_type}. "
                f"Supported configurations: {', '.join(config_descriptions)}"
            )
        else:
            raise TypeError(
                f"Unsupported MMA Scaled element type combination: "
                f"operand={lhs_element_type}, scale={lhs_scale_element_type}"
            )

    return return_results(
        _cuda_tile.MmaFScaledOp(
            lhs=lhs,
            rhs=rhs,
            acc=acc,
            lhs_scale=lhs_scale,
            rhs_scale=rhs_scale,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def extract(
    result,
    source,
    indices,
    *,
    loc=None,
    ip=None,
) -> Tile:
    """Extracts a slice from the source tile at the specified indices."""
    if isinstance(result, TileType) is False:
        raise Exception("result type must be cuda_tile.TileType")

    # Verify that source and result have the same element type
    if source.element_type != result.element_type:
        raise ValueError("Expected source and result to have the same element type.")

    # Verify that the number of indices matches the rank of the source
    source_rank = len(source.tile_type.shape)
    if len(indices) != source_rank:
        raise ValueError(f"Expected {source_rank} indices, but got {len(indices)}")

    # Verify that source dimensions are evenly divisible by result dimensions
    source_shape = source.tile_type.shape
    result_shape = result.shape
    if len(source_shape) != len(result_shape):
        raise ValueError("Expected source and result to have the same rank.")

    for i, (source_dim, result_dim) in enumerate(zip(source_shape, result_shape)):
        if source_dim % result_dim != 0:
            raise ValueError(
                f"Source dimension {i} size ({source_dim}) must be evenly divisible by "
                f"result dimension {i} size ({result_dim})"
            )

    # Verify indices are valid and perform bounds checking
    for i, index in enumerate(indices):
        if not isinstance(index, Tile):
            raise TypeError(f"Index {i} must be a Tile, got {type(index).__name__}")

        # Check if index is scalar
        index_shape = index.tile_type.shape
        if len(index_shape) != 0:
            raise ValueError(
                f"Index {i} must be a scalar tile, got rank {len(index_shape)}"
            )

        # Check if index has i32 element type
        if index.element_type != _ods_ir.IntegerType.get_signless(32):
            raise ValueError(f"Index {i} must have i32 element type")

    # Constant bounds checking is done by the C++ verifier.

    op = _cuda_tile.ExtractOp(
        result=result, source=source, indices=indices, loc=loc, ip=ip
    )
    return return_results(op)


@cuda_tile_op
def insert(source, destination, indices, *, loc=None, ip=None) -> Tile:
    """Inserts a source subtile into a destination tile at the specified indices."""
    source_shape = source.tile_type.shape
    dest_shape = destination.tile_type.shape

    if source.element_type != destination.element_type:
        raise ValueError(
            "Expected source and destination to have the same element type."
        )

    if len(source_shape) != len(dest_shape):
        raise ValueError("Expected source and destination to have the same rank.")

    source_rank = len(source_shape)
    if len(indices) != source_rank:
        raise ValueError(f"Expected {source_rank} indices, but got {len(indices)}")

    for i, (src_dim, dst_dim) in enumerate(zip(source_shape, dest_shape)):
        if dst_dim % src_dim != 0:
            raise ValueError(
                f"Destination dimension {i} size ({dst_dim}) must be evenly divisible by "
                f"source dimension {i} size ({src_dim})"
            )

    for i, index in enumerate(indices):
        if not isinstance(index, Tile):
            raise TypeError(f"Index {i} must be a Tile, got {type(index).__name__}")

        index_shape = index.tile_type.shape
        if len(index_shape) != 0:
            raise ValueError(
                f"Index {i} must be a scalar tile, got rank {len(index_shape)}"
            )

        if index.element_type != _ods_ir.IntegerType.get_signless(32):
            raise ValueError(f"Index {i} must have i32 element type")

    return return_results(
        _cuda_tile.InsertOp(
            source=source,
            destination=destination,
            indices=indices,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def get_tile_block_id(*, loc=None, ip=None) -> Tile:
    """Get the ID of the current tile block."""
    return return_results(_cuda_tile.GetTileBlockIdOp(loc=loc, ip=ip))


@cuda_tile_op
def get_num_tile_blocks(*, loc=None, ip=None) -> Tile:
    """Get number of tile blocks."""
    return return_results(_cuda_tile.GetNumTileBlocksOp(loc=loc, ip=ip))


@cuda_tile_op
def trunci(el_type, from_, *, overflow=None, loc=None, ip=None) -> Tile:
    """Truncates the source integer to the specified target type."""
    if not isinstance(el_type, type) or not isinstance(
        el_type.mlir_type, _ods_ir.IntegerType
    ):
        raise TypeError(f"expected integer destination el_type, but got {el_type}")
    src_el_type = from_.tile_type.element_type
    if not isinstance(src_el_type, _ods_ir.IntegerType):
        raise TypeError(f"expected integer tile type for source, but got {src_el_type}")
    if src_el_type.width <= el_type.mlir_type.width:
        raise TypeError(
            f"source type {src_el_type} has a bitwidth smaller than or equal to destination type {el_type.mlir_type}"
        )

    overflow_attr = (
        get_integer_overflow_attr(overflow) if overflow is not None else None
    )
    result_type = make_tile_type(el_type, from_.tile_type.shape)
    return return_results(
        _cuda_tile.TruncIOp(
            to=result_type, from_=from_, overflow=overflow_attr, loc=loc, ip=ip
        )
    )


@cuda_tile_op
def load_ptr_tko(
    result: TileType,
    source,
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.WEAK,
    memory_scope=None,
    input_token=None,
    mask=None,
    padding_value=None,
    return_token: bool = False,
    arch=None,
    latency=None,
    suppress_optimization_hints: bool = False,
    loc=None,
    ip=None,
) -> Tile | Tuple[Tile, Token]:
    """Load data from memory with specified ordering and optional masking.

    Loads data from the given source pointer(s) using the specified memory
    synchronization semantics. Supports scalar and tile loads, as well as
    optional masking with a padding value for masked-out elements.

    :param result: The result tile type (shape and element type)
    :type result: TileType
    :param source: Tile of pointers to load from; must match result shape
    :type source: Tile
    :param memory_ordering_semantics: Memory ordering guarantees ("relaxed", "strong", or "weak")
    :type memory_ordering_semantics: str
    :param input_token: Optional synchronization token for ordering
    :type input_token: Optional[Token]
    :param memory_scope: Memory visibility scope ("device", "sys", "tl_blk", or None)
    :type memory_scope: Optional[str]
    :param mask: Optional boolean mask (i1 tile) matching result shape
    :type mask: Optional[Tile]
    :param padding_value: Value used for masked-out elements (requires mask)
    :type padding_value: Optional[Tile]
    :param return_token: Whether to return a synchronization token alongside the result
    :type return_token: bool
    :param arch: Architecture name to use for OptimizationHint ("sm_80", "sm_90", "sm_100", "sm_103", "sm_120")
    :type arch: Optional[str]
    :param latency: Latency Hint value in the range [1, 10]
    :type latency: Optional[int]
    :param loc: Source location for MLIR operation tracking
    :type loc: Optional[Location]
    :param ip: Insertion point for MLIR operation
    :type ip: Optional[InsertionPoint]

    :return: A Tile containing the loaded data, or (Tile, Token) if return_token is True
    :rtype: Tile | Tuple[Tile, Token]

    :raises ValueError: If validation fails (e.g., mismatched shapes or invalid parameters)
    """
    if not isinstance(result, TileType):
        raise Exception("result type must be cuda_tile.TileType")

    if not PointerType.isinstance(source.element_type):
        raise ValueError("source must be a pointer or tile of pointers")

    if source.tile_type.shape != result.shape:
        raise ValueError("source must have the same shape as the result")

    if input_token is not None:
        if not isinstance(input_token, Token):
            raise ValueError("input_token must be a Token")

    if mask is not None:
        if not isinstance(mask, Tile):
            raise ValueError("mask must be a Tile")
        if not (
            isinstance(mask.element_type, _ods_ir.IntegerType)
            and mask.element_type.width == 1
        ):
            raise ValueError("mask must have boolean element type")
        if mask.tile_type.shape != result.shape:
            raise ValueError("mask must have the same shape as the result")

    if padding_value is not None:
        if mask is None:
            raise ValueError("padding_value only supported if mask is provided")
        if not isinstance(padding_value, Tile):
            raise ValueError("padding_value must be a Tile")
        if padding_value.element_type != result.element_type:
            raise ValueError(
                "padding_value must have the same element type as the result"
            )
        if padding_value.tile_type.shape != result.shape:
            raise ValueError("padding_value must have the same shape as the result")

    if memory_ordering_semantics not in [
        MemoryOrderingSemantics.RELAXED,
        MemoryOrderingSemantics.ACQUIRE,
        MemoryOrderingSemantics.WEAK,
    ]:
        raise ValueError(
            "memory_ordering_semantics must be one of: relaxed, acquire, or weak"
        )

    memory_ordering_semantics_attr = get_memory_ordering_semantics_attr(
        memory_ordering_semantics
    )

    memory_scope_attr = None
    if memory_ordering_semantics != MemoryOrderingSemantics.WEAK:
        memory_scope_attr = get_memory_scope_attr(memory_scope)

    optimization_hints = None
    if not isinstance(suppress_optimization_hints, bool):
        raise TypeError("suppress_optimization_hints must be a bool")
    if suppress_optimization_hints and any(
        value is not None
        for value in (
            arch,
            latency,
        )
    ):
        raise ValueError(
            "suppress_optimization_hints cannot be combined with arch, "
            "latency"
        )
    if not suppress_optimization_hints:
        arch, latency, allow_tma_resolved = _resolve_load_store_hint(
            result, arch, latency
        )
        if (arch is not None) and (latency is not None):
            optimization_hints = OptimizationHintsAttr.getLoadStoreOpHint(
                arch,
                allow_tma_resolved,
                latency,
                context=_ods_get_default_loc_context(loc),
            )
        elif latency is not None:
            raise ValueError(
                f"Expected arch to be specified for OptimizationHint: latency = {latency}"
            )
    # Create the load_ptr_tko operation, which returns both a tile and a token
    result_token_type = TokenType.get()
    load_op = _cuda_tile.LoadPtrTkoOp(
        result=result,
        result_token=result_token_type,
        memory_ordering_semantics=memory_ordering_semantics_attr,
        memory_scope=memory_scope_attr,
        source=source,
        mask=mask,
        paddingValue=padding_value,
        token=input_token,
        optimization_hints=optimization_hints,
        loc=loc,
        ip=ip,
    )

    # Return both tile and token if requested
    if return_token:
        return return_results(load_op)

    # Otherwise, return only the tile result
    return Tile(load_op.results[0], load_op.results[0].type)


@cuda_tile_op
def load_view_tko(
    view: TileView,
    indices: Sequence[Tile | int],
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.WEAK,
    memory_scope: MemoryScope = None,
    input_token=None,
    return_token: bool = False,
    arch=None,
    allow_tma=None,
    latency=None,
    suppress_optimization_hints: bool = False,
    loc=None,
    ip=None,
) -> Tile | Tuple[Tile, Token]:
    """Load data from a tile view with specified memory ordering and scope."""
    if not isinstance(view, TileView):
        raise TypeError(f"Expected a tile view, got {view}")

    if view.view_index_rank != len(indices):
        raise ValueError(
            f"Expected {view.view_index_rank} index values, got {len(indices)}"
        )

    if input_token is not None:
        if not isinstance(input_token, Token):
            raise ValueError("input_token must be a Token")

    # Add memory ordering semantics validation aligned with C++ implementation
    if memory_ordering_semantics not in [
        MemoryOrderingSemantics.WEAK,
        MemoryOrderingSemantics.RELAXED,
        MemoryOrderingSemantics.ACQUIRE,
    ]:
        raise ValueError(
            "memory_ordering_semantics must be one of: weak, relaxed, or acquire"
        )

    # Add memory scope validation aligned with C++ implementation
    if (
        memory_ordering_semantics == MemoryOrderingSemantics.WEAK
        and memory_scope is not None
    ):
        raise ValueError("weak load must not have memory scope")

    if not isinstance(suppress_optimization_hints, bool):
        raise TypeError("suppress_optimization_hints must be a bool")
    if suppress_optimization_hints and any(
        value is not None for value in (arch, allow_tma, latency)
    ):
        raise ValueError(
            "suppress_optimization_hints cannot be combined with arch, "
            "allow_tma, or latency"
        )
    index_tiles = _view_index_list_to_tiles(view, indices)
    sem_attr = get_memory_ordering_semantics_attr(memory_ordering_semantics)

    memory_scope_attr = None
    if memory_ordering_semantics != MemoryOrderingSemantics.WEAK:
        memory_scope_attr = get_memory_scope_attr(memory_scope)

    result_token_type = TokenType.get()

    # Preserve explicit hints for all views. Only ordinary views may receive
    # inferred hints from the resolver.
    if suppress_optimization_hints:
        arch, latency, allow_tma = None, None, None
    elif (
        isinstance(view, GatherScatterView)
        and arch is None
        and latency is None
        and allow_tma is None
    ):
        arch, latency, allow_tma = None, None, None
    else:
        arch, latency, allow_tma = _resolve_load_store_hint(
            view.view_tile_type,
            arch,
            latency,
            allow_tma=allow_tma,
            apply_tile_size_gate=False,
        )
    optimization_hints = None
    if (arch is not None) and ((allow_tma is not None) or (latency is not None)):
        optimization_hints = OptimizationHintsAttr.getLoadStoreOpHint(
            arch,
            allow_tma,  # Pass None/True/False as-is to C++ binding
            0 if latency is None else latency,
            context=_ods_get_default_loc_context(loc),
        )
    elif (allow_tma is not None) or (latency is not None):
        # (arch == None) and hint values are specified
        raise ValueError(
            "Expected arch to be specified for OptimizationHint:"
            f" allow_tma = {allow_tma}, latency = {latency}"
        )
    inbounds_attr = _ods_ir.DenseBoolArrayAttr.get([False] * len(index_tiles))

    load_op = _cuda_tile.LoadViewTkoOp(
        tile=view.view_tile_type,
        result_token=result_token_type,
        memory_ordering_semantics=sem_attr,
        memory_scope=memory_scope_attr,
        view=view,
        index=index_tiles,
        token=input_token,
        optimization_hints=optimization_hints,
        inbounds=inbounds_attr,
        loc=loc,
        ip=ip,
    )

    # Return both tile and token if requested
    if return_token:
        return return_results(load_op)

    # Otherwise return only tile result
    return Tile(load_op.results[0], load_op.results[0].type)


@cuda_tile_op
def permute(source: Tile, permutation, *, loc=None, ip=None) -> Tile:
    """Rearranges the elements of the source tile according to the permutation."""

    src_shape = source.tile_type.shape
    rank = len(src_shape)
    if rank < 2:
        raise Exception(f"expected at least rank 2, but got: {rank}")

    # Verify permutation.
    permutation_sz = len(permutation)
    if permutation_sz != rank:
        raise Exception(
            f"expected permutation size {permutation_sz} to equal the rank of the source {rank}"
        )
    if len(tuple(set(permutation))) != rank:
        raise Exception(f"expected permutation elements {permutation} to be unique")
    for idx, perm in enumerate(permutation):
        if perm < 0 or perm >= rank:
            raise Exception(
                f"permutation element at index {idx} '{perm}' is out of bounds [0, {rank})"
            )

    return return_results(
        _cuda_tile.PermuteOp(source=source, permutation=permutation, loc=loc, ip=ip)
    )


@cuda_tile_op
def reshape(
    shape: List[int],
    source: Tile,
    *,
    loc=None,
    ip=None,
) -> Tile:
    result_type = TileType.get(shape, source.element_type)
    op = _cuda_tile.ReshapeOp(result=result_type, source=source, loc=loc, ip=ip)
    return return_results(op)


@cuda_tile_op
def make_token(*, loc=None, ip=None) -> Token:
    return return_results(_cuda_tile.MakeTokenOp(loc=loc, ip=ip))


@cuda_tile_op
def memory_fence_alias_tko(input_token: Token, *, loc=None, ip=None) -> Token:
    """Token-ordered proxy alias fence.

    Along the token chain, this fence orders all earlier target memory
    operations that access virtually aliased memory before the fence and
    requires their completion before operations consume the result token.
    """
    if not isinstance(input_token, Token):
        raise TypeError("input_token must be a Token")

    return return_results(
        _cuda_tile.MemoryFenceAliasTkoOp(
            token=input_token,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def join_tokens(*tokens, loc=None, ip=None) -> Token:
    """Join multiple tokens into a single token.

    Args:
        *tokens: Variable number of Token objects to join
        loc: Source location
        ip: Insertion point

    Returns:
        A new Token that represents the join of all input tokens
    """
    # Ensure all inputs are Token objects
    if not all(isinstance(token, Token) for token in tokens):
        raise TypeError("All arguments must be Token objects")

    return return_results(_cuda_tile.JoinTokensOp(tokens=tokens, loc=loc, ip=ip))


@cuda_tile_op
def store_ptr_tko(
    destination,
    value,
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.WEAK,
    memory_scope=None,
    input_token=None,
    mask=None,
    arch=None,
    latency=None,
    suppress_optimization_hints: bool = False,
    loc=None,
    ip=None,
) -> Token:
    """Store a value into memory with specified ordering and optional masking.

    Performs memory stores to the specified destination pointer(s) using the given
    memory synchronization semantics. Supports both scalar and tile stores,
    and allows optional masking to conditionally store values.

    :param destination: Tile of pointers to store to; must match the shape of value
    :type destination: Tile
    :param value: Tile containing the data to store
    :type value: Tile
    :param memory_ordering_semantics: Memory ordering guarantees ("relaxed", "strong", or "weak")
    :type memory_ordering_semantics: str
    :param input_token: Optional synchronization token for ordering
    :type input_token: Optional[Token]
    :param memory_scope: Memory visibility scope ("device", "sys", "tl_blk", or None)
    :type memory_scope: Optional[str]
    :param mask: Optional boolean mask (i1 tile) matching the shape of value
    :type mask: Optional[Tile]
    :param arch: Architecture name to use for OptimizationHint ("sm_80", "sm_90", "sm_100", "sm_103", "sm_120")
    :type arch: Optional[str]
    :param latency: Latency Hint value in the range [1, 10]
    :type latency: Optional[int]
    :param loc: Source location for MLIR operation tracking
    :type loc: Optional[Location]
    :param ip: Insertion point for MLIR operation
    :type ip: Optional[InsertionPoint]

    :return: A synchronization token for use in subsequent memory operations
    :rtype: Token

    :raises ValueError: If validation fails (e.g., incompatible shapes or invalid parameters)
    """
    if not PointerType.isinstance(destination.element_type):
        raise ValueError("destination must be a pointer or tile of pointers")

    if not isinstance(value, Tile):
        raise ValueError("value must be a Tile")
    if destination.tile_type.shape != value.tile_type.shape:
        raise ValueError("destination must have the same shape as the value")

    if input_token is not None:
        if not isinstance(input_token, Token):
            raise ValueError("input_token must be a Token")

    if mask is not None:
        if not isinstance(mask, Tile):
            raise ValueError("mask must be a Tile")
        if not (
            isinstance(mask.element_type, _ods_ir.IntegerType)
            and mask.element_type.width == 1
        ):
            raise ValueError("mask must have boolean element type")
        if mask.tile_type.shape != value.tile_type.shape:
            raise ValueError("mask must have the same shape as the value")

    if memory_ordering_semantics not in [
        MemoryOrderingSemantics.RELAXED,
        MemoryOrderingSemantics.RELEASE,
        MemoryOrderingSemantics.WEAK,
    ]:
        raise ValueError(
            "memory_ordering_semantics must be one of: relaxed, release, or weak"
        )

    memory_ordering_semantics_attr = get_memory_ordering_semantics_attr(
        memory_ordering_semantics
    )

    if memory_scope is not None:
        if memory_scope not in [
            MemoryScope.DEVICE,
            MemoryScope.SYS,
            MemoryScope.TL_BLK,
        ]:
            raise ValueError(
                "memory_ordering_semantics must be one of: device, sys and tl_blk"
            )
        memory_scope_attr = get_memory_scope_attr(memory_scope)
    else:
        memory_scope_attr = None

    optimization_hints = None
    if not isinstance(suppress_optimization_hints, bool):
        raise TypeError("suppress_optimization_hints must be a bool")
    if suppress_optimization_hints and any(
        value is not None
        for value in (
            arch,
            latency,
        )
    ):
        raise ValueError(
            "suppress_optimization_hints cannot be combined with arch, "
            "latency"
        )
    if not suppress_optimization_hints:
        arch, latency, allow_tma_resolved = _resolve_load_store_hint(
            value.tile_type, arch, latency
        )
        if (arch is not None) and (latency is not None):
            optimization_hints = OptimizationHintsAttr.getLoadStoreOpHint(
                arch,
                allow_tma_resolved,
                latency,
                context=_ods_get_default_loc_context(loc),
            )
        elif latency is not None:
            raise ValueError(
                f"Expected arch to be specified for OptimizationHint: latency = {latency}"
            )
    return return_results(
        _cuda_tile.StorePtrTkoOp(
            memory_ordering_semantics=memory_ordering_semantics_attr,
            memory_scope=memory_scope_attr,
            destination=destination,
            value=value,
            mask=mask,
            token=input_token,
            optimization_hints=optimization_hints,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def store_view_tko(
    tile: Tile,
    view: TileView,
    indices: Sequence[Tile | int],
    *,
    memory_ordering_semantics: MemoryOrderingSemantics = MemoryOrderingSemantics.WEAK,
    memory_scope: MemoryScope = None,
    input_token=None,
    arch=None,
    allow_tma=None,
    latency=None,
    suppress_optimization_hints: bool = False,
    loc=None,
    ip=None,
) -> Token:
    """Store a tile to a tile view with specified memory ordering and scope."""

    if not isinstance(view, TileView):
        raise TypeError(f"Expected a tile view, got {view}")

    if tile.tile_type != view.view_tile_type:
        raise TypeError(
            f"Expected tile type to be {view.view_tile_type}, got {tile.tile_type}"
        )

    if input_token is not None:
        if not isinstance(input_token, Token):
            raise ValueError("input_token must be a Token")

    # Add memory ordering semantics validation aligned with C++ implementation
    if memory_ordering_semantics not in [
        MemoryOrderingSemantics.WEAK,
        MemoryOrderingSemantics.RELAXED,
        MemoryOrderingSemantics.RELEASE,
    ]:
        raise ValueError(
            "memory_ordering_semantics must be one of: weak, relaxed, or release"
        )

    # Add memory scope validation aligned with C++ implementation
    if (
        memory_ordering_semantics == MemoryOrderingSemantics.WEAK
        and memory_scope is not None
    ):
        raise ValueError("weak store must not have memory scope")

    if not isinstance(suppress_optimization_hints, bool):
        raise TypeError("suppress_optimization_hints must be a bool")
    if suppress_optimization_hints and any(
        value is not None for value in (arch, allow_tma, latency)
    ):
        raise ValueError(
            "suppress_optimization_hints cannot be combined with arch, "
            "allow_tma, or latency"
        )
    # Add index count validation
    if view.view_index_rank != len(indices):
        raise ValueError(
            f"Expected {view.view_index_rank} index values, got {len(indices)}"
        )

    index_tiles = _view_index_list_to_tiles(view, indices)
    sem_attr = get_memory_ordering_semantics_attr(memory_ordering_semantics)

    scope_attr = None
    if memory_ordering_semantics != MemoryOrderingSemantics.WEAK:
        scope_attr = get_memory_scope_attr(memory_scope)
    # Preserve explicit hints for all views. Only ordinary views may receive
    # inferred hints from the resolver.
    if suppress_optimization_hints:
        arch, latency, allow_tma = None, None, None
    elif (
        isinstance(view, GatherScatterView)
        and arch is None
        and latency is None
        and allow_tma is None
    ):
        arch, latency, allow_tma = None, None, None
    else:
        arch, latency, allow_tma = _resolve_load_store_hint(
            tile.tile_type,
            arch,
            latency,
            allow_tma=allow_tma,
            apply_tile_size_gate=False,
        )
    optimization_hints = None
    if (arch is not None) and ((allow_tma is not None) or (latency is not None)):
        optimization_hints = OptimizationHintsAttr.getLoadStoreOpHint(
            arch,
            allow_tma,  # Pass None/True/False as-is to C++ binding
            0 if latency is None else latency,
            context=_ods_get_default_loc_context(loc),
        )
    elif (allow_tma is not None) or (latency is not None):
        # (arch == None) and hint values are specified
        raise ValueError(
            "Expected arch to be specified for OptimizationHint:"
            f" allow_tma = {allow_tma}, latency = {latency}"
        )
    inbounds_attr = _ods_ir.DenseBoolArrayAttr.get([False] * len(index_tiles))

    store_op = _cuda_tile.StoreViewTkoOp(
        memory_ordering_semantics=sem_attr,
        memory_scope=scope_attr,
        tile=tile,
        view=view,
        index=index_tiles,
        token=input_token,
        optimization_hints=optimization_hints,
        inbounds=inbounds_attr,
        loc=loc,
        ip=ip,
    )
    return return_results(store_op)


@cuda_tile_op
def select(condition, trueval, falseval, *, loc=None, ip=None) -> Tile:
    if trueval.element_type != falseval.element_type:
        raise TypeError("trueval and falseval must have the same element type")
    if (
        not isinstance(condition.element_type, _ods_ir.IntegerType)
        or condition.element_type.width != 1
    ):
        raise TypeError("condition must have boolean element type")
    if (
        trueval.tile_type.shape != falseval.tile_type.shape
        or condition.tile_type.shape != trueval.tile_type.shape
    ):
        raise TypeError("trueval, falseval and condition must have the same shape")
    return return_results(
        _cuda_tile.SelectOp(condition, trueval, falseval, loc=loc, ip=ip)
    )


# Modes accepted by the FToFOp `OnlyVariants` constraint. Anything outside
# this set (e.g. APPROX, FULL) is always invalid for ftof.
_FTOF_VALID_MODES = (
    RoundingMode.NEAREST_EVEN,
    RoundingMode.ZERO,
    RoundingMode.NEGATIVE_INF,
    RoundingMode.POSITIVE_INF,
    RoundingMode.NEAREST_AWAY,
)


def _is_ftof_low_precision_float_type(ty) -> bool:
    """Return whether a float type is narrower than f16."""
    return ty.width < 16


def _get_ftof_float_semantics(ty):
    """Return float semantics for known ftof types.

    The tuple is (mantissa_bits, min_exponent, max_exponent).  The ftof
    strict-widening check uses these values to prove that the destination can
    exactly represent every source value.  Returning None means the type is not
    known here, so validation must not use the strict-widening rule.
    """
    semantics = (
        (Float4E2M1FN.mlir_type, (2, -1, 2)),
        (Float8E4M3FN.mlir_type, (4, -6, 8)),
        (Float8E4M3FNUZ.mlir_type, (4, -7, 8)),
        (Float8E5M2.mlir_type, (3, -14, 15)),
        (Float8E8M0FNU.mlir_type, (1, -127, 127)),
        (Float16.mlir_type, (11, -14, 15)),
        (BFloat16.mlir_type, (8, -126, 127)),
        (TFloat32.mlir_type, (11, -126, 127)),
        (Float32.mlir_type, (24, -126, 127)),
        (Float64.mlir_type, (53, -1022, 1023)),
    )
    nv8e5m3fnu = globals().get("Float8E5M3FNU")
    if nv8e5m3fnu is not None:
        semantics += ((nv8e5m3fnu.mlir_type, (4, -15, 16)),)

    for float_ty, float_semantics in semantics:
        if ty == float_ty:
            return float_semantics
    return None


def _is_ftof_strict_widening(src_ty, dst_ty) -> bool:
    """Return whether ftof can ignore rounding for this widening pair."""
    src_semantics = _get_ftof_float_semantics(src_ty)
    dst_semantics = _get_ftof_float_semantics(dst_ty)
    if src_semantics is None or dst_semantics is None:
        return False

    src_precision, src_min_exp, src_max_exp = src_semantics
    dst_precision, dst_min_exp, dst_max_exp = dst_semantics
    return (
        dst_precision >= src_precision
        and dst_min_exp <= src_min_exp
        and dst_max_exp >= src_max_exp
    )


def _validate_ftof_rounding_mode(src_ty, dst_ty, mode: RoundingMode) -> None:
    """Mirror the FToFOp verifier's ordered rounding-mode rules."""
    if mode not in _FTOF_VALID_MODES:
        names = ", ".join(repr(m.value) for m in _FTOF_VALID_MODES)
        raise ValueError(
            f"invalid rounding mode {mode.value!r} for ftof; supported: {names}"
        )

    if isinstance(dst_ty, _ods_ir.Float8E8M0FNUType):
        if mode not in (RoundingMode.ZERO, RoundingMode.POSITIVE_INF):
            raise ValueError(
                f"invalid rounding mode {mode.value!r} for ftof to f8E8M0FNU; "
                f"supported: 'zero', 'positive_inf'"
            )
        return

    if _is_ftof_low_precision_float_type(dst_ty):
        if mode != RoundingMode.NEAREST_EVEN:
            raise ValueError(
                f"invalid rounding mode {mode.value!r} for ftof to "
                f"low-precision type; supported: 'nearest_even'"
            )
        return

    if _is_ftof_strict_widening(src_ty, dst_ty):
        return

    if src_ty == Float64.mlir_type and dst_ty == Float32.mlir_type:
        if mode not in (
            RoundingMode.NEAREST_EVEN,
            RoundingMode.ZERO,
            RoundingMode.NEGATIVE_INF,
            RoundingMode.POSITIVE_INF,
        ):
            raise ValueError(
                f"invalid rounding mode {mode.value!r} for ftof from f64 to f32; "
                f"supported: 'nearest_even', 'zero', 'negative_inf', 'positive_inf'"
            )
        return

    if src_ty == Float32.mlir_type and dst_ty == TFloat32.mlir_type:
        if mode not in (
            RoundingMode.NEAREST_EVEN,
            RoundingMode.ZERO,
            RoundingMode.NEAREST_AWAY,
        ):
            raise ValueError(
                f"invalid rounding mode {mode.value!r} for ftof from f32 to tf32; "
                f"supported: 'nearest_even', 'zero', 'nearest_away'"
            )
        return

    if mode not in (RoundingMode.NEAREST_EVEN, RoundingMode.ZERO):
        raise ValueError(
            f"invalid rounding mode {mode.value!r} for ftof narrowing conversion; "
            f"supported: 'nearest_even', 'zero'"
        )


@cuda_tile_op
def ftof(
    el_type,
    from_,
    *,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    loc=None,
    ip=None,
) -> Tile:
    if not isinstance(el_type, type) or not isinstance(
        el_type.mlir_type, _ods_ir.FloatType
    ):
        raise TypeError(f"expected float destination el_type, but got {el_type}")
    src_el_type = from_.tile_type.element_type
    if not isinstance(src_el_type, _ods_ir.FloatType):
        raise TypeError(f"expected float tile type for source, but got {src_el_type}")

    if src_el_type == el_type.mlir_type:
        raise TypeError(f"source and destination types are identical: {src_el_type}")

    _validate_ftof_rounding_mode(src_el_type, el_type.mlir_type, rounding_mode)
    result_type = make_tile_type(el_type, from_.tile_type.shape)
    return return_results(
        _cuda_tile.FToFOp(
            to=result_type,
            from_=from_,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def ftoi(
    el_type,
    from_,
    *,
    signedness: Signedness = Signedness.SIGNED,
    saturating: bool = False,
    loc=None,
    ip=None,
) -> Tile:
    if not isinstance(el_type, type) or not isinstance(
        el_type.mlir_type, _ods_ir.IntegerType
    ):
        raise TypeError(
            f"expected integer destination el_type, but got {el_type.mlir_type}"
        )
    src_el_type = from_.tile_type.element_type
    if not isinstance(src_el_type, _ods_ir.FloatType):
        raise TypeError(f"expected float tile type for source, but got {src_el_type}")

    result_type = make_tile_type(el_type, from_.tile_type.shape)
    return return_results(
        _cuda_tile.FToIOp(
            to=result_type,
            from_=from_,
            signedness=get_signedness_attr(signedness),
            rounding_mode=get_rounding_mode_attr(RoundingMode.NEAREST_INT_TO_ZERO),
            saturating=saturating,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def iota(n: int, el_type, *, loc=None, ip=None) -> Tile:
    mlir_type = _get_mlir_type(el_type)
    if not isinstance(mlir_type, _ods_ir.IntegerType):
        raise TypeError(f"iota requires an integer element type, got {mlir_type}")
    bitwidth = mlir_type.width
    if n > (1 << bitwidth):
        raise Exception(
            f"the number of elements {n} exceeds the maximum value of {bitwidth}-bit integer"
        )
    result_type = make_tile_type(mlir_type, (n,))
    return return_results(_cuda_tile.IotaOp(result=result_type, loc=loc, ip=ip))


@cuda_tile_op
def exti(
    el_type, from_, *, signedness: Signedness = Signedness.SIGNED, loc=None, ip=None
) -> Tile:
    if not isinstance(el_type, type) or not isinstance(
        el_type.mlir_type, _ods_ir.IntegerType
    ):
        raise TypeError(
            f"expected integer destination el_type, but got {el_type.mlir_type}"
        )
    src_el_type = from_.tile_type.element_type
    if not isinstance(src_el_type, _ods_ir.IntegerType):
        raise TypeError(f"expected integer tile type for source, but got {src_el_type}")

    if src_el_type.width >= el_type.mlir_type.width:
        raise TypeError(
            f"source type {src_el_type} has a bitwidth greater than or equal to destination type {el_type.mlir_type}"
        )

    result_type = make_tile_type(el_type, from_.tile_type.shape)
    return return_results(
        _cuda_tile.ExtIOp(
            to=result_type,
            from_=from_,
            signedness=get_signedness_attr(signedness),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def itof(
    el_type,
    from_,
    *,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
):
    if not isinstance(el_type, type) or not isinstance(
        el_type.mlir_type, _ods_ir.FloatType
    ):
        raise TypeError(
            f"expected float destination el_type, but got {el_type.mlir_type}"
        )
    src_el_type = from_.tile_type.element_type
    if not isinstance(src_el_type, _ods_ir.IntegerType):
        raise TypeError(f"expected integer tile type for source, but got {src_el_type}")
    if el_type.mlir_type == Float8E8M0FNU.mlir_type:
        raise ValueError(
            f"Integer to f8E8M0FNU conversion is not supported, please first convert to another float type"
        )
    if rounding_mode != RoundingMode.NEAREST_EVEN:
        raise ValueError(
            f"Invalid rounding mode for itof: {rounding_mode}, expected NEAREST_EVEN"
        )

    result_type = make_tile_type(el_type, from_.tile_type.shape)
    return return_results(
        _cuda_tile.IToFOp(
            to=result_type,
            from_=from_,
            signedness=get_signedness_attr(signedness),
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def if_generate(
    condition: Tile,
    then_body: Callable,
    else_body: Optional[Callable] | None = None,
    input_args: List[Union[Tile, Token]] | None = None,
    return_types: List[Union[Tile, Token, _ods_ir.Type]] | None = None,
    *,
    loc=None,
    ip=None,
) -> Tile:
    input_args = input_args or []
    return_types = return_types or []

    result_types = [
        t.value.type if isinstance(t, (Tile, Token)) else t
        for t in return_types
    ]

    if_op = _cuda_tile.IfOp(results_=result_types, condition=condition, loc=loc, ip=ip)
    if_op.thenRegion.blocks.append()

    with _ods_ir.InsertionPoint(if_op.thenRegion.blocks[0]):
        args = then_body(*input_args)
        if args is None:
            _ensure_if_region_terminator(if_op.thenRegion.blocks[0], loc=loc, ip=ip)
        elif isinstance(args, (Tile, Token)):
            _cuda_tile.YieldOp(operands_=[args.value], loc=loc, ip=ip)
        else:
            yielded = [a.value for a in args]
            _cuda_tile.YieldOp(operands_=yielded, loc=loc, ip=ip)

    if else_body is not None:
        if_op.elseRegion.blocks.append()
        with _ods_ir.InsertionPoint(if_op.elseRegion.blocks[0]):
            args = else_body(*input_args)
            if args is None:
                _ensure_if_region_terminator(if_op.elseRegion.blocks[0], loc=loc, ip=ip)
            elif isinstance(args, (Tile, Token)):
                _cuda_tile.YieldOp(operands_=[args.value], loc=loc, ip=ip)
            else:
                yielded = [a.value for a in args]
                _cuda_tile.YieldOp(operands_=yielded, loc=loc, ip=ip)

    return return_results(if_op)


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def _muli(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.MulIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def _mulf(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool,
    rounding_mode: RoundingMode,
    loc=None,
    ip=None,
) -> Tile:

    if rounding_mode not in [
        RoundingMode.NEAREST_EVEN,
        RoundingMode.ZERO,
        RoundingMode.NEGATIVE_INF,
        RoundingMode.POSITIVE_INF,
    ]:
        raise ValueError(
            f"Invalid rounding mode for mulf: {rounding_mode}, expected one of NEAREST_EVEN, ZERO, NEGATIVE_INF, POSITIVE_INF"
        )

    return return_results(
        _cuda_tile.MulFOp(
            lhs,
            rhs,
            flush_to_zero=flush_to_zero,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def mulhii(lhs: Tile, rhs: Tile, *, loc=None, ip=None) -> Tile:
    # Performs element-wise high-n bits of multiplication of two tiles.
    el_type = lhs.element_type
    if el_type.width not in [1, 8, 16, 32, 64]:
        raise TypeError(
            f"expected i1, i8, i16, i32, or i64 element types in rhs/lhs, but got ({el_type})"
        )
    return return_results(_cuda_tile.MulhiIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
def mul(
    lhs: Tile,
    rhs: Tile,
    *,
    flush_to_zero: bool = False,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    loc=None,
    ip=None,
) -> Tile:
    """Performs element-wise multiplication of two tiles."""
    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        if flush_to_zero:
            raise ValueError(
                "flush_to_zero is only valid for floating-point operations"
            )
        return _muli(lhs, rhs, loc=loc, ip=ip)
    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        return _mulf(
            lhs,
            rhs,
            flush_to_zero=flush_to_zero,
            rounding_mode=rounding_mode,
            loc=loc,
            ip=ip,
        )
    else:
        raise TypeError("expected integer or float element type")


@cuda_tile_op
def loop_generate(
    inputs: Iterable[Union[Tile, Token]],
    loop_body: Callable,
    *,
    result_types: Iterable[Union[Tile, Token, _ods_ir.Type]] | None = None,
    loc=None,
    ip=None,
) -> Union[
    Tile,
    Token,
    Tuple[Union[Tile, Token], ...],
    None,
]:
    inputs = tuple(inputs)
    input_types = []
    for i in inputs:
        if not isinstance(i, (Tile, Token)):
            raise TypeError("All elements in 'inputs' must be Tile or Token values.")
        if isinstance(i, Tile):
            input_types.append(i.tile_type)
        else:
            input_types.append(i.value.type)
    if result_types is None:
        output_types = input_types
    else:
        output_types = []
        for result_type in result_types:
            if isinstance(result_type, (Tile, Token)):
                result_type = result_type.value.type
            if not isinstance(result_type, _ods_ir.Type) or not (
                TileType.isinstance(result_type)
                or TokenType.isinstance(result_type)
            ):
                raise TypeError(
                    "All elements in 'result_types' must be Tile, Token, or their "
                    "supported MLIR types."
                )
            output_types.append(result_type)
    while_op = _cuda_tile.LoopOp(output_types, inputs, loc=loc, ip=ip)
    while_op.region.blocks.append(*input_types)
    block = while_op.region.blocks[0]
    with _ods_ir.InsertionPoint(block):
        loop_args = []
        for barg in block.arguments:
            loop_args.append(_wrap_result_value(barg))
        loop_body(*loop_args)

    return return_results(while_op)


@cuda_tile_op
def loop_break(
    operands: Union[
        Tile,
        Token,
        Iterable[Union[Tile, Token]],
    ],
    *,
    loc=None,
    ip=None,
):
    # Normalize operands into an iterable
    if isinstance(operands, (Tile, Token)):
        operands = [operands]

    mlir_values = []
    for i in operands:
        if not isinstance(i, (Tile, Token)):
            raise TypeError("All loop break operands must be Tile or Token.")
        mlir_values.append(i.value)
    _cuda_tile.BreakOp(operands_=mlir_values, loc=loc, ip=ip)


@cuda_tile_op
def loop_continue(
    operands: Union[
        Tile,
        Token,
        Iterable[Union[Tile, Token]],
    ],
    *,
    loc=None,
    ip=None,
):
    # Normalize operands into an iterable
    if isinstance(operands, (Tile, Token)):
        operands = [operands]

    mlir_values = []
    for i in operands:
        if not isinstance(i, (Tile, Token)):
            raise TypeError("All loop continue operands must be Tile or Token.")
        mlir_values.append(i.value)
    _cuda_tile.ContinueOp(operands_=mlir_values, loc=loc, ip=ip)


@cuda_tile_op
def for_loop(
    body: Callable,
    lower_bound: int | Tile,
    upper_bound: int | Tile,
    step: int | Tile = 1,
    init_values: Sequence[Union[Tile, Token]] = (),
    el_type=Int32,
    *,
    unsigned: bool = False,
    loc=None,
    ip=None,
) -> Union[
    Tile,
    Token,
    Tuple[Union[Tile, Token], ...],
    None,
]:
    """
    Constructs a for loop with the provided body. The body is a function taking
    as argument the iteration variables and building the operations within the
    body (including continue and break).

    By default, only the induction variable is created. If initializers for
    additional iteration variables are provided in `init_values`, additional
    iteration variables will be passed to the body and returned from the
    operation.

    By default, the induction variable element type is Int32, which can be
    overriden by setting `el_type`.

    By default, signed comparison is used for loop termination. Set `unsigned=True`
    to use unsigned integer comparison.
    """

    index_type = el_type.mlir_type

    def check_scalar(x: int | Tile, name: str) -> Tile:
        nonlocal index_type
        if isinstance(x, int):
            return constant(x, el_type)
        elif isinstance(x, Tile) and x.element_type == index_type and len(x.shape) == 0:
            return x
        else:
            raise TypeError(
                f"For loop {name} must be an integer or a scalar {el_type} tile value, got {x}"
            )

    lower_bound = check_scalar(lower_bound, "lower bound")
    upper_bound = check_scalar(upper_bound, "upper bound")
    step = check_scalar(step, "step")

    iter_arg_types = []
    iter_arg_values = []
    for x in init_values:
        if isinstance(x, Tile):
            iter_arg_types.append(x.tile_type)
            iter_arg_values.append(x.value)
        elif isinstance(x, Token):
            iter_arg_types.append(x.value.type)
            iter_arg_values.append(x.value)
        else:
            raise TypeError(
                "For loop init values must be Tile or Token values, "
                f"got {type(x).__name__}"
            )
    iter_arg_types = tuple(iter_arg_types)
    _for_op = _cuda_tile.ForOp(
        resultValues=iter_arg_types,
        lowerBound=lower_bound,
        upperBound=upper_bound,
        step=step,
        initValues=iter_arg_values,
        unsignedCmp=unsigned,
        loc=loc,
        ip=ip,
    )
    block_arg_types = list(chain((step.value.type,), iter_arg_types))
    body_block = _ods_ir.Block.create_at_start(_for_op.region, block_arg_types)
    iteration_variables = []
    for arg in body_block.arguments:
        iteration_variables.append(_wrap_result_value(arg))
    with _ods_ir.InsertionPoint(body_block):
        body(*iteration_variables)

    return return_results(_for_op)


def entry(
    sym_name,
    function_type,
    *,
    arch=None,
    arg_attrs=None,
    num_cta=None,
    num_worker_warps=None,
    occupancy=None,
    loc=None,
    ip=None,
) -> Tile:
    optimization_hints = None
    if (arch != None) and (
        (num_cta != None) or (num_worker_warps != None) or (occupancy != None)
    ):
        optimization_hints = OptimizationHintsAttr.getEntryOpHint(
            arch,
            0 if num_cta is None else num_cta,
            0 if num_worker_warps is None else num_worker_warps,
            0 if occupancy is None else occupancy,
            context=_ods_get_default_loc_context(loc),
        )
    elif (num_cta != None) or (num_worker_warps != None) or (occupancy != None):
        # (arch == None) and hint values are specified
        raise ValueError(
            "Expected arch to be specified for OptimizationHint:"
            f" num_cta = {num_cta}, num_worker_warps = {num_worker_warps}, occupancy = {occupancy}"
        )
    return _cuda_tile.EntryOp(
        sym_name=sym_name,
        function_type=function_type,
        arg_attrs=arg_attrs,
        optimization_hints=optimization_hints,
        loc=loc,
        ip=ip,
    )


@cuda_tile_op
def ret(args: Iterable[Tile], *, loc=None, ip=None):
    """Return values from a function."""
    _cuda_tile.ReturnOp(args, loc=loc, ip=ip)


@cuda_tile_op
def make_tensor_view(
    base_ptr,
    el_type,
    shape: List[int | Tile] | None = None,
    strides: List[int | Tile] | None = None,
    *,
    ptr_attr: PtrAttr | None = None,
    loc=None,
    ip=None,
) -> TensorView:
    def tile_to_none(x):
        return None if isinstance(x, Tile) else x

    shape = shape or []
    strides = strides or []

    def valid_dim(dim):
        return (isinstance(dim, int) and dim >= 0) or (
            isinstance(dim, Tile)
            and dim.shape == []
            and isinstance(dim.element_type, _ods_ir.IntegerType)
        )

    if not all(valid_dim(dim) for dim in shape):
        raise ValueError(
            f"Shape must be a list of non-negative ints or scalar integer tile values, got {shape}"
        )

    if not all(valid_dim(dim) for dim in strides):
        raise ValueError(
            f"Strides must be a list of non-negative ints or scalar integer tile values, got {strides}"
        )

    if not (
        isinstance(base_ptr, Tile)
        and base_ptr.shape == []
        and PointerType.isinstance(base_ptr.element_type)
    ):
        raise ValueError(
            f"Base pointer must be a scalar tile of pointer, got {base_ptr}"
        )

    tensor_view_type = make_tensor_view_type(
        el_type,
        list(map(tile_to_none, shape)),
        list(map(tile_to_none, strides)),
        ptr_attr=ptr_attr,
    )
    dynamic_shape = list(filter(lambda x: not isinstance(x, int), shape))
    dynamic_strides = list(filter(lambda x: not isinstance(x, int), strides))
    return return_tensor_view(
        _cuda_tile.MakeTensorViewOp(
            result=tensor_view_type,
            base=base_ptr,
            dynamicShape=dynamic_shape,
            dynamicStrides=dynamic_strides,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def maxf(
    lhs: Tile,
    rhs: Tile,
    *,
    propagate_nan: bool = False,
    flush_to_zero: bool = False,
    loc=None,
    ip=None,
) -> Tile:
    return return_results(
        _cuda_tile.MaxFOp(
            lhs,
            rhs,
            propagate_nan=propagate_nan,
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def maxi(
    lhs: Tile,
    rhs: Tile,
    *,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
) -> Tile:
    return return_results(
        _cuda_tile.MaxIOp(
            lhs,
            rhs,
            get_signedness_attr(signedness),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def max(
    lhs: Tile,
    rhs: Tile,
    *,
    propagate_nan: bool = False,
    flush_to_zero: bool = False,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
) -> Tile:
    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        if propagate_nan or flush_to_zero:
            raise Exception(
                "nan modifier or flush_to_zero modifier can only be used with floating point tiles"
            )
        signedness = Signedness.SIGNED if not signedness else signedness
        return maxi(lhs=lhs, rhs=rhs, signedness=signedness, loc=loc, ip=ip)

    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        return maxf(
            lhs=lhs,
            rhs=rhs,
            propagate_nan=propagate_nan,
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    else:
        raise TypeError("expected integer or float element type")


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def mini(
    lhs: Tile,
    rhs: Tile,
    *,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
) -> Tile:
    return return_results(
        _cuda_tile.MinIOp(
            lhs=lhs,
            rhs=rhs,
            signedness=get_signedness_attr(signedness),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.FloatType)
@check_data_type_binary("rhs", _ods_ir.FloatType)
@check_same_type
def minf(
    lhs: Tile,
    rhs: Tile,
    *,
    propagate_nan: bool = False,
    flush_to_zero: bool = False,
    loc=None,
    ip=None,
) -> Tile:
    return return_results(
        _cuda_tile.MinFOp(
            lhs,
            rhs,
            propagate_nan=propagate_nan,
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def min(
    lhs: Tile,
    rhs: Tile,
    *,
    propagate_nan: bool = False,
    flush_to_zero: bool = False,
    signedness: Signedness = Signedness.SIGNED,
    loc=None,
    ip=None,
) -> Tile:
    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        if propagate_nan or flush_to_zero:
            raise Exception(
                "propagate_nan modifier or flush_to_zero modifier can only be used with floating point tiles"
            )
        signedness = Signedness.SIGNED if not signedness else signedness
        return mini(lhs=lhs, rhs=rhs, signedness=signedness, loc=loc, ip=ip)
    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        return minf(
            lhs=lhs,
            rhs=rhs,
            propagate_nan=propagate_nan,
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    else:
        raise TypeError("expected integer or float element type")


@cuda_tile_op
def optimization_barrier(
    value: Tile, keep_axis_info: bool = False, *, loc=None, ip=None
) -> Tile:
    return value


# Helper function for both reduce and scan operations
def _identity_attr(element_type, identity):
    if isinstance(element_type, _ods_ir.IntegerType):
        return _ods_ir.IntegerAttr.get(element_type, identity)
    if isinstance(element_type, _ods_ir.FloatType):
        return _ods_ir.FloatAttr.get(element_type, identity)
    raise TypeError("Tile operand is not integer or float data type")


def _prepare_aggregate_op(operand, dim, reverse, identities, operation_type):
    """Helper function for reduce and scan operations.
    Prepares common components such as element type handling and attribute creation.

    Args:
        operand: The input tile
        dim: The dimension along which to perform the operation
        identities: Identity values for the operation
        operation_type: "reduce" or "scan" to determine shape transformation

    Returns:
        A tuple of (result_type, dim_attr, reverse_attr, identities_attr, bb_arg_type, el_type)
    """
    el_type = operand.element_type
    attr = _identity_attr(el_type, identities)

    # Create result shape - for reduce, remove the dimension; for scan, keep the same shape
    shape = operand.tile_type.shape
    if operation_type == "reduce":
        result_shape = [d for i, d in enumerate(shape) if i != dim]
    else:  # scan
        result_shape = shape

    result_type = make_tile_type(el_type, result_shape)

    # Create dimension and identities attributes
    i32 = _ods_ir.IntegerType.get_signless(32)
    dim_attr = _ods_ir.IntegerAttr.get(i32, dim)
    reverse_attr = _ods_ir.BoolAttr.get(reverse)
    identities_attr = _ods_ir.ArrayAttr.get([attr])

    # Create block argument type
    bb_arg_ty = _cuda_tile_capi.TileType.get([], el_type)

    return (result_type, dim_attr, reverse_attr, identities_attr, bb_arg_ty, el_type)


def _as_tile_tuple(operands):
    if isinstance(operands, Tile):
        return (operands,)
    if isinstance(operands, (list, tuple)):
        if not operands:
            raise ValueError("Expected at least one Tile operand")
        if not all(isinstance(operand, Tile) for operand in operands):
            raise TypeError("Expected every aggregate operand to be a Tile")
        return tuple(operands)
    raise TypeError("Expected a Tile or a tuple/list of Tiles")


def _as_identity_tuple(identities, num_operands):
    if isinstance(identities, (list, tuple)):
        identity_tuple = tuple(identities)
    else:
        identity_tuple = (identities,)
    if len(identity_tuple) != num_operands:
        raise ValueError(
            "Expected one identity per aggregate operand, got "
            f"{len(identity_tuple)} identities for {num_operands} operands"
        )
    return identity_tuple


def _prepare_aggregate_op_multi(operands, dim, reverse, identities, operation_type):
    tile_operands = _as_tile_tuple(operands)
    identity_tuple = _as_identity_tuple(identities, len(tile_operands))
    shape = tile_operands[0].tile_type.shape
    for operand in tile_operands:
        if operand.tile_type.shape != shape:
            raise ValueError("Expected all aggregate operands to have the same shape")

    if operation_type == "reduce":
        result_shape = [d for i, d in enumerate(shape) if i != dim]
    else:
        result_shape = shape

    result_types = []
    identities_attrs = []
    bb_arg_types = []
    scalar_tile_types = []
    for operand, identity in zip(tile_operands, identity_tuple):
        el_type = operand.element_type
        result_types.append(make_tile_type(el_type, result_shape))
        identities_attrs.append(_identity_attr(el_type, identity))
        bb_arg_ty = _cuda_tile_capi.TileType.get([], el_type)
        bb_arg_types.extend([bb_arg_ty, bb_arg_ty])
        scalar_tile_types.extend(
            [make_tile_type(el_type, []), make_tile_type(el_type, [])]
        )

    i32 = _ods_ir.IntegerType.get_signless(32)
    dim_attr = _ods_ir.IntegerAttr.get(i32, dim)
    reverse_attr = _ods_ir.BoolAttr.get(reverse)
    identities_attr = _ods_ir.ArrayAttr.get(identities_attrs)
    operand_values = [operand.value for operand in tile_operands]

    return (
        result_types,
        operand_values,
        dim_attr,
        reverse_attr,
        identities_attr,
        bb_arg_types,
        scalar_tile_types,
        len(tile_operands),
    )


def _build_aggregate_region(
    op, body, bb_arg_types, scalar_tile_types, num_operands, loc, ip
):
    block = op.regions[0].blocks.append(*bb_arg_types)
    with _ods_ir.InsertionPoint(block):
        block_args = [
            Tile(block.arguments[i], scalar_tile_types[i])
            for i in range(len(scalar_tile_types))
        ]
        args = block_args
        if num_operands > 1:
            args = block_args[0::2] + block_args[1::2]
        values = body(*args)
        if isinstance(values, Tile):
            values = (values,)
        elif isinstance(values, list):
            values = tuple(values)
        if not isinstance(values, tuple) or len(values) != num_operands:
            raise TypeError(
                "Aggregate body must return one Tile per operand, got " f"{values}"
            )
        if not all(isinstance(value, Tile) for value in values):
            raise TypeError("Aggregate body must return only Tile values")
        _cuda_tile.YieldOp(operands_=[value.value for value in values], loc=loc, ip=ip)


@cuda_tile_op
def reduce(
    operand: Tile,
    dim,
    identities,
    reduce_body: Callable,
    *,
    loc=None,
    ip=None,
):
    # Prepare common components
    result_type, dim_attr, _, identities_attr, bb_arg_ty, el_type = (
        _prepare_aggregate_op(operand, dim, False, identities, "reduce")
    )

    # Create reduce operation
    reduce_op = _cuda_tile.ReduceOp(
        [result_type], [operand.value], dim_attr, identities_attr, loc=loc, ip=ip
    )

    # Set up the block and body
    block = reduce_op.regions[0].blocks.append(bb_arg_ty, bb_arg_ty)
    with _ods_ir.InsertionPoint(block):
        values = reduce_body(
            Tile(block.arguments[0], make_tile_type(el_type, [])),
            Tile(block.arguments[1], make_tile_type(el_type, [])),
        )
        if isinstance(values, Tile) is False:
            error = f"Expected a tile type but it received {values}"
            raise Exception(error)
        _cuda_tile.YieldOp(operands_=[values.value], loc=loc, ip=ip)

    return return_results(reduce_op)


@cuda_tile_op
def reduce_multi(
    operands, dim, identities, reduce_body: Callable, *, loc=None, ip=None
):
    (
        result_types,
        operand_values,
        dim_attr,
        _,
        identities_attr,
        bb_arg_types,
        scalar_tile_types,
        num_operands,
    ) = _prepare_aggregate_op_multi(operands, dim, False, identities, "reduce")

    reduce_op = _cuda_tile.ReduceOp(
        result_types, operand_values, dim_attr, identities_attr, loc=loc, ip=ip
    )
    _build_aggregate_region(
        reduce_op,
        reduce_body,
        bb_arg_types,
        scalar_tile_types,
        num_operands,
        loc,
        ip,
    )
    return return_results(reduce_op)


@cuda_tile_op
def scan(
    operand: Tile, dim, reverse, identities, scan_body: Callable, *, loc=None, ip=None
):
    # Prepare common components
    result_type, dim_attr, reverse_attr, identities_attr, bb_arg_ty, el_type = (
        _prepare_aggregate_op(operand, dim, reverse, identities, "scan")
    )

    # Create scan operation
    scan_op = _cuda_tile.ScanOp(
        [result_type],
        [operand.value],
        dim_attr,
        reverse_attr,
        identities=identities_attr,
        loc=loc,
        ip=ip,
    )

    # Set up the block and body
    block = scan_op.regions[0].blocks.append(bb_arg_ty, bb_arg_ty)
    with _ods_ir.InsertionPoint(block):
        values = scan_body(
            Tile(block.arguments[0], make_tile_type(el_type, [])),
            Tile(block.arguments[1], make_tile_type(el_type, [])),
        )
        if isinstance(values, Tile) is False:
            error = f"Expected a tile type but it received {values}"
            raise Exception(error)
        _cuda_tile.YieldOp(operands_=[values.value], loc=loc, ip=ip)

    return return_results(scan_op)


@cuda_tile_op
def scan_multi(
    operands, dim, reverse, identities, scan_body: Callable, *, loc=None, ip=None
):
    (
        result_types,
        operand_values,
        dim_attr,
        reverse_attr,
        identities_attr,
        bb_arg_types,
        scalar_tile_types,
        num_operands,
    ) = _prepare_aggregate_op_multi(operands, dim, reverse, identities, "scan")

    scan_op = _cuda_tile.ScanOp(
        result_types,
        operand_values,
        dim_attr,
        reverse_attr,
        identities=identities_attr,
        loc=loc,
        ip=ip,
    )
    _build_aggregate_region(
        scan_op, scan_body, bb_arg_types, scalar_tile_types, num_operands, loc, ip
    )
    return return_results(scan_op)


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def sin(source: Tile, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.SinOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def sinh(source: Tile, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.SinHOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def shli(lhs, rhs, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.ShLIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def shri(
    lhs, rhs, *, signedness: Signedness = Signedness.SIGNED, loc=None, ip=None
) -> Tile:
    return return_results(
        _cuda_tile.ShRIOp(
            lhs=lhs, rhs=rhs, signedness=get_signedness_attr(signedness), loc=loc, ip=ip
        )
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def tan(source: Tile, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.TanOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def tanh(
    source: Tile,
    *,
    rounding_mode: RoundingMode = RoundingMode.FULL,
    loc=None,
    ip=None,
) -> Tile:
    """Compute the hyperbolic tangent of source element-wise."""

    if rounding_mode not in [RoundingMode.APPROX, RoundingMode.FULL]:
        raise ValueError(
            f"Invalid rounding mode for tanh: {rounding_mode}, expected APPROX or FULL"
        )
    if rounding_mode == RoundingMode.APPROX and not isinstance(
        source.element_type, _ods_ir.F32Type
    ):
        raise ValueError(
            f"approx rounding mode only supported for f32 data type, but got: {source.element_type}"
        )

    return return_results(
        _cuda_tile.TanHOp(
            source=source,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def cmpf(
    comparison_predicate: ComparisonPredicates,
    comparison_ordering: ComparisonOrdering,
    lhs: Tile,
    rhs: Tile,
    *,
    loc=None,
    ip=None,
) -> Tile:
    """Float comparison operation."""

    return return_results(
        _cuda_tile.CmpFOp(
            comparison_predicate=get_comparison_predicate_attr(comparison_predicate),
            comparison_ordering=get_comparison_ordering_attr(comparison_ordering),
            lhs=lhs,
            rhs=rhs,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def cmpi(
    comparison_predicate: ComparisonPredicates,
    lhs: Tile,
    rhs: Tile,
    signedness: Signedness,
    *,
    loc=None,
    ip=None,
) -> Tile:
    """Integer comparison operation."""

    return return_results(
        _cuda_tile.CmpIOp(
            comparison_predicate=get_comparison_predicate_attr(comparison_predicate),
            lhs=lhs,
            rhs=rhs,
            signedness=get_signedness_attr(signedness),
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def cmp(
    comparison_predicate: ComparisonPredicates,
    lhs,
    rhs,
    signedness: Signedness = Signedness.SIGNED,
    comparison_ordering: ComparisonOrdering = ComparisonOrdering.ORDERED,
    *,
    loc=None,
    ip=None,
) -> Tile:
    """Performs element-wise comparison of two tiles."""

    if not isinstance(comparison_predicate, ComparisonPredicates):
        raise ValueError(
            f"Invalid cuda_tile.cmpf 'comparison_predicate' argument: {comparison_predicate}"
        )

    if lhs.tile_type.element_type != rhs.tile_type.element_type:
        raise ValueError("expected matching lhs/rhs tile types")
    if lhs.tile_type.shape != rhs.tile_type.shape:
        raise ValueError("expected matching lhs/rhs tile shapes")

    if isinstance(lhs.element_type, _ods_ir.IntegerType):
        return cmpi(
            comparison_predicate, lhs, rhs, signedness=signedness, loc=loc, ip=ip
        )
    elif isinstance(lhs.element_type, _ods_ir.FloatType):
        return cmpf(
            comparison_predicate,
            comparison_ordering,
            lhs,
            rhs,
            loc=loc,
            ip=ip,
        )
    else:
        raise TypeError("expected integer or float element type")


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def floordivi(lhs, rhs, *, loc=None, ip=None) -> Tile:
    """Signed integer floor division operation."""
    return return_results(
        _cuda_tile.DivIOp(
            lhs=lhs,
            rhs=rhs,
            signedness=get_signedness_attr(Signedness.SIGNED),
            loc=loc,
            ip=ip,
            rounding=get_rounding_mode_attr(RoundingMode.NEGATIVE_INF),
        )
    )


def _flatten_constants(value):
    """
    Helper function for cuda_tile.constant and cuda_tile.global that
    flattens values and determines the shape.
    """
    shape = []
    flattened_values = []
    if isinstance(value, list):
        # Compute the shape of the constant.
        def compute_shape(val):
            assert isinstance(val, list), "expected list"
            if len(val) == 0:
                shape.append(0)
            else:
                shape.append(len(val))
                if isinstance(val[0], list):
                    assert all(
                        isinstance(v, list) for v in val
                    ), "inconsistent nesting level"
                    compute_shape(val[0])
                else:
                    assert not any(
                        isinstance(v, list) for v in val
                    ), "inconsistent nesting level"

        compute_shape(value)

        # Flatten the list.
        def flatten(val, depth):
            assert depth < len(shape), "out of bounds depth"
            assert len(val) == shape[depth], "inconsistent nesting level"
            if depth + 1 < len(shape):
                for v in val:
                    flatten(v, depth + 1)
            else:
                flattened_values.extend(val)

        flatten(value, 0)
    else:
        assert isinstance(value, int) or isinstance(
            value, float
        ), "expected int or float"
        flattened_values = [value]
    assert len(flattened_values) > 0, "empty tiles are not allowed"
    return shape, flattened_values


def _validate_integer_constant_range(values, el_type):
    mlir_type = _get_mlir_type(el_type)
    if not isinstance(mlir_type, _ods_ir.IntegerType):
        return

    width = mlir_type.width
    signed = getattr(el_type, "signed", None)
    if signed is None:
        type_name = getattr(el_type, "__name__", "")
        if type_name.startswith("Uint"):
            signed = False
        elif type_name.startswith("Int"):
            signed = True

    if width == 1:
        lower, upper = 0, 1
    elif signed is True or mlir_type.is_signed:
        lower, upper = -(1 << (width - 1)), (1 << width) - 1
    elif signed is False or mlir_type.is_unsigned:
        lower, upper = 0, (1 << width) - 1
    else:
        lower, upper = -(1 << (width - 1)), (1 << width) - 1

    for value in values:
        if isinstance(value, int) and not lower <= value <= upper:
            raise ValueError(
                f"Scalar {value} is out of range for type {mlir_type}"
            )


@cuda_tile_op
@promote_rhs_to_tile
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
@check_same_type
def ceildivi(
    lhs, rhs, *, signedness: Signedness = Signedness.SIGNED, loc=None, ip=None
) -> Tile:
    """Integer ceiling division operation."""
    return return_results(
        _cuda_tile.DivIOp(
            lhs=lhs,
            rhs=rhs,
            signedness=get_signedness_attr(signedness),
            loc=loc,
            ip=ip,
            rounding=get_rounding_mode_attr(RoundingMode.POSITIVE_INF),
        )
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def ceil(source: Tile, *, loc=None, ip=None) -> Tile:
    """Floating point ceiling operation."""
    return return_results(_cuda_tile.CeilOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
def constant(
    value, el_type=None, tile_type: TileType = None, loc=None, ip=None
) -> Tile:
    """
    Helper function that builds a cuda_tile.constant op for the given value,
    which is either a scalar (integer/float) or a Python list. Nested lists
    are supported and are turned into multi-dimensional tile constants. The
    shape of the constant is inferred from the nesting of the Python lists.
    """

    if tile_type is not None and isinstance(tile_type, TileType) is False:
        issue = f'tile_type must be "TileType" type but it is {tile_type}'
        raise Exception(issue)

    shape, flattened_values = _flatten_constants(value)

    selected_el_type = tile_type.element_type if tile_type is not None else el_type
    if tile_type is None:
        # type is optional. Try to infer it from the first input value.
        if el_type is None:
            el_type = _infer_mlir_type_from_python(flattened_values[0])

        selected_el_type = el_type
        tile_type = make_tile_type(el_type, shape)
        assert tile_type != None, "cannot create tile type"
    elif el_type is not None and _get_mlir_type(el_type) == tile_type.element_type:
        selected_el_type = el_type

    _validate_integer_constant_range(flattened_values, selected_el_type)

    constant_op = _ConstantOp(tile_type, flattened_values, loc=loc, ip=ip)

    return return_results(constant_op)


# A counter for global ops to ensure that we generate unique symbols.
_cuda_tile.GlobalOp.counter = 0


@cuda_tile_op
def global_(
    symbol_name,
    value,
    el_type=None,
    tile_type: TileType = None,
    alignment=None,
    symbol_visibility: SymbolVisibility | None = None,
    constant: bool = False,
    loc=None,
    ip=None,
):
    """
    Create a cuda_tile.global in the enclosing cuda_tile.module.
    """

    current_ip = _ods_ir.InsertionPoint.current
    assert current_ip, "current insertion point is unknown"
    current_op = current_ip.block.owner
    while current_op and current_op.name != "cuda_tile.module":
        current_op = current_op.parent
    assert (
        current_op and current_op.name == "cuda_tile.module"
    ), "could not find enclosing module"

    if tile_type is not None and isinstance(tile_type, TileType) is False:
        issue = f'tile_type must be "TileType" type but it is {tile_type}'
        raise Exception(issue)

    shape, flattened_values = _flatten_constants(value)

    if tile_type is None:
        # type is optional. Try to infer it from the first input value.
        if el_type is None:
            el_type = _infer_mlir_type_from_python(flattened_values[0])

        tile_type = make_tile_type(el_type, shape)
        assert tile_type != None, "cannot create tile type"

    # Insert cuda_tile.global op.
    if len(tile_type.shape) != 1:
        raise ValueError(f"type must have rank 1, but found {len(tile_type.shape)}")
    with _ods_ir.InsertionPoint(current_op.regions[0].blocks[0]):
        return _GlobalOp(
            tile_type,
            symbol_name,
            flattened_values,
            alignment=alignment,
            symbol_visibility=symbol_visibility,
            constant=constant,
            loc=loc,
            ip=ip,
        )


@cuda_tile_op
def get_global(global_op, loc=None, ip=None):
    if not isinstance(global_op, _ods_ir.OpView):
        raise TypeError("expected cuda_tile.global op")
    if global_op.name != "cuda_tile.global":
        raise TypeError("expected cuda_tile.global op")

    # Insert cuda_tile.get_global op.
    tile_type = TileType.upcast_type(global_op.value.type)
    ptr_type = PointerType.get(tile_type.element_type)
    ptr_tile_ty = TileType.get([], ptr_type)
    return return_results(
        _cuda_tile.GetGlobalOp(ptr_tile_ty, global_op.sym_name.value, loc=loc, ip=ip)
    )


@cuda_tile_op
def create_and_get_global(
    value,
    el_type=None,
    tile_type: TileType = None,
    alignment=None,
    symbol_visibility: SymbolVisibility | None = None,
    constant: bool = False,
    loc=None,
    ip=None,
):
    """
    Helper function that inserts a new cuda_tile.global in the enclosing module
    and a cuda_tile.get_global at the current insertion point.
    """

    # Generate a unique symbol.
    symbol_name = f"_global_{_cuda_tile.GlobalOp.counter}"
    _cuda_tile.GlobalOp.counter += 1

    # Insert cuda_tile.global op and cuda_tile.get_global op.
    global_op = global_(
        symbol_name,
        value,
        el_type,
        tile_type,
        alignment=alignment,
        symbol_visibility=symbol_visibility,
        constant=constant,
        loc=loc,
        ip=ip,
    )
    return get_global(global_op, loc=loc, ip=ip)


@cuda_tile_op
def get_index_space_shape(
    view: TileView, result_type=Int64, loc=None, ip=None
) -> Tuple[Tile, ...]:
    if not isinstance(view, TileView):
        raise TypeError(f"view must be a TileView")

    result_types = [make_tile_type(result_type, [])] * view.view_index_rank

    return return_results(
        _cuda_tile.GetIndexSpaceShapeOp(result_types, view, loc=loc, ip=ip)
    )


@cuda_tile_op
def get_tensor_shape(
    view: TensorView, result_type=Int64, loc=None, ip=None
) -> Tuple[Tile, ...]:
    if not isinstance(view, TensorView):
        raise TypeError(f"view must be a TensorView")

    result_types = [make_tile_type(result_type, [])] * len(view.shape)

    return return_results(
        _cuda_tile.GetTensorShapeOp(result_types, view, loc=loc, ip=ip)
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def log(source: Tile, loc=None, ip=None) -> Tile:
    # Base-e logarithm of source
    return return_results(_cuda_tile.LogOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def log10(source: Tile, loc=None, ip=None) -> Tile:
    # Base-10 logarithm of source.
    return return_results(_cuda_tile.Log10Op(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def log1p(source: Tile, loc=None, ip=None) -> Tile:
    # Base-e logarithm of one plus source.
    return return_results(_cuda_tile.Log1pOp(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def log2(source: Tile, loc=None, ip=None) -> Tile:
    # Base-2 logarithm of source.
    return return_results(_cuda_tile.Log2Op(source=source, loc=loc, ip=ip))


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def rsqrt(
    source: Tile,
    *,
    flush_to_zero: bool = False,
    loc=None,
    ip=None,
) -> Tile:
    """Compute the approximate reciprocal square root of source."""
    return return_results(
        _cuda_tile.RsqrtOp(
            source=source,
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
@check_data_type_unary("source", _ods_ir.FloatType)
def sqrt(
    source: Tile,
    *,
    rounding_mode: RoundingMode = RoundingMode.NEAREST_EVEN,
    flush_to_zero: bool = False,
    loc=None,
    ip=None,
) -> Tile:
    """Compute the square root of source."""

    if rounding_mode not in [
        RoundingMode.NEAREST_EVEN,
        RoundingMode.ZERO,
        RoundingMode.NEGATIVE_INF,
        RoundingMode.POSITIVE_INF,
        RoundingMode.APPROX,
    ]:
        raise ValueError(
            f"Invalid rounding mode for sqrt: {rounding_mode}, expected one of NEAREST_EVEN, ZERO, NEGATIVE_INF, POSITIVE_INF, APPROX"
        )

    return return_results(
        _cuda_tile.SqrtOp(
            source=source,
            rounding_mode=get_rounding_mode_attr(rounding_mode),
            flush_to_zero=flush_to_zero,
            loc=loc,
            ip=ip,
        )
    )


@cuda_tile_op
def _continue(operands_, *, loc=None, ip=None) -> Tile:
    return _cuda_tile.ContinueOp(operands_=operands_, loc=loc, ip=ip)


@cuda_tile_op
def make_partition_view(
    tensor_view: TensorView,
    tile_shape: List[int],
    *,
    dim_map: List[int] | None = None,
    padding_value: PaddingValue | None = None,
    loc=None,
    ip=None,
) -> PartitionView:
    # Input validation
    if not isinstance(tensor_view, TensorView):
        raise TypeError(f"Expected TensorView, got {type(tensor_view).__name__}")

    partition_view_type = make_partition_view_type(
        tensor_view.tensor_view_type, tile_shape, dim_map, padding_value
    )
    return return_partition_view(
        _cuda_tile.MakePartitionViewOp(partition_view_type, tensor_view, loc=loc, ip=ip)
    )


@cuda_tile_op
def make_strided_view(
    tensor_view: TensorView,
    tile_shape: List[int],
    traversal_strides: List[int],
    *,
    dim_map: List[int] | None = None,
    padding_value: PaddingValue | None = None,
    loc=None,
    ip=None,
) -> StridedView:
    # Input validation
    if not isinstance(tensor_view, TensorView):
        raise TypeError(f"Expected TensorView, got {type(tensor_view).__name__}")

    strided_view_type = make_strided_view_type(
        tensor_view.tensor_view_type,
        tile_shape,
        traversal_strides,
        dim_map,
        padding_value,
    )
    return return_strided_view(
        _cuda_tile.MakeStridedViewOp(strided_view_type, tensor_view, loc=loc, ip=ip)
    )


@cuda_tile_op
def make_gather_scatter_view(
    tensor_view: TensorView,
    tile_shape: List[int],
    *,
    sparse_dim: int,
    padding_value: PaddingValue | None = None,
    loc=None,
    ip=None,
) -> GatherScatterView:
    if not isinstance(tensor_view, TensorView):
        raise TypeError(f"Expected TensorView, got {type(tensor_view).__name__}")

    gather_scatter_view_type = make_gather_scatter_view_type(
        tensor_view.tensor_view_type, tile_shape, sparse_dim, padding_value
    )
    return return_gather_scatter_view(
        _cuda_tile.MakeGatherScatterViewOp(
            gather_scatter_view_type, tensor_view, loc=loc, ip=ip
        )
    )


@cuda_tile_op
@promote_rhs_to_tile
@check_same_type
@check_data_type_binary("lhs", _ods_ir.IntegerType)
@check_data_type_binary("rhs", _ods_ir.IntegerType)
def xori(lhs, rhs, *, loc=None, ip=None) -> Tile:
    return return_results(_cuda_tile.XOrIOp(lhs, rhs, loc=loc, ip=ip))


@cuda_tile_op
def pack(source: Tile, *, loc=None, ip=None) -> Tile:
    """Pack a numeric tile into an i8 byte array, preserving bit values."""
    if len(source.shape) != 1:
        raise TypeError(f"pack expects rank-1 tile, got rank-{len(source.shape)}")

    elem_bits = _get_element_bit_width(source.element_type)
    src_bits = elem_bits * source.num_elements

    if src_bits % 8 != 0:
        raise TypeError(
            f"pack expects source tile to be byte-aligned, got {src_bits} bits"
        )

    num_bytes = src_bits // 8
    result_type = TileType.get([num_bytes], _ods_ir.IntegerType.get_signless(8))

    return return_results(
        _cuda_tile.PackOp(result=result_type, source=source, loc=loc, ip=ip)
    )


@cuda_tile_op
def unpack(el_type, source: Tile, *, loc=None, ip=None) -> Tile:
    """Unpack an i8 byte array into a numeric tile, preserving bit values."""
    el_type = _get_mlir_type(el_type)

    if len(source.shape) != 1:
        raise TypeError(f"unpack expects rank-1 tile, got rank-{len(source.shape)}")

    src_elem = source.element_type
    if not (isinstance(src_elem, _ods_ir.IntegerType) and src_elem.width == 8):
        raise TypeError(f"unpack source must be i8 tile, got {src_elem}")

    num_bytes = source.shape[0]
    src_bits = num_bytes * 8
    elem_bits = _get_element_bit_width(el_type)

    if src_bits % elem_bits != 0:
        raise TypeError(
            f"unpack: {num_bytes} bytes not divisible by {elem_bits}-bit elements"
        )

    num_elements = src_bits // elem_bits
    res_bits = num_elements * elem_bits

    if res_bits % 8 != 0:
        raise TypeError(
            f"unpack expects result tile to be byte-aligned, got {res_bits} bits"
        )

    result_type = TileType.get([num_elements], el_type)

    return return_results(
        _cuda_tile.UnpackOp(result=result_type, source=source, loc=loc, ip=ip)
    )


# =============================================================================
# Classes
# =============================================================================


@_ods_cext.register_operation(_Dialect, replace=True)
class ModuleOp(_cuda_tile.ModuleOp):
    """Specialization for the module op class."""

    def __init__(self, sym_name, *, loc=None, ip=None):
        super().__init__(sym_name, loc=loc, ip=ip)
        body = self.regions[0].blocks.append()

    @property
    def body(self):
        return self.regions[0].blocks[0]


# =============================================================================
# Generator
# =============================================================================


class EntryContext:
    def __init__(self, kernel_name, loc, arg_types):
        self.arg_types = arg_types
        func_type = _ods_ir.TypeAttr.get(_ods_ir.FunctionType.get(arg_types, []))
        self.entry = entry(kernel_name, func_type, loc=loc)
        self.entry.sym_visibility = _ods_ir.StringAttr.get("public")
        self.body_start = self.entry.body.blocks.append(*(func_type).value.inputs)

    def __enter__(self):
        self.ipoint_op = _ods_ir.InsertionPoint(self.body_start)
        self.ipoint_op.__enter__()
        args = self.entry.regions[0].blocks[0].arguments
        tile_args = []
        for arg in args:
            tile_args.append(Tile(arg, self.arg_types[0]))
        return tile_args

    def __exit__(self, exc_type, exc_value, traceback):
        ret([])
        self.ipoint_op.__exit__(exc_type, exc_value, traceback)


class TileIrGenerator:
    """
    A class to generate CUDA Tile IR python bindings.

    Example usage:
    ```
    module_manager = cuda_tile.TileIrGenerator()

    with module_manager.tile_ir_start(), module_manager.location():
        with module_manager.create_tile_ir_module():
            cuda_tile.entry ...


    # Optionally print the generated IR
    module_manager.print_ir(False)
    ```
    """

    def __init__(self):
        """
        Initializes the TileIrGenerator instance.
        """
        pass

    def tile_ir_start(self):
        """
        Starts the CUDA Tile IR context.
        """
        return _ods_ir.Context()

    def create_tile_ir_module(self, module_name="tile_ir_module"):
        """
        Creates a CUDA Tile IR module.
        """
        self.loc = _ods_ir.Location.unknown()
        self.module = ModuleOp(module_name, loc=self.loc)
        return _ods_ir.InsertionPoint(self.module.body)

    def location(self):
        """
        Gets an unknown location for the CUDA Tile IR.
        """
        return _ods_ir.Location.unknown()

    def create_entry(self, kernel_name, arg_types, module_name="module"):
        """
        Creates a kernel entry in the CUDA Tile IR module.

        Args:
            kernel_name (str): The name of the kernel entry.
            arg_types (list): The argument types for the kernel entry.
            module_name (str): The name of the module. Defaults to "module".

        Returns:
            EntryContext: The context for the kernel entry.
        """
        entry_context = EntryContext(kernel_name, self.loc, arg_types)
        return entry_context

    def print_ir(self, enable_location=True):
        """
        Prints the CUDA Tile IR module.
        """
        self.module.operation.print(enable_debug_info=enable_location)


@cuda_tile_op
def gdc_launch_dependents_tko(*, input_token=None, loc=None, ip=None) -> Token:
    """Signal dependent kernels can be scheduled (PDL).

    Emits ``gdc_launch_dependents_tko``.  Every CTA must execute this
    (or complete via EXIT) before the dependent kernel is scheduled.
    On SM<90 this is a no-op.
    """
    if input_token is not None and not isinstance(input_token, Token):
        raise TypeError("input_token must be a Token")
    return return_results(
        _cuda_tile.GdcLaunchDependentsTkoOp(token=input_token, loc=loc, ip=ip)
    )


@cuda_tile_op
def gdc_wait_tko(*, input_token=None, loc=None, ip=None) -> Token:
    """Wait for predecessor kernel completion (PDL).

    Emits ``gdc_wait_tko``.  Acquire semantics: stores from the
    predecessor kernel are visible after this completes.
    On SM<90 this is a no-op.
    """
    if input_token is not None and not isinstance(input_token, Token):
        raise TypeError("input_token must be a Token")
    return return_results(_cuda_tile.GdcWaitTkoOp(token=input_token, loc=loc, ip=ip))


def _ensure_func_sequence(values, field_name):
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    raise TypeError(f"{field_name} must be a list or tuple")


_TILE_LIBRARIES_ATTR = "cuda_tile.tile_libraries"


def _get_tile_libraries_attr(op):
    """Return the tile-library paths registered on a CudaTile module."""
    if _TILE_LIBRARIES_ATTR not in op.attributes:
        return []
    attr = _ods_ir.ArrayAttr(op.attributes[_TILE_LIBRARIES_ATTR])
    return [_ods_ir.StringAttr(item).value for item in attr]


def _walk_operations(op):
    """Yield an operation and all nested operations in preorder."""
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk_operations(child)


def _get_func_mlir_type(type_like):
    if isinstance(type_like, _ods_ir.Type):
        return type_like
    if hasattr(type_like, "mlir_type"):
        return type_like.mlir_type
    raise TypeError(
        f"Expected an MLIR type or wrapper with mlir_type, got {type(type_like).__name__}"
    )


def _normalize_func_attr_dict(attr, field_name):
    if attr is None:
        return _ods_ir.Attribute.parse("{}")
    if isinstance(attr, str):
        attr = _ods_ir.Attribute.parse(attr)
    if not isinstance(attr, _ods_ir.Attribute):
        raise TypeError(
            f"{field_name} entries must be MLIR attributes or parseable strings, "
            f"got {type(attr).__name__}"
        )

    attr_asm = str(attr)
    if attr_asm.startswith("{"):
        return attr
    raise TypeError(
        f"{field_name} entries must be dictionary attrs, got {attr_asm}"
    )


def _normalize_func_attr_array(attrs, expected_len, field_name):
    if attrs is None:
        return None
    attrs = _ensure_func_sequence(attrs, field_name)
    if len(attrs) != expected_len:
        raise ValueError(
            f"{field_name} must have {expected_len} entries, got {len(attrs)}"
        )
    return _ods_ir.ArrayAttr.get(
        [_normalize_func_attr_dict(attr, field_name) for attr in attrs]
    )


def _normalize_func_const_param(param):
    if isinstance(param, _ods_ir.Attribute):
        return param
    if isinstance(param, str):
        return _ods_ir.Attribute.parse(param)
    raise TypeError(
        "const_params entries must be MLIR attributes or parseable strings, "
        f"got {type(param).__name__}"
    )


def _normalize_func_const_params(const_params):
    return _ods_ir.ArrayAttr.get(
        [
            _normalize_func_const_param(param)
            for param in _ensure_func_sequence(const_params, "const_params")
        ]
    )


def _get_enclosing_cuda_tile_module_op(ip):
    if ip is not None:
        current_op = ip.block.owner
    else:
        current_ip = _ods_ir.InsertionPoint.current
        if current_ip is None:
            raise RuntimeError(
                "current insertion point is unknown while resolving the "
                "enclosing cuda_tile.module insertion point"
            )
        current_op = current_ip.block.owner

    while current_op and current_op.name != "cuda_tile.module":
        current_op = current_op.parent
    if current_op is None:
        raise RuntimeError("could not find enclosing cuda_tile.module")
    return current_op


def _get_enclosing_cuda_tile_module_ip(ip):
    module_op = _get_enclosing_cuda_tile_module_op(ip)
    return _ods_ir.InsertionPoint(module_op.regions[0].blocks[0])


@cuda_tile_op
def use_tile_library(path, *, loc=None, ip=None) -> None:
    """Link the enclosing module against a tile library.

    The library's interface is spliced into the program by the compiler, so a
    kernel calling into it must not declare the callee itself. Repeated paths
    are ignored while distinct paths preserve registration order.

    Args:
        path: Path to a ``.tilelib`` file.
    """
    path = _os.fsdecode(path)
    if not path:
        raise ValueError("tile library path must be non-empty")
    module_op = _get_enclosing_cuda_tile_module_op(ip)
    existing = _get_tile_libraries_attr(module_op)
    if path in existing:
        return
    existing.append(path)
    module_op.attributes[_TILE_LIBRARIES_ATTR] = _ods_ir.ArrayAttr.get(
        [_ods_ir.StringAttr.get(library) for library in existing]
    )


def _consume_module_tile_libraries(module):
    """Remove and return tile-library paths recorded on CudaTile modules."""
    root_op = module.operation if hasattr(module, "operation") else module
    tile_libraries = []
    for op in _walk_operations(root_op):
        if op.name != "cuda_tile.module":
            continue
        tile_libraries.extend(_get_tile_libraries_attr(op))
        if _TILE_LIBRARIES_ATTR in op.attributes:
            del op.attributes[_TILE_LIBRARIES_ATTR]
    return tile_libraries


@cuda_tile_op
def extern_func(
    sym_name: str,
    inputs: Sequence,
    results: Sequence,
    *,
    arg_attrs=None,
    res_attrs=None,
    const_params=None,
    loc=None,
    ip=None,
):
    """Create an extern public-preview function declaration."""
    inputs = [
        _get_func_mlir_type(t) for t in _ensure_func_sequence(inputs, "inputs")
    ]
    results = [
        _get_func_mlir_type(t) for t in _ensure_func_sequence(results, "results")
    ]

    op = _cuda_tile.FuncOp(
        sym_name=sym_name,
        is_extern=True,
        function_type=_ods_ir.TypeAttr.get(
            _ods_ir.FunctionType.get(inputs, results)
        ),
        arg_attrs=_normalize_func_attr_array(
            arg_attrs, len(inputs), "arg_attrs"
        ),
        res_attrs=_normalize_func_attr_array(
            res_attrs, len(results), "res_attrs"
        ),
        const_params=_normalize_func_const_params(const_params),
        loc=loc,
        ip=_get_enclosing_cuda_tile_module_ip(ip),
    )
    op.operation.attributes["sym_visibility"] = _ods_ir.StringAttr.get(
        "private", context=op.operation.context
    )
    return op


_EXTERN_SYNCHRONIZATION_BY_NAME = {
    "none": _cuda_tile_enum.Synchronization.None_,
    "warp": _cuda_tile_enum.Synchronization.Warp,
    "warpgroup": _cuda_tile_enum.Synchronization.Warpgroup,
    "cta": _cuda_tile_enum.Synchronization.CTA,
}


def _normalize_extern_const_arg(arg):
    if isinstance(arg, int):
        return _ods_ir.Attribute.parse(f"#cuda_tile.preview$expr_int<{arg}>")
    if isinstance(arg, (list, tuple)):
        values = ", ".join(
            str(_normalize_extern_const_arg(value)) for value in arg
        )
        return _ods_ir.Attribute.parse(
            f"#cuda_tile.preview$symbolic_tuple<[{values}]>"
        )
    if isinstance(arg, str):
        stripped = arg.strip()
        if stripped.startswith("#"):
            return _ods_ir.Attribute.parse(stripped)
        return _ods_ir.Attribute.parse(
            f'#cuda_tile.preview$expr_var<"{stripped}">'
        )
    if isinstance(arg, _ods_ir.Type) or hasattr(arg, "mlir_type"):
        type_asm = str(_get_func_mlir_type(arg))
        return _ods_ir.Attribute.parse(
            f"#cuda_tile.preview$concrete_type<{type_asm}>"
        )
    if isinstance(arg, _ods_ir.Attribute):
        return arg
    raise TypeError(f"Unsupported const_args entry of type {type(arg).__name__}")


def _normalize_extern_const_args(const_args):
    if const_args is None:
        return None
    return _ods_ir.ArrayAttr.get(
        [
            _normalize_extern_const_arg(arg)
            for arg in _ensure_func_sequence(const_args, "const_args")
        ]
    )


def _normalize_extern_const_params(const_params):
    if const_params is None:
        return None
    return _ods_ir.ArrayAttr.get(
        [
            _normalize_func_const_param(param)
            for param in _ensure_func_sequence(const_params, "const_params")
        ]
    )


def _normalize_extern_synchronization(value, field_name):
    if value is None or isinstance(value, _ods_ir.Attribute):
        return value
    if isinstance(value, str):
        normalized = _EXTERN_SYNCHRONIZATION_BY_NAME.get(value.strip().lower())
        if normalized is None:
            raise ValueError(
                f"{field_name} must be one of "
                f"{sorted(_EXTERN_SYNCHRONIZATION_BY_NAME)}, got {value}"
            )
        value = normalized
    if isinstance(value, _cuda_tile_enum.Synchronization):
        return _ods_ir.Attribute.parse(
            f"#cuda_tile.preview$synchronization<{value}>"
        )
    raise TypeError(
        f"{field_name} must be a string, Synchronization, or MLIR attribute, "
        f"got {type(value).__name__}"
    )


def _normalize_extern_arch(arch):
    if arch is None:
        return None
    return _ods_ir.ArrayAttr.get(
        [
            value
            if isinstance(value, _ods_ir.Attribute)
            else _ods_ir.StringAttr.get(value)
            for value in _ensure_func_sequence(arch, "arch")
        ]
    )


@cuda_tile_op
def extern_impl(
    sym_name: str,
    implements: str,
    inputs: Sequence,
    results: Sequence,
    *,
    const_params=None,
    const_args=None,
    arg_attrs=None,
    res_attrs=None,
    synchronization=None,
    arch=None,
    loc=None,
    ip=None,
):
    """Create a public-preview extern implementation op.

    ``synchronization`` accepts ``none``, ``warp``, ``warpgroup``, or ``cta``.

    The pre-compiled code that defines ``sym_name`` travels in a tile library
    rather than alongside this op; see :func:`use_tile_library`.
    """
    module_op = _get_enclosing_cuda_tile_module_op(ip)
    module_ip = _ods_ir.InsertionPoint(module_op.regions[0].blocks[0])
    inputs = [
        _get_func_mlir_type(t) for t in _ensure_func_sequence(inputs, "inputs")
    ]
    results = [
        _get_func_mlir_type(t) for t in _ensure_func_sequence(results, "results")
    ]
    op = _cuda_tile.ExternImplOp(
        sym_name=sym_name,
        implements=implements,
        function_type=_ods_ir.TypeAttr.get(
            _ods_ir.FunctionType.get(inputs, results)
        ),
        const_params=_normalize_extern_const_params(const_params),
        const_args=_normalize_extern_const_args(const_args),
        arg_attrs=_normalize_func_attr_array(arg_attrs, len(inputs), "arg_attrs"),
        res_attrs=_normalize_func_attr_array(res_attrs, len(results), "res_attrs"),
        synchronization=_normalize_extern_synchronization(
            synchronization, "synchronization"
        ),
        arch=_normalize_extern_arch(arch),
        loc=loc,
        ip=module_ip,
    )
    op.operation.attributes["sym_visibility"] = _ods_ir.StringAttr.get(
        "private", context=op.operation.context
    )
    return op


@cuda_tile_op
def call(
    callee: str,
    operands: Sequence[Tile],
    result_types: Sequence[Any] | None = None,
    *,
    const_args=None,
    arg_attrs=None,
    res_attrs=None,
    loc=None,
    ip=None,
):
    """Create a public-preview call op.

    ``result_types`` defaults to the first operand type for unary elementwise
    calls. Calls without operands must provide their result types explicitly.
    """
    operands = _ensure_func_sequence(operands, "operands")
    if result_types is None:
        if not operands:
            raise ValueError(
                "result_types must be provided for calls without operands"
            )
        result_types = [operands[0].type]
    result_types = [
        _get_func_mlir_type(type_)
        for type_ in _ensure_func_sequence(result_types, "result_types")
    ]
    normalized_const_args = _normalize_extern_const_args(const_args)
    if normalized_const_args is None:
        normalized_const_args = _ods_ir.ArrayAttr.get([])
    return return_results(
        _cuda_tile.CallOp(
            results_=result_types,
            callee=callee,
            operands_=operands,
            const_args=normalized_const_args,
            arg_attrs=_normalize_func_attr_array(
                arg_attrs, len(operands), "arg_attrs"
            ),
            res_attrs=_normalize_func_attr_array(
                res_attrs, len(result_types), "res_attrs"
            ),
            loc=loc,
            ip=ip,
        )
    )


