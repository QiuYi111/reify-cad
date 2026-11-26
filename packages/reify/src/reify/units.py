"""Explicit physical unit wrappers (mm, deg, rad) for Reify AST nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reify.ast import Param


class UnitError(TypeError):
    """Raised when a value has missing or incompatible physical units."""


class Dimension(str, Enum):
    LENGTH = "length"
    ANGLE = "angle"


@dataclass(frozen=True)
class Quantity:
    """Scalar physical quantity with explicit unit."""

    value: float
    unit: str
    dimension: Dimension

    def as_mm(self) -> float:
        if self.dimension != Dimension.LENGTH:
            raise UnitError(f"Expected length unit (mm), got {self.unit!r}")
        if self.unit == "mm":
            return float(self.value)
        raise UnitError(f"Unsupported length unit: {self.unit!r}")

    def as_deg(self) -> float:
        if self.dimension != Dimension.ANGLE:
            raise UnitError(f"Expected angle unit (deg/rad), got {self.unit!r}")
        if self.unit == "deg":
            return float(self.value)
        if self.unit == "rad":
            return math.degrees(float(self.value))
        raise UnitError(f"Unsupported angle unit: {self.unit!r}")

    def as_rad(self) -> float:
        if self.dimension != Dimension.ANGLE:
            raise UnitError(f"Expected angle unit (deg/rad), got {self.unit!r}")
        if self.unit == "rad":
            return float(self.value)
        if self.unit == "deg":
            return math.radians(float(self.value))
        raise UnitError(f"Unsupported angle unit: {self.unit!r}")

    def __neg__(self) -> Quantity:
        return Quantity(-self.value, self.unit, self.dimension)

    def __add__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity) or self.dimension != other.dimension:
            raise UnitError(f"Cannot add {self} and {other}")
        if self.dimension == Dimension.LENGTH:
            return mm(self.as_mm() + other.as_mm())
        return deg(self.as_deg() + other.as_deg())

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity) or self.dimension != other.dimension:
            raise UnitError(f"Cannot subtract {other} from {self}")
        if self.dimension == Dimension.LENGTH:
            return mm(self.as_mm() - other.as_mm())
        return deg(self.as_deg() - other.as_deg())


def mm(val: float | int) -> Quantity:
    """Create a length quantity in millimeters."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise UnitError(f"mm() expects numeric scalar, got {type(val).__name__}")
    return Quantity(float(val), "mm", Dimension.LENGTH)


def deg(val: float | int) -> Quantity:
    """Create an angle quantity in degrees."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise UnitError(f"deg() expects numeric scalar, got {type(val).__name__}")
    return Quantity(float(val), "deg", Dimension.ANGLE)


def rad(val: float | int) -> Quantity:
    """Create an angle quantity in radians."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise UnitError(f"rad() expects numeric scalar, got {type(val).__name__}")
    return Quantity(float(val), "rad", Dimension.ANGLE)


def require_length(val: Any, name: str = "value") -> Quantity | "Param":
    """Validate that val is a length Quantity (mm) or length Param."""
    from reify.ast import Param

    if isinstance(val, Quantity):
        if val.dimension != Dimension.LENGTH:
            raise UnitError(f"{name} requires length unit (mm), got {val.unit!r}")
        return val
    if isinstance(val, Param):
        if val.unit != "mm":
            raise UnitError(f"{name} requires Param with unit='mm', got {val.unit!r}")
        return val
    raise UnitError(
        f"{name} requires explicit length unit via mm(...) or Param(..., unit='mm'), got {val!r}"
    )


def require_angle(val: Any, name: str = "angle") -> Quantity | "Param":
    """Validate that val is an angle Quantity (deg/rad) or angle Param."""
    from reify.ast import Param

    if isinstance(val, Quantity):
        if val.dimension != Dimension.ANGLE:
            raise UnitError(f"{name} requires angle unit (deg/rad), got {val.unit!r}")
        return val
    if isinstance(val, Param):
        if val.unit not in ("deg", "rad"):
            raise UnitError(f"{name} requires Param with unit='deg'|'rad', got {val.unit!r}")
        return val
    raise UnitError(
        f"{name} requires explicit angle unit via deg(...)/rad(...) or angle Param, got {val!r}"
    )
