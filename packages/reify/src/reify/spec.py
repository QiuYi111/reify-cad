"""Read-only in-process engineering constraint specification (Spec)."""

from __future__ import annotations

from dataclasses import dataclass

from reify.ast import FreezeSpec, Node
from reify.types import NumericalBudget


@dataclass(frozen=True)
class ClearanceRule:
    """Minimum external separation clearance against an obstacle geometry."""

    obstacle: Node
    min_clearance_mm: float = 0.0
    name: str = "obstacle"


@dataclass(frozen=True)
class HoleMarginRule:
    """Minimum radial wall margin around a cylindrical hole inside a host solid."""

    hole_node_id: str
    host_node_id: str
    min_margin_mm: float
    name: str = "hole_margin"


@dataclass(frozen=True)
class Spec:
    """Immutable in-process verification and solver constraint specification."""

    obstacles: tuple[ClearanceRule, ...] = ()
    freezes: tuple[FreezeSpec, ...] = ()
    hole_margins: tuple[HoleMarginRule, ...] = ()
    budget: NumericalBudget = NumericalBudget()
