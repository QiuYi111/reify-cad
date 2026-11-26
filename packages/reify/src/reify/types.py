"""Core enums, numerical budget, and structured result types for Reify."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class OutOfDomainError(ValueError):
    """Raised when an input violates an operator's explicit domain guard."""


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


class PointClass(str, Enum):
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    UNCERTAIN = "UNCERTAIN"


class GuaranteeLevel(str, Enum):
    CERTIFIED_INTERVAL = "CERTIFIED_INTERVAL"
    SOLVER_TOLERANCE = "SOLVER_TOLERANCE"


class FailReason(str, Enum):
    PENETRATION = "PENETRATION"
    CLEARANCE_VIOLATION = "CLEARANCE_VIOLATION"
    FREEZE_VIOLATION = "FREEZE_VIOLATION"
    KINEMATIC_COLLISION = "KINEMATIC_COLLISION"
    WALL_THICKNESS = "WALL_THICKNESS"
    TOOL_COLLISION = "TOOL_COLLISION"
    PHYSICS_LIMIT = "PHYSICS_LIMIT"
    TOPOLOGY_INVALID = "TOPOLOGY_INVALID"


class GradientStatus(str, Enum):
    UNIQUE = "UNIQUE"
    MULTI_BRANCH = "MULTI_BRANCH"
    NON_DIFFERENTIABLE = "NON_DIFFERENTIABLE"


class CertificationTier(str, Enum):
    CORE_CERTIFIED = "CORE_CERTIFIED"
    BREP_ESCAPED = "BREP_ESCAPED"


@dataclass(frozen=True)
class NumericalBudget:
    """Scale- and depth-aware floating-point error budget."""

    s_max_mm: float = 1e4
    d_max: int = 64
    eps_abs_f64: float = 1e-9
    eps_rel_f64: float = 1e-12
    eps_abs_f32: float = 1e-5
    eps_rel_f32: float = 1e-6

    def validate_domain(self, scale_mm: float, depth: int) -> None:
        if scale_mm > self.s_max_mm:
            raise OutOfDomainError(
                f"Local geometric scale {scale_mm:.3g} mm exceeds s_max_mm={self.s_max_mm:.3g} mm"
            )
        if depth > self.d_max:
            raise OutOfDomainError(
                f"AST depth {depth} exceeds d_max={self.d_max}"
            )

    def eps_num(self, scale_mm: float, depth: int, dtype: Any) -> float:
        import torch

        self.validate_domain(scale_mm, depth)
        eff_depth = max(int(depth), 1)
        eff_scale = max(float(scale_mm), 0.0)
        if dtype == torch.float64:
            return self.eps_abs_f64 + self.eps_rel_f64 * eff_depth * eff_scale
        if dtype == torch.float32:
            return self.eps_abs_f32 + self.eps_rel_f32 * eff_depth * eff_scale
        raise OutOfDomainError(f"Unsupported numerical dtype: {dtype}")


@dataclass(frozen=True)
class DistanceResult:
    """Structured distance query result separating certified intervals from solver tolerances."""

    value: float
    lower_bound: float
    upper_bound: float
    solver_deflection: float | None
    witness_a: tuple[float, float, float] | None
    witness_b: tuple[float, float, float] | None
    containment: bool
    guarantee_level: GuaranteeLevel
    status: Status
    is_done: bool = True


@dataclass
class SurfaceCover:
    """Static analytic boundary covering patches for certified verification."""

    centers: "torch.Tensor"  # (N, 3) float64
    radii: "torch.Tensor"  # (N,) float64
    normals: "torch.Tensor"  # (N, 3) float64
    node_ids: list[str]


@dataclass(frozen=True)
class ProvenanceResult:
    """Active CSG branch provenance at a query point."""

    active_branches: list[str]
    gradient_status: GradientStatus
    tie_tol: float


@dataclass(frozen=True)
class VerificationReport:
    """Four-state structured diagnostic output from verifiers and exporters."""

    status: Status
    reason: FailReason | None = None
    witness_points: list[tuple[float, float, float]] = field(default_factory=list)
    active_branches: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""
