"""Joint-distribution IID Clopper-Pearson statistical yield and deterministic MMC/LMC checks (M2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import scipy.stats
import torch

from reify.ast import Node, Part
from reify.query import GeometryQuery
from reify.types import FailReason, Status, VerificationReport
from reify.verify import check_clearance


class JointDistribution(Protocol):
    """Explicit joint error distribution supporting strictly IID sampling."""

    names: list[str]

    def sample_iid(self, n_samples: int, seed: int = 0) -> np.ndarray: ...


@dataclass(frozen=True)
class MultivariateNormalDistribution:
    """Explicit joint multivariate normal error distribution (mean, cov)."""

    mean: Sequence[float]
    cov: Sequence[Sequence[float]]
    names: list[str]

    def __post_init__(self) -> None:
        m = np.asarray(self.mean, dtype=np.float64)
        c = np.asarray(self.cov, dtype=np.float64)
        if m.ndim != 1 or c.shape != (len(m), len(m)) or len(self.names) != len(m):
            raise ValueError("Incompatible mean, cov, or names dimensions")

    def sample_iid(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        return rng.multivariate_normal(
            np.asarray(self.mean, dtype=np.float64),
            np.asarray(self.cov, dtype=np.float64),
            size=int(n_samples),
        )


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Explicit empirical joint error distribution via IID bootstrap resampling."""

    samples: np.ndarray
    names: list[str]

    def sample_iid(self, n_samples: int, seed: int = 0) -> np.ndarray:
        arr = np.asarray(self.samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.names):
            raise ValueError("EmpiricalDistribution samples shape must match len(names)")
        rng = np.random.Generator(np.random.PCG64(seed))
        idx = rng.integers(0, arr.shape[0], size=int(n_samples))
        return arr[idx]


def check_yield(
    eval_pass_fn: Callable[[dict[str, float]], bool],
    distribution: JointDistribution,
    n_samples: int = 2000,
    confidence: float = 0.95,
    min_yield: float = 0.95,
    seed: int = 0,
) -> VerificationReport:
    """Compute one-sided Clopper-Pearson yield bounds using strictly IID PCG64 sampling."""
    if not (0.5 < confidence < 1.0):
        raise ValueError("confidence must be in (0.5, 1.0)")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    draws = distribution.sample_iid(n_samples, seed=seed)
    names = distribution.names
    k_pass = 0
    for row in draws:
        sample_dict = {names[j]: float(row[j]) for j in range(len(names))}
        if bool(eval_pass_fn(sample_dict)):
            k_pass += 1

    alpha = 1.0 - confidence
    res = scipy.stats.binomtest(k_pass, n_samples)
    ci_two = res.proportion_ci(confidence_level=1.0 - 2.0 * alpha, method="exact")
    cp_lower = float(ci_two.low) if k_pass > 0 else 0.0
    cp_upper = float(ci_two.high) if k_pass < n_samples else 1.0

    metrics = {
        "k_pass": k_pass,
        "n_samples": int(n_samples),
        "sample_pass_rate": k_pass / float(n_samples),
        "clopper_pearson_lower": cp_lower,
        "clopper_pearson_upper": cp_upper,
        "confidence": confidence,
        "min_yield": min_yield,
        "sampling_mode": "IID_PCG64",
    }
    if cp_lower >= min_yield:
        return VerificationReport(
            status=Status.PASS,
            metrics=metrics,
            message=f"Clopper-Pearson {confidence:.0%} lower bound {cp_lower:.4f} >= {min_yield:.4f}.",
        )
    if cp_upper < min_yield:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.PHYSICS_LIMIT,
            metrics=metrics,
            message=f"Clopper-Pearson {confidence:.0%} upper bound {cp_upper:.4f} < {min_yield:.4f}.",
        )
    return VerificationReport(
        status=Status.UNKNOWN,
        metrics=metrics,
        message=(
            f"Sample budget N={n_samples} insufficient: "
            f"min_yield={min_yield:.4f} inside [{cp_lower:.4f}, {cp_upper:.4f}]."
        ),
    )


class _OffsetSolidNode(Node):
    """Requicha offset solid f_offset(x) = f(x) - offset_mm (positive offset expands solid)."""

    brep_exportable = False

    def __init__(self, child: Node, offset_mm: float) -> None:
        super().__init__(node_id=f"offset_{child.node_id}")
        self.child = child
        self.offset_mm = float(offset_mm)

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self.child.lipschitz_bound(params)

    def ast_depth(self) -> int:
        return 1 + self.child.ast_depth()

    def local_bounds(self, params: dict[str, Any] | None = None):
        bmin, bmax = self.child.local_bounds(params)
        pad = max(self.offset_mm, 0.0)
        return (
            (bmin[0] - pad, bmin[1] - pad, bmin[2] - pad),
            (bmax[0] + pad, bmax[1] + pad, bmax[2] + pad),
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return self.child.eval_sdf(pts, params=params, dtype=dtype) - self.offset_mm


def check_material_condition(
    part_a: Node | Part,
    part_b: Node | Part,
    tol_a_mm: float,
    tol_b_mm: float,
    condition: str = "MMC",
    min_clearance_mm: float = 0.0,
    max_depth: int = 4,
) -> VerificationReport:
    """Deterministic Requicha offset-solid check under MMC (expanded) or LMC (contracted)."""
    if condition not in ("MMC", "LMC"):
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message=f"Unsupported material condition: {condition!r}",
        )
    ra = part_a.root if isinstance(part_a, Part) else part_a
    rb = part_b.root if isinstance(part_b, Part) else part_b
    sign = +1.0 if condition == "MMC" else -1.0
    off_a = _OffsetSolidNode(ra, sign * float(tol_a_mm))
    off_b = _OffsetSolidNode(rb, sign * float(tol_b_mm))
    rep = check_clearance(
        GeometryQuery(off_a),
        GeometryQuery(off_b),
        min_clearance_mm=min_clearance_mm,
        max_depth=max_depth,
        initial_radius=0.5,
    )
    return VerificationReport(
        status=rep.status,
        reason=rep.reason,
        witness_points=rep.witness_points,
        metrics={**rep.metrics, "condition": condition, "tol_a_mm": tol_a_mm, "tol_b_mm": tol_b_mm},
        message=f"[{condition}] {rep.message}",
    )


def check_mating_pin_hole_condition(
    pin_dia_mm: float,
    pin_tol_mm: float,
    hole_dia_mm: float,
    hole_tol_mm: float,
    condition: str = "MMC",
    min_clearance_mm: float = 0.0,
) -> VerificationReport:
    """Deterministic GD&T MMC/LMC mating clearance for cylindrical pin-in-hole assembly."""
    if condition not in ("MMC", "LMC"):
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message=f"Unsupported material condition: {condition!r}",
        )
    if condition == "MMC":
        # Worst-case tightest fit: largest pin inside smallest hole
        eff_pin = pin_dia_mm + abs(pin_tol_mm)
        eff_hole = hole_dia_mm - abs(hole_tol_mm)
    else:
        # LMC: smallest pin inside largest hole
        eff_pin = pin_dia_mm - abs(pin_tol_mm)
        eff_hole = hole_dia_mm + abs(hole_tol_mm)

    radial_clearance = 0.5 * (eff_hole - eff_pin)
    metrics = {
        "condition": condition,
        "effective_pin_dia_mm": eff_pin,
        "effective_hole_dia_mm": eff_hole,
        "radial_clearance_mm": radial_clearance,
        "min_clearance_mm": min_clearance_mm,
    }
    if radial_clearance >= min_clearance_mm - 1e-12:
        return VerificationReport(
            status=Status.PASS,
            metrics=metrics,
            message=f"[{condition}] radial clearance {radial_clearance:.4f} mm >= {min_clearance_mm:.4f} mm.",
        )
    return VerificationReport(
        status=Status.FAIL,
        reason=FailReason.CLEARANCE_VIOLATION,
        metrics=metrics,
        message=f"[{condition}] radial clearance {radial_clearance:.4f} mm < {min_clearance_mm:.4f} mm.",
    )
