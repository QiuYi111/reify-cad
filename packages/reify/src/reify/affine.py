"""First-order float64 Affine Arithmetic for wrapping-free conservative interval bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from reify.ast import (
    Box,
    Cylinder,
    ExtrudeRoundedRect,
    FreeImplicit,
    Intersect,
    Node,
    Scale,
    SmoothUnion,
    Sphere,
    Subtract,
    Transform,
    Union,
    _val_deg_float,
    _val_mm_float,
)
from reify.features import (
    CircularPattern,
    ExtrudeSketch,
    Frustum,
    LinearPattern,
    Offset,
    RevolveSketch,
    Shell,
    TaperedExtrude,
)
from reify.nurbs import BSplineSurfaceShell, LoftSketches, SweepAlongSpline


@dataclass(frozen=True)
class AffineForm:
    """First-order affine form: c0 + sum(ci * eps_i) + err * eps_new, eps in [-1, 1]."""

    center: float
    coeffs: np.ndarray  # shape (3,) for x, y, z noise symbols
    radius: float  # non-negative accumulated nonlinear error

    @classmethod
    def constant(cls, val: float) -> AffineForm:
        return cls(float(val), np.zeros(3, dtype=np.float64), 0.0)

    def bounds(self) -> tuple[float, float]:
        r = float(np.sum(np.abs(self.coeffs))) + self.radius
        return self.center - r, self.center + r

    def __add__(self, other: AffineForm | float) -> AffineForm:
        if isinstance(other, (int, float)):
            return AffineForm(self.center + float(other), self.coeffs, self.radius)
        return AffineForm(
            self.center + other.center,
            self.coeffs + other.coeffs,
            self.radius + other.radius,
        )

    def __radd__(self, other: float) -> AffineForm:
        return self + other

    def __sub__(self, other: AffineForm | float) -> AffineForm:
        if isinstance(other, (int, float)):
            return AffineForm(self.center - float(other), self.coeffs, self.radius)
        return AffineForm(
            self.center - other.center,
            self.coeffs - other.coeffs,
            self.radius + other.radius,
        )

    def __rsub__(self, other: float) -> AffineForm:
        return AffineForm(float(other) - self.center, -self.coeffs, self.radius)

    def __neg__(self) -> AffineForm:
        return AffineForm(-self.center, -self.coeffs, self.radius)

    def __mul__(self, scalar: float) -> AffineForm:
        s = float(scalar)
        return AffineForm(self.center * s, self.coeffs * s, self.radius * abs(s))

    def __rmul__(self, scalar: float) -> AffineForm:
        return self * scalar

    def abs(self) -> AffineForm:
        lo, hi = self.bounds()
        if lo >= 0.0:
            return self
        if hi <= 0.0:
            return -self
        # Chebyshev linear approximation for |t| on [lo, hi] where lo < 0 < hi
        slope = (hi + lo) / (hi - lo)
        # |t| - slope * t has extrema at t=0 (val=0) and endpoints (val = hi*(1-slope) = -2*lo*hi/(hi-lo))
        mid = -lo * hi / (hi - lo)
        return AffineForm(
            center=slope * self.center + mid,
            coeffs=slope * self.coeffs,
            radius=abs(slope) * self.radius + mid,
        )

    def square(self) -> AffineForm:
        lin_c = self.center * self.center
        lin_coeffs = 2.0 * self.center * self.coeffs
        r_total = float(np.sum(np.abs(self.coeffs))) + self.radius
        rem_mid = 0.5 * r_total * r_total
        new_c = lin_c + rem_mid
        new_r = 2.0 * abs(self.center) * self.radius + rem_mid
        return AffineForm(new_c, lin_coeffs, new_r)

    def clamp_min_zero(self) -> AffineForm:
        lo, hi = self.bounds()
        if lo >= 0.0:
            return self
        if hi <= 0.0:
            return AffineForm.constant(0.0)
        return 0.5 * (self + self.abs())

    def clamp_max_zero(self) -> AffineForm:
        lo, hi = self.bounds()
        if hi <= 0.0:
            return self
        if lo >= 0.0:
            return AffineForm.constant(0.0)
        return 0.5 * (self - self.abs())

    def sqrt_nonneg(self) -> AffineForm:
        lo, hi = self.bounds()
        lo = max(0.0, lo)
        hi = max(lo, hi)
        if hi == 0.0:
            return AffineForm.constant(0.0)
        s_lo = math.sqrt(lo)
        s_hi = math.sqrt(hi)
        if abs(hi - lo) < 1e-15:
            return AffineForm.constant(0.5 * (s_lo + s_hi))
        # Concave sqrt: secant slope from (lo, s_lo) to (hi, s_hi)
        slope = (s_hi - s_lo) / (hi - lo)
        # Max gap at t* = 1/(4*slope^2): sqrt(t*) - (s_lo + slope*(t* - lo))
        t_star = 1.0 / (4.0 * slope * slope)
        max_gap = math.sqrt(t_star) - (s_lo + slope * (t_star - lo))
        offset = (s_lo - slope * lo) + 0.5 * max_gap
        return AffineForm(
            center=slope * self.center + offset,
            coeffs=slope * self.coeffs,
            radius=abs(slope) * self.radius + 0.5 * max_gap,
        )

    @staticmethod
    def minimum(a: AffineForm, b: AffineForm) -> AffineForm:
        return 0.5 * (a + b - (a - b).abs())

    @staticmethod
    def maximum(a: AffineForm, b: AffineForm) -> AffineForm:
        return 0.5 * (a + b + (a - b).abs())


def _rot_matrix_np(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cy * cz, -cy * sz, sy],
            [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
            [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
        ],
        dtype=np.float64,
    )


def eval_affine_form(
    node: Node,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    params: dict[str, Any] | None = None,
) -> AffineForm:
    """Conservatively bound node.eval_sdf(x) over [box_min, box_max] as a first-order AffineForm."""
    c = 0.5 * (np.array(box_min, dtype=np.float64) + np.array(box_max, dtype=np.float64))
    h = 0.5 * (np.array(box_max, dtype=np.float64) - np.array(box_min, dtype=np.float64))
    ax = AffineForm(c[0], np.array([h[0], 0.0, 0.0]), 0.0)
    ay = AffineForm(c[1], np.array([0.0, h[1], 0.0]), 0.0)
    az = AffineForm(c[2], np.array([0.0, 0.0, h[2]]), 0.0)
    return _eval_affine_node(node, ax, ay, az, params)


def eval_affine_box(
    node: Node,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    params: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Conservatively bound node.eval_sdf(x) over [box_min, box_max] using affine arithmetic."""
    return eval_affine_form(node, box_min, box_max, params).bounds()


def _eval_affine_node(
    node: Node,
    x: AffineForm,
    y: AffineForm,
    z: AffineForm,
    params: dict[str, Any] | None,
) -> AffineForm:
    if isinstance(node, Box):
        hx = 0.5 * _val_mm_float(node.lx, params)
        hy = 0.5 * _val_mm_float(node.ly, params)
        hz = 0.5 * _val_mm_float(node.lz, params)
        qx = x.abs() - hx
        qy = y.abs() - hy
        qz = z.abs() - hz
        out = (
            qx.clamp_min_zero().square()
            + qy.clamp_min_zero().square()
            + qz.clamp_min_zero().square()
        ).sqrt_nonneg()
        inside = AffineForm.maximum(AffineForm.maximum(qx, qy), qz).clamp_max_zero()
        return out + inside

    if isinstance(node, Cylinder):
        r = _val_mm_float(node.radius, params)
        hz = 0.5 * _val_mm_float(node.height, params)
        radial = (x.square() + y.square()).sqrt_nonneg()
        dx = radial - r
        dy = z.abs() - hz
        out = (dx.clamp_min_zero().square() + dy.clamp_min_zero().square()).sqrt_nonneg()
        inside = AffineForm.maximum(dx, dy).clamp_max_zero()
        return out + inside

    if isinstance(node, Sphere):
        r = _val_mm_float(node.radius, params)
        return (x.square() + y.square() + z.square()).sqrt_nonneg() - r

    if isinstance(node, ExtrudeRoundedRect):
        hx = 0.5 * _val_mm_float(node.lx, params)
        hy = 0.5 * _val_mm_float(node.ly, params)
        hz = 0.5 * _val_mm_float(node.height, params)
        r = _val_mm_float(node.fillet_r, params)
        qx = x.abs() - (hx - r)
        qy = y.abs() - (hy - r)
        d2d = (
            (qx.clamp_min_zero().square() + qy.clamp_min_zero().square()).sqrt_nonneg()
            + AffineForm.maximum(qx, qy).clamp_max_zero()
            - r
        )
        dz = z.abs() - hz
        out = (d2d.clamp_min_zero().square() + dz.clamp_min_zero().square()).sqrt_nonneg()
        inside = AffineForm.maximum(d2d, dz).clamp_max_zero()
        return out + inside

    if isinstance(node, Transform):
        t = np.array([_val_mm_float(c, params) for c in node.translation], dtype=np.float64)
        R = _rot_matrix_np(
            _val_deg_float(node.rotation[0], params),
            _val_deg_float(node.rotation[1], params),
            _val_deg_float(node.rotation[2], params),
        )
        dx = x - t[0]
        dy = y - t[1]
        dz = z - t[2]
        # local = R.T @ (p - t)
        lx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
        ly = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
        lz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
        return _eval_affine_node(node.child, lx, ly, lz, params)

    if isinstance(node, Scale):
        s_min = min(node.sx, node.sy, node.sz)
        return _eval_affine_node(
            node.child, x * (1.0 / node.sx), y * (1.0 / node.sy), z * (1.0 / node.sz), params
        ) * s_min

    if isinstance(node, Union):
        acc = _eval_affine_node(node.children[0], x, y, z, params)
        for ch in node.children[1:]:
            acc = AffineForm.minimum(acc, _eval_affine_node(ch, x, y, z, params))
        return acc

    if isinstance(node, Intersect):
        acc = _eval_affine_node(node.children[0], x, y, z, params)
        for ch in node.children[1:]:
            acc = AffineForm.maximum(acc, _eval_affine_node(ch, x, y, z, params))
        return acc

    if isinstance(node, Subtract):
        fb = _eval_affine_node(node.base, x, y, z, params)
        ft = _eval_affine_node(node.tool, x, y, z, params)
        return AffineForm.maximum(fb, -ft)

    if isinstance(node, ExtrudeSketch):
        hz = 0.5 * _val_mm_float(node.height, params)
        x_lo, x_hi = x.bounds()
        y_lo, y_hi = y.bounds()
        d2_lo, d2_hi = node.sketch.interval_bound_2d((x_lo, y_lo), (x_hi, y_hi), params=params)
        d2_af = AffineForm(0.5 * (d2_lo + d2_hi), np.zeros(3), 0.5 * (d2_hi - d2_lo))
        dz = z.abs() - hz
        out = (d2_af.clamp_min_zero().square() + dz.clamp_min_zero().square()).sqrt_nonneg()
        inside = AffineForm.maximum(d2_af, dz).clamp_max_zero()
        return out + inside

    if isinstance(node, TaperedExtrude):
        h = _val_mm_float(node.height, params)
        hz = 0.5 * h
        theta = math.radians(_val_deg_float(node.draft_angle, params))
        x_lo, x_hi = x.bounds()
        y_lo, y_hi = y.bounds()
        d2_lo, d2_hi = node.sketch.interval_bound_2d((x_lo, y_lo), (x_hi, y_hi), params=params)
        d2_af = AffineForm(0.5 * (d2_lo + d2_hi), np.zeros(3), 0.5 * (d2_hi - d2_lo))
        d_side = d2_af * math.cos(theta) + (z + hz) * math.sin(theta)
        dz = z.abs() - hz
        return AffineForm.maximum(d_side, dz)

    if isinstance(node, Offset):
        r = _val_mm_float(node.radius, params)
        return _eval_affine_node(node.child, x, y, z, params) - r

    if isinstance(node, Shell) and not node.open_top:
        t = _val_mm_float(node.thickness, params)
        fc = _eval_affine_node(node.child, x, y, z, params)
        return AffineForm.maximum(fc, -(fc + t))

    if isinstance(node, LinearPattern):
        sx = _val_mm_float(node.spacing[0], params)
        sy = _val_mm_float(node.spacing[1], params)
        sz = _val_mm_float(node.spacing[2], params)
        acc = _eval_affine_node(node.child, x, y, z, params)
        for k in range(1, node.count):
            acc = AffineForm.minimum(acc, _eval_affine_node(node.child, x - k * sx, y - k * sy, z - k * sz, params))
        return acc

    if isinstance(node, CircularPattern):
        rad = _val_mm_float(node.radius, params)
        a0 = math.radians(_val_deg_float(node.start_angle_deg, params))
        acc: AffineForm | None = None
        for k in range(node.count):
            ang = a0 + (2.0 * math.pi * k / node.count)
            ca, sa = math.cos(ang), math.sin(ang)
            lx = x * ca + y * sa - rad
            ly = -x * sa + y * ca
            cur = _eval_affine_node(node.child, lx, ly, z, params)
            acc = cur if acc is None else AffineForm.minimum(acc, cur)
        assert acc is not None
        return acc

    if isinstance(node, BSplineSurfaceShell):
        cx, cy, cz = x.center, y.center, z.center
        rx = float(np.sum(np.abs(x.coeffs))) + x.radius
        ry = float(np.sum(np.abs(y.coeffs))) + y.radius
        rz = float(np.sum(np.abs(z.coeffs))) + z.radius
        bmin = (cx - rx, cy - ry, cz - rz)
        bmax = (cx + rx, cy + ry, cz + rz)
        d_lb = node.surface.interval_bound_distance(bmin, bmax, params=params)
        t_half = 0.5 * _val_mm_float(node.thickness, params)
        fc = float(node.eval_sdf(torch.tensor([[cx, cy, cz]], dtype=torch.float64), params=params)[0].item())
        half_diag = math.sqrt(rx * rx + ry * ry + rz * rz)
        lo = max(d_lb - t_half, fc - half_diag)
        hi = fc + half_diag
        return AffineForm(0.5 * (lo + hi), np.zeros(3), 0.5 * (hi - lo))

    if isinstance(node, SweepAlongSpline) and not isinstance(node.profile, Sketch2D):
        cx, cy, cz = x.center, y.center, z.center
        rx = float(np.sum(np.abs(x.coeffs))) + x.radius
        ry = float(np.sum(np.abs(y.coeffs))) + y.radius
        rz = float(np.sum(np.abs(z.coeffs))) + z.radius
        bmin = (cx - rx, cy - ry, cz - rz)
        bmax = (cx + rx, cy + ry, cz + rz)
        d_lb = node.path.interval_bound_distance(bmin, bmax, params=params)
        r_tube = _val_mm_float(node.profile, params)
        fc = float(node.eval_sdf(torch.tensor([[cx, cy, cz]], dtype=torch.float64), params=params)[0].item())
        half_diag = math.sqrt(rx * rx + ry * ry + rz * rz)
        lo = max(d_lb - r_tube, fc - half_diag)
        hi = fc + half_diag
        return AffineForm(0.5 * (lo + hi), np.zeros(3), 0.5 * (hi - lo))

    if isinstance(node, SmoothUnion):
        fa = _eval_affine_node(node.a, x, y, z, params)
        fb = _eval_affine_node(node.b, x, y, z, params)
        k = _val_mm_float(node.k, params)
        if k <= 1e-12:
            return AffineForm.minimum(fa, fb)
        diff = (fa - fb).abs()
        h = (AffineForm.constant(k) - diff).clamp_min_zero() * (1.0 / k)
        return AffineForm.minimum(fa, fb) - h.square() * (0.25 * k)

    # General Lipschitz box bound fallback (handles Frustum, Shell, LoftSketches,
    # internal _RigidMatrixNode / _OffsetSolidNode wrappers, and custom Node subclasses)
    cx, cy, cz = x.center, y.center, z.center
    rx = float(np.sum(np.abs(x.coeffs))) + x.radius
    ry = float(np.sum(np.abs(y.coeffs))) + y.radius
    rz = float(np.sum(np.abs(z.coeffs))) + z.radius
    fc = float(node.eval_sdf(torch.tensor([[cx, cy, cz]], dtype=torch.float64), params=params)[0].item())
    lip_r = node.lipschitz_bound(params) * math.sqrt(rx * rx + ry * ry + rz * rz)
    return AffineForm(fc, np.zeros(3), lip_r)
