"""M5: First-class Class-1 parametric engineering features (ExtrudeSketch, TaperedExtrude, Frustum, RevolveSketch, Offset, Shell, Patterns)."""

from __future__ import annotations

import math
from typing import Any

import torch

from reify.ast import (
    Node,
    Param,
    _match_pts_shape,
    _reject_finalized,
    _val_deg_float,
    _val_mm_float,
    _val_mm_tensor,
    _val_rad_tensor,
)
from reify.sketch import Sketch2D
from reify.types import OutOfDomainError
from reify.units import Quantity, deg, mm, require_angle, require_length


class ExtrudeSketch(Node):
    """Z-centered extrusion of a 2D Sketch2D (exact L=1)."""

    def __init__(
        self,
        sketch: Sketch2D,
        height: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.sketch = sketch
        self.height = require_length(height, "height")

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmin2, bmax2 = self.sketch.bounds_2d(params)
        hz = 0.5 * _val_mm_float(self.height, params)
        return (bmin2[0], bmin2[1], -hz), (bmax2[0], bmax2[1], hz)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        hz = 0.5 * _val_mm_tensor(self.height, params, dtype).to(device=p.device)
        if hz.ndim == 1 and p.ndim == 2:
            p = p.unsqueeze(0)
        hz = _match_pts_shape(hz, p)
        d2d = self.sketch.eval_sdf_2d(p[..., :2], params=params, dtype=dtype)
        dz = torch.abs(p[..., 2]) - hz
        if d2d.ndim == 2 and dz.ndim == 1:
            dz = dz.unsqueeze(0).expand_as(d2d)
        elif d2d.ndim == 1 and dz.ndim == 2:
            d2d = d2d.unsqueeze(0).expand_as(dz)
        d = torch.stack([d2d, dz], dim=-1)
        return torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1) + torch.clamp(
            torch.maximum(d2d, dz), max=0.0
        )


class TaperedExtrude(Node):
    """Z-centered drafted extrusion of a 2D Sketch2D with inward draft angle theta >= 0 (exact L=1)."""

    def __init__(
        self,
        sketch: Sketch2D,
        height: Quantity | Param,
        draft_angle: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.sketch = sketch
        self.height = require_length(height, "height")
        self.draft_angle = require_angle(draft_angle, "draft_angle")

    def check_degeneracy(self, params: dict[str, Any] | None = None) -> None:
        h = _val_mm_float(self.height, params)
        ang_deg = _val_deg_float(self.draft_angle, params)
        if ang_deg < 0.0 or ang_deg >= 89.0:
            raise OutOfDomainError(f"TaperedExtrude {self.node_id}: draft_angle={ang_deg} deg out of [0, 89)")
        bmin2, bmax2 = self.sketch.bounds_2d(params)
        half_span = 0.5 * min(bmax2[0] - bmin2[0], bmax2[1] - bmin2[1])
        inset = h * math.tan(math.radians(ang_deg))
        if inset >= half_span - 1e-4:
            raise OutOfDomainError(
                f"TaperedExtrude {self.node_id}: draft inset {inset:.4f} mm collapses profile (half_span={half_span:.4f})"
            )

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmin2, bmax2 = self.sketch.bounds_2d(params)
        hz = 0.5 * _val_mm_float(self.height, params)
        return (bmin2[0], bmin2[1], -hz), (bmax2[0], bmax2[1], hz)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        h = _val_mm_tensor(self.height, params, dtype).to(device=p.device)
        theta = _val_rad_tensor(self.draft_angle, params, dtype).to(device=p.device)
        if (h.ndim == 1 or theta.ndim == 1) and p.ndim == 2:
            p = p.unsqueeze(0)
        h = _match_pts_shape(h, p)
        theta = _match_pts_shape(theta, p)
        hz = 0.5 * h

        d2d = self.sketch.eval_sdf_2d(p[..., :2], params=params, dtype=dtype)
        pz = p[..., 2]
        if d2d.ndim == 2 and pz.ndim == 1:
            pz = pz.unsqueeze(0).expand_as(d2d)
            hz = hz.unsqueeze(0).expand_as(d2d) if hz.ndim == 1 else hz
            theta = theta.unsqueeze(0).expand_as(d2d) if theta.ndim == 1 else theta
        elif d2d.ndim == 1 and pz.ndim == 2:
            d2d = d2d.unsqueeze(0).expand_as(pz)

        # Side normal distance: d2d * cos(theta) + (z + hz) * sin(theta), ||grad|| = 1
        d_side = d2d * torch.cos(theta) + (pz + hz) * torch.sin(theta)
        dz = torch.abs(pz) - hz
        return torch.maximum(d_side, dz)


class Frustum(Node):
    """Z-centered conical frustum (r_bottom at -h/2, r_top at +h/2) with exact SDF (L=1)."""

    is_convex = True

    def __init__(
        self,
        r_bottom: Quantity | Param,
        r_top: Quantity | Param,
        height: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.r_bottom = require_length(r_bottom, "r_bottom")
        self.r_top = require_length(r_top, "r_top")
        self.height = require_length(height, "height")

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        rmax = max(_val_mm_float(self.r_bottom, params), _val_mm_float(self.r_top, params))
        hz = 0.5 * _val_mm_float(self.height, params)
        return (-rmax, -rmax, -hz), (rmax, rmax, hz)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        r1 = _val_mm_tensor(self.r_bottom, params, dtype).to(device=p.device)
        r2 = _val_mm_tensor(self.r_top, params, dtype).to(device=p.device)
        hz = 0.5 * _val_mm_tensor(self.height, params, dtype).to(device=p.device)
        if any(v.ndim == 1 for v in (r1, r2, hz)) and p.ndim == 2:
            p = p.unsqueeze(0)
        r1, r2, hz = (_match_pts_shape(v, p) for v in (r1, r2, hz))

        qr = torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + 1e-30)
        qz = p[..., 2]
        k1r, k1z = r2, hz
        k2r, k2z = r2 - r1, 2.0 * hz

        r_cap = torch.where(qz < 0.0, r1, r2)
        car = qr - torch.minimum(qr, r_cap)
        caz = torch.abs(qz) - hz
        ca_d2 = car * car + caz * caz

        k2_d2 = (k2r * k2r + k2z * k2z).clamp(min=1e-24)
        proj = torch.clamp(((k1r - qr) * k2r + (k1z - qz) * k2z) / k2_d2, 0.0, 1.0)
        cbr = qr - k1r + k2r * proj
        cbz = qz - k1z + k2z * proj
        cb_d2 = cbr * cbr + cbz * cbz

        inside = (cbr < 0.0) & (caz < 0.0)
        sign = torch.where(inside, -1.0, 1.0).to(dtype=dtype)
        return sign * torch.sqrt(torch.minimum(ca_d2, cb_d2) + 1e-30)


class RevolveSketch(Node):
    """Revolution of a 2D (r >= 0, z) Sketch2D around the Z axis (exact L=1)."""

    def __init__(
        self,
        sketch: Sketch2D,
        angle_deg: Quantity | Param = deg(360.0),
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.sketch = sketch
        self.angle_deg = require_angle(angle_deg, "angle_deg")
        bmin2, _ = sketch.bounds_2d(None)
        if bmin2[0] < -1e-6:
            raise OutOfDomainError(f"RevolveSketch {self.node_id}: sketch r_min={bmin2[0]} < 0")

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmin2, bmax2 = self.sketch.bounds_2d(params)
        rmax = max(0.0, bmax2[0])
        return (-rmax, -rmax, bmin2[1]), (rmax, rmax, bmax2[1])

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        alpha = _val_rad_tensor(self.angle_deg, params, dtype).to(device=p.device)
        if alpha.ndim == 1 and p.ndim == 2:
            p = p.unsqueeze(0)
        alpha = _match_pts_shape(alpha, p)

        r = torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + 1e-30)
        rz = torch.stack([r, p[..., 2]], dim=-1)
        d_sec = self.sketch.eval_sdf_2d(rz, params=params, dtype=dtype)
        if d_sec.ndim == 1 and p.ndim == 3:
            d_sec = d_sec.unsqueeze(0).expand(p.shape[0], -1)

        if isinstance(self.angle_deg, Quantity) and self.angle_deg.as_deg() >= 360.0 - 1e-6:
            return d_sec

        # Partial revolution [0, alpha]
        phi = torch.remainder(torch.atan2(p[..., 1], p[..., 0]), 2.0 * math.pi)
        in_sector = (phi <= alpha) | (alpha >= 2.0 * math.pi - 1e-6)

        # Distance to cap 0 (phi=0: plane y=0, x=x)
        d2_0 = self.sketch.eval_sdf_2d(torch.stack([p[..., 0], p[..., 2]], dim=-1), params=params, dtype=dtype)
        if d2_0.ndim == 1 and p.ndim == 3:
            d2_0 = d2_0.unsqueeze(0).expand(p.shape[0], -1)
        dist_cap0 = torch.sqrt(p[..., 1] ** 2 + torch.clamp(d2_0, min=0.0) ** 2 + 1e-30)

        # Distance to cap alpha
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        x_rot = p[..., 0] * ca + p[..., 1] * sa
        y_rot = -p[..., 0] * sa + p[..., 1] * ca
        d2_a = self.sketch.eval_sdf_2d(torch.stack([x_rot, p[..., 2]], dim=-1), params=params, dtype=dtype)
        if d2_a.ndim == 1 and p.ndim == 3:
            d2_a = d2_a.unsqueeze(0).expand(p.shape[0], -1)
        dist_capa = torch.sqrt(y_rot ** 2 + torch.clamp(d2_a, min=0.0) ** 2 + 1e-30)

        dist_caps = torch.minimum(dist_cap0, dist_capa)
        # Inside sector: max(d_sec, -dist_caps); outside sector: dist_caps
        inside_val = torch.maximum(d_sec, -dist_caps)
        return torch.where(in_sector, inside_val, dist_caps)


class Offset(Node):
    """Uniform isotropic offset of a 1-Lipschitz solid (preserves L=1)."""

    def __init__(
        self,
        child: Node,
        radius: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        _reject_finalized(child, "Offset")
        if child.lipschitz_bound() > 1.0 + 1e-9:
            raise OutOfDomainError(f"Offset requires child L=1, got {child.lipschitz_bound()}")
        self.child = child
        self.radius = require_length(radius, "radius")

    def ast_depth(self) -> int:
        return 1 + self.child.ast_depth()

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return self.child.find_node(node_id)

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmin, bmax = self.child.local_bounds(params)
        r = _val_mm_float(self.radius, params)
        pad = max(0.0, r)
        return (bmin[0] - pad, bmin[1] - pad, bmin[2] - pad), (bmax[0] + pad, bmax[1] + pad, bmax[2] + pad)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        r = _val_mm_tensor(self.radius, params, dtype).to(device=p.device)
        f = self.child.eval_sdf(p, params=params, dtype=dtype)
        if r.ndim == 1 and f.ndim == 1:
            f = f.unsqueeze(0)
        r = _match_pts_shape(r, f if f.ndim == 2 else p)
        return f - r


class Shell(Node):
    """Constant-thickness hollow shell (closed or open_top=True removing top +Z face, L=1)."""

    def __init__(
        self,
        child: Node,
        thickness: Quantity | Param,
        open_top: bool = False,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        _reject_finalized(child, "Shell")
        if child.lipschitz_bound() > 1.0 + 1e-9:
            raise OutOfDomainError(f"Shell requires child L=1, got {child.lipschitz_bound()}")
        self.child = child
        self.thickness = require_length(thickness, "thickness")
        self.open_top = bool(open_top)

    def ast_depth(self) -> int:
        return 1 + self.child.ast_depth()

    def check_degeneracy(self, params: dict[str, Any] | None = None) -> None:
        t = _val_mm_float(self.thickness, params)
        if t <= 0.0:
            raise OutOfDomainError(f"Shell {self.node_id}: thickness={t} must be > 0")
        bmin, bmax = self.child.local_bounds(params)
        h = bmax[2] - bmin[2]
        min_h = t if self.open_top else 2.0 * t
        if min_h >= h - 1e-4:
            raise OutOfDomainError(
                f"Shell {self.node_id}: thickness={t:.4f} mm exceeds child height={h:.4f} mm"
            )
        if isinstance(self.child, TaperedExtrude):
            self.child.check_degeneracy(params)
            bmin2, bmax2 = self.child.sketch.bounds_2d(params)
            half_span = 0.5 * min(bmax2[0] - bmin2[0], bmax2[1] - bmin2[1])
            ang_rad = math.radians(_val_deg_float(self.child.draft_angle, params))
            h_c = _val_mm_float(self.child.height, params)
            total_top_inset = h_c * math.tan(ang_rad) + t / math.cos(ang_rad)
            if total_top_inset >= half_span - 1e-4:
                raise OutOfDomainError(
                    f"Shell {self.node_id}: drafted wall breakthrough at top (inset+wall={total_top_inset:.4f} >= half_span={half_span:.4f})"
                )

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return self.child.find_node(node_id)

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self.child.local_bounds(params)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        t = _val_mm_tensor(self.thickness, params, dtype).to(device=p.device)
        f_outer = self.child.eval_sdf(p, params=params, dtype=dtype)
        if not self.open_top:
            if t.ndim == 1 and f_outer.ndim == 1:
                f_outer = f_outer.unsqueeze(0)
            t_b = _match_pts_shape(t, p) if f_outer.ndim == 2 else t
            return torch.maximum(f_outer, -(f_outer + t_b))

        # Open-top shell: inner cavity extends cleanly upward through z_max
        bmin, bmax = self.child.local_bounds(params)
        z_bot, z_top = bmin[2], bmax[2]
        if isinstance(self.child, TaperedExtrude):
            # Evaluate drafted side + bottom cap without top cap so cavity opens cleanly at z_top
            h_c = _val_mm_tensor(self.child.height, params, dtype).to(device=p.device)
            th_c = _val_rad_tensor(self.child.draft_angle, params, dtype).to(device=p.device)
            if (h_c.ndim == 1 or th_c.ndim == 1 or t.ndim == 1) and p.ndim == 2:
                p = p.unsqueeze(0)
                f_outer = self.child.eval_sdf(p, params=params, dtype=dtype)
            h_c = _match_pts_shape(h_c, p)
            th_c = _match_pts_shape(th_c, p)
            hz_c = 0.5 * h_c
            d2d = self.child.sketch.eval_sdf_2d(p[..., :2], params=params, dtype=dtype)
            if d2d.ndim == 1 and p.ndim == 3:
                d2d = d2d.unsqueeze(0).expand(p.shape[0], -1)
            d_side = d2d * torch.cos(th_c) + (p[..., 2] + hz_c) * torch.sin(th_c)
            d_bot = -hz_c - p[..., 2]
            f_inner_open = torch.maximum(d_side, d_bot)
            t_b = _match_pts_shape(t, p) if f_outer.ndim == 2 else t
            return torch.maximum(f_outer, -(f_inner_open + t_b))

        # Constant-section children (Box, Cylinder, ExtrudeSketch, etc.): clamp z to midpoint for inner cavity
        z_mid = 0.5 * (z_bot + z_top)
        z_cap = torch.tensor(z_mid, dtype=dtype, device=p.device)
        if t.ndim == 1 and p.ndim == 2:
            p = p.unsqueeze(0)
            f_outer = self.child.eval_sdf(p, params=params, dtype=dtype)
        p_in = torch.cat([p[..., :2], torch.minimum(p[..., 2], z_cap).unsqueeze(-1)], dim=-1)
        f_inner = self.child.eval_sdf(p_in, params=params, dtype=dtype)
        t_b = _match_pts_shape(t, p) if f_outer.ndim == 2 else t
        return torch.maximum(f_outer, -(f_inner + t_b))


class LinearPattern(Node):
    """Linear array of `count` instances translated by k * spacing (preserves child L)."""

    def __init__(
        self,
        child: Node,
        count: int,
        spacing: tuple[Quantity | Param, Quantity | Param, Quantity | Param],
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        if count < 1:
            raise ValueError("LinearPattern count must be >= 1")
        self.child = child
        self.count = int(count)
        self.spacing = tuple(require_length(s, f"spacing[{i}]") for i, s in enumerate(spacing))
        self.is_convex = (count == 1) and child.is_convex

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self.child.lipschitz_bound(params)

    def ast_depth(self) -> int:
        return 1 + self.child.ast_depth()

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return self.child.find_node(node_id)

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmin, bmax = self.child.local_bounds(params)
        s = [_val_mm_float(c, params) for c in self.spacing]
        n = self.count - 1
        return (
            tuple(min(bmin[i], bmin[i] + n * s[i]) for i in range(3)),  # type: ignore[return-value]
            tuple(max(bmax[i], bmax[i] + n * s[i]) for i in range(3)),  # type: ignore[return-value]
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        sx = _val_mm_tensor(self.spacing[0], params, dtype).to(device=p.device)
        sy = _val_mm_tensor(self.spacing[1], params, dtype).to(device=p.device)
        sz = _val_mm_tensor(self.spacing[2], params, dtype).to(device=p.device)
        if any(v.ndim == 1 for v in (sx, sy, sz)) and p.ndim == 2:
            p = p.unsqueeze(0)
        sx, sy, sz = (_match_pts_shape(v, p) for v in (sx, sy, sz))
        svec = torch.stack([sx, sy, sz], dim=-1)
        vals = [self.child.eval_sdf(p - k * svec, params=params, dtype=dtype) for k in range(self.count)]
        return torch.amin(torch.stack(vals, dim=0), dim=0)


class CircularPattern(Node):
    """Circular array of `count` instances around Z axis at `radius` (preserves child L)."""

    def __init__(
        self,
        child: Node,
        count: int,
        radius: Quantity | Param = mm(0.0),
        start_angle_deg: Quantity | Param = deg(0.0),
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        if count < 1:
            raise ValueError("CircularPattern count must be >= 1")
        self.child = child
        self.count = int(count)
        self.radius = require_length(radius, "radius")
        self.start_angle_deg = require_angle(start_angle_deg, "start_angle_deg")
        self.is_convex = (count == 1) and child.is_convex

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self.child.lipschitz_bound(params)

    def ast_depth(self) -> int:
        return 1 + self.child.ast_depth()

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return self.child.find_node(node_id)

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmin, bmax = self.child.local_bounds(params)
        r = _val_mm_float(self.radius, params)
        rmax = max(abs(bmin[0]), abs(bmax[0])) + max(abs(bmin[1]), abs(bmax[1])) + abs(r)
        return (-rmax, -rmax, bmin[2]), (rmax, rmax, bmax[2])

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        rad = _val_mm_tensor(self.radius, params, dtype).to(device=p.device)
        a0 = _val_rad_tensor(self.start_angle_deg, params, dtype).to(device=p.device)
        if (rad.ndim == 1 or a0.ndim == 1) and p.ndim == 2:
            p = p.unsqueeze(0)
        rad = _match_pts_shape(rad, p)
        a0 = _match_pts_shape(a0, p)

        vals = []
        for k in range(self.count):
            ang = a0 + (2.0 * math.pi * k / self.count)
            ca, sa = torch.cos(ang), torch.sin(ang)
            # Rotate world p by -ang around Z, then shift by -rad along X
            lx = p[..., 0] * ca + p[..., 1] * sa - rad
            ly = -p[..., 0] * sa + p[..., 1] * ca
            lp = torch.stack([lx, ly, p[..., 2]], dim=-1)
            vals.append(self.child.eval_sdf(lp, params=params, dtype=dtype))
        return torch.amin(torch.stack(vals, dim=0), dim=0)
