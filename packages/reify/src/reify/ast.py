"""Declarative parametric AST with explicit units, differentiable Anchors, and Freeze specs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch

from reify.types import OutOfDomainError, Status
from reify.units import Quantity, deg, mm, require_angle, require_length

_NODE_COUNTER = 0


def _reject_finalized(obj: Any, target: str) -> None:
    if type(obj).__name__ == "BRepFinalized":
        raise TypeError(
            f"Cannot pass irreversible BRepFinalized into {target}; "
            "it is a terminal BREP_ESCAPED object isolated from the Class 1 analytic AST."
        )


def _next_node_id(prefix: str) -> str:
    global _NODE_COUNTER
    _NODE_COUNTER += 1
    return f"{prefix}_{_NODE_COUNTER}"


class ParamKind(str, Enum):
    SMOOTH = "SMOOTH"
    PIECEWISE = "PIECEWISE"
    DISCRETE = "DISCRETE"


@dataclass
class Param:
    """Typed design parameter with explicit physical unit and bounds."""

    name: str
    value: float
    low: float
    high: float
    unit: str = "mm"
    kind: ParamKind = ParamKind.SMOOTH

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Param {self.name}: low ({self.low}) > high ({self.high})")
        if self.unit not in ("mm", "deg", "rad"):
            raise ValueError(f"Param {self.name}: unsupported unit {self.unit!r}")

    def to_tensor(
        self,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        if params is not None and self.name in params:
            v = params[self.name]
            if isinstance(v, torch.Tensor):
                return v.to(dtype=dtype)
            return torch.tensor(float(v), dtype=dtype)
        return torch.tensor(float(self.value), dtype=dtype)

    def to_float(self, params: dict[str, Any] | None = None) -> float:
        if params is not None and self.name in params:
            v = params[self.name]
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)
        return float(self.value)

    def to_rad_tensor(
        self,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        t = self.to_tensor(params, dtype=dtype)
        if self.unit == "deg":
            return t * (math.pi / 180.0)
        if self.unit == "rad":
            return t
        raise ValueError(f"Param {self.name} has unit {self.unit!r}, expected angle")

    def to_deg_float(self, params: dict[str, Any] | None = None) -> float:
        v = self.to_float(params)
        if self.unit == "deg":
            return v
        if self.unit == "rad":
            return math.degrees(v)
        raise ValueError(f"Param {self.name} has unit {self.unit!r}, expected angle")


def _val_mm_tensor(
    val: Quantity | Param,
    params: dict[str, Any] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(val, Param):
        return val.to_tensor(params, dtype=dtype)
    return torch.tensor(val.as_mm(), dtype=dtype)


def _val_mm_float(val: Quantity | Param, params: dict[str, Any] | None) -> float:
    if isinstance(val, Param):
        return val.to_float(params)
    return val.as_mm()


def _val_rad_tensor(
    val: Quantity | Param,
    params: dict[str, Any] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(val, Param):
        return val.to_rad_tensor(params, dtype=dtype)
    return torch.tensor(val.as_rad(), dtype=dtype)


def _val_deg_float(val: Quantity | Param, params: dict[str, Any] | None) -> float:
    if isinstance(val, Param):
        return val.to_deg_float(params)
    return val.as_deg()


def _match_pts_shape(val: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Broadcast a 0-D scalar or 1-D (B,) parameter tensor against pts (..., 3)."""
    if val.ndim == 1 and pts.ndim == 3:
        return val[:, None]
    return val


def _euler_xyz_intrinsic_matrix(
    rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor
) -> torch.Tensor:
    """Intrinsic XYZ rotation matrix R = Rx @ Ry @ Rz (matches build123d.Rotation)."""
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    r00 = cy * cz
    r01 = -cy * sz
    r02 = sy

    r10 = cx * sz + sx * sy * cz
    r11 = cx * cz - sx * sy * sz
    r12 = -sx * cy

    r20 = sx * sz - cx * sy * cz
    r21 = sx * cz + cx * sy * sz
    r22 = cx * cy

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


@dataclass
class Anchor:
    """Differentiable local coordinate frame attached to an AST node."""

    origin: torch.Tensor  # (3,)
    x_axis: torch.Tensor  # (3,)
    y_axis: torch.Tensor  # (3,)
    z_axis: torch.Tensor  # (3,)
    name: str = "anchor"

    def rotation_matrix(self) -> torch.Tensor:
        return torch.stack([self.x_axis, self.y_axis, self.z_axis], dim=-1)

    def to_world(self, local_pts: torch.Tensor) -> torch.Tensor:
        R = self.rotation_matrix().to(dtype=local_pts.dtype, device=local_pts.device)
        o = self.origin.to(dtype=local_pts.dtype, device=local_pts.device)
        return local_pts @ R.transpose(-1, -2) + o

    def from_world(self, world_pts: torch.Tensor) -> torch.Tensor:
        R = self.rotation_matrix().to(dtype=world_pts.dtype, device=world_pts.device)
        o = self.origin.to(dtype=world_pts.dtype, device=world_pts.device)
        return (world_pts - o) @ R


class FreezeScope(str, Enum):
    WORLD_SHELL = "world_shell"
    DATUM_AXIS = "datum_axis"


@dataclass(frozen=True)
class FreezeSpec:
    """Scoped freeze constraint for a named AST feature."""

    node_id: str
    scope: FreezeScope
    tol_mm: float = 1e-3
    shell_delta_mm: float = 0.5


@dataclass(frozen=True)
class RevoluteJoint:
    name: str
    parent_link: str
    child_link: str
    axis_origin_mm: tuple[float, float, float]
    axis_dir: tuple[float, float, float]
    q_min_rad: float
    q_max_rad: float


@dataclass(frozen=True)
class PrismaticJoint:
    name: str
    parent_link: str
    child_link: str
    axis_dir: tuple[float, float, float]
    q_min_mm: float
    q_max_mm: float


@dataclass
class JointGraph:
    """Kinematic joint graph with static closed-loop detection."""

    joints: list[RevoluteJoint | PrismaticJoint]
    root_link: str = "base"

    def has_cycle(self) -> bool:
        adj: dict[str, list[str]] = {}
        nodes: set[str] = {self.root_link}
        for j in self.joints:
            adj.setdefault(j.parent_link, []).append(j.child_link)
            adj.setdefault(j.child_link, []).append(j.parent_link)
            nodes.add(j.parent_link)
            nodes.add(j.child_link)
        visited: set[str] = set()
        for start in nodes:
            if start in visited:
                continue
            stack: list[tuple[str, str | None]] = [(start, None)]
            while stack:
                u, parent = stack.pop()
                if u in visited:
                    return True
                visited.add(u)
                for v in adj.get(u, []):
                    if v != parent:
                        if v in visited:
                            return True
                        stack.append((v, u))
        return False

    def check_domain(self) -> Status:
        if self.has_cycle():
            return Status.OUT_OF_DOMAIN
        return Status.PASS


class Node:
    """Base class for all declarative parametric geometry AST nodes."""

    brep_exportable: bool = True
    is_convex: bool = False
    smoothness_class: str | None = None

    def __init__(self, node_id: str | None = None) -> None:
        self.node_id: str = node_id or _next_node_id(self.__class__.__name__.lower())

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return 1.0

    def ast_depth(self) -> int:
        return 1

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        raise NotImplementedError

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        raise NotImplementedError

    def eval_sdf_with_provenance(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        tie_tol: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, list[list[str]]]:
        vals = self.eval_sdf(pts, params=params, dtype=dtype)
        n = vals.shape[0]
        return vals, [[self.node_id] for _ in range(n)]

    def anchor(
        self,
        name: str = "center",
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Anchor:
        if name != "center":
            raise KeyError(f"Unknown anchor {name!r} on {self.node_id}")
        zero = torch.zeros(3, dtype=dtype)
        eye = torch.eye(3, dtype=dtype)
        return Anchor(zero, eye[:, 0], eye[:, 1], eye[:, 2], name=name)

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return None

    def is_brep_exportable(self) -> bool:
        return self.brep_exportable


class Box(Node):
    """Axis-aligned box centered at local origin with exact SDF (L=1)."""

    is_convex = True

    def __init__(
        self,
        lx: Quantity | Param,
        ly: Quantity | Param,
        lz: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.lx = require_length(lx, "lx")
        self.ly = require_length(ly, "ly")
        self.lz = require_length(lz, "lz")

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        hx = 0.5 * _val_mm_float(self.lx, params)
        hy = 0.5 * _val_mm_float(self.ly, params)
        hz = 0.5 * _val_mm_float(self.lz, params)
        return (-hx, -hy, -hz), (hx, hy, hz)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        lx = _val_mm_tensor(self.lx, params, dtype).to(device=p.device)
        ly = _val_mm_tensor(self.ly, params, dtype).to(device=p.device)
        lz = _val_mm_tensor(self.lz, params, dtype).to(device=p.device)
        if lx.ndim == 1 or ly.ndim == 1 or lz.ndim == 1:
            lx, ly, lz = torch.broadcast_tensors(lx, ly, lz)
            if p.ndim == 2:
                p = p.unsqueeze(0)
            half = 0.5 * torch.stack([lx, ly, lz], dim=-1)[:, None, :]  # (B, 1, 3)
        else:
            half = 0.5 * torch.stack([lx, ly, lz])
        q = torch.abs(p) - half
        out = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(torch.amax(q, dim=-1), max=0.0)
        return out + inside

    def anchor(
        self,
        name: str = "center",
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Anchor:
        hx = 0.5 * _val_mm_tensor(self.lx, params, dtype)
        hy = 0.5 * _val_mm_tensor(self.ly, params, dtype)
        hz = 0.5 * _val_mm_tensor(self.lz, params, dtype)
        z = torch.zeros((), dtype=dtype)
        eye = torch.eye(3, dtype=dtype)
        anchors = {
            "center": torch.stack([z, z, z]),
            "top": torch.stack([z, z, hz]),
            "bottom": torch.stack([z, z, -hz]),
            "front": torch.stack([hx, z, z]),
            "back": torch.stack([-hx, z, z]),
            "right": torch.stack([z, hy, z]),
            "left": torch.stack([z, -hy, z]),
        }
        if name not in anchors:
            raise KeyError(f"Unknown anchor {name!r} on Box {self.node_id}")
        return Anchor(anchors[name], eye[:, 0], eye[:, 1], eye[:, 2], name=name)


class Cylinder(Node):
    """Z-aligned cylinder centered at local origin with exact SDF (L=1)."""

    is_convex = True

    def __init__(
        self,
        radius: Quantity | Param,
        height: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.radius = require_length(radius, "radius")
        self.height = require_length(height, "height")

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        r = _val_mm_float(self.radius, params)
        hz = 0.5 * _val_mm_float(self.height, params)
        return (-r, -r, -hz), (r, r, hz)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        r = _val_mm_tensor(self.radius, params, dtype).to(device=p.device)
        hz = 0.5 * _val_mm_tensor(self.height, params, dtype).to(device=p.device)
        if (r.ndim == 1 or hz.ndim == 1) and p.ndim == 2:
            p = p.unsqueeze(0)
        r = _match_pts_shape(r, p)
        hz = _match_pts_shape(hz, p)
        radial = torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + 1e-30)
        dx = radial - r
        dy = torch.abs(p[..., 2]) - hz
        d = torch.stack([dx, dy], dim=-1)
        out = torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1)
        inside = torch.clamp(torch.maximum(dx, dy), max=0.0)
        return out + inside

    def anchor(
        self,
        name: str = "center",
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Anchor:
        hz = 0.5 * _val_mm_tensor(self.height, params, dtype)
        z = torch.zeros((), dtype=dtype)
        eye = torch.eye(3, dtype=dtype)
        if name == "center":
            o = torch.stack([z, z, z])
        elif name == "top":
            o = torch.stack([z, z, hz])
        elif name == "bottom":
            o = torch.stack([z, z, -hz])
        else:
            raise KeyError(f"Unknown anchor {name!r} on Cylinder {self.node_id}")
        return Anchor(o, eye[:, 0], eye[:, 1], eye[:, 2], name=name)


class Sphere(Node):
    """Sphere centered at local origin with exact SDF (L=1)."""

    is_convex = True

    def __init__(self, radius: Quantity | Param, node_id: str | None = None) -> None:
        super().__init__(node_id)
        self.radius = require_length(radius, "radius")

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        r = _val_mm_float(self.radius, params)
        return (-r, -r, -r), (r, r, r)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        r = _val_mm_tensor(self.radius, params, dtype).to(device=p.device)
        if r.ndim == 1 and p.ndim == 2:
            p = p.unsqueeze(0)
        r = _match_pts_shape(r, p)
        return torch.sqrt(torch.sum(p * p, dim=-1) + 1e-30) - r


class ExtrudeRoundedRect(Node):
    """Z-extruded rounded rectangle centered at local origin with exact SDF (L=1)."""

    is_convex = True

    def __init__(
        self,
        lx: Quantity | Param,
        ly: Quantity | Param,
        height: Quantity | Param,
        fillet_r: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.lx = require_length(lx, "lx")
        self.ly = require_length(ly, "ly")
        self.height = require_length(height, "height")
        self.fillet_r = require_length(fillet_r, "fillet_r")

    def check_degeneracy(self, params: dict[str, Any] | None = None) -> str | None:
        lx = _val_mm_float(self.lx, params)
        ly = _val_mm_float(self.ly, params)
        r = _val_mm_float(self.fillet_r, params)
        half_min = 0.5 * min(lx, ly)
        if r < 0.0 or r > half_min + 1e-12:
            raise OutOfDomainError(
                f"ExtrudeRoundedRect {self.node_id}: fillet_r={r} out of [0, {half_min}]"
            )
        if half_min - r < 1e-3:
            return f"DEGENERATE_FILLET: fillet_r={r:.6f} approaches half_min={half_min:.6f}"
        return None

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        hx = 0.5 * _val_mm_float(self.lx, params)
        hy = 0.5 * _val_mm_float(self.ly, params)
        hz = 0.5 * _val_mm_float(self.height, params)
        return (-hx, -hy, -hz), (hx, hy, hz)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        hx = 0.5 * _val_mm_tensor(self.lx, params, dtype).to(device=p.device)
        hy = 0.5 * _val_mm_tensor(self.ly, params, dtype).to(device=p.device)
        hz = 0.5 * _val_mm_tensor(self.height, params, dtype).to(device=p.device)
        r = _val_mm_tensor(self.fillet_r, params, dtype).to(device=p.device)
        if any(v.ndim == 1 for v in (hx, hy, hz, r)) and p.ndim == 2:
            p = p.unsqueeze(0)
        hx, hy, hz, r = (_match_pts_shape(v, p) for v in (hx, hy, hz, r))

        inner_x = hx - r
        inner_y = hy - r
        qx = torch.abs(p[..., 0]) - inner_x
        qy = torch.abs(p[..., 1]) - inner_y
        q2 = torch.stack([qx, qy], dim=-1)
        d_2d = (
            torch.linalg.norm(torch.clamp(q2, min=0.0), dim=-1)
            + torch.clamp(torch.maximum(qx, qy), max=0.0)
            - r
        )
        dz = torch.abs(p[..., 2]) - hz
        d = torch.stack([d_2d, dz], dim=-1)
        return torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1) + torch.clamp(
            torch.maximum(d_2d, dz), max=0.0
        )


class Transform(Node):
    """Rigid SE(3) isometry transform (L = L_child)."""

    def __init__(
        self,
        child: Node,
        translation: tuple[Quantity | Param, Quantity | Param, Quantity | Param] = (
            mm(0.0),
            mm(0.0),
            mm(0.0),
        ),
        rotation_deg: tuple[Quantity | Param, Quantity | Param, Quantity | Param] = (
            deg(0.0),
            deg(0.0),
            deg(0.0),
        ),
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        _reject_finalized(child, "Transform")
        self.child = child
        self.translation = tuple(require_length(t, f"t[{i}]") for i, t in enumerate(translation))
        self.rotation = tuple(require_angle(a, f"rot[{i}]") for i, a in enumerate(rotation_deg))
        self.is_convex = child.is_convex

    def is_brep_exportable(self) -> bool:
        return self.child.is_brep_exportable()

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self.child.lipschitz_bound(params)

    def ast_depth(self) -> int:
        return 1 + self.child.ast_depth()

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return self.child.find_node(node_id)

    def world_matrix_tensors(
        self, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ts = [_val_mm_tensor(c, params, dtype) for c in self.translation]
        rx = _val_rad_tensor(self.rotation[0], params, dtype)
        ry = _val_rad_tensor(self.rotation[1], params, dtype)
        rz = _val_rad_tensor(self.rotation[2], params, dtype)
        all_t = ts + [rx, ry, rz]
        if any(v.ndim == 1 for v in all_t):
            dev = next(v.device for v in all_t if v.ndim == 1)
            all_b = torch.broadcast_tensors(*[v.to(device=dev) for v in all_t])
            t = torch.stack(list(all_b[:3]), dim=-1)  # (B, 3)
            R = _euler_xyz_intrinsic_matrix(all_b[3], all_b[4], all_b[5])  # (B, 3, 3)
            return R, t
        t = torch.stack(ts)
        R = _euler_xyz_intrinsic_matrix(rx, ry, rz)
        return R, t

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmin, bmax = self.child.local_bounds(params)
        corners = torch.tensor(
            [
                [x, y, z]
                for x in (bmin[0], bmax[0])
                for y in (bmin[1], bmax[1])
                for z in (bmin[2], bmax[2])
            ],
            dtype=torch.float64,
        )
        R, t = self.world_matrix_tensors(params, dtype=torch.float64)
        wc = corners @ R.T + t
        wmin = wc.min(dim=0).values.tolist()
        wmax = wc.max(dim=0).values.tolist()
        return (wmin[0], wmin[1], wmin[2]), (wmax[0], wmax[1], wmax[2])

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        R, t = self.world_matrix_tensors(params, dtype=dtype)
        R = R.to(device=p.device)
        t = t.to(device=p.device)
        if R.ndim == 3:
            if p.ndim == 2:
                p = p.unsqueeze(0)
            local_p = torch.matmul(p - t[:, None, :], R)
        else:
            local_p = (p - t) @ R
        return self.child.eval_sdf(local_p, params=params, dtype=dtype)

    def eval_sdf_with_provenance(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        tie_tol: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, list[list[str]]]:
        p = pts.to(dtype=dtype)
        R, t = self.world_matrix_tensors(params, dtype=dtype)
        R = R.to(device=p.device)
        t = t.to(device=p.device)
        local_p = (p - t) @ R
        return self.child.eval_sdf_with_provenance(
            local_p, params=params, tie_tol=tie_tol, dtype=dtype
        )

    def anchor(
        self,
        name: str = "center",
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Anchor:
        ca = self.child.anchor(name, params=params, dtype=dtype)
        R, t = self.world_matrix_tensors(params, dtype=dtype)
        return Anchor(
            origin=R @ ca.origin + t,
            x_axis=R @ ca.x_axis,
            y_axis=R @ ca.y_axis,
            z_axis=R @ ca.z_axis,
            name=name,
        )


class Scale(Node):
    """Non-uniform scale node with explicit Lipschitz tracking (non-BRep-exportable)."""

    brep_exportable = False

    def __init__(
        self,
        child: Node,
        sx: float,
        sy: float,
        sz: float,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
            raise ValueError("Scale factors must be strictly positive")
        self.child = child
        self.sx = float(sx)
        self.sy = float(sy)
        self.sz = float(sz)

    def is_brep_exportable(self) -> bool:
        return False

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self.child.lipschitz_bound(params) / min(self.sx, self.sy, self.sz)

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
        return (
            (bmin[0] * self.sx, bmin[1] * self.sy, bmin[2] * self.sz),
            (bmax[0] * self.sx, bmax[1] * self.sy, bmax[2] * self.sz),
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        s = torch.tensor([self.sx, self.sy, self.sz], dtype=dtype, device=p.device)
        return self.child.eval_sdf(p / s, params=params, dtype=dtype) * min(
            self.sx, self.sy, self.sz
        )


class Union(Node):
    """Hard min CSG union preserving Lipschitz bound max_i L_i."""

    def __init__(self, children: list[Node], node_id: str | None = None) -> None:
        super().__init__(node_id)
        if len(children) < 1:
            raise ValueError("Union requires at least 1 child")
        for ch in children:
            _reject_finalized(ch, "Union")
        self.children = list(children)

    def is_brep_exportable(self) -> bool:
        return all(c.is_brep_exportable() for c in self.children)

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return max(c.lipschitz_bound(params) for c in self.children)

    def ast_depth(self) -> int:
        return 1 + max(c.ast_depth() for c in self.children)

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        for c in self.children:
            found = c.find_node(node_id)
            if found is not None:
                return found
        return None

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmins, bmaxs = zip(*(c.local_bounds(params) for c in self.children))
        return (
            tuple(min(b[i] for b in bmins) for i in range(3)),
            tuple(max(b[i] for b in bmaxs) for i in range(3)),
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        vals = torch.stack(
            [c.eval_sdf(pts, params=params, dtype=dtype) for c in self.children], dim=0
        )
        return torch.amin(vals, dim=0)

    def eval_sdf_with_provenance(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        tie_tol: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, list[list[str]]]:
        res = [
            c.eval_sdf_with_provenance(pts, params=params, tie_tol=tie_tol, dtype=dtype)
            for c in self.children
        ]
        vals = torch.stack([r[0] for r in res], dim=0)
        winner = torch.amin(vals, dim=0)
        n = winner.shape[0]
        branches: list[list[str]] = []
        for i in range(n):
            w = float(winner[i].item())
            active: list[str] = []
            for c_idx, (cv, cb) in enumerate(res):
                if abs(float(cv[i].item()) - w) <= tie_tol:
                    for b in cb[i]:
                        if b not in active:
                            active.append(b)
            branches.append(active)
        return winner, branches


class Intersect(Node):
    """Hard max CSG intersection preserving Lipschitz bound max_i L_i."""

    def __init__(self, children: list[Node], node_id: str | None = None) -> None:
        super().__init__(node_id)
        if len(children) < 1:
            raise ValueError("Intersect requires at least 1 child")
        for ch in children:
            _reject_finalized(ch, "Intersect")
        self.children = list(children)
        self.is_convex = all(c.is_convex for c in self.children)

    def is_brep_exportable(self) -> bool:
        return all(c.is_brep_exportable() for c in self.children)

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return max(c.lipschitz_bound(params) for c in self.children)

    def ast_depth(self) -> int:
        return 1 + max(c.ast_depth() for c in self.children)

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        for c in self.children:
            found = c.find_node(node_id)
            if found is not None:
                return found
        return None

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bmins, bmaxs = zip(*(c.local_bounds(params) for c in self.children))
        return (
            tuple(max(b[i] for b in bmins) for i in range(3)),
            tuple(min(b[i] for b in bmaxs) for i in range(3)),
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        vals = torch.stack(
            [c.eval_sdf(pts, params=params, dtype=dtype) for c in self.children], dim=0
        )
        return torch.amax(vals, dim=0)

    def eval_sdf_with_provenance(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        tie_tol: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, list[list[str]]]:
        res = [
            c.eval_sdf_with_provenance(pts, params=params, tie_tol=tie_tol, dtype=dtype)
            for c in self.children
        ]
        vals = torch.stack([r[0] for r in res], dim=0)
        winner = torch.amax(vals, dim=0)
        n = winner.shape[0]
        branches: list[list[str]] = []
        for i in range(n):
            w = float(winner[i].item())
            active: list[str] = []
            for cv, cb in res:
                if abs(float(cv[i].item()) - w) <= tie_tol:
                    for b in cb[i]:
                        if b not in active:
                            active.append(b)
            branches.append(active)
        return winner, branches


class Subtract(Node):
    """Hard CSG difference max(base, -tool) preserving Lipschitz bound."""

    def __init__(self, base: Node, tool: Node, node_id: str | None = None) -> None:
        super().__init__(node_id)
        _reject_finalized(base, "Subtract")
        _reject_finalized(tool, "Subtract")
        self.base = base
        self.tool = tool

    def is_brep_exportable(self) -> bool:
        return self.base.is_brep_exportable() and self.tool.is_brep_exportable()

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return max(self.base.lipschitz_bound(params), self.tool.lipschitz_bound(params))

    def ast_depth(self) -> int:
        return 1 + max(self.base.ast_depth(), self.tool.ast_depth())

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return self.base.find_node(node_id) or self.tool.find_node(node_id)

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self.base.local_bounds(params)

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        fb = self.base.eval_sdf(pts, params=params, dtype=dtype)
        ft = self.tool.eval_sdf(pts, params=params, dtype=dtype)
        return torch.maximum(fb, -ft)

    def eval_sdf_with_provenance(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        tie_tol: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, list[list[str]]]:
        fb, bb = self.base.eval_sdf_with_provenance(
            pts, params=params, tie_tol=tie_tol, dtype=dtype
        )
        ft, bt = self.tool.eval_sdf_with_provenance(
            pts, params=params, tie_tol=tie_tol, dtype=dtype
        )
        neg_ft = -ft
        winner = torch.maximum(fb, neg_ft)
        branches: list[list[str]] = []
        for i in range(winner.shape[0]):
            w = float(winner[i].item())
            active: list[str] = []
            if abs(float(fb[i].item()) - w) <= tie_tol:
                active.extend(bb[i])
            if abs(float(neg_ft[i].item()) - w) <= tie_tol:
                for b in bt[i]:
                    if b not in active:
                        active.append(b)
            branches.append(active)
        return winner, branches


class SmoothUnion(Node):
    """Polynomial smooth union (C1 quad_smin or C2 cubic_smin), non-BRep-exportable."""

    brep_exportable = False

    def __init__(
        self,
        a: Node,
        b: Node,
        k: Quantity,
        kind: str = "quad_smin",
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        if kind not in ("quad_smin", "cubic_smin"):
            raise ValueError(f"Unsupported SmoothUnion kind: {kind!r}")
        _reject_finalized(a, "SmoothUnion")
        _reject_finalized(b, "SmoothUnion")
        self.a = a
        self.b = b
        self.k = require_length(k, "k")
        self.kind = kind
        self.smoothness_class = "C1" if kind == "quad_smin" else "C2"

    def is_brep_exportable(self) -> bool:
        return False

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return max(self.a.lipschitz_bound(params), self.b.lipschitz_bound(params))

    def ast_depth(self) -> int:
        return 1 + max(self.a.ast_depth(), self.b.ast_depth())

    def find_node(self, node_id: str) -> Node | None:
        if self.node_id == node_id:
            return self
        return self.a.find_node(node_id) or self.b.find_node(node_id)

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        ba_min, ba_max = self.a.local_bounds(params)
        bb_min, bb_max = self.b.local_bounds(params)
        k_val = _val_mm_float(self.k, params)
        return (
            tuple(min(ba_min[i], bb_min[i]) - k_val for i in range(3)),
            tuple(max(ba_max[i], bb_max[i]) + k_val for i in range(3)),
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        fa = self.a.eval_sdf(pts, params=params, dtype=dtype)
        fb = self.b.eval_sdf(pts, params=params, dtype=dtype)
        k = max(_val_mm_float(self.k, params), 1e-12)
        if self.kind == "quad_smin":
            h = torch.clamp(0.5 + 0.5 * (fb - fa) / k, 0.0, 1.0)
            return fb * (1.0 - h) + fa * h - k * h * (1.0 - h)
        h = torch.clamp(k - torch.abs(fa - fb), min=0.0) / k
        return torch.minimum(fa, fb) - (1.0 / 6.0) * k * h * h * h


class FreeImplicit(Node):
    """Arbitrary user-supplied implicit field with explicit Lipschitz & bounds (non-BRep)."""

    brep_exportable = False

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        lipschitz: float,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        if lipschitz < 1.0:
            raise ValueError("lipschitz must be >= 1.0")
        self.fn = fn
        self._lipschitz = float(lipschitz)
        self._bounds = bounds

    def is_brep_exportable(self) -> bool:
        return False

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self._lipschitz

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self._bounds

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return self.fn(pts.to(dtype=dtype)).to(dtype=dtype)


@dataclass
class Part:
    """Top-level parametric part wrapper with freeze and craft feature metadata."""

    root: Node
    name: str = "part"
    freezes: list[FreezeSpec] = field(default_factory=list)
    holes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _reject_finalized(self.root, "Part")

    def freeze(
        self, node_id: str, scope: FreezeScope, tol_mm: float = 1e-3, shell_delta_mm: float = 0.5
    ) -> Part:
        if self.root.find_node(node_id) is None:
            raise KeyError(f"Freeze target node_id={node_id!r} not found in Part {self.name!r}")
        self.freezes.append(FreezeSpec(node_id=node_id, scope=scope, tol_mm=tol_mm, shell_delta_mm=shell_delta_mm))
        return self
