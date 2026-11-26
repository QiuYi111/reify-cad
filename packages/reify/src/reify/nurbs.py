"""M6: 3D spline sweep (RMF), parametric loft, and differentiable certified NURBS surface shell."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFill import BRepFill
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell, BRepOffsetAPI_ThruSections
from OCP.collections import Array1_double, Array1_gp_Pnt, Array1_int, Array2_gp_Pnt
from OCP.Geom import Geom_BezierCurve, Geom_BSplineSurface, Geom_OffsetSurface
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from reify.ast import Node, Param, _match_pts_shape, _val_mm_float, _val_mm_tensor
from reify.sketch import (
    Sketch2D,
    _bernstein_basis_np,
    _boehm_extraction_matrix,
    _de_casteljau_split_matrix,
    _eval_bezier_np,
    _subdivide_bezier_ctrl,
)
from reify.types import OutOfDomainError
from reify.units import Quantity, mm, require_length

Point3DLike = tuple[Quantity | Param, Quantity | Param, Quantity | Param]


def _require_pt3d(pt: Point3DLike, name: str = "pt") -> Point3DLike:
    if len(pt) != 3:
        raise ValueError(f"{name} must be a 3-tuple (x, y, z)")
    return (
        require_length(pt[0], f"{name}[0]"),
        require_length(pt[1], f"{name}[1]"),
        require_length(pt[2], f"{name}[2]"),
    )


def _pt3d_float(pt: Point3DLike, params: dict[str, Any] | None) -> tuple[float, float, float]:
    return (_val_mm_float(pt[0], params), _val_mm_float(pt[1], params), _val_mm_float(pt[2], params))


def _pt3d_tensor(
    pt: Point3DLike, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    x = _val_mm_tensor(pt[0], params, dtype).to(device=device)
    y = _val_mm_tensor(pt[1], params, dtype).to(device=device)
    z = _val_mm_tensor(pt[2], params, dtype).to(device=device)
    if x.ndim == 1 or y.ndim == 1 or z.ndim == 1:
        x, y, z = torch.broadcast_tensors(x, y, z)
    return torch.stack([x, y, z], dim=-1)


def _stack_pts3d_tensor(
    pts_list: Sequence[Point3DLike],
    params: dict[str, Any] | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensors = [_pt3d_tensor(p, params, dtype, device) for p in pts_list]
    if not any(t.ndim == 2 for t in tensors):
        return torch.stack(tensors, dim=0)
    B = next(t.shape[0] for t in tensors if t.ndim == 2)
    expanded = [t if t.ndim == 2 else t.unsqueeze(0).expand(B, 3) for t in tensors]
    return torch.stack(expanded, dim=1)


def _dist_aabb_3d(
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    smin: tuple[float, float, float],
    smax: tuple[float, float, float],
) -> float:
    dx = max(smin[0] - bmax[0], bmin[0] - smax[0], 0.0)
    dy = max(smin[1] - bmax[1], bmin[1] - smax[1], 0.0)
    dz = max(smin[2] - bmax[2], bmin[2] - smax[2], 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class _Bezier3DDistanceFn(torch.autograd.Function):
    """Exact closest-point distance to multi-span 3D Bézier with Envelope Theorem backward."""

    @staticmethod
    def forward(ctx: Any, pts: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        pts_was_2d = pts.ndim == 2
        ctrl_was_3d = ctrl.ndim == 3

        p_in = pts.detach().to(torch.float64).cpu()
        c_in = ctrl.detach().to(torch.float64).cpu()
        if p_in.ndim == 2:
            p_in = p_in.unsqueeze(0)
        if c_in.ndim == 3:
            c_in = c_in.unsqueeze(0)

        B = max(p_in.shape[0], c_in.shape[0])
        if p_in.shape[0] == 1 and B > 1:
            p_in = p_in.expand(B, -1, -1)
        if c_in.shape[0] == 1 and B > 1:
            c_in = c_in.expand(B, -1, -1, -1)

        p_np = p_in.numpy()
        c_np = c_in.numpy()
        N = p_np.shape[1]
        _, S, K, _ = c_np.shape
        deg_k = K - 1

        M = 17
        t_grid = np.linspace(0.0, 1.0, M, dtype=np.float64)
        b_grid = _bernstein_basis_np(deg_k, t_grid)

        best_dist = np.full((B, N), np.inf, dtype=np.float64)
        best_span = np.zeros((B, N), dtype=np.int64)
        best_t = np.zeros((B, N), dtype=np.float64)
        best_pt = np.zeros((B, N, 3), dtype=np.float64)

        for b_idx in range(B):
            cb = c_np[b_idx]
            pb = p_np[b_idx]
            samples = np.einsum("mk,skd->smd", b_grid, cb)
            diff = pb[:, None, None, :] - samples[None, :, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            flat_d2 = d2.reshape(N, S * M)
            top2 = np.argpartition(flat_d2, kth=min(1, S * M - 1), axis=1)[:, :2]

            cand_dists = np.full((N, 2), np.inf, dtype=np.float64)
            cand_ts = np.zeros((N, 2), dtype=np.float64)
            cand_spans = np.zeros((N, 2), dtype=np.int64)
            cand_pts = np.zeros((N, 2, 3), dtype=np.float64)

            for r in range(top2.shape[1]):
                idx = top2[:, r]
                sp = idx // M
                m_idx = idx % M
                t_cur = t_grid[m_idx].copy()
                ctrl_sel = cb[sp]
                c_cur, dc_cur, d2c_cur = _eval_bezier_np(ctrl_sel, t_cur)
                err_cur = c_cur - pb
                d2_cur = np.sum(err_cur * err_cur, axis=-1)

                for _ in range(6):
                    g = np.sum(err_cur * dc_cur, axis=-1)
                    gp = np.sum(dc_cur * dc_cur, axis=-1) + np.sum(err_cur * d2c_cur, axis=-1)
                    valid = np.abs(gp) > 1e-14
                    step = np.where(valid, g / np.where(valid, gp, 1.0), 0.0)
                    step = np.clip(step, -0.25, 0.25)

                    t_try = np.clip(t_cur - step, 0.0, 1.0)
                    c_try, _, _ = _eval_bezier_np(ctrl_sel, t_try)
                    d2_try = np.sum((c_try - pb) ** 2, axis=-1)
                    worse = d2_try > d2_cur
                    if np.any(worse):
                        t_half = np.clip(t_cur - 0.5 * step, 0.0, 1.0)
                        c_half, _, _ = _eval_bezier_np(ctrl_sel, t_half)
                        d2_half = np.sum((c_half - pb) ** 2, axis=-1)
                        use_half = worse & (d2_half < d2_cur)
                        t_try = np.where(use_half, t_half, np.where(worse, t_cur, t_try))

                    t_cur = t_try
                    c_cur, dc_cur, d2c_cur = _eval_bezier_np(ctrl_sel, t_cur)
                    err_cur = c_cur - pb
                    d2_cur = np.sum(err_cur * err_cur, axis=-1)

                cand_dists[:, r] = np.sqrt(np.maximum(d2_cur, 0.0))
                cand_ts[:, r] = t_cur
                cand_spans[:, r] = sp
                cand_pts[:, r, :] = c_cur

            win = np.argmin(cand_dists, axis=1)
            rows = np.arange(N)
            best_dist[b_idx] = cand_dists[rows, win]
            best_t[b_idx] = cand_ts[rows, win]
            best_span[b_idx] = cand_spans[rows, win]
            best_pt[b_idx] = cand_pts[rows, win, :]

        ctx.pts_was_2d = pts_was_2d
        ctx.ctrl_was_3d = ctrl_was_3d
        ctx.pts_dtype = pts.dtype
        ctx.pts_device = pts.device
        ctx.ctrl_dtype = ctrl.dtype
        ctx.ctrl_device = ctrl.device
        ctx.B = B
        ctx.S = S
        ctx.K = K
        ctx.save_for_backward(
            p_in,
            torch.from_numpy(best_span),
            torch.from_numpy(best_t),
            torch.from_numpy(best_pt),
            torch.from_numpy(best_dist),
        )
        out = torch.from_numpy(best_dist).to(dtype=pts.dtype, device=pts.device)
        if pts_was_2d and ctrl_was_3d:
            out = out.squeeze(0)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        p_in, best_span, best_t, best_pt, best_dist = ctx.saved_tensors
        B, S, K = ctx.B, ctx.S, ctx.K
        deg_k = K - 1

        g_out = grad_output.detach().to(torch.float64).cpu()
        if g_out.ndim == 1:
            g_out = g_out.unsqueeze(0)
        if g_out.shape[0] == 1 and B > 1:
            g_out = g_out.expand(B, -1)

        diff = p_in - best_pt
        safe_d = best_dist.clamp(min=1e-15).unsqueeze(-1)
        u = torch.where(best_dist.unsqueeze(-1) > 1e-15, diff / safe_d, torch.zeros_like(diff))
        grad_p = g_out.unsqueeze(-1) * u

        bern = torch.from_numpy(_bernstein_basis_np(deg_k, best_t.numpy()))
        g_ctrl_contrib = -grad_p.unsqueeze(-2) * bern.unsqueeze(-1)
        grad_ctrl_b = torch.zeros((B, S, K, 3), dtype=torch.float64)
        idx = best_span[:, :, None, None].expand(B, p_in.shape[1], K, 3)
        grad_ctrl_b.scatter_add_(1, idx, g_ctrl_contrib)

        if ctx.pts_was_2d and ctx.ctrl_was_3d:
            grad_p_ret = grad_p.squeeze(0)
        elif ctx.pts_was_2d:
            grad_p_ret = grad_p.sum(dim=0)
        else:
            grad_p_ret = grad_p
        grad_ctrl_ret = grad_ctrl_b.sum(dim=0) if ctx.ctrl_was_3d else grad_ctrl_b

        return (
            grad_p_ret.to(dtype=ctx.pts_dtype, device=ctx.pts_device) if ctx.needs_input_grad[0] else None,
            grad_ctrl_ret.to(dtype=ctx.ctrl_dtype, device=ctx.ctrl_device) if ctx.needs_input_grad[1] else None,
        )


class BSplineCurve3D:
    """3D clamped B-Spline curve with Boehm Bezier extraction, RMF frames, and Envelope Theorem distance."""

    def __init__(
        self,
        control_points: Sequence[Point3DLike],
        degree: int = 3,
        knots: Sequence[float] | None = None,
    ) -> None:
        if degree not in (2, 3):
            raise ValueError("BSplineCurve3D supports degree 2 or 3")
        n = len(control_points)
        if n < degree + 1:
            raise ValueError(f"BSplineCurve3D degree={degree} requires >= {degree + 1} control points")
        self.control_points = [_require_pt3d(p, f"ctrl[{i}]") for i, p in enumerate(control_points)]
        self.degree = int(degree)
        if knots is None:
            n_int = n - degree - 1
            internal = np.linspace(0.0, 1.0, n_int + 2)[1:-1] if n_int > 0 else np.array([])
            u_arr = np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])
        else:
            u_arr = np.asarray(knots, dtype=np.float64)
        self._extract_mat, _ = _boehm_extraction_matrix(n, degree, u_arr)

    def bezier_ctrl_tensor(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        q = _stack_pts3d_tensor(self.control_points, params, dtype, device)
        M = torch.from_numpy(self._extract_mat).to(dtype=dtype, device=device)
        if q.ndim == 2:
            return torch.einsum("skc,cd->skd", M, q)
        return torch.einsum("skc,bcd->bskd", M, q)

    def bezier_ctrl_np(self, params: dict[str, Any] | None = None) -> np.ndarray:
        return self.bezier_ctrl_tensor(params, torch.float64, torch.device("cpu")).detach().numpy()

    def eval_unsigned_distance(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        ctrl = self.bezier_ctrl_tensor(params, dtype, pts.device)
        return _Bezier3DDistanceFn.apply(pts, ctrl)

    def subdivided_aabbs(
        self, params: dict[str, Any] | None = None
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        sub = _subdivide_bezier_ctrl(self.bezier_ctrl_np(params), depth=4)
        out = []
        for s in range(sub.shape[0]):
            mn, mx = sub[s].min(axis=0), sub[s].max(axis=0)
            out.append(((float(mn[0]), float(mn[1]), float(mn[2])), (float(mx[0]), float(mx[1]), float(mx[2]))))
        return out

    def interval_bound_distance(
        self,
        box_min: tuple[float, float, float],
        box_max: tuple[float, float, float],
        params: dict[str, Any] | None = None,
    ) -> float:
        d_lb = float("inf")
        for smin, smax in self.subdivided_aabbs(params):
            d_lb = min(d_lb, _dist_aabb_3d(box_min, box_max, smin, smax))
        return d_lb

    def compute_rmf_stations(
        self, n_per_span: int = 24, params: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute Double-Reflection Rotation-Minimizing Frames (X, T, N, B) and max curvature."""
        ctrl = self.bezier_ctrl_np(params)
        S = ctrl.shape[0]
        ts = np.linspace(0.0, 1.0, n_per_span + 1)[:-1]
        xs, d1s, d2s = [], [], []
        for s in range(S):
            t_arr = np.linspace(0.0, 1.0, n_per_span + 1) if s == S - 1 else ts
            c, dc, d2c = _eval_bezier_np(ctrl[s : s + 1], t_arr[:, None])
            xs.append(c[:, 0, :])
            d1s.append(dc[:, 0, :])
            d2s.append(d2c[:, 0, :])
        X = np.concatenate(xs, axis=0)
        d1 = np.concatenate(d1s, axis=0)
        d2 = np.concatenate(d2s, axis=0)
        d1_norm = np.linalg.norm(d1, axis=-1, keepdims=True).clip(min=1e-12)
        T = d1 / d1_norm
        cross_12 = np.linalg.norm(np.cross(d1, d2), axis=-1)
        kappa_max = float(np.max(cross_12 / (d1_norm.squeeze(-1) ** 3).clip(min=1e-12)))

        M_pts = X.shape[0]
        N = np.zeros_like(T)
        B = np.zeros_like(T)
        ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n0 = np.cross(T[0], ref)
        n0 /= max(np.linalg.norm(n0), 1e-12)
        N[0] = n0
        B[0] = np.cross(T[0], N[0])

        for i in range(M_pts - 1):
            v1 = X[i + 1] - X[i]
            c1 = float(np.dot(v1, v1))
            if c1 < 1e-20:
                N[i + 1] = N[i]
                B[i + 1] = B[i]
                continue
            rL = N[i] - (2.0 / c1) * float(np.dot(v1, N[i])) * v1
            tL = T[i] - (2.0 / c1) * float(np.dot(v1, T[i])) * v1
            v2 = T[i + 1] - tL
            c2 = float(np.dot(v2, v2))
            if c2 < 1e-20:
                r_next = rL
            else:
                r_next = rL - (2.0 / c2) * float(np.dot(v2, rL)) * v2
            r_next = r_next - float(np.dot(r_next, T[i + 1])) * T[i + 1]
            r_next /= max(np.linalg.norm(r_next), 1e-12)
            N[i + 1] = r_next
            B[i + 1] = np.cross(T[i + 1], N[i + 1])

        return X, T, N, B, kappa_max

    def to_occt_wire(self, params: dict[str, Any] | None = None) -> Any:
        ctrl = self.bezier_ctrl_np(params)
        S, K, _ = ctrl.shape
        wb = BRepBuilderAPI_MakeWire()
        for s in range(S):
            arr = Array1_gp_Pnt(1, K)
            for k in range(K):
                arr.SetValue(k + 1, gp_Pnt(float(ctrl[s, k, 0]), float(ctrl[s, k, 1]), float(ctrl[s, k, 2])))
            wb.Add(BRepBuilderAPI_MakeEdge(Geom_BezierCurve(arr)).Edge())
        if not wb.IsDone():
            raise RuntimeError("OCCT MakeWire failed for BSplineCurve3D")
        return wb.Wire()


class SweepAlongSpline(Node):
    """3D sweep of a circular radius or Sketch2D profile along a BSplineCurve3D using RMF."""

    def __init__(
        self,
        path: BSplineCurve3D,
        profile: Quantity | Param | Sketch2D,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.path = path
        if isinstance(profile, Sketch2D):
            self.profile: Quantity | Param | Sketch2D = profile
            bmin2, bmax2 = profile.bounds_2d()
            self._r_max_nom = max(math.hypot(x, y) for x in (bmin2[0], bmax2[0]) for y in (bmin2[1], bmax2[1]))
        else:
            self.profile = require_length(profile, "profile")
            self._r_max_nom = _val_mm_float(self.profile, None)

        _, _, _, _, kappa_max = path.compute_rmf_stations()
        if kappa_max * self._r_max_nom >= 0.95:
            raise OutOfDomainError(
                f"SweepAlongSpline {self.node_id}: curvature*radius={kappa_max * self._r_max_nom:.3f} >= 0.95"
            )
        self._lip = 1.0 if not isinstance(self.profile, Sketch2D) else 1.0 / max(1.0 - kappa_max * self._r_max_nom, 0.05)

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self._lip

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        aabbs = self.path.subdivided_aabbs(params)
        r = self._r_max_nom if isinstance(self.profile, Sketch2D) else _val_mm_float(self.profile, params)
        pad = r + 0.5
        return (
            (min(a[0][0] for a in aabbs) - pad, min(a[0][1] for a in aabbs) - pad, min(a[0][2] for a in aabbs) - pad),
            (max(a[1][0] for a in aabbs) + pad, max(a[1][1] for a in aabbs) + pad, max(a[1][2] for a in aabbs) + pad),
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        X, T, N_rmf, B_rmf, _ = self.path.compute_rmf_stations(n_per_span=24, params=params)
        # Project p onto closest RMF station in detached numpy for end-cap & frame selection
        p_in = p if p.ndim == 3 else p.unsqueeze(0)
        B_sz, N_pts, _ = p_in.shape
        X_t = torch.from_numpy(X).to(dtype=dtype, device=p.device)
        T_t = torch.from_numpy(T).to(dtype=dtype, device=p.device)
        N_t = torch.from_numpy(N_rmf).to(dtype=dtype, device=p.device)
        B_t = torch.from_numpy(B_rmf).to(dtype=dtype, device=p.device)

        if not isinstance(self.profile, Sketch2D):
            r = _val_mm_tensor(self.profile, params, dtype).to(device=p.device)
            d_curve = self.path.eval_unsigned_distance(p, params=params, dtype=dtype)
            if r.ndim == 1 and d_curve.ndim == 1:
                d_curve = d_curve.unsqueeze(0)
            r_b = _match_pts_shape(r, p_in) if d_curve.ndim == 2 else r
            d_tube = d_curve - r_b

            # Flat end-cap planes at s=0 and s=1
            ax0 = -torch.sum((p - X_t[0]) * T_t[0], dim=-1)
            ax1 = torch.sum((p - X_t[-1]) * T_t[-1], dim=-1)
            # Axial overhang outside caps
            over0 = torch.clamp(ax0, min=0.0)
            over1 = torch.clamp(ax1, min=0.0)
            rad0 = torch.sqrt(torch.clamp(torch.sum((p - X_t[0]) ** 2, dim=-1) - ax0 ** 2, min=0.0) + 1e-30)
            rad1 = torch.sqrt(torch.clamp(torch.sum((p - X_t[-1]) ** 2, dim=-1) - ax1 ** 2, min=0.0) + 1e-30)
            cap0 = torch.sqrt(over0 ** 2 + torch.clamp(rad0 - r_b, min=0.0) ** 2 + 1e-30)
            cap1 = torch.sqrt(over1 ** 2 + torch.clamp(rad1 - r_b, min=0.0) ** 2 + 1e-30)

            out_cap = torch.where(ax0 > 0.0, cap0, torch.where(ax1 > 0.0, cap1, d_tube))
            in_cap = torch.maximum(d_tube, torch.maximum(ax0, ax1))
            return torch.where((ax0 > 0.0) | (ax1 > 0.0), out_cap, in_cap)

        # Sketch2D profile via nearest RMF station
        with torch.no_grad():
            d2 = torch.sum((p_in[:, :, None, :] - X_t[None, None, :, :]) ** 2, dim=-1)
            idx = torch.argmin(d2, dim=-1)  # (B, N)
        diff = p_in - X_t[idx]
        u = torch.sum(diff * N_t[idx], dim=-1)
        v = torch.sum(diff * B_t[idx], dim=-1)
        d2d = self.profile.eval_sdf_2d(torch.stack([u, v], dim=-1).reshape(-1, 2), params=params, dtype=dtype).reshape(B_sz, N_pts)
        ax0 = -torch.sum((p_in - X_t[0]) * T_t[0], dim=-1)
        ax1 = torch.sum((p_in - X_t[-1]) * T_t[-1], dim=-1)
        d_res = torch.maximum(d2d, torch.maximum(ax0, ax1))
        return d_res.squeeze(0) if p.ndim == 2 else d_res


def _eval_bezier_surface_np(
    ctrl: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate bicubic/biquadratic Bezier patch S(u,v), dS/du, dS/dv for ctrl (..., Ku, Kv, 3)."""
    du = ctrl.shape[-3] - 1
    dv = ctrl.shape[-2] - 1
    bu = _bernstein_basis_np(du, u)  # (..., Ku)
    bv = _bernstein_basis_np(dv, v)  # (..., Kv)
    s = np.einsum("...i,...j,...ijd->...d", bu, bv, ctrl)

    du_ctrl = du * (ctrl[..., 1:, :, :] - ctrl[..., :-1, :, :])
    bu1 = _bernstein_basis_np(du - 1, u)
    su = np.einsum("...i,...j,...ijd->...d", bu1, bv, du_ctrl)

    dv_ctrl = dv * (ctrl[..., :, 1:, :] - ctrl[..., :, :-1, :])
    bv1 = _bernstein_basis_np(dv - 1, v)
    sv = np.einsum("...i,...j,...ijd->...d", bu, bv1, dv_ctrl)
    return s, su, sv


def _subdivide_patch_2d_np(ctrl: np.ndarray, depth: int = 2) -> np.ndarray:
    """Subdivide Bezier patches (S, Ku, Kv, 3) into (S * 4^depth, Ku, Kv, 3)."""
    du = ctrl.shape[-3] - 1
    dv = ctrl.shape[-2] - 1
    Lu, Ru = _de_casteljau_split_matrix(du)
    Lv, Rv = _de_casteljau_split_matrix(dv)
    cur = ctrl
    for _ in range(depth):
        # Split u
        l_u = np.einsum("ai,sijd->sajd", Lu, cur)
        r_u = np.einsum("ai,sijd->sajd", Ru, cur)
        both_u = np.concatenate([l_u, r_u], axis=0)
        # Split v
        l_v = np.einsum("bj,sijd->sibd", Lv, both_u)
        r_v = np.einsum("bj,sijd->sibd", Rv, both_u)
        cur = np.concatenate([l_v, r_v], axis=0)
    return cur


class _BezierSurfaceDistanceFn(torch.autograd.Function):
    """Exact closest-point distance to multi-patch tensor-product Bezier surface with Envelope Theorem backward."""

    @staticmethod
    def forward(ctx: Any, pts: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # pts: (N, 3) or (B, N, 3); patches: (S, Ku, Kv, 3) or (B, S, Ku, Kv, 3)
        pts_was_2d = pts.ndim == 2
        pat_was_4d = patches.ndim == 4

        p_in = pts.detach().to(torch.float64).cpu()
        c_in = patches.detach().to(torch.float64).cpu()
        if p_in.ndim == 2:
            p_in = p_in.unsqueeze(0)
        if c_in.ndim == 4:
            c_in = c_in.unsqueeze(0)

        B = max(p_in.shape[0], c_in.shape[0])
        if p_in.shape[0] == 1 and B > 1:
            p_in = p_in.expand(B, -1, -1)
        if c_in.shape[0] == 1 and B > 1:
            c_in = c_in.expand(B, -1, -1, -1, -1)

        p_np = p_in.numpy()
        c_np = c_in.numpy()
        N = p_np.shape[1]
        _, S, Ku, Kv, _ = c_np.shape
        du, dv = Ku - 1, Kv - 1

        # 5x5 uniform seed grid per patch
        M = 5
        g1 = np.linspace(0.0, 1.0, M, dtype=np.float64)
        gu, gv = np.meshgrid(g1, g1, indexing="ij")
        gu_f, gv_f = gu.ravel(), gv.ravel()  # (M*M,)
        bu_g = _bernstein_basis_np(du, gu_f)  # (M2, Ku)
        bv_g = _bernstein_basis_np(dv, gv_f)  # (M2, Kv)
        M2 = M * M

        best_dist = np.full((B, N), np.inf, dtype=np.float64)
        best_patch = np.zeros((B, N), dtype=np.int64)
        best_u = np.zeros((B, N), dtype=np.float64)
        best_v = np.zeros((B, N), dtype=np.float64)
        best_pt = np.zeros((B, N, 3), dtype=np.float64)

        for b_idx in range(B):
            cb = c_np[b_idx]  # (S, Ku, Kv, 3)
            pb = p_np[b_idx]  # (N, 3)
            samples = np.einsum("mi,mj,sijd->smd", bu_g, bv_g, cb)  # (S, M2, 3)
            k_top = min(4, S)
            if S > 8:
                from scipy.spatial import cKDTree

                tree = cKDTree(samples.reshape(S * M2, 3))
                k_q = min(max(32, k_top * 8), S * M2)
                _, q_idx = tree.query(pb, k=k_q)  # (N, k_q)
                if q_idx.ndim == 1:
                    q_idx = q_idx[:, None]
                q_sp = q_idx // M2
                q_m = q_idx % M2
                top_sp = np.zeros((N, k_top), dtype=np.int64)
                top_m = np.zeros((N, k_top), dtype=np.int64)
                for n_i in range(N):
                    seen: list[int] = []
                    for j_i in range(q_sp.shape[1]):
                        sp_j = int(q_sp[n_i, j_i])
                        if sp_j not in seen:
                            top_sp[n_i, len(seen)] = sp_j
                            top_m[n_i, len(seen)] = int(q_m[n_i, j_i])
                            seen.append(sp_j)
                            if len(seen) == k_top:
                                break
                    while len(seen) < k_top:
                        top_sp[n_i, len(seen)] = top_sp[n_i, 0]
                        top_m[n_i, len(seen)] = top_m[n_i, 0]
                        seen.append(-1)
            else:
                diff = pb[:, None, None, :] - samples[None, :, :, :]
                d2 = np.sum(diff * diff, axis=-1)  # (N, S, M2)
                patch_best_m = np.argmin(d2, axis=2)  # (N, S)
                patch_best_d2 = np.take_along_axis(d2, patch_best_m[:, :, None], axis=2).squeeze(2)  # (N, S)
                top_sp = np.argpartition(patch_best_d2, kth=k_top - 1, axis=1)[:, :k_top]  # (N, k_top)
                rows_n = np.arange(N)
                top_m = patch_best_m[rows_n[:, None], top_sp]

            cand_dists = np.full((N, k_top), np.inf, dtype=np.float64)
            cand_us = np.zeros((N, k_top), dtype=np.float64)
            cand_vs = np.zeros((N, k_top), dtype=np.float64)
            cand_sp = np.zeros((N, k_top), dtype=np.int64)
            cand_pts = np.zeros((N, k_top, 3), dtype=np.float64)

            for r in range(k_top):
                sp = top_sp[:, r]
                m_idx = top_m[:, r]
                uc = gu_f[m_idx].copy()
                vc = gv_f[m_idx].copy()
                ctrl_sel = cb[sp]  # (N, Ku, Kv, 3)

                s_cur, su, sv = _eval_bezier_surface_np(ctrl_sel, uc, vc)
                err = s_cur - pb
                d2_cur = np.sum(err * err, axis=-1)

                # 8 damped 2D Gauss-Newton / LM steps clamped to [0, 1]^2
                for _ in range(8):
                    j11 = np.sum(su * su, axis=-1) + 1e-8
                    j12 = np.sum(su * sv, axis=-1)
                    j22 = np.sum(sv * sv, axis=-1) + 1e-8
                    rhs1 = -np.sum(su * err, axis=-1)
                    rhs2 = -np.sum(sv * err, axis=-1)
                    det = (j11 * j22 - j12 * j12).clip(min=1e-14)
                    du_step = np.clip((j22 * rhs1 - j12 * rhs2) / det, -0.3, 0.3)
                    dv_step = np.clip((-j12 * rhs1 + j11 * rhs2) / det, -0.3, 0.3)
                    if float(np.max(np.abs(du_step))) < 1e-10 and float(np.max(np.abs(dv_step))) < 1e-10:
                        break

                    u_try = np.clip(uc + du_step, 0.0, 1.0)
                    v_try = np.clip(vc + dv_step, 0.0, 1.0)
                    s_try, _, _ = _eval_bezier_surface_np(ctrl_sel, u_try, v_try)
                    d2_try = np.sum((s_try - pb) ** 2, axis=-1)
                    worse = d2_try > d2_cur
                    if np.any(worse):
                        u_h = np.clip(uc + 0.5 * du_step, 0.0, 1.0)
                        v_h = np.clip(vc + 0.5 * dv_step, 0.0, 1.0)
                        s_h, _, _ = _eval_bezier_surface_np(ctrl_sel, u_h, v_h)
                        d2_h = np.sum((s_h - pb) ** 2, axis=-1)
                        use_h = worse & (d2_h < d2_cur)
                        u_try = np.where(use_h, u_h, np.where(worse, uc, u_try))
                        v_try = np.where(use_h, v_h, np.where(worse, vc, v_try))

                    uc, vc = u_try, v_try
                    s_cur, su, sv = _eval_bezier_surface_np(ctrl_sel, uc, vc)
                    err = s_cur - pb
                    d2_cur = np.sum(err * err, axis=-1)

                cand_dists[:, r] = np.sqrt(np.maximum(d2_cur, 0.0))
                cand_us[:, r] = uc
                cand_vs[:, r] = vc
                cand_sp[:, r] = sp
                cand_pts[:, r, :] = s_cur

            win = np.argmin(cand_dists, axis=1)
            rows = np.arange(N)
            best_dist[b_idx] = cand_dists[rows, win]
            best_u[b_idx] = cand_us[rows, win]
            best_v[b_idx] = cand_vs[rows, win]
            best_patch[b_idx] = cand_sp[rows, win]
            best_pt[b_idx] = cand_pts[rows, win, :]

        ctx.pts_was_2d = pts_was_2d
        ctx.pat_was_4d = pat_was_4d
        ctx.pts_dtype = pts.dtype
        ctx.pts_device = pts.device
        ctx.pat_dtype = patches.dtype
        ctx.pat_device = patches.device
        ctx.B, ctx.S, ctx.Ku, ctx.Kv = B, S, Ku, Kv
        ctx.save_for_backward(
            p_in,
            torch.from_numpy(best_patch),
            torch.from_numpy(best_u),
            torch.from_numpy(best_v),
            torch.from_numpy(best_pt),
            torch.from_numpy(best_dist),
        )
        out = torch.from_numpy(best_dist).to(dtype=pts.dtype, device=pts.device)
        if pts_was_2d and pat_was_4d:
            out = out.squeeze(0)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        p_in, best_patch, best_u, best_v, best_pt, best_dist = ctx.saved_tensors
        B, S, Ku, Kv = ctx.B, ctx.S, ctx.Ku, ctx.Kv
        g_out = grad_output.detach().to(torch.float64).cpu()
        if g_out.ndim == 1:
            g_out = g_out.unsqueeze(0)
        if g_out.shape[0] == 1 and B > 1:
            g_out = g_out.expand(B, -1)

        diff = p_in - best_pt
        safe_d = best_dist.clamp(min=1e-15).unsqueeze(-1)
        u_vec = torch.where(best_dist.unsqueeze(-1) > 1e-15, diff / safe_d, torch.zeros_like(diff))
        grad_p = g_out.unsqueeze(-1) * u_vec  # (B, N, 3)

        bu = torch.from_numpy(_bernstein_basis_np(Ku - 1, best_u.numpy()))  # (B, N, Ku)
        bv = torch.from_numpy(_bernstein_basis_np(Kv - 1, best_v.numpy()))  # (B, N, Kv)
        w = torch.einsum("bni,bnj->bnij", bu, bv)  # (B, N, Ku, Kv)
        contrib = -grad_p[:, :, None, None, :] * w[:, :, :, :, None]  # (B, N, Ku, Kv, 3)

        grad_pat_b = torch.zeros((B, S, Ku, Kv, 3), dtype=torch.float64)
        idx = best_patch[:, :, None, None, None].expand(B, p_in.shape[1], Ku, Kv, 3)
        grad_pat_b.scatter_add_(1, idx, contrib)

        if ctx.pts_was_2d and ctx.pat_was_4d:
            grad_p_ret = grad_p.squeeze(0)
        elif ctx.pts_was_2d:
            grad_p_ret = grad_p.sum(dim=0)
        else:
            grad_p_ret = grad_p
        grad_pat_ret = grad_pat_b.sum(dim=0) if ctx.pat_was_4d else grad_pat_b

        return (
            grad_p_ret.to(dtype=ctx.pts_dtype, device=ctx.pts_device) if ctx.needs_input_grad[0] else None,
            grad_pat_ret.to(dtype=ctx.pat_dtype, device=ctx.pat_device) if ctx.needs_input_grad[1] else None,
        )


class BSplineSurface:
    """Tensor-product B-Spline surface with Boehm patch extraction, BVH lower bounds, and OCCT NURBS export."""

    def __init__(
        self,
        control_net: Sequence[Sequence[Point3DLike]],
        degree_u: int = 3,
        degree_v: int = 3,
    ) -> None:
        self.nu = len(control_net)
        self.nv = len(control_net[0])
        if self.nu < degree_u + 1 or self.nv < degree_v + 1:
            raise ValueError("Control net smaller than degree + 1")
        self.degree_u = int(degree_u)
        self.degree_v = int(degree_v)
        self.control_net = [
            [_require_pt3d(pt, f"ctrl[{i}][{j}]") for j, pt in enumerate(row)]
            for i, row in enumerate(control_net)
        ]

        def _clamped_knots(n: int, deg_k: int) -> np.ndarray:
            ni = n - deg_k - 1
            internal = np.linspace(0.0, 1.0, ni + 2)[1:-1] if ni > 0 else np.array([])
            return np.concatenate([np.zeros(deg_k + 1), internal, np.ones(deg_k + 1)])

        self.knots_u = _clamped_knots(self.nu, self.degree_u)
        self.knots_v = _clamped_knots(self.nv, self.degree_v)
        self._Mu, _ = _boehm_extraction_matrix(self.nu, self.degree_u, self.knots_u)
        self._Mv, _ = _boehm_extraction_matrix(self.nv, self.degree_v, self.knots_v)
        self._cached_aabbs_np: np.ndarray | None = None

    def control_net_tensor(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        rows = [_stack_pts3d_tensor(row, params, dtype, device) for row in self.control_net]
        if any(r.ndim == 3 for r in rows):
            B = next(r.shape[0] for r in rows if r.ndim == 3)
            rows = [r if r.ndim == 3 else r.unsqueeze(0).expand(B, -1, -1) for r in rows]
            return torch.stack(rows, dim=1)  # (B, nu, nv, 3)
        return torch.stack(rows, dim=0)  # (nu, nv, 3)

    def bezier_patches_tensor(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        q = self.control_net_tensor(params, dtype, device)
        Mu = torch.from_numpy(self._Mu).to(dtype=dtype, device=device)  # (Su, Ku, nu)
        Mv = torch.from_numpy(self._Mv).to(dtype=dtype, device=device)  # (Sv, Kv, nv)
        if q.ndim == 3:
            p = torch.einsum("aiu,bjv,uvd->abijd", Mu, Mv, q)
            return p.reshape(-1, self.degree_u + 1, self.degree_v + 1, 3)
        p = torch.einsum("aiu,bjv,buvd->babijd".replace("bab", "ab").replace("abijd", "abijd"), Mu, Mv, q) if False else torch.einsum("aiu,bjv,quvd->qabijd", Mu, Mv, q)
        return p.reshape(q.shape[0], -1, self.degree_u + 1, self.degree_v + 1, 3)

    def bezier_patches_np(self, params: dict[str, Any] | None = None) -> np.ndarray:
        return self.bezier_patches_tensor(params, torch.float64, torch.device("cpu")).detach().numpy()

    def eval_unsigned_distance(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        patches = self.bezier_patches_tensor(params, dtype, pts.device)
        return _BezierSurfaceDistanceFn.apply(pts, patches)

    def eval_shell_sdf(
        self,
        pts: torch.Tensor,
        thickness: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Exact L=1 shell SDF matching OCCT flat side-skirt thickening at patch boundaries."""
        d = self.eval_unsigned_distance(pts, params=params, dtype=dtype)
        t_b = _match_pts_shape(thickness, pts if pts.ndim == 3 else (pts.unsqueeze(0) if d.ndim == 2 else pts))
        f_int = d - 0.5 * t_b
        # For non-batched or scalar evaluation, refine flat boundary skirt where (u*, v*) hits {0, 1}
        if pts.ndim == 2 and d.ndim == 1:
            with torch.no_grad():
                p_np = pts.detach().to(torch.float64).cpu().numpy()
                patches_np = self.bezier_patches_np(params)
                N = p_np.shape[0]
                S = patches_np.shape[0]
                g1 = np.linspace(0.0, 1.0, 5)
                gu, gv = np.meshgrid(g1, g1, indexing="ij")
                gu_f, gv_f = gu.ravel(), gv.ravel()
                bu_g = _bernstein_basis_np(self.degree_u, gu_f)
                bv_g = _bernstein_basis_np(self.degree_v, gv_f)
                samples = np.einsum("mi,mj,sijd->smd", bu_g, bv_g, patches_np)
                d2 = np.sum((p_np[:, None, None, :] - samples[None, :, :, :]) ** 2, axis=-1).reshape(N, S * 25)
                idx = np.argmin(d2, axis=1)
                sp = idx // 25
                m_idx = idx % 25
                uc, vc = gu_f[m_idx].copy(), gv_f[m_idx].copy()
                ctrl_sel = patches_np[sp]
                for _ in range(8):
                    s_cur, su, sv = _eval_bezier_surface_np(ctrl_sel, uc, vc)
                    err = s_cur - p_np
                    j11 = np.sum(su * su, axis=-1) + 1e-8
                    j12 = np.sum(su * sv, axis=-1)
                    j22 = np.sum(sv * sv, axis=-1) + 1e-8
                    rhs1 = -np.sum(su * err, axis=-1)
                    rhs2 = -np.sum(sv * err, axis=-1)
                    det = (j11 * j22 - j12 * j12).clip(min=1e-14)
                    uc = np.clip(uc + np.clip((j22 * rhs1 - j12 * rhs2) / det, -0.3, 0.3), 0.0, 1.0)
                    vc = np.clip(vc + np.clip((-j12 * rhs1 + j11 * rhs2) / det, -0.3, 0.3), 0.0, 1.0)
                s_cur, su, sv = _eval_bezier_surface_np(ctrl_sel, uc, vc)
                nrm = np.cross(su, sv)
                nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
                on_bnd = (uc < 1e-4) | (uc > 1.0 - 1e-4) | (vc < 1e-4) | (vc > 1.0 - 1e-4)
                diff = p_np - s_cur
                proj_n = np.sum(diff * nrm, axis=-1)
                d_n = np.abs(proj_n) - 0.5 * float(thickness.detach().cpu().item())
                d_tau = np.linalg.norm(diff - proj_n[:, None] * nrm, axis=-1)
                f_bnd = np.sqrt(np.maximum(d_n, 0.0) ** 2 + d_tau ** 2)
                f_bnd_t = torch.from_numpy(np.where(on_bnd, f_bnd, 0.0)).to(dtype=dtype, device=pts.device)
                on_bnd_t = torch.from_numpy(on_bnd).to(device=pts.device)
                f_int = torch.where(on_bnd_t, f_bnd_t, f_int)
        return f_int

    def _aabbs_np(self, params: dict[str, Any] | None = None, depth: int = 2) -> np.ndarray:
        if params is None and depth == 2 and self._cached_aabbs_np is not None:
            return self._cached_aabbs_np
        sub = _subdivide_patch_2d_np(self.bezier_patches_np(params), depth=depth)
        flat = sub.reshape(sub.shape[0], -1, 3)
        mn = flat.min(axis=1)
        mx = flat.max(axis=1)
        arr = np.concatenate([mn, mx], axis=1)  # (S_sub, 6)
        if params is None and depth == 2:
            self._cached_aabbs_np = arr
        return arr

    def subdivided_aabbs(
        self, params: dict[str, Any] | None = None, depth: int = 2
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        arr = self._aabbs_np(params, depth=depth)
        return [((float(r[0]), float(r[1]), float(r[2])), (float(r[3]), float(r[4]), float(r[5]))) for r in arr]

    def interval_bound_distance(
        self,
        box_min: tuple[float, float, float],
        box_max: tuple[float, float, float],
        params: dict[str, Any] | None = None,
    ) -> float:
        arr = self._aabbs_np(params, depth=2)
        dx = np.maximum(np.maximum(arr[:, 0] - box_max[0], box_min[0] - arr[:, 3]), 0.0)
        dy = np.maximum(np.maximum(arr[:, 1] - box_max[1], box_min[1] - arr[:, 4]), 0.0)
        dz = np.maximum(np.maximum(arr[:, 2] - box_max[2], box_min[2] - arr[:, 5]), 0.0)
        return float(np.min(np.sqrt(dx * dx + dy * dy + dz * dz)))

    def sample_points_tensor(
        self, n_u: int = 12, n_v: int = 12, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        patches = self.bezier_patches_tensor(params, dtype, device)
        u = np.linspace(0.0, 1.0, n_u)
        v = np.linspace(0.0, 1.0, n_v)
        gu, gv = np.meshgrid(u, v, indexing="ij")
        bu = torch.from_numpy(_bernstein_basis_np(self.degree_u, gu.ravel())).to(dtype=dtype, device=device)
        bv = torch.from_numpy(_bernstein_basis_np(self.degree_v, gv.ravel())).to(dtype=dtype, device=device)
        if patches.ndim == 4:
            pts = torch.einsum("mi,mj,sijd->smd", bu, bv, patches).reshape(-1, 3)
            return pts
        pts = torch.einsum("mi,mj,bsijd->bsmd", bu, bv, patches).reshape(patches.shape[0], -1, 3)
        return pts

    def to_occt_surface(self, params: dict[str, Any] | None = None) -> Geom_BSplineSurface:
        poles = Array2_gp_Pnt(1, self.nu, 1, self.nv)
        for i in range(self.nu):
            for j in range(self.nv):
                x, y, z = _pt3d_float(self.control_net[i][j], params)
                poles.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))

        def _unique_knots_mults(k_arr: np.ndarray) -> tuple[Array1_double, Array1_int]:
            vals: list[float] = []
            mults: list[int] = []
            for k in k_arr:
                if not vals or abs(float(k) - vals[-1]) > 1e-12:
                    vals.append(float(k))
                    mults.append(1)
                else:
                    mults[-1] += 1
            ak = Array1_double(1, len(vals))
            am = Array1_int(1, len(mults))
            for idx, (kv, mv) in enumerate(zip(vals, mults)):
                ak.SetValue(idx + 1, kv)
                am.SetValue(idx + 1, mv)
            return ak, am

        uk, um = _unique_knots_mults(self.knots_u)
        vk, vm = _unique_knots_mults(self.knots_v)
        return Geom_BSplineSurface(poles, uk, vk, um, vm, self.degree_u, self.degree_v)


class BSplineSurfaceShell(Node):
    """Differentiable certified thickened freeform NURBS surface shell (L=1)."""

    def __init__(
        self,
        surface: BSplineSurface,
        thickness: Quantity | Param,
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        self.surface = surface
        self.thickness = require_length(thickness, "thickness")

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        aabbs = self.surface.subdivided_aabbs(params, depth=1)
        pad = 0.5 * _val_mm_float(self.thickness, params) + 0.5
        return (
            (min(a[0][0] for a in aabbs) - pad, min(a[0][1] for a in aabbs) - pad, min(a[0][2] for a in aabbs) - pad),
            (max(a[1][0] for a in aabbs) + pad, max(a[1][1] for a in aabbs) + pad, max(a[1][2] for a in aabbs) + pad),
        )

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        t = _val_mm_tensor(self.thickness, params, dtype).to(device=p.device)
        return self.surface.eval_shell_sdf(p, t, params=params, dtype=dtype)


class LoftSketches(Node):
    """Smooth parametric solid loft across >= 2 closed Loop2D/Sketch2D sections at increasing Z levels."""

    def __init__(
        self,
        sections: Sequence[Sketch2D],
        z_levels: Sequence[Quantity | Param],
        node_id: str | None = None,
    ) -> None:
        super().__init__(node_id)
        if len(sections) < 2 or len(sections) != len(z_levels):
            raise ValueError("LoftSketches requires >= 2 sections matching z_levels")
        self.sections = list(sections)
        self.z_levels = [require_length(z, f"z_levels[{i}]") for i, z in enumerate(z_levels)]
        self._cached_patches: torch.Tensor | None = None

    def local_bounds(
        self, params: dict[str, Any] | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        b2s = [s.bounds_2d(params) for s in self.sections]
        zs = [_val_mm_float(z, params) for z in self.z_levels]
        return (
            (min(b[0][0] for b in b2s), min(b[0][1] for b in b2s), min(zs)),
            (max(b[1][0] for b in b2s), max(b[1][1] for b in b2s), max(zs)),
        )

    def _build_lateral_patches(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Sample sections segment-by-segment (1:1 topology) and build ruled patches."""
        if params is None and self._cached_patches is not None and self._cached_patches.dtype == dtype and self._cached_patches.device == device:
            return self._cached_patches
        n_segs = len(self.sections[0].outer.segments)
        same_topo = all(len(s.outer.segments) == n_segs for s in self.sections)
        n_per_seg = max(4, 48 // n_segs) if same_topo else 16
        rings: list[torch.Tensor] = []
        for sec, zq in zip(self.sections, self.z_levels):
            pts2 = sec.outer.sample_boundary_points(n_per_seg=n_per_seg, params=params, dtype=dtype, device=device)
            if pts2.ndim == 2:
                if not same_topo:
                    idx = torch.linspace(0, pts2.shape[0] - 1, 48, dtype=torch.long)
                    pts2 = pts2[idx]
                zv = _val_mm_tensor(zq, params, dtype).to(device=device).expand(pts2.shape[0], 1)
                rings.append(torch.cat([pts2, zv], dim=-1))
            else:
                B = pts2.shape[0]
                if not same_topo:
                    idx = torch.linspace(0, pts2.shape[1] - 1, 48, dtype=torch.long)
                    pts2 = pts2[:, idx, :]
                zv = _val_mm_tensor(zq, params, dtype).to(device=device)
                zv = (zv.expand(B) if zv.ndim == 0 else zv)[:, None, None].expand(B, pts2.shape[1], 1)
                rings.append(torch.cat([pts2, zv], dim=-1))
        # Build (S, 2, 2, 3) bilinear Bezier patches around ring
        patches = []
        for k in range(len(rings) - 1):
            r0 = rings[k]
            r1 = rings[k + 1]
            r0_next = torch.roll(r0, -1, dims=-2)
            r1_next = torch.roll(r1, -1, dims=-2)
            # Each quad: [[r0, r1], [r0_next, r1_next]]
            q_k = torch.stack(
                [torch.stack([r0, r1], dim=-2), torch.stack([r0_next, r1_next], dim=-2)],
                dim=-3,
            )  # (..., n_u, 2, 2, 3)
            patches.append(q_k)
        out = torch.cat(patches, dim=-4)
        if params is None:
            self._cached_patches = out
        return out

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        patches = self._build_lateral_patches(params, dtype, p.device)
        d_lat = _BezierSurfaceDistanceFn.apply(p, patches)
        zs = [_val_mm_float(z, params) for z in self.z_levels]
        z_min, z_max = min(zs), max(zs)
        dz = torch.maximum(z_min - p[..., 2], p[..., 2] - z_max)

        # Approximate interior sign by linearly interpolating 2D section SDF along Z
        t_norm = torch.clamp((p[..., 2] - z_min) / max(z_max - z_min, 1e-9), 0.0, 1.0)
        d0 = self.sections[0].eval_sdf_2d(p[..., :2], params=params, dtype=dtype)
        d1 = self.sections[-1].eval_sdf_2d(p[..., :2], params=params, dtype=dtype)
        # For >2 sections, interpolate nearest bracket
        d_sec = (1.0 - t_norm) * d0 + t_norm * d1
        inside = (d_sec <= 0.0) & (dz <= 0.0)
        dist = torch.where(
            dz > 0.0,
            torch.sqrt(dz * dz + torch.where(d_sec <= 0.0, torch.zeros_like(d_lat), d_lat) ** 2 + 1e-30),
            torch.where(inside, -torch.minimum(d_lat, -dz), d_lat),
        )
        return dist
