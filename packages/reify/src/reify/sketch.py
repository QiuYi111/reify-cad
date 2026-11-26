"""M4: 2D complete sketch and spline kernel (lines, arcs, Bézier, B-Spline, Envelope Theorem autograd, OCCT)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.collections import Array1_gp_Pnt
from OCP.GC import GC_MakeArcOfCircle
from OCP.Geom import Geom_BezierCurve
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Wire

from reify.ast import Param, _val_deg_float, _val_mm_float, _val_mm_tensor, _val_rad_tensor
from reify.types import OutOfDomainError
from reify.units import Quantity, deg, mm, require_angle, require_length

Point2DLike = tuple[Quantity | Param, Quantity | Param]


def _require_pt2d(pt: Point2DLike, name: str = "pt") -> tuple[Quantity | Param, Quantity | Param]:
    if len(pt) != 2:
        raise ValueError(f"{name} must be a 2-tuple (x, y)")
    return (require_length(pt[0], f"{name}[0]"), require_length(pt[1], f"{name}[1]"))


def _pt2d_float(
    pt: tuple[Quantity | Param, Quantity | Param], params: dict[str, Any] | None
) -> tuple[float, float]:
    return (_val_mm_float(pt[0], params), _val_mm_float(pt[1], params))


def _pt2d_tensor(
    pt: tuple[Quantity | Param, Quantity | Param],
    params: dict[str, Any] | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    x = _val_mm_tensor(pt[0], params, dtype).to(device=device)
    y = _val_mm_tensor(pt[1], params, dtype).to(device=device)
    if x.ndim == 1 or y.ndim == 1:
        x, y = torch.broadcast_tensors(x, y)
    return torch.stack([x, y], dim=-1)  # (2,) or (B, 2)


def _stack_pts2d_tensor(
    pts_list: Sequence[tuple[Quantity | Param, Quantity | Param]],
    params: dict[str, Any] | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Stack M 2D points into shape (M, 2) or (B, M, 2) if any coord is batched."""
    tensors = [_pt2d_tensor(p, params, dtype, device) for p in pts_list]
    has_batch = any(t.ndim == 2 for t in tensors)
    if not has_batch:
        return torch.stack(tensors, dim=0)  # (M, 2)
    b = next(t.shape[0] for t in tensors if t.ndim == 2)
    expanded = [t if t.ndim == 2 else t.unsqueeze(0).expand(b, 2) for t in tensors]
    return torch.stack(expanded, dim=1)  # (B, M, 2)


def _dist_aabb_2d(
    bmin: tuple[float, float],
    bmax: tuple[float, float],
    smin: tuple[float, float],
    smax: tuple[float, float],
) -> float:
    dx = max(smin[0] - bmax[0], bmin[0] - smax[0], 0.0)
    dy = max(smin[1] - bmax[1], bmin[1] - smax[1], 0.0)
    return math.hypot(dx, dy)


def _bernstein_basis_np(degree: int, t: np.ndarray) -> np.ndarray:
    """Evaluate Bernstein polynomials B_{i, degree}(t) for t of shape (...), returning (..., degree+1)."""
    omt = 1.0 - t
    if degree == 2:
        return np.stack([omt * omt, 2.0 * omt * t, t * t], axis=-1)
    if degree == 3:
        omt2 = omt * omt
        t2 = t * t
        return np.stack([omt2 * omt, 3.0 * omt2 * t, 3.0 * omt * t2, t2 * t], axis=-1)
    # General degree via de Casteljau table
    cols = []
    for i in range(degree + 1):
        c = math.comb(degree, i)
        cols.append(c * (t**i) * (omt ** (degree - i)))
    return np.stack(cols, axis=-1)


def _eval_bezier_np(ctrl: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Bézier curve C(t), C'(t), C''(t) for ctrl (..., K, D) and t (...)."""
    deg_k = ctrl.shape[-2] - 1
    b = _bernstein_basis_np(deg_k, t)  # (..., K)
    c = np.einsum("...k,...kd->...d", b, ctrl)

    d1_ctrl = deg_k * (ctrl[..., 1:, :] - ctrl[..., :-1, :])
    b1 = _bernstein_basis_np(deg_k - 1, t)
    dc = np.einsum("...k,...kd->...d", b1, d1_ctrl)

    if deg_k >= 2:
        d2_ctrl = (deg_k - 1) * (d1_ctrl[..., 1:, :] - d1_ctrl[..., :-1, :])
        if deg_k == 2:
            d2c = np.broadcast_to(d2_ctrl[..., 0, :], c.shape)
        else:
            b2 = _bernstein_basis_np(deg_k - 2, t)
            d2c = np.einsum("...k,...kd->...d", b2, d2_ctrl)
    else:
        d2c = np.zeros_like(c)
    return c, dc, d2c


def _de_casteljau_split_matrix(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return constant (K, K) linear matrices (L, R) splitting a degree-p Bézier at t=0.5."""
    k = degree + 1
    L = np.zeros((k, k), dtype=np.float64)
    R = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1):
            L[i, j] = math.comb(i, j) * (0.5**i)
            R[k - 1 - i, k - 1 - j] = math.comb(i, j) * (0.5**i)
    return L, R


def _subdivide_bezier_ctrl(ctrl: np.ndarray, depth: int) -> np.ndarray:
    """Subdivide Bézier spans (S, K, D) `depth` times into (S * 2^depth, K, D)."""
    if depth <= 0:
        return ctrl
    deg_k = ctrl.shape[-2] - 1
    L, R = _de_casteljau_split_matrix(deg_k)
    cur = ctrl
    for _ in range(depth):
        left = np.einsum("ij,sjd->sid", L, cur)
        right = np.einsum("ij,sjd->sid", R, cur)
        cur = np.concatenate([left, right], axis=0)
    return cur


class _Bezier2DDistanceFn(torch.autograd.Function):
    """Exact closest-point distance to multi-span 2D Bézier with Envelope Theorem backward."""

    @staticmethod
    def forward(ctx: Any, pts: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        # pts: (N, 2) or (B, N, 2); ctrl: (S, K, 2) or (B, S, K, 2)
        pts_was_2d = pts.ndim == 2
        ctrl_was_3d = ctrl.ndim == 3

        p_in = pts.detach().to(torch.float64).cpu()
        c_in = ctrl.detach().to(torch.float64).cpu()

        if p_in.ndim == 2:
            p_in = p_in.unsqueeze(0)  # (1, N, 2)
        if c_in.ndim == 3:
            c_in = c_in.unsqueeze(0)  # (1, S, K, 2)

        B = max(p_in.shape[0], c_in.shape[0])
        if p_in.shape[0] == 1 and B > 1:
            p_in = p_in.expand(B, -1, -1)
        if c_in.shape[0] == 1 and B > 1:
            c_in = c_in.expand(B, -1, -1, -1)

        p_np = p_in.numpy()  # (B, N, 2)
        c_np = c_in.numpy()  # (B, S, K, 2)
        N = p_np.shape[1]
        _, S, K, _ = c_np.shape
        deg_k = K - 1

        # Uniform sample grid per span for global multi-seed initialization
        M = 17
        t_grid = np.linspace(0.0, 1.0, M, dtype=np.float64)  # (M,)
        b_grid = _bernstein_basis_np(deg_k, t_grid)  # (M, K)

        best_dist = np.full((B, N), np.inf, dtype=np.float64)
        best_span = np.zeros((B, N), dtype=np.int64)
        best_t = np.zeros((B, N), dtype=np.float64)
        best_pt = np.zeros((B, N, 2), dtype=np.float64)

        for b_idx in range(B):
            cb = c_np[b_idx]  # (S, K, 2)
            pb = p_np[b_idx]  # (N, 2)

            # Sample curves: (S, M, 2)
            samples = np.einsum("mk,skd->smd", b_grid, cb)
            if S > 8:
                from scipy.spatial import cKDTree

                tree = cKDTree(samples.reshape(S * M, 2))
                _, top2 = tree.query(pb, k=min(2, S * M))
                if top2.ndim == 1:
                    top2 = top2[:, None]
            else:
                diff = pb[:, None, None, :] - samples[None, :, :, :]
                d2 = np.sum(diff * diff, axis=-1)
                flat_d2 = d2.reshape(N, S * M)
                top2 = np.argpartition(flat_d2, kth=min(1, S * M - 1), axis=1)[:, :2]  # (N, 2)

            cand_dists = np.full((N, 2), np.inf, dtype=np.float64)
            cand_ts = np.zeros((N, 2), dtype=np.float64)
            cand_spans = np.zeros((N, 2), dtype=np.int64)
            cand_pts = np.zeros((N, 2, 2), dtype=np.float64)

            for r in range(top2.shape[1]):
                idx = top2[:, r]
                sp = idx // M
                m_idx = idx % M
                t_cur = t_grid[m_idx].copy()
                ctrl_sel = cb[sp]  # (N, K, 2)

                c_cur, dc_cur, d2c_cur = _eval_bezier_np(ctrl_sel, t_cur)
                err_cur = c_cur - pb
                d2_cur = np.sum(err_cur * err_cur, axis=-1)

                # Safeguarded Newton steps on g(t) = (C(t)-p) . C'(t) = 0
                for _ in range(6):
                    g = np.sum(err_cur * dc_cur, axis=-1)
                    gp = np.sum(dc_cur * dc_cur, axis=-1) + np.sum(err_cur * d2c_cur, axis=-1)
                    valid = np.abs(gp) > 1e-14
                    step = np.where(valid, g / np.where(valid, gp, 1.0), 0.0)
                    step = np.clip(step, -0.25, 0.25)
                    if float(np.max(np.abs(step))) < 1e-11:
                        break

                    # Backtracking line search
                    alpha = np.ones(N, dtype=np.float64)
                    t_try = np.clip(t_cur - alpha * step, 0.0, 1.0)
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
        B = ctx.B
        S = ctx.S
        K = ctx.K
        deg_k = K - 1

        g_out = grad_output.detach().to(torch.float64).cpu()
        if g_out.ndim == 1:
            g_out = g_out.unsqueeze(0)
        if g_out.shape[0] == 1 and B > 1:
            g_out = g_out.expand(B, -1)

        diff = p_in - best_pt  # (B, N, 2)
        safe_d = best_dist.clamp(min=1e-15).unsqueeze(-1)
        u = torch.where(best_dist.unsqueeze(-1) > 1e-15, diff / safe_d, torch.zeros_like(diff))

        grad_p = g_out.unsqueeze(-1) * u  # (B, N, 2)

        t_np = best_t.numpy()
        bern = torch.from_numpy(_bernstein_basis_np(deg_k, t_np))  # (B, N, K)
        # -g_out * u * B_k(t*)
        g_ctrl_contrib = -grad_p.unsqueeze(-2) * bern.unsqueeze(-1)  # (B, N, K, 2)

        grad_ctrl_b = torch.zeros((B, S, K, 2), dtype=torch.float64)
        idx = best_span[:, :, None, None].expand(B, p_in.shape[1], K, 2)
        grad_ctrl_b.scatter_add_(1, idx, g_ctrl_contrib)

        if ctx.pts_was_2d and ctx.ctrl_was_3d:
            grad_p_ret = grad_p.squeeze(0)
        elif ctx.pts_was_2d:
            grad_p_ret = grad_p.sum(dim=0)
        else:
            grad_p_ret = grad_p

        if ctx.ctrl_was_3d:
            grad_ctrl_ret = grad_ctrl_b.sum(dim=0)
        else:
            grad_ctrl_ret = grad_ctrl_b

        return (
            grad_p_ret.to(dtype=ctx.pts_dtype, device=ctx.pts_device) if ctx.needs_input_grad[0] else None,
            grad_ctrl_ret.to(dtype=ctx.ctrl_dtype, device=ctx.ctrl_device) if ctx.needs_input_grad[1] else None,
        )


def _boehm_extraction_matrix(
    n_ctrl: int, degree: int, knots: np.ndarray
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Compute exact linear matrix M of shape (n_spans, degree+1, n_ctrl) mapping B-Spline ctrl to Bézier spans."""
    p = degree
    # Identify unique non-empty knot spans inside [knots[p], knots[n_ctrl]]
    u_min = float(knots[p])
    u_max = float(knots[n_ctrl])
    tol = 1e-12

    # Work on a mutable knot list and linear weight matrix W (current_ctrl x n_ctrl)
    U = list(map(float, knots))
    W = np.eye(n_ctrl, dtype=np.float64)

    # Collect unique breakpoints in [u_min, u_max]
    breaks: list[float] = []
    for u in U:
        if u_min - tol <= u <= u_max + tol:
            if not breaks or abs(u - breaks[-1]) > tol:
                breaks.append(u)

    def _mult_and_last_idx(u_val: float) -> tuple[int, int]:
        cnt = 0
        last = -1
        for idx, uk in enumerate(U):
            if abs(uk - u_val) <= tol:
                cnt += 1
                last = idx
        return cnt, last

    # Insert each internal breakpoint until multiplicity == p
    for u_val in breaks[1:-1]:
        mult, k = _mult_and_last_idx(u_val)
        r = p - mult
        for _ in range(max(0, r)):
            # Boehm single knot insertion at u_val after index k
            n_cur = W.shape[0]
            W_new = np.zeros((n_cur + 1, n_ctrl), dtype=np.float64)
            W_new[: k - p + 1] = W[: k - p + 1]
            for i in range(k - p + 1, k + 1):
                denom = U[i + p] - U[i]
                alpha = (u_val - U[i]) / denom if abs(denom) > 1e-15 else 0.0
                W_new[i] = (1.0 - alpha) * W[i - 1] + alpha * W[i]
            W_new[k + 1 :] = W[k:]
            U.insert(k + 1, u_val)
            k += 1
            W = W_new

    # Now every span [breaks[s], breaks[s+1]] has p+1 consecutive rows in W
    # Find starting control index for the first span
    _, k0 = _mult_and_last_idx(breaks[0])
    start_ctrl = k0 - p
    n_spans = len(breaks) - 1
    M = np.zeros((n_spans, p + 1, n_ctrl), dtype=np.float64)
    span_ranges: list[tuple[float, float]] = []
    for s in range(n_spans):
        idx0 = start_ctrl + s * p
        M[s] = W[idx0 : idx0 + p + 1]
        span_ranges.append((breaks[s], breaks[s + 1]))
    return M, span_ranges


class Segment2D:
    """Base class for 2D oriented sketch segments."""

    def endpoints_float(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        raise NotImplementedError

    def eval_unsigned_distance(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        raise NotImplementedError

    def winding_number_contrib(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, n_sub: int = 1
    ) -> torch.Tensor:
        """Exact 1-chord winding angle for straight line segment."""
        poly = self.sample_polyline_tensor(1, params=params, dtype=dtype, device=pts.device)
        return _polyline_winding_contrib(pts, poly)

    def sample_polyline_tensor(
        self, n_sub: int, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        raise NotImplementedError

    def winding_number_contrib(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, n_sub: int = 16
    ) -> torch.Tensor:
        """Exact or high-precision signed angle / (2*pi) contribution."""
        poly = self.sample_polyline_tensor(n_sub, params=params, dtype=dtype, device=pts.device)
        # poly: (M+1, 2) or (B, M+1, 2)
        return _polyline_winding_contrib(pts, poly)

    def subdivided_aabbs(
        self, params: dict[str, Any] | None = None
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        raise NotImplementedError

    def to_occt_edges(self, params: dict[str, Any] | None = None, z: float = 0.0) -> list[Any]:
        raise NotImplementedError


def _polyline_winding_contrib(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Compute sum of signed angles / (2*pi) for polyline (M+1, 2) or (B, M+1, 2) against pts (N, 2) or (B, N, 2)."""
    if poly.ndim == 2:
        p0 = poly[:-1]  # (M, 2)
        p1 = poly[1:]   # (M, 2)
        if pts.ndim == 2:
            v0 = p0[None, :, :] - pts[:, None, :]
            v1 = p1[None, :, :] - pts[:, None, :]
        else:
            v0 = p0[None, None, :, :] - pts[:, :, None, :]
            v1 = p1[None, None, :, :] - pts[:, :, None, :]
    else:
        p0 = poly[:, :-1, :]  # (B, M, 2)
        p1 = poly[:, 1:, :]
        if pts.ndim == 2:
            v0 = p0[:, None, :, :] - pts[None, :, None, :]
            v1 = p1[:, None, :, :] - pts[None, :, None, :]
        else:
            v0 = p0[:, None, :, :] - pts[:, :, None, :]
            v1 = p1[:, None, :, :] - pts[:, :, None, :]
    cross = v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]
    dot = v0[..., 0] * v1[..., 0] + v0[..., 1] * v1[..., 1]
    ang = torch.atan2(cross, dot)
    return torch.sum(ang, dim=-1) / (2.0 * math.pi)


class LineSeg2D(Segment2D):
    """Straight 2D line segment from p0 to p1."""

    def __init__(self, p0: Point2DLike, p1: Point2DLike) -> None:
        self.p0 = _require_pt2d(p0, "p0")
        self.p1 = _require_pt2d(p1, "p1")

    def endpoints_float(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        return _pt2d_float(self.p0, params), _pt2d_float(self.p1, params)

    def eval_unsigned_distance(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        a = _pt2d_tensor(self.p0, params, dtype, pts.device)
        b = _pt2d_tensor(self.p1, params, dtype, pts.device)
        if (a.ndim == 2 or b.ndim == 2) and pts.ndim == 2:
            pts = pts.unsqueeze(0)
        if a.ndim == 2:
            a = a[:, None, :]
        if b.ndim == 2:
            b = b[:, None, :]
        ab = b - a
        ap = pts - a
        denom = torch.sum(ab * ab, dim=-1).clamp(min=1e-24)
        t = torch.clamp(torch.sum(ap * ab, dim=-1) / denom, 0.0, 1.0)
        proj = a + t.unsqueeze(-1) * ab
        return torch.sqrt(torch.sum((pts - proj) ** 2, dim=-1) + 1e-30)

    def sample_polyline_tensor(
        self, n_sub: int, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        a = _pt2d_tensor(self.p0, params, dtype, device)
        b = _pt2d_tensor(self.p1, params, dtype, device)
        m = max(1, n_sub)
        s = torch.linspace(0.0, 1.0, m + 1, dtype=dtype, device=device)
        if a.ndim == 2 or b.ndim == 2:
            B = a.shape[0] if a.ndim == 2 else b.shape[0]
            if a.ndim == 1:
                a = a.unsqueeze(0).expand(B, 2)
            if b.ndim == 1:
                b = b.unsqueeze(0).expand(B, 2)
            return a[:, None, :] + (b - a)[:, None, :] * s[None, :, None]
        return a[None, :] + (b - a)[None, :] * s[:, None]

    def subdivided_aabbs(
        self, params: dict[str, Any] | None = None
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        p0, p1 = self.endpoints_float(params)
        return [((min(p0[0], p1[0]), min(p0[1], p1[1])), (max(p0[0], p1[0]), max(p0[1], p1[1])))]

    def to_occt_edges(self, params: dict[str, Any] | None = None, z: float = 0.0) -> list[Any]:
        p0, p1 = self.endpoints_float(params)
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-9:
            return []
        return [BRepBuilderAPI_MakeEdge(gp_Pnt(p0[0], p0[1], z), gp_Pnt(p1[0], p1[1], z)).Edge()]


class Arc2D(Segment2D):
    """Circular 2D arc defined by center, radius, and start/end angles."""

    def __init__(
        self,
        center: Point2DLike,
        radius: Quantity | Param,
        start_deg: Quantity | Param,
        end_deg: Quantity | Param,
    ) -> None:
        self.center = _require_pt2d(center, "center")
        self.radius = require_length(radius, "radius")
        self.start_deg = require_angle(start_deg, "start_deg")
        self.end_deg = require_angle(end_deg, "end_deg")

    @classmethod
    def from_endpoints(
        cls,
        p0: tuple[float, float],
        p1: tuple[float, float],
        center: tuple[float, float],
        ccw: bool = True,
    ) -> Arc2D:
        r0 = math.hypot(p0[0] - center[0], p0[1] - center[1])
        r1 = math.hypot(p1[0] - center[0], p1[1] - center[1])
        if abs(r0 - r1) > 1e-4 or r0 <= 1e-9:
            raise OutOfDomainError(f"Arc2D.from_endpoints radial mismatch: r0={r0}, r1={r1}")
        a0 = math.degrees(math.atan2(p0[1] - center[1], p0[0] - center[0]))
        a1 = math.degrees(math.atan2(p1[1] - center[1], p1[0] - center[0]))
        if ccw:
            while a1 <= a0:
                a1 += 360.0
        else:
            while a1 >= a0:
                a1 -= 360.0
        return cls((mm(center[0]), mm(center[1])), mm(0.5 * (r0 + r1)), deg(a0), deg(a1))

    def endpoints_float(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = _pt2d_float(self.center, params)
        r = _val_mm_float(self.radius, params)
        a0 = math.radians(_val_deg_float(self.start_deg, params))
        a1 = math.radians(_val_deg_float(self.end_deg, params))
        return (
            (cx + r * math.cos(a0), cy + r * math.sin(a0)),
            (cx + r * math.cos(a1), cy + r * math.sin(a1)),
        )

    def eval_unsigned_distance(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        c = _pt2d_tensor(self.center, params, dtype, pts.device)
        r = _val_mm_tensor(self.radius, params, dtype).to(device=pts.device)
        a0 = _val_rad_tensor(self.start_deg, params, dtype).to(device=pts.device)
        a1 = _val_rad_tensor(self.end_deg, params, dtype).to(device=pts.device)

        has_b = any(t.ndim >= (2 if idx == 0 else 1) for idx, t in enumerate((c, r, a0, a1)))
        if has_b and pts.ndim == 2:
            pts = pts.unsqueeze(0)
        if c.ndim == 2:
            c = c[:, None, :]
        if r.ndim == 1:
            r = r[:, None]
        if a0.ndim == 1:
            a0 = a0[:, None]
        if a1.ndim == 1:
            a1 = a1[:, None]

        rel = pts - c
        rp = torch.sqrt(torch.sum(rel * rel, dim=-1) + 1e-30)
        ang = torch.atan2(rel[..., 1], rel[..., 0])

        delta = a1 - a0
        # Unwrap ang relative to a0 in direction of delta
        diff_ang = torch.remainder(ang - a0, 2.0 * math.pi)
        in_ccw = (delta >= 0.0) & (diff_ang <= delta + 1e-12)
        diff_cw = torch.remainder(a0 - ang, 2.0 * math.pi)
        in_cw = (delta < 0.0) & (diff_cw <= (-delta) + 1e-12)
        in_sector = in_ccw | in_cw | (torch.abs(delta) >= 2.0 * math.pi - 1e-9)

        d_arc = torch.abs(rp - r)
        p0 = torch.stack([c[..., 0] + r * torch.cos(a0), c[..., 1] + r * torch.sin(a0)], dim=-1)
        p1 = torch.stack([c[..., 0] + r * torch.cos(a1), c[..., 1] + r * torch.sin(a1)], dim=-1)
        d0 = torch.sqrt(torch.sum((pts - p0) ** 2, dim=-1) + 1e-30)
        d1 = torch.sqrt(torch.sum((pts - p1) ** 2, dim=-1) + 1e-30)
        return torch.where(in_sector, d_arc, torch.minimum(d0, d1))

    def sample_polyline_tensor(
        self, n_sub: int, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        c = _pt2d_tensor(self.center, params, dtype, device)
        r = _val_mm_tensor(self.radius, params, dtype).to(device=device)
        a0 = _val_rad_tensor(self.start_deg, params, dtype).to(device=device)
        a1 = _val_rad_tensor(self.end_deg, params, dtype).to(device=device)
        m = max(8, n_sub)
        s = torch.linspace(0.0, 1.0, m + 1, dtype=dtype, device=device)
        if any(t.ndim >= (2 if i == 0 else 1) for i, t in enumerate((c, r, a0, a1))):
            B = next(
                t.shape[0]
                for i, t in enumerate((c, r, a0, a1))
                if t.ndim >= (2 if i == 0 else 1)
            )
            if c.ndim == 1:
                c = c.unsqueeze(0).expand(B, 2)
            if r.ndim == 0:
                r = r.unsqueeze(0).expand(B)
            if a0.ndim == 0:
                a0 = a0.unsqueeze(0).expand(B)
            if a1.ndim == 0:
                a1 = a1.unsqueeze(0).expand(B)
            ang = a0[:, None] + (a1 - a0)[:, None] * s[None, :]
            return torch.stack(
                [c[:, None, 0] + r[:, None] * torch.cos(ang), c[:, None, 1] + r[:, None] * torch.sin(ang)],
                dim=-1,
            )
        ang = a0 + (a1 - a0) * s
        return torch.stack([c[0] + r * torch.cos(ang), c[1] + r * torch.sin(ang)], dim=-1)

    def subdivided_aabbs(
        self, params: dict[str, Any] | None = None
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        cx, cy = _pt2d_float(self.center, params)
        r = _val_mm_float(self.radius, params)
        a0 = math.radians(_val_deg_float(self.start_deg, params))
        a1 = math.radians(_val_deg_float(self.end_deg, params))
        # Subdivide arc into 8 sub-arcs, adding exact chord sagitta margin
        n = 8
        ts = np.linspace(a0, a1, n + 1)
        dtheta = abs(a1 - a0) / n
        sag = r * (1.0 - math.cos(0.5 * dtheta))
        out = []
        for i in range(n):
            x0, y0 = cx + r * math.cos(ts[i]), cy + r * math.sin(ts[i])
            x1, y1 = cx + r * math.cos(ts[i + 1]), cy + r * math.sin(ts[i + 1])
            out.append((
                (min(x0, x1) - sag, min(y0, y1) - sag),
                (max(x0, x1) + sag, max(y0, y1) + sag),
            ))
        return out

    def to_occt_edges(self, params: dict[str, Any] | None = None, z: float = 0.0) -> list[Any]:
        cx, cy = _pt2d_float(self.center, params)
        r = _val_mm_float(self.radius, params)
        a0 = math.radians(_val_deg_float(self.start_deg, params))
        a1 = math.radians(_val_deg_float(self.end_deg, params))
        delta = a1 - a0
        if abs(abs(delta) - 2.0 * math.pi) < 1e-6:
            ax = gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0.0, 0.0, 1.0 if delta >= 0 else -1.0))
            return [BRepBuilderAPI_MakeEdge(gp_Circ(ax, r)).Edge()]
        amid = 0.5 * (a0 + a1)
        p_start = gp_Pnt(cx + r * math.cos(a0), cy + r * math.sin(a0), z)
        p_mid = gp_Pnt(cx + r * math.cos(amid), cy + r * math.sin(amid), z)
        p_end = gp_Pnt(cx + r * math.cos(a1), cy + r * math.sin(a1), z)
        arc = GC_MakeArcOfCircle(p_start, p_mid, p_end).Value()
        return [BRepBuilderAPI_MakeEdge(arc).Edge()]


class _BezierSpans2DBase(Segment2D):
    """Shared base for 2D Quadratic/Cubic Bézier and B-Spline curves."""

    def _bezier_ctrl_tensor(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError

    def _bezier_ctrl_np(self, params: dict[str, Any] | None) -> np.ndarray:
        return self._bezier_ctrl_tensor(params, torch.float64, torch.device("cpu")).detach().numpy()

    def eval_unsigned_distance(
        self, pts: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        ctrl = self._bezier_ctrl_tensor(params, dtype, pts.device)
        return _Bezier2DDistanceFn.apply(pts, ctrl)

    def sample_polyline_tensor(
        self, n_sub: int, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        ctrl = self._bezier_ctrl_tensor(params, dtype, device)  # (S, K, 2) or (B, S, K, 2)
        deg_k = ctrl.shape[-2] - 1
        m = max(4, n_sub)
        t = np.linspace(0.0, 1.0, m + 1, dtype=np.float64)
        B_mat = torch.from_numpy(_bernstein_basis_np(deg_k, t)).to(dtype=dtype, device=device)  # (m+1, K)
        if ctrl.ndim == 3:
            pts = torch.einsum("mk,skd->smd", B_mat, ctrl)  # (S, m+1, 2)
            # Concatenate spans dropping duplicate seam points
            parts = [pts[0]] + [pts[s, 1:] for s in range(1, pts.shape[0])]
            return torch.cat(parts, dim=0)
        pts = torch.einsum("mk,bskd->bsmd", B_mat, ctrl)
        parts = [pts[:, 0]] + [pts[:, s, 1:] for s in range(1, pts.shape[1])]
        return torch.cat(parts, dim=1)

    def subdivided_aabbs(
        self, params: dict[str, Any] | None = None
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        ctrl = self._bezier_ctrl_np(params)
        sub = _subdivide_bezier_ctrl(ctrl, depth=4)
        aabbs = []
        for s in range(sub.shape[0]):
            mn = sub[s].min(axis=0)
            mx = sub[s].max(axis=0)
            aabbs.append(((float(mn[0]), float(mn[1])), (float(mx[0]), float(mx[1]))))
        return aabbs

    def to_occt_edges(self, params: dict[str, Any] | None = None, z: float = 0.0) -> list[Any]:
        ctrl = self._bezier_ctrl_np(params)
        S, K, _ = ctrl.shape
        edges = []
        for s in range(S):
            arr = Array1_gp_Pnt(1, K)
            for k in range(K):
                arr.SetValue(k + 1, gp_Pnt(float(ctrl[s, k, 0]), float(ctrl[s, k, 1]), z))
            curve = Geom_BezierCurve(arr)
            edges.append(BRepBuilderAPI_MakeEdge(curve).Edge())
        return edges


class QuadBezier2D(_BezierSpans2DBase):
    """Quadratic 2D Bézier segment."""

    def __init__(self, p0: Point2DLike, p1: Point2DLike, p2: Point2DLike) -> None:
        self.pts = [_require_pt2d(p0, "p0"), _require_pt2d(p1, "p1"), _require_pt2d(p2, "p2")]

    def endpoints_float(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        return _pt2d_float(self.pts[0], params), _pt2d_float(self.pts[-1], params)

    def _bezier_ctrl_tensor(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        stacked = _stack_pts2d_tensor(self.pts, params, dtype, device)
        return stacked.unsqueeze(-3)  # (1, 3, 2) or (B, 1, 3, 2)


class CubicBezier2D(_BezierSpans2DBase):
    """Cubic 2D Bézier segment."""

    def __init__(self, p0: Point2DLike, p1: Point2DLike, p2: Point2DLike, p3: Point2DLike) -> None:
        self.pts = [
            _require_pt2d(p0, "p0"),
            _require_pt2d(p1, "p1"),
            _require_pt2d(p2, "p2"),
            _require_pt2d(p3, "p3"),
        ]

    def endpoints_float(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        return _pt2d_float(self.pts[0], params), _pt2d_float(self.pts[-1], params)

    def _bezier_ctrl_tensor(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        stacked = _stack_pts2d_tensor(self.pts, params, dtype, device)
        return stacked.unsqueeze(-3)


class BSpline2D(_BezierSpans2DBase):
    """2D clamped or periodic B-Spline with exact Boehm Bézier extraction matrix (autograd-safe)."""

    def __init__(
        self,
        control_points: Sequence[Point2DLike],
        degree: int = 3,
        knots: Sequence[float] | None = None,
        closed_periodic: bool = False,
    ) -> None:
        if degree not in (2, 3):
            raise ValueError("BSpline2D supports degree 2 or 3")
        n = len(control_points)
        if n < degree + 1:
            raise ValueError(f"BSpline2D degree={degree} requires at least {degree + 1} control points")
        self.control_points = [_require_pt2d(p, f"ctrl[{i}]") for i, p in enumerate(control_points)]
        self.degree = int(degree)
        self.closed_periodic = bool(closed_periodic)

        if closed_periodic:
            # Uniform periodic B-spline-to-Bezier constant stencil per span
            if degree == 2:
                C = np.array(
                    [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
                    dtype=np.float64,
                )
            else:
                C = np.array(
                    [
                        [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0],
                        [0.0, 4.0 / 6.0, 2.0 / 6.0, 0.0],
                        [0.0, 2.0 / 6.0, 4.0 / 6.0, 0.0],
                        [0.0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0],
                    ],
                    dtype=np.float64,
                )
            M = np.zeros((n, degree + 1, n), dtype=np.float64)
            for s in range(n):
                for j in range(degree + 1):
                    M[s, :, (s + j) % n] += C[:, j]
            self._extract_mat = M
        else:
            if knots is None:
                n_internal = n - degree - 1
                internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1] if n_internal > 0 else np.array([])
                u_arr = np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])
            else:
                u_arr = np.asarray(knots, dtype=np.float64)
                if len(u_arr) != n + degree + 1:
                    raise ValueError(f"Expected {n + degree + 1} knots, got {len(u_arr)}")
            self._extract_mat, _ = _boehm_extraction_matrix(n, degree, u_arr)

    def endpoints_float(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        ctrl = self._bezier_ctrl_np(params)
        return (float(ctrl[0, 0, 0]), float(ctrl[0, 0, 1])), (float(ctrl[-1, -1, 0]), float(ctrl[-1, -1, 1]))

    def _bezier_ctrl_tensor(
        self, params: dict[str, Any] | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        q = _stack_pts2d_tensor(self.control_points, params, dtype, device)  # (N_c, 2) or (B, N_c, 2)
        M = torch.from_numpy(self._extract_mat).to(dtype=dtype, device=device)  # (S, K, N_c)
        if q.ndim == 2:
            return torch.einsum("skc,cd->skd", M, q)
        return torch.einsum("skc,bcd->bskd", M, q)


class Loop2D:
    """Closed 2D contour made of Segment2D pieces."""

    def __init__(self, segments: Sequence[Segment2D], check_closure: bool = True) -> None:
        if len(segments) < 1:
            raise ValueError("Loop2D requires at least 1 segment")
        self.segments = list(segments)
        self._cached_aabbs_np: np.ndarray | None = None
        if check_closure:
            self.validate_closure()

    def validate_closure(self, params: dict[str, Any] | None = None, tol: float = 1e-4) -> None:
        n = len(self.segments)
        for i in range(n):
            _, end_i = self.segments[i].endpoints_float(params)
            start_next, _ = self.segments[(i + 1) % n].endpoints_float(params)
            gap = math.hypot(end_i[0] - start_next[0], end_i[1] - start_next[1])
            if gap > tol:
                raise OutOfDomainError(f"Loop2D not closed at segment {i}: gap={gap:.6g} mm")

    @classmethod
    def circle(cls, center: Point2DLike, radius: Quantity | Param) -> Loop2D:
        return cls([Arc2D(center, radius, deg(0.0), deg(360.0))], check_closure=True)

    @classmethod
    def rounded_rect(
        cls, lx: Quantity | Param, ly: Quantity | Param, fillet_r: Quantity | Param
    ) -> Loop2D:
        hx = 0.5 * _val_mm_float(require_length(lx, "lx"), None)
        hy = 0.5 * _val_mm_float(require_length(ly, "ly"), None)
        r = _val_mm_float(require_length(fillet_r, "fillet_r"), None)
        if r <= 1e-9:
            return cls.polygon([(mm(hx), mm(-hy)), (mm(hx), mm(hy)), (mm(-hx), mm(hy)), (mm(-hx), mm(-hy))])
        ix, iy = hx - r, hy - r
        segs: list[Segment2D] = []
        if iy > 1e-9:
            segs.append(LineSeg2D((mm(hx), mm(-iy)), (mm(hx), mm(iy))))
        segs.append(Arc2D((mm(ix), mm(iy)), mm(r), deg(0.0), deg(90.0)))
        if ix > 1e-9:
            segs.append(LineSeg2D((mm(ix), mm(hy)), (mm(-ix), mm(hy))))
        segs.append(Arc2D((mm(-ix), mm(iy)), mm(r), deg(90.0), deg(180.0)))
        if iy > 1e-9:
            segs.append(LineSeg2D((mm(-hx), mm(iy)), (mm(-hx), mm(-iy))))
        segs.append(Arc2D((mm(-ix), mm(-iy)), mm(r), deg(180.0), deg(270.0)))
        if ix > 1e-9:
            segs.append(LineSeg2D((mm(-ix), mm(-hy)), (mm(ix), mm(-hy))))
        segs.append(Arc2D((mm(ix), mm(-iy)), mm(r), deg(270.0), deg(360.0)))
        return cls(segs)

    @classmethod
    def polygon(
        cls,
        points: Sequence[Point2DLike | tuple[float, float]],
        fillet_r: float | Quantity | None = None,
        chamfer_d: float | Quantity | None = None,
    ) -> Loop2D:
        if fillet_r is not None and chamfer_d is not None:
            raise ValueError("Specify at most one of fillet_r or chamfer_d")
        pts_q: list[Point2DLike] = []
        for p in points:
            if isinstance(p[0], (int, float)):
                pts_q.append((mm(float(p[0])), mm(float(p[1]))))
            else:
                pts_q.append(_require_pt2d(p))  # type: ignore[arg-type]

        if fillet_r is None and chamfer_d is None:
            n = len(pts_q)
            return cls([LineSeg2D(pts_q[i], pts_q[(i + 1) % n]) for i in range(n)])

        pts_f = [_pt2d_float(p, None) for p in pts_q]
        n = len(pts_f)
        if fillet_r is not None:
            rad = fillet_r.as_mm() if isinstance(fillet_r, Quantity) else float(fillet_r)
        else:
            rad = 0.0
        if chamfer_d is not None:
            ch = chamfer_d.as_mm() if isinstance(chamfer_d, Quantity) else float(chamfer_d)
        else:
            ch = 0.0

        segs: list[Segment2D] = []
        trim_starts: list[tuple[float, float]] = []
        trim_ends: list[tuple[float, float]] = []
        corner_segs: list[Segment2D | None] = []

        for i in range(n):
            p_prev = pts_f[(i - 1) % n]
            p_cur = pts_f[i]
            p_next = pts_f[(i + 1) % n]
            u = (p_prev[0] - p_cur[0], p_prev[1] - p_cur[1])
            v = (p_next[0] - p_cur[0], p_next[1] - p_cur[1])
            lu = math.hypot(u[0], u[1])
            lv = math.hypot(v[0], v[1])
            if lu < 1e-9 or lv < 1e-9:
                raise OutOfDomainError("Zero-length polygon edge")
            u = (u[0] / lu, u[1] / lu)
            v = (v[0] / lv, v[1] / lv)
            dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
            ang = math.acos(dot)
            if ch > 0.0:
                d_trim = ch
                if d_trim >= 0.49 * min(lu, lv):
                    raise OutOfDomainError("Chamfer too large for polygon edge")
                a_pt = (p_cur[0] + d_trim * u[0], p_cur[1] + d_trim * u[1])
                b_pt = (p_cur[0] + d_trim * v[0], p_cur[1] + d_trim * v[1])
                trim_starts.append(a_pt)
                trim_ends.append(b_pt)
                corner_segs.append(LineSeg2D((mm(a_pt[0]), mm(a_pt[1])), (mm(b_pt[0]), mm(b_pt[1]))))
            elif rad > 0.0 and ang < math.pi - 1e-4:
                d_trim = rad / math.tan(0.5 * ang)
                if d_trim >= 0.49 * min(lu, lv):
                    raise OutOfDomainError("Fillet radius too large for polygon edge")
                a_pt = (p_cur[0] + d_trim * u[0], p_cur[1] + d_trim * u[1])
                b_pt = (p_cur[0] + d_trim * v[0], p_cur[1] + d_trim * v[1])
                bis = (u[0] + v[0], u[1] + v[1])
                lb = math.hypot(bis[0], bis[1])
                bis = (bis[0] / lb, bis[1] / lb)
                dist_c = rad / math.sin(0.5 * ang)
                center = (p_cur[0] + dist_c * bis[0], p_cur[1] + dist_c * bis[1])
                cross = u[0] * v[1] - u[1] * v[0]
                # If u x v < 0 (left turn from prev->cur to cur->next), arc is CCW
                ccw = cross < 0.0
                trim_starts.append(a_pt)
                trim_ends.append(b_pt)
                corner_segs.append(Arc2D.from_endpoints(a_pt, b_pt, center, ccw=ccw))
            else:
                trim_starts.append(p_cur)
                trim_ends.append(p_cur)
                corner_segs.append(None)

        for i in range(n):
            if corner_segs[i] is not None:
                segs.append(corner_segs[i])  # type: ignore[arg-type]
            p_from = trim_ends[i]
            p_to = trim_starts[(i + 1) % n]
            if math.hypot(p_to[0] - p_from[0], p_to[1] - p_from[1]) > 1e-7:
                segs.append(LineSeg2D((mm(p_from[0]), mm(p_from[1])), (mm(p_to[0]), mm(p_to[1]))))
        return cls(segs)

    @classmethod
    def closed_bspline(cls, control_points: Sequence[Point2DLike], degree: int = 3) -> Loop2D:
        return cls([BSpline2D(control_points, degree=degree, closed_periodic=True)], check_closure=True)

    def signed_area(self, params: dict[str, Any] | None = None) -> float:
        polys = [
            seg.sample_polyline_tensor(16, params=params, dtype=torch.float64).detach().numpy()
            for seg in self.segments
        ]
        pts = np.concatenate([p[:-1] for p in polys], axis=0)
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def _aabbs_np(self, params: dict[str, Any] | None = None) -> np.ndarray:
        if params is None and self._cached_aabbs_np is not None:
            return self._cached_aabbs_np
        aabbs = []
        for seg in self.segments:
            for smin, smax in seg.subdivided_aabbs(params):
                aabbs.append((smin[0], smin[1], smax[0], smax[1]))
        arr = np.asarray(aabbs, dtype=np.float64)
        if params is None:
            self._cached_aabbs_np = arr
        return arr

    def bounds_2d(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        arr = self._aabbs_np(params)
        return (
            (float(np.min(arr[:, 0])), float(np.min(arr[:, 1]))),
            (float(np.max(arr[:, 2])), float(np.max(arr[:, 3]))),
        )

    def eval_sdf_2d(
        self, pts_2d: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        seg_dists = [seg.eval_unsigned_distance(pts_2d, params=params, dtype=dtype) for seg in self.segments]
        if any(d.ndim == 2 for d in seg_dists):
            B = next(d.shape[0] for d in seg_dists if d.ndim == 2)
            seg_dists = [d if d.ndim == 2 else d.unsqueeze(0).expand(B, -1) for d in seg_dists]
        dists = torch.stack(seg_dists, dim=0)
        unsigned_d = torch.amin(dists, dim=0)
        with torch.no_grad():
            if pts_2d.ndim == 2 and unsigned_d.ndim == 1:
                bmin, bmax = self.bounds_2d(params)
                in_box = (
                    (pts_2d[:, 0] >= bmin[0] - 1e-9)
                    & (pts_2d[:, 0] <= bmax[0] + 1e-9)
                    & (pts_2d[:, 1] >= bmin[1] - 1e-9)
                    & (pts_2d[:, 1] <= bmax[1] + 1e-9)
                )
                sign = torch.ones_like(unsigned_d)
                if torch.any(in_box):
                    p_in = pts_2d[in_box]
                    w_coarse = torch.zeros(p_in.shape[0], dtype=dtype, device=pts_2d.device)
                    for seg in self.segments:
                        w_coarse = w_coarse + seg.winding_number_contrib(p_in, params=params, dtype=dtype, n_sub=16)
                    # Refine near curved boundary where chord sagitta could flip winding sign
                    near_bnd = unsigned_d[in_box] < 0.25
                    if torch.any(near_bnd):
                        p_near = p_in[near_bnd]
                        w_fine = torch.zeros(p_near.shape[0], dtype=dtype, device=pts_2d.device)
                        for seg in self.segments:
                            w_fine = w_fine + seg.winding_number_contrib(p_near, params=params, dtype=dtype, n_sub=64)
                        w_coarse[near_bnd] = w_fine
                    sign[in_box] = torch.where(torch.abs(w_coarse) > 0.5, -1.0, 1.0).to(dtype=dtype)
            else:
                winding = torch.zeros_like(unsigned_d)
                for seg in self.segments:
                    w = seg.winding_number_contrib(pts_2d, params=params, dtype=dtype, n_sub=16)
                    if w.ndim < winding.ndim:
                        w = w.unsqueeze(0).expand_as(winding)
                    winding = winding + w
                sign = torch.where(torch.abs(winding) > 0.5, -1.0, 1.0).to(dtype=dtype)
        return unsigned_d * sign

    def interval_bound_2d(
        self,
        box_min: tuple[float, float],
        box_max: tuple[float, float],
        params: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        arr = self._aabbs_np(params)
        dx = np.maximum(np.maximum(arr[:, 0] - box_max[0], box_min[0] - arr[:, 2]), 0.0)
        dy = np.maximum(np.maximum(arr[:, 1] - box_max[1], box_min[1] - arr[:, 3]), 0.0)
        d_lb = float(np.min(np.hypot(dx, dy)))
        cx = 0.5 * (box_min[0] + box_max[0])
        cy = 0.5 * (box_min[1] + box_max[1])
        half_diag = 0.5 * math.hypot(box_max[0] - box_min[0], box_max[1] - box_min[1])
        fc = float(
            self.eval_sdf_2d(
                torch.tensor([[cx, cy]], dtype=torch.float64), params=params, dtype=torch.float64
            )[0].item()
        )
        lo_lip = fc - half_diag
        hi_lip = fc + half_diag
        if d_lb > 0.0:
            if fc < 0.0:
                return lo_lip, min(hi_lip, -d_lb)
            return max(lo_lip, d_lb), hi_lip
        return lo_lip, hi_lip

    def sample_boundary_points(
        self, n_per_seg: int = 24, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        pts_list = [
            seg.sample_polyline_tensor(n_per_seg, params=params, dtype=dtype, device=device)
            for seg in self.segments
        ]
        if any(p.ndim == 3 for p in pts_list):
            B = next(p.shape[0] for p in pts_list if p.ndim == 3)
            expanded = [
                p[:, :-1, :] if p.ndim == 3 else p[:-1].unsqueeze(0).expand(B, -1, 2)
                for p in pts_list
            ]
            return torch.cat(expanded, dim=1)
        return torch.cat([p[:-1] for p in pts_list], dim=0)

    def to_occt_wire(self, params: dict[str, Any] | None = None, z: float = 0.0) -> TopoDS_Wire:
        wb = BRepBuilderAPI_MakeWire()
        for seg in self.segments:
            for edge in seg.to_occt_edges(params=params, z=z):
                wb.Add(edge)
        if not wb.IsDone():
            raise RuntimeError("OCCT BRepBuilderAPI_MakeWire failed for Loop2D")
        return wb.Wire()


@dataclass
class Sketch2D:
    """2D parametric sketch with outer loop and optional inner hole loops (L=1)."""

    outer: Loop2D
    holes: list[Loop2D] | None = None

    def __post_init__(self) -> None:
        if self.holes is None:
            self.holes = []

    def bounds_2d(self, params: dict[str, Any] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.outer.bounds_2d(params)

    def eval_sdf_2d(
        self, pts_2d: torch.Tensor, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        f = self.outer.eval_sdf_2d(pts_2d, params=params, dtype=dtype)
        for h in self.holes or []:
            fh = h.eval_sdf_2d(pts_2d, params=params, dtype=dtype)
            f = torch.maximum(f, -fh)
        return f

    def interval_bound_2d(
        self,
        box_min: tuple[float, float],
        box_max: tuple[float, float],
        params: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        lo, hi = self.outer.interval_bound_2d(box_min, box_max, params=params)
        for h in self.holes or []:
            h_lo, h_hi = h.interval_bound_2d(box_min, box_max, params=params)
            lo = max(lo, -h_hi)
            hi = max(hi, -h_lo)
        return lo, hi

    def sample_boundary_points(
        self, n_per_seg: int = 24, params: dict[str, Any] | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        parts = [self.outer.sample_boundary_points(n_per_seg, params=params, dtype=dtype, device=device)]
        for h in self.holes or []:
            parts.append(h.sample_boundary_points(n_per_seg, params=params, dtype=dtype, device=device))
        dim = 1 if parts[0].ndim == 3 else 0
        return torch.cat(parts, dim=dim)

    def to_occt_face(self, params: dict[str, Any] | None = None, z: float = 0.0) -> TopoDS_Face:
        outer_wire = self.outer.to_occt_wire(params=params, z=z)
        if self.outer.signed_area(params) < 0.0:
            outer_wire = TopoDS.Wire(outer_wire.Reversed())
        fb = BRepBuilderAPI_MakeFace(outer_wire, True)
        for h in self.holes or []:
            hw = h.to_occt_wire(params=params, z=z)
            if h.signed_area(params) > 0.0:
                hw = TopoDS.Wire(hw.Reversed())
            fb.Add(hw)
        if not fb.IsDone():
            raise RuntimeError("OCCT BRepBuilderAPI_MakeFace failed for Sketch2D")
        face = fb.Face()
        if not BRepCheck_Analyzer(face).IsValid():
            raise RuntimeError("Compiled Sketch2D OCCT face failed BRepCheck_Analyzer")
        return face
