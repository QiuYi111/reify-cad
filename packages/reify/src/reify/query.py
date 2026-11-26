"""GeometryQuery unified protocol: scale-aware bounds, inside/outside split, surface covering, and Hausdorff."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_OUT, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

from reify.affine import eval_affine_box, eval_affine_form
from reify.ast import (
    Box,
    Cylinder,
    ExtrudeRoundedRect,
    FreeImplicit,
    Intersect,
    Node,
    Param,
    Part,
    Scale,
    SmoothUnion,
    Sphere,
    Subtract,
    Transform,
    Union,
    _val_deg_float,
    _val_mm_float,
    _val_mm_tensor,
    _val_rad_tensor,
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
from reify.types import (
    DistanceResult,
    FailReason,
    GradientStatus,
    GuaranteeLevel,
    NumericalBudget,
    OutOfDomainError,
    PointClass,
    ProvenanceResult,
    Status,
    SurfaceCover,
    VerificationReport,
)


def _iter_transformed_leaves(
    node: Node,
    params: dict[str, Any] | None,
    R_acc: torch.Tensor,
    t_acc: torch.Tensor,
):
    """Yield (leaf_node, R_world, t_world) for rigid-transform CSG trees."""
    if isinstance(node, Transform):
        R_n, t_n = node.world_matrix_tensors(params, dtype=torch.float64)
        R_new = R_acc @ R_n
        t_new = R_acc @ t_n + t_acc
        yield from _iter_transformed_leaves(node.child, params, R_new, t_new)
    elif hasattr(node, "R") and hasattr(node, "t") and hasattr(node, "child"):
        R_new = R_acc @ node.R
        t_new = R_acc @ node.t + t_acc
        yield from _iter_transformed_leaves(node.child, params, R_new, t_new)
    elif isinstance(node, (Union, Intersect)):
        for ch in node.children:
            yield from _iter_transformed_leaves(ch, params, R_acc, t_acc)
    elif isinstance(node, Subtract):
        yield from _iter_transformed_leaves(node.base, params, R_acc, t_acc)
        yield from _iter_transformed_leaves(node.tool, params, R_acc, t_acc)
    elif isinstance(node, SmoothUnion):
        yield from _iter_transformed_leaves(node.a, params, R_acc, t_acc)
        yield from _iter_transformed_leaves(node.b, params, R_acc, t_acc)
    elif isinstance(node, LinearPattern):
        sx = _val_mm_float(node.spacing[0], params)
        sy = _val_mm_float(node.spacing[1], params)
        sz = _val_mm_float(node.spacing[2], params)
        for k in range(node.count):
            t_shift = torch.tensor([k * sx, k * sy, k * sz], dtype=torch.float64)
            yield from _iter_transformed_leaves(node.child, params, R_acc, R_acc @ t_shift + t_acc)
    elif isinstance(node, CircularPattern):
        rad = _val_mm_float(node.radius, params)
        a0 = math.radians(_val_deg_float(node.start_angle_deg, params))
        for k in range(node.count):
            ang = a0 + (2.0 * math.pi * k / node.count)
            ca, sa = math.cos(ang), math.sin(ang)
            R_k = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
            t_k = torch.tensor([rad * ca, rad * sa, 0.0], dtype=torch.float64)
            yield from _iter_transformed_leaves(node.child, params, R_acc @ R_k, R_acc @ t_k + t_acc)
    else:
        yield node, R_acc, t_acc


def _tile_rect_2d(
    hx: float, hy: float, max_radius: float
) -> tuple[np.ndarray, float]:
    """Tile [-hx, hx] x [-hy, hy] with cells of circumradius <= max_radius."""
    cell_side = max(max_radius * math.sqrt(2.0), 1e-3)
    nx = max(1, int(math.ceil(2.0 * hx / cell_side)))
    ny = max(1, int(math.ceil(2.0 * hy / cell_side)))
    dx = (2.0 * hx) / nx
    dy = (2.0 * hy) / ny
    r = 0.5 * math.sqrt(dx * dx + dy * dy)
    xs = np.linspace(-hx + 0.5 * dx, hx - 0.5 * dx, nx)
    ys = np.linspace(-hy + 0.5 * dy, hy - 0.5 * dy, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    return pts, r


def _tile_leaf_surface(
    leaf: Node, max_radius: float, params: dict[str, Any] | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate static boundary covering patches (centers, radii, normals) in leaf local frame."""
    centers_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []

    if isinstance(leaf, Box):
        hx = 0.5 * _val_mm_float(leaf.lx, params)
        hy = 0.5 * _val_mm_float(leaf.ly, params)
        hz = 0.5 * _val_mm_float(leaf.lz, params)
        faces = [
            (2, +hz, hx, hy, (0.0, 0.0, 1.0)),
            (2, -hz, hx, hy, (0.0, 0.0, -1.0)),
            (0, +hx, hy, hz, (1.0, 0.0, 0.0)),
            (0, -hx, hy, hz, (-1.0, 0.0, 0.0)),
            (1, +hy, hx, hz, (0.0, 1.0, 0.0)),
            (1, -hy, hx, hz, (0.0, -1.0, 0.0)),
        ]
        for axis, offset, ha, hb, nrm in faces:
            uv, rad = _tile_rect_2d(ha, hb, max_radius)
            pts = np.zeros((uv.shape[0], 3), dtype=np.float64)
            if axis == 2:
                pts[:, 0], pts[:, 1], pts[:, 2] = uv[:, 0], uv[:, 1], offset
            elif axis == 0:
                pts[:, 0], pts[:, 1], pts[:, 2] = offset, uv[:, 0], uv[:, 1]
            else:
                pts[:, 0], pts[:, 1], pts[:, 2] = uv[:, 0], offset, uv[:, 1]
            centers_list.append(pts)
            radii_list.append(np.full(uv.shape[0], rad, dtype=np.float64))
            normals_list.append(np.tile(np.array(nrm, dtype=np.float64), (uv.shape[0], 1)))

    elif isinstance(leaf, Cylinder):
        r = _val_mm_float(leaf.radius, params)
        hz = 0.5 * _val_mm_float(leaf.height, params)
        # Top and bottom caps (disk inside [-r, r]^2)
        uv, rad = _tile_rect_2d(r, r, max_radius)
        mask = (uv[:, 0] ** 2 + uv[:, 1] ** 2) <= (r + rad) ** 2
        uv = uv[mask]
        for sign in (+1.0, -1.0):
            pts = np.column_stack([uv, np.full(uv.shape[0], sign * hz)])
            centers_list.append(pts)
            radii_list.append(np.full(uv.shape[0], rad, dtype=np.float64))
            normals_list.append(np.tile([0.0, 0.0, sign], (uv.shape[0], 1)))
        # Lateral wall
        circ = 2.0 * math.pi * max(r, 1e-6)
        side = max(max_radius * math.sqrt(2.0), 1e-3)
        n_theta = max(8, int(math.ceil(circ / side)))
        n_z = max(1, int(math.ceil(2.0 * hz / side)))
        dtheta = 2.0 * math.pi / n_theta
        dz = 2.0 * hz / n_z
        arc_chord = 2.0 * r * math.sin(0.5 * dtheta)
        lat_rad = 0.5 * math.sqrt(arc_chord * arc_chord + dz * dz) + r * (1.0 - math.cos(0.5 * dtheta))
        thetas = np.linspace(0.5 * dtheta, 2.0 * math.pi - 0.5 * dtheta, n_theta)
        zs = np.linspace(-hz + 0.5 * dz, hz - 0.5 * dz, n_z)
        gt, gz = np.meshgrid(thetas, zs, indexing="ij")
        ct, st = np.cos(gt.ravel()), np.sin(gt.ravel())
        pts = np.column_stack([r * ct, r * st, gz.ravel()])
        nrms = np.column_stack([ct, st, np.zeros_like(ct)])
        centers_list.append(pts)
        radii_list.append(np.full(pts.shape[0], lat_rad, dtype=np.float64))
        normals_list.append(nrms)

    elif isinstance(leaf, Sphere):
        r = _val_mm_float(leaf.radius, params)
        side = max(max_radius, 1e-3)
        n_lat = max(6, int(math.ceil(math.pi * max(r, 1e-6) / side)))
        n_lon = max(12, int(math.ceil(2.0 * math.pi * max(r, 1e-6) / side)))
        dphi = math.pi / n_lat
        dtheta = 2.0 * math.pi / n_lon
        patch_rad = r * math.sqrt((0.5 * dphi) ** 2 + (0.5 * dtheta) ** 2) * 1.15
        phis = np.linspace(0.5 * dphi, math.pi - 0.5 * dphi, n_lat)
        thetas = np.linspace(0.5 * dtheta, 2.0 * math.pi - 0.5 * dtheta, n_lon)
        gp, gt = np.meshgrid(phis, thetas, indexing="ij")
        sp, cp = np.sin(gp.ravel()), np.cos(gp.ravel())
        ct, st = np.cos(gt.ravel()), np.sin(gt.ravel())
        nrms = np.column_stack([sp * ct, sp * st, cp])
        centers_list.append(r * nrms)
        radii_list.append(np.full(nrms.shape[0], patch_rad, dtype=np.float64))
        normals_list.append(nrms)

    elif isinstance(leaf, ExtrudeRoundedRect):
        hx = 0.5 * _val_mm_float(leaf.lx, params)
        hy = 0.5 * _val_mm_float(leaf.ly, params)
        hz = 0.5 * _val_mm_float(leaf.height, params)
        r = _val_mm_float(leaf.fillet_r, params)
        # Caps
        uv, rad = _tile_rect_2d(hx, hy, max_radius)
        for sign in (+1.0, -1.0):
            pts = np.column_stack([uv, np.full(uv.shape[0], sign * hz)])
            centers_list.append(pts)
            radii_list.append(np.full(uv.shape[0], rad, dtype=np.float64))
            normals_list.append(np.tile([0.0, 0.0, sign], (uv.shape[0], 1)))
        # Side perimeter sampled uniformly
        perim = 4.0 * (hx - r + hy - r) + 2.0 * math.pi * r
        side = max(max_radius * math.sqrt(2.0), 1e-3)
        n_p = max(16, int(math.ceil(perim / side)))
        n_z = max(1, int(math.ceil(2.0 * hz / side)))
        dz = 2.0 * hz / n_z
        # Sample rounded rectangle perimeter via 4 corners + 4 flats
        pts_2d: list[tuple[float, float, float, float]] = []
        ix, iy = max(hx - r, 0.0), max(hy - r, 0.0)
        # Straight segments
        for (x0, y0, x1, y1, nx, ny) in [
            (ix, -iy, ix, iy, 1.0, 0.0),
            (-ix, iy, ix, iy, 0.0, 1.0),
            (-ix, iy, -ix, -iy, -1.0, 0.0),
            (-ix, -iy, ix, -iy, 0.0, -1.0),
        ]:
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len > 1e-9:
                ns = max(1, int(math.ceil(seg_len / side)))
                for k in range(ns):
                    t = (k + 0.5) / ns
                    px = (1 - t) * x0 + t * x1 + nx * r
                    py = (1 - t) * y0 + t * y1 + ny * r
                    pts_2d.append((px, py, nx, ny))
        # Corner arcs
        if r > 1e-9:
            nc = max(2, int(math.ceil(0.5 * math.pi * r / side)))
            for cx, cy, a0 in [(ix, iy, 0.0), (-ix, iy, 0.5 * math.pi), (-ix, -iy, math.pi), (ix, -iy, 1.5 * math.pi)]:
                for k in range(nc):
                    a = a0 + (k + 0.5) / nc * (0.5 * math.pi)
                    ca, sa = math.cos(a), math.sin(a)
                    pts_2d.append((cx + r * ca, cy + r * sa, ca, sa))
        p2 = np.array(pts_2d, dtype=np.float64)
        zs = np.linspace(-hz + 0.5 * dz, hz - 0.5 * dz, n_z)
        lat_rad = 0.5 * math.sqrt(side * side + dz * dz) * 1.15
        for zv in zs:
            pts = np.column_stack([p2[:, 0], p2[:, 1], np.full(len(p2), zv)])
            nrms = np.column_stack([p2[:, 2], p2[:, 3], np.zeros(len(p2))])
            centers_list.append(pts)
            radii_list.append(np.full(len(p2), lat_rad, dtype=np.float64))
            normals_list.append(nrms)
    elif isinstance(leaf, (ExtrudeSketch, TaperedExtrude)):
        bmin2, bmax2 = leaf.sketch.bounds_2d(params)
        hx = 0.5 * (bmax2[0] - bmin2[0])
        hy = 0.5 * (bmax2[1] - bmin2[1])
        cx2 = 0.5 * (bmin2[0] + bmax2[0])
        cy2 = 0.5 * (bmin2[1] + bmax2[1])
        hz = 0.5 * _val_mm_float(leaf.height, params)
        uv, rad = _tile_rect_2d(max(hx, 1e-3), max(hy, 1e-3), max_radius)
        uv[:, 0] += cx2
        uv[:, 1] += cy2
        for sign in (+1.0, -1.0):
            pts = np.column_stack([uv, np.full(uv.shape[0], sign * hz)])
            centers_list.append(pts)
            radii_list.append(np.full(uv.shape[0], rad, dtype=np.float64))
            normals_list.append(np.tile([0.0, 0.0, sign], (uv.shape[0], 1)))
        bnd2 = leaf.sketch.sample_boundary_points(n_per_seg=max(16, int(math.ceil(4.0 / max(max_radius, 0.1)))), params=params, dtype=torch.float64).detach().numpy()
        side = max(max_radius * math.sqrt(2.0), 1e-3)
        n_z = max(1, int(math.ceil(2.0 * hz / side)))
        dz = 2.0 * hz / n_z
        zs = np.linspace(-hz + 0.5 * dz, hz - 0.5 * dz, n_z)
        tan_t = math.tan(math.radians(_val_deg_float(leaf.draft_angle, params))) if isinstance(leaf, TaperedExtrude) else 0.0
        lat_rad = max(0.5 * math.sqrt(side * side + dz * dz) * 1.25, 0.5)
        # Compute true outward 2D normals from boundary tangents
        bnd2_next = np.roll(bnd2, -1, axis=0)
        bnd2_prev = np.roll(bnd2, 1, axis=0)
        tang2 = bnd2_next - bnd2_prev
        tnorm2 = np.maximum(np.linalg.norm(tang2, axis=-1, keepdims=True), 1e-9)
        tang2 = tang2 / tnorm2
        # For CCW loop, outward normal is (ty, -tx)
        out_n2 = np.column_stack([tang2[:, 1], -tang2[:, 0]])
        cos_t = math.cos(math.radians(_val_deg_float(leaf.draft_angle, params))) if isinstance(leaf, TaperedExtrude) else 1.0
        sin_t = math.sin(math.radians(_val_deg_float(leaf.draft_angle, params))) if isinstance(leaf, TaperedExtrude) else 0.0
        nrm_lat = np.column_stack([out_n2 * cos_t, np.full(len(bnd2), sin_t)])
        for zv in zs:
            pts = np.column_stack([bnd2[:, 0], bnd2[:, 1], np.full(len(bnd2), zv)])
            if tan_t > 0.0:
                inset = (zv + hz) * tan_t
                pts[:, :2] = bnd2 - out_n2 * min(inset, 0.5 * min(hx, hy))
            centers_list.append(pts)
            radii_list.append(np.full(len(bnd2), lat_rad, dtype=np.float64))
            normals_list.append(nrm_lat)
    elif isinstance(leaf, BSplineSurfaceShell):
        patches = leaf.surface.bezier_patches_np(params)
        from reify.nurbs import _eval_bezier_surface_np

        n_g = 12
        g1 = np.linspace(0.0, 1.0, n_g)
        gu, gv = np.meshgrid(g1, g1, indexing="ij")
        gu_f, gv_f = gu.ravel(), gv.ravel()
        t_val = _val_mm_float(leaf.thickness, params)
        pts_list_s, nrm_list_s = [], []
        for s_idx in range(patches.shape[0]):
            s_pt, su, sv = _eval_bezier_surface_np(patches[s_idx : s_idx + 1], gu_f[:, None], gv_f[:, None])
            s_pt, su, sv = s_pt[:, 0, :], su[:, 0, :], sv[:, 0, :]
            nrm = np.cross(su, sv)
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
            pts_list_s.append(s_pt + 0.5 * t_val * nrm)
            nrm_list_s.append(nrm)
            pts_list_s.append(s_pt - 0.5 * t_val * nrm)
            nrm_list_s.append(-nrm)
        pts_all_s = np.concatenate(pts_list_s, axis=0)
        nrm_all_s = np.concatenate(nrm_list_s, axis=0)
        centers_list.append(pts_all_s)
        radii_list.append(np.full(pts_all_s.shape[0], max(max_radius, 0.5), dtype=np.float64))
        normals_list.append(nrm_all_s)
    elif isinstance(leaf, SweepAlongSpline):
        X, _, N_rmf, B_rmf, _ = leaf.path.compute_rmf_stations(n_per_span=20, params=params)
        r = leaf._r_max_nom if hasattr(leaf.profile, "bounds_2d") else _val_mm_float(leaf.profile, params)
        angs = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        ca, sa = np.cos(angs), np.sin(angs)
        ring = N_rmf[:, None, :] * ca[None, :, None] + B_rmf[:, None, :] * sa[None, :, None]
        pts_s = (X[:, None, :] + r * ring).reshape(-1, 3)
        nrms_s = ring.reshape(-1, 3)
        centers_list.append(pts_s)
        radii_list.append(np.full(pts_s.shape[0], max(max_radius, 0.5), dtype=np.float64))
        normals_list.append(nrms_s)
    elif isinstance(leaf, Shell):
        c_loc, r_loc, n_loc = _tile_leaf_surface(leaf.child, max_radius, params)
        t_val = _val_mm_float(leaf.thickness, params)
        centers_list.append(c_loc.numpy())
        radii_list.append(r_loc.numpy())
        normals_list.append(n_loc.numpy())
        inner_c = c_loc.numpy() - t_val * n_loc.numpy()
        if leaf.open_top:
            _, bmax_c = leaf.child.local_bounds(params)
            keep_in = inner_c[:, 2] < bmax_c[2] - 1e-3
            inner_c = inner_c[keep_in]
            inner_n = -n_loc.numpy()[keep_in]
            inner_r = r_loc.numpy()[keep_in]
        else:
            inner_n = -n_loc.numpy()
            inner_r = r_loc.numpy()
        if inner_c.shape[0] > 0:
            centers_list.append(inner_c)
            radii_list.append(np.maximum(inner_r, 0.5))
            normals_list.append(inner_n)
    elif isinstance(leaf, Offset):
        c_loc, r_loc, n_loc = _tile_leaf_surface(leaf.child, max_radius, params)
        r_off = _val_mm_float(leaf.radius, params)
        centers_list.append(c_loc.numpy() + r_off * n_loc.numpy())
        radii_list.append(np.maximum(r_loc.numpy(), 0.5))
        normals_list.append(n_loc.numpy())
    elif isinstance(leaf, RevolveSketch):
        ang_rad = math.radians(_val_deg_float(leaf.angle_deg, params))
        bnd2 = leaf.sketch.sample_boundary_points(n_per_seg=12, params=params, dtype=torch.float64).detach().numpy()
        rmax = max(float(np.max(bnd2[:, 0])), 1e-3)
        n_ang = max(12, int(math.ceil(ang_rad * rmax / max(max_radius, 0.25))))
        angs = np.linspace(0.0, ang_rad, n_ang, endpoint=(ang_rad < 2.0 * math.pi - 1e-6))
        ca, sa = np.cos(angs), np.sin(angs)
        pts_rev = np.stack([
            bnd2[None, :, 0] * ca[:, None],
            bnd2[None, :, 0] * sa[:, None],
            np.broadcast_to(bnd2[None, :, 1], (len(angs), len(bnd2))),
        ], axis=-1).reshape(-1, 3)
        nrm_rev = np.stack([ca[:, None] * np.ones((1, len(bnd2))), sa[:, None] * np.ones((1, len(bnd2))), np.zeros((len(angs), len(bnd2)))], axis=-1).reshape(-1, 3)
        centers_list.append(pts_rev)
        radii_list.append(np.full(pts_rev.shape[0], max(max_radius * 1.5, 0.5), dtype=np.float64))
        normals_list.append(nrm_rev)
    elif isinstance(leaf, Frustum):
        r1 = _val_mm_float(leaf.r_bottom, params)
        r2 = _val_mm_float(leaf.r_top, params)
        hz = 0.5 * _val_mm_float(leaf.height, params)
        rmax = max(r1, r2, 1e-3)
        uv, rad = _tile_rect_2d(rmax, rmax, max_radius)
        for rv, zv, sign in ((r1, -hz, -1.0), (r2, hz, 1.0)):
            mask = (uv[:, 0] ** 2 + uv[:, 1] ** 2) <= (rv + rad) ** 2
            uvm = uv[mask]
            centers_list.append(np.column_stack([uvm, np.full(len(uvm), zv)]))
            radii_list.append(np.full(len(uvm), rad, dtype=np.float64))
            normals_list.append(np.tile([0.0, 0.0, sign], (len(uvm), 1)))
        n_ang = max(16, int(math.ceil(2.0 * math.pi * rmax / max(max_radius, 0.25))))
        n_z = max(2, int(math.ceil(2.0 * hz / max(max_radius, 0.25))))
        angs = np.linspace(0.0, 2.0 * math.pi, n_ang, endpoint=False)
        ts_z = np.linspace(0.0, 1.0, n_z)
        ga, gt = np.meshgrid(angs, ts_z, indexing="ij")
        rs = (1.0 - gt.ravel()) * r1 + gt.ravel() * r2
        zs = -hz + gt.ravel() * (2.0 * hz)
        ca, sa = np.cos(ga.ravel()), np.sin(ga.ravel())
        centers_list.append(np.column_stack([rs * ca, rs * sa, zs]))
        radii_list.append(np.full(len(rs), max(max_radius * 1.5, 0.5), dtype=np.float64))
        normals_list.append(np.column_stack([ca, sa, np.zeros_like(ca)]))
    elif isinstance(leaf, LoftSketches):
        patches = leaf._build_lateral_patches(params, torch.float64, torch.device("cpu")).numpy()
        g1 = np.linspace(0.0, 1.0, 3)
        gu, gv = np.meshgrid(g1, g1, indexing="ij")
        w00 = ((1 - gu) * (1 - gv)).ravel()[:, None]
        w01 = ((1 - gu) * gv).ravel()[:, None]
        w10 = (gu * (1 - gv)).ravel()[:, None]
        w11 = (gu * gv).ravel()[:, None]
        pts_lat = (
            w00[None, :, :] * patches[:, None, 0, 0, :]
            + w01[None, :, :] * patches[:, None, 0, 1, :]
            + w10[None, :, :] * patches[:, None, 1, 0, :]
            + w11[None, :, :] * patches[:, None, 1, 1, :]
        ).reshape(-1, 3)
        centers_list.append(pts_lat)
        radii_list.append(np.full(pts_lat.shape[0], max(max_radius * 2.0, 1.0), dtype=np.float64))
        normals_list.append(np.tile([1.0, 0.0, 0.0], (pts_lat.shape[0], 1)))
        for sec, zq, sign in ((leaf.sections[0], leaf.z_levels[0], -1.0), (leaf.sections[-1], leaf.z_levels[-1], 1.0)):
            b2min, b2max = sec.bounds_2d(params)
            uv, rad = _tile_rect_2d(0.5 * (b2max[0] - b2min[0]), 0.5 * (b2max[1] - b2min[1]), max(max_radius, 1.0))
            uv[:, 0] += 0.5 * (b2min[0] + b2max[0])
            uv[:, 1] += 0.5 * (b2min[1] + b2max[1])
            zv = _val_mm_float(zq, params)
            centers_list.append(np.column_stack([uv, np.full(len(uv), zv)]))
            radii_list.append(np.full(len(uv), rad, dtype=np.float64))
            normals_list.append(np.tile([0.0, 0.0, sign], (len(uv), 1)))
    else:
        # Generic bounding-box fallback grid for FreeImplicit/Scale
        bmin, bmax = leaf.local_bounds(params)
        hx = 0.5 * (bmax[0] - bmin[0])
        hy = 0.5 * (bmax[1] - bmin[1])
        hz = 0.5 * (bmax[2] - bmin[2])
        cx, cy, cz = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1]), 0.5 * (bmin[2] + bmax[2])
        side = max(max_radius, 0.25)
        nx = max(2, int(math.ceil(2 * hx / side)))
        ny = max(2, int(math.ceil(2 * hy / side)))
        nz = max(2, int(math.ceil(2 * hz / side)))
        gx, gy, gz = np.meshgrid(
            np.linspace(bmin[0], bmax[0], nx),
            np.linspace(bmin[1], bmax[1], ny),
            np.linspace(bmin[2], bmax[2], nz),
            indexing="ij",
        )
        pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        rad = 0.5 * math.sqrt((2 * hx / nx) ** 2 + (2 * hy / ny) ** 2 + (2 * hz / nz) ** 2)
        centers_list.append(pts)
        radii_list.append(np.full(pts.shape[0], rad, dtype=np.float64))
        normals_list.append(np.tile([0.0, 0.0, 1.0], (pts.shape[0], 1)))

    c_all = torch.from_numpy(np.concatenate(centers_list, axis=0)).to(torch.float64)
    r_all = torch.from_numpy(np.concatenate(radii_list, axis=0)).to(torch.float64)
    n_all = torch.from_numpy(np.concatenate(normals_list, axis=0)).to(torch.float64)
    return c_all, r_all, n_all


def _tessellate_brep_surface_points(shape: Any, deflection: float = 0.01) -> np.ndarray:
    """Sample exact on-surface vertices from OCCT B-Rep face triangulations."""
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.15, True)
    pts: list[list[float]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            n_nodes = tri.NbNodes()
            for idx in range(1, n_nodes + 1):
                p = tri.Node(idx).Transformed(trsf)
                pts.append([p.X(), p.Y(), p.Z()])
        exp.Next()
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


class GeometryQuery:
    """Unified query protocol separating field, inside/outside, clearance, interval, and provenance."""

    def __init__(
        self,
        node: Node | Part,
        budget: NumericalBudget | None = None,
    ) -> None:
        self.root: Node = node.root if isinstance(node, Part) else node
        self.budget: NumericalBudget = budget or NumericalBudget()

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self.root.lipschitz_bound(params)

    def ast_depth(self) -> int:
        return self.root.ast_depth()

    def scale_mm(
        self, pts: torch.Tensor | None = None, params: dict[str, Any] | None = None
    ) -> float:
        bmin, bmax = self.root.local_bounds(params)
        corner_norm = max(
            math.sqrt(x * x + y * y + z * z)
            for x in (bmin[0], bmax[0])
            for y in (bmin[1], bmax[1])
            for z in (bmin[2], bmax[2])
        )
        if pts is not None and pts.numel() > 0:
            p_norm = float(torch.amax(torch.linalg.norm(pts.detach().to(torch.float64), dim=-1)).item())
            return max(corner_norm, p_norm)
        return corner_norm

    def eps_num(
        self,
        pts: torch.Tensor | None = None,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> float:
        s = self.scale_mm(pts=pts, params=params)
        return self.budget.eps_num(s, self.ast_depth(), dtype)

    def eval_field(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return self.root.eval_sdf(pts, params=params, dtype=dtype)

    def classify_point(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
    ) -> list[PointClass]:
        pts64 = pts.to(torch.float64)
        L = self.lipschitz_bound(params)
        eps = self.eps_num(pts=pts64, params=params, dtype=torch.float64)
        thresh = L * eps
        f = self.eval_field(pts64, params=params, dtype=torch.float64)
        out: list[PointClass] = []
        for v in f.detach().cpu().tolist():
            if v > thresh:
                out.append(PointClass.OUTSIDE)
            elif v < -thresh:
                out.append(PointClass.INSIDE)
            else:
                out.append(PointClass.UNCERTAIN)
        return out

    def external_clearance_lower(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Certified external clearance lower bound; valid ONLY for points classified OUTSIDE."""
        pts64 = pts.to(torch.float64)
        classes = self.classify_point(pts64, params=params)
        if any(c != PointClass.OUTSIDE for c in classes):
            raise ValueError(
                "external_clearance_lower() is valid ONLY for points certified OUTSIDE; "
                "call classify_point() first."
            )
        L = self.lipschitz_bound(params)
        eps = self.eps_num(pts=pts64, params=params, dtype=torch.float64)
        f = self.eval_field(pts64, params=params, dtype=torch.float64)
        return torch.clamp(f / L - eps, min=0.0)

    def unsigned_boundary_distance_lower(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Certified unsigned distance lower bound to boundary across all space (for WoS/stepping)."""
        pts64 = pts.to(torch.float64)
        L = self.lipschitz_bound(params)
        eps = self.eps_num(pts=pts64, params=params, dtype=torch.float64)
        f = self.eval_field(pts64, params=params, dtype=torch.float64)
        return torch.clamp(torch.abs(f) / L - eps, min=0.0)

    def interval_bound(
        self,
        box_min: tuple[float, float, float],
        box_max: tuple[float, float, float],
        params: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        corners = torch.tensor(
            [
                [x, y, z]
                for x in (box_min[0], box_max[0])
                for y in (box_min[1], box_max[1])
                for z in (box_min[2], box_max[2])
            ],
            dtype=torch.float64,
        )
        eps = self.eps_num(pts=corners, params=params, dtype=torch.float64)
        lo, hi = eval_affine_box(self.root, box_min, box_max, params=params)
        return lo - eps, hi + eps

    def cover_surface(
        self,
        max_radius: float = 0.5,
        params: dict[str, Any] | None = None,
    ) -> SurfaceCover:
        """Static certified boundary covering at fixed params (for verify)."""
        eye = torch.eye(3, dtype=torch.float64)
        zero = torch.zeros(3, dtype=torch.float64)
        all_c: list[torch.Tensor] = []
        all_r: list[torch.Tensor] = []
        all_n: list[torch.Tensor] = []
        all_ids: list[str] = []

        for leaf, R_w, t_w in _iter_transformed_leaves(self.root, params, eye, zero):
            c_loc, r_loc, n_loc = _tile_leaf_surface(leaf, max_radius, params)
            c_w = c_loc @ R_w.T + t_w
            n_w = n_loc @ R_w.T
            all_c.append(c_w)
            all_r.append(r_loc)
            all_n.append(n_w)
            all_ids.extend([leaf.node_id] * c_w.shape[0])

        centers = torch.cat(all_c, dim=0)
        radii = torch.cat(all_r, dim=0)
        normals = torch.cat(all_n, dim=0)
        L = self.lipschitz_bound(params)
        eps = self.eps_num(pts=centers, params=params, dtype=torch.float64)
        vals = self.eval_field(centers, params=params, dtype=torch.float64)
        # Keep every patch that could intersect f=0 under Lipschitz bound
        keep = (torch.abs(vals) / L) <= (radii + eps)
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        kept_ids = [all_ids[i] for i in keep_idx.tolist()]
        return SurfaceCover(
            centers=centers[keep_idx],
            radii=radii[keep_idx],
            normals=normals[keep_idx],
            node_ids=kept_ids,
        )

    def sample_parametric_points(
        self,
        n_per_primitive: int = 128,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Differentiable parametric surface sampling in local primitive coordinates (supports (B,) batch)."""
        batch_size: int | None = None
        dev = torch.device("cpu")
        if params is not None:
            for v in params.values():
                if isinstance(v, torch.Tensor):
                    dev = v.device
                    if v.ndim == 1:
                        batch_size = int(v.shape[0])
                        break

        if batch_size is not None:
            eye = torch.eye(3, dtype=dtype, device=dev).unsqueeze(0).expand(batch_size, 3, 3)
            zero = torch.zeros(batch_size, 3, dtype=dtype, device=dev)
        else:
            eye = torch.eye(3, dtype=dtype, device=dev)
            zero = torch.zeros(3, dtype=dtype, device=dev)

        def _collect(node: Node, R_acc: torch.Tensor, t_acc: torch.Tensor) -> list[torch.Tensor]:
            from reify.ast import _val_mm_tensor

            if isinstance(node, Transform):
                R_n, t_n = node.world_matrix_tensors(params, dtype=dtype)
                R_n, t_n = R_n.to(device=dev), t_n.to(device=dev)
                if R_acc.ndim == 3 or R_n.ndim == 3:
                    R_new = torch.matmul(R_acc, R_n)
                    t_new = torch.matmul(R_acc, t_n.unsqueeze(-1)).squeeze(-1) + t_acc
                else:
                    R_new = R_acc @ R_n
                    t_new = R_acc @ t_n + t_acc
                return _collect(node.child, R_new, t_new)
            if isinstance(node, (Union, Intersect)):
                pts_out: list[torch.Tensor] = []
                for ch in node.children:
                    pts_out.extend(_collect(ch, R_acc, t_acc))
                return pts_out
            if isinstance(node, Subtract):
                return _collect(node.base, R_acc, t_acc) + _collect(node.tool, R_acc, t_acc)
            if isinstance(node, SmoothUnion):
                return _collect(node.a, R_acc, t_acc) + _collect(node.b, R_acc, t_acc)
            if isinstance(node, (Offset, Shell)):
                return _collect(node.child, R_acc, t_acc)
            if isinstance(node, LinearPattern):
                sx = _val_mm_tensor(node.spacing[0], params, dtype).to(device=dev)
                sy = _val_mm_tensor(node.spacing[1], params, dtype).to(device=dev)
                sz = _val_mm_tensor(node.spacing[2], params, dtype).to(device=dev)
                pts_out_lp: list[torch.Tensor] = []
                for k in range(node.count):
                    if batch_size is not None:
                        sxb = (sx.expand(batch_size) if sx.ndim == 0 else sx) * k
                        syb = (sy.expand(batch_size) if sy.ndim == 0 else sy) * k
                        szb = (sz.expand(batch_size) if sz.ndim == 0 else sz) * k
                        tk = torch.stack([sxb, syb, szb], dim=-1)
                        t_new = torch.matmul(R_acc, tk.unsqueeze(-1)).squeeze(-1) + t_acc
                    else:
                        tk = torch.stack([sx, sy, sz]) * k
                        t_new = R_acc @ tk + t_acc
                    pts_out_lp.extend(_collect(node.child, R_acc, t_new))
                return pts_out_lp
            if isinstance(node, CircularPattern):
                rad = _val_mm_tensor(node.radius, params, dtype).to(device=dev)
                a0 = _val_rad_tensor(node.start_angle_deg, params, dtype).to(device=dev)
                pts_out_cp: list[torch.Tensor] = []
                for k in range(node.count):
                    ang = a0 + (2.0 * math.pi * k / node.count)
                    ca, sa = torch.cos(ang), torch.sin(ang)
                    if batch_size is not None:
                        cab = ca.expand(batch_size) if ca.ndim == 0 else ca
                        sab = sa.expand(batch_size) if sa.ndim == 0 else sa
                        rb = rad.expand(batch_size) if rad.ndim == 0 else rad
                        z0 = torch.zeros_like(cab)
                        o1 = torch.ones_like(cab)
                        Rk = torch.stack([
                            torch.stack([cab, -sab, z0], dim=-1),
                            torch.stack([sab, cab, z0], dim=-1),
                            torch.stack([z0, z0, o1], dim=-1),
                        ], dim=-2)
                        tk = torch.stack([rb * cab, rb * sab, z0], dim=-1)
                        R_new = torch.matmul(R_acc, Rk)
                        t_new = torch.matmul(R_acc, tk.unsqueeze(-1)).squeeze(-1) + t_acc
                    else:
                        Rk = torch.tensor([[float(ca), -float(sa), 0.0], [float(sa), float(ca), 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=dev)
                        tk = torch.stack([rad * ca, rad * sa, torch.zeros_like(rad)])
                        R_new = R_acc @ Rk
                        t_new = R_acc @ tk + t_acc
                    pts_out_cp.extend(_collect(node.child, R_new, t_new))
                return pts_out_cp

            # Sample unit local normalized coordinates and scale differentiably by params
            if isinstance(node, Box):
                u = torch.linspace(-1.0, 1.0, max(2, int(round((n_per_primitive / 6) ** 0.5))), dtype=dtype, device=dev)
                g1, g2 = torch.meshgrid(u, u, indexing="ij")
                g1, g2 = g1.ravel(), g2.ravel()
                ones = torch.ones_like(g1)
                local_norm = torch.cat(
                    [
                        torch.stack([g1, g2, ones], dim=-1),
                        torch.stack([g1, g2, -ones], dim=-1),
                        torch.stack([ones, g1, g2], dim=-1),
                        torch.stack([-ones, g1, g2], dim=-1),
                        torch.stack([g1, ones, g2], dim=-1),
                        torch.stack([g1, -ones, g2], dim=-1),
                    ],
                    dim=0,
                )
                lx = _val_mm_tensor(node.lx, params, dtype).to(device=dev)
                ly = _val_mm_tensor(node.ly, params, dtype).to(device=dev)
                lz = _val_mm_tensor(node.lz, params, dtype).to(device=dev)
                if batch_size is not None:
                    lx, ly, lz = torch.broadcast_tensors(
                        lx.expand(batch_size) if lx.ndim == 0 else lx,
                        ly.expand(batch_size) if ly.ndim == 0 else ly,
                        lz.expand(batch_size) if lz.ndim == 0 else lz,
                    )
                    half = 0.5 * torch.stack([lx, ly, lz], dim=-1)[:, None, :]  # (B, 1, 3)
                    local_pts = local_norm.unsqueeze(0) * half
                else:
                    half = 0.5 * torch.stack([lx, ly, lz])
                    local_pts = local_norm * half
            elif isinstance(node, Cylinder):
                n_ang = max(8, n_per_primitive // 2)
                theta = torch.linspace(0.0, 2.0 * math.pi, n_ang + 1, dtype=dtype, device=dev)[:-1]
                r = _val_mm_tensor(node.radius, params, dtype).to(device=dev)
                hz = 0.5 * _val_mm_tensor(node.height, params, dtype).to(device=dev)
                ct, st = torch.cos(theta), torch.sin(theta)
                if batch_size is not None:
                    rb = (r.expand(batch_size) if r.ndim == 0 else r)[:, None]
                    hb = (hz.expand(batch_size) if hz.ndim == 0 else hz)[:, None]
                    ctb, stb = ct.unsqueeze(0).expand(batch_size, n_ang), st.unsqueeze(0).expand(batch_size, n_ang)
                    hb_exp = hb.expand(batch_size, n_ang)
                    ring_top = torch.stack([rb * ctb, rb * stb, hb_exp], dim=-1)
                    ring_bot = torch.stack([rb * ctb, rb * stb, -hb_exp], dim=-1)
                    ring_mid = torch.stack([rb * ctb, rb * stb, torch.zeros_like(hb_exp)], dim=-1)
                    local_pts = torch.cat([ring_top, ring_bot, ring_mid], dim=1)
                else:
                    ring_top = torch.stack([r * ct, r * st, hz.expand_as(ct)], dim=-1)
                    ring_bot = torch.stack([r * ct, r * st, -hz.expand_as(ct)], dim=-1)
                    ring_mid = torch.stack([r * ct, r * st, torch.zeros_like(ct)], dim=-1)
                    local_pts = torch.cat([ring_top, ring_bot, ring_mid], dim=0)
            elif isinstance(node, Sphere):
                n_ang = max(8, int(math.sqrt(n_per_primitive)))
                phi = torch.linspace(0.1, math.pi - 0.1, n_ang, dtype=dtype, device=dev)
                th = torch.linspace(0.0, 2.0 * math.pi, n_ang + 1, dtype=dtype, device=dev)[:-1]
                gp, gt = torch.meshgrid(phi, th, indexing="ij")
                unit_pts = torch.stack(
                    [
                        torch.sin(gp.ravel()) * torch.cos(gt.ravel()),
                        torch.sin(gp.ravel()) * torch.sin(gt.ravel()),
                        torch.cos(gp.ravel()),
                    ],
                    dim=-1,
                )
                r = _val_mm_tensor(node.radius, params, dtype).to(device=dev)
                if batch_size is not None:
                    rb = (r.expand(batch_size) if r.ndim == 0 else r)[:, None, None]
                    local_pts = rb * unit_pts.unsqueeze(0)
                else:
                    local_pts = r * unit_pts
            elif isinstance(node, (ExtrudeSketch, TaperedExtrude)):
                bnd2 = node.sketch.sample_boundary_points(n_per_seg=12, params=params, dtype=dtype, device=dev)
                hz = 0.5 * _val_mm_tensor(node.height, params, dtype).to(device=dev)
                if batch_size is not None:
                    if bnd2.ndim == 2:
                        bnd2 = bnd2.unsqueeze(0).expand(batch_size, -1, 2)
                    hb = (hz.expand(batch_size) if hz.ndim == 0 else hz)[:, None].expand(batch_size, bnd2.shape[1])
                    top = torch.cat([bnd2, hb.unsqueeze(-1)], dim=-1)
                    bot = torch.cat([bnd2, -hb.unsqueeze(-1)], dim=-1)
                    mid = torch.cat([bnd2, torch.zeros_like(hb).unsqueeze(-1)], dim=-1)
                    local_pts = torch.cat([top, bot, mid], dim=1)
                else:
                    h_exp = hz.expand(bnd2.shape[0]).unsqueeze(-1)
                    top = torch.cat([bnd2, h_exp], dim=-1)
                    bot = torch.cat([bnd2, -h_exp], dim=-1)
                    mid = torch.cat([bnd2, torch.zeros_like(h_exp)], dim=-1)
                    local_pts = torch.cat([top, bot, mid], dim=0)
            elif isinstance(node, BSplineSurfaceShell):
                local_pts = node.surface.sample_points_tensor(n_u=8, n_v=8, params=params, dtype=dtype, device=dev)
                if batch_size is not None and local_pts.ndim == 2:
                    local_pts = local_pts.unsqueeze(0).expand(batch_size, -1, 3)
            elif isinstance(node, SweepAlongSpline):
                X, _, _, _, _ = node.path.compute_rmf_stations(n_per_span=12, params=params)
                local_pts = torch.from_numpy(X).to(dtype=dtype, device=dev)
                if batch_size is not None:
                    local_pts = local_pts.unsqueeze(0).expand(batch_size, -1, 3)
            else:
                bmin, bmax = node.local_bounds(params)
                u = torch.linspace(-1.0, 1.0, 4, dtype=dtype, device=dev)
                gx, gy, gz = torch.meshgrid(u, u, u, indexing="ij")
                c = torch.tensor([0.5 * (bmin[i] + bmax[i]) for i in range(3)], dtype=dtype, device=dev)
                h = torch.tensor([0.5 * (bmax[i] - bmin[i]) for i in range(3)], dtype=dtype, device=dev)
                local_pts = c + h * torch.stack([gx.ravel(), gy.ravel(), gz.ravel()], dim=-1)
                if batch_size is not None:
                    local_pts = local_pts.unsqueeze(0).expand(batch_size, -1, 3)

            if batch_size is not None:
                return [torch.matmul(local_pts, R_acc.transpose(-1, -2)) + t_acc[:, None, :]]
            return [local_pts @ R_acc.T + t_acc]

        return torch.cat(_collect(self.root, eye, zero), dim=1 if batch_size is not None else 0)

    def provenance(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        tie_tol: float = 1e-6,
    ) -> list[ProvenanceResult]:
        pts64 = pts.to(torch.float64)
        _, branch_lists = self.root.eval_sdf_with_provenance(
            pts64, params=params, tie_tol=tie_tol, dtype=torch.float64
        )
        results: list[ProvenanceResult] = []
        for branches in branch_lists:
            gstatus = (
                GradientStatus.MULTI_BRANCH
                if len(branches) > 1
                else GradientStatus.UNIQUE
            )
            results.append(
                ProvenanceResult(
                    active_branches=branches,
                    gradient_status=gstatus,
                    tie_tol=tie_tol,
                )
            )
        return results

    def distance_query(
        self,
        other: GeometryQuery,
        params: dict[str, Any] | None = None,
        deflection: float = 1e-4,
    ) -> DistanceResult:
        """Pure-SDF Two-Stage 3D Octree Lagrangian-Affine Branch-and-Bound separation distance."""
        import heapq

        def _closest_pair(
            ba: tuple[tuple[float, float, float], tuple[float, float, float]],
            bb: tuple[tuple[float, float, float], tuple[float, float, float]],
        ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
            pa = [min(max(0.5 * (bb[0][i] + bb[1][i]), ba[0][i]), ba[1][i]) for i in range(3)]
            pb = [min(max(pa[i], bb[0][i]), bb[1][i]) for i in range(3)]
            pa = [min(max(pb[i], ba[0][i]), ba[1][i]) for i in range(3)]
            return (pa[0], pa[1], pa[2]), (pb[0], pb[1], pb[2])

        def _alt_proj(pa0: torch.Tensor, pb0: torch.Tensor, rounds: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
            pa, pb = pa0.clone(), pb0.clone()
            for _ in range(rounds):
                pa = self._project_to_implicit_surface(pb, params=params, steps=3)
                pb = other._project_to_implicit_surface(pa, params=params, steps=3)
            pa = self._project_to_implicit_surface(pb, params=params, steps=4)
            pb = other._project_to_implicit_surface(pa, params=params, steps=4)
            return pa, pb

        box_a = self.root.local_bounds(params)
        box_b = other.root.local_bounds(params)
        L_a = max(1.0, self.lipschitz_bound(params))
        L_b = max(1.0, other.lipschitz_bound(params))

        ma0 = [0.5 * (box_a[0][i] + box_a[1][i]) for i in range(3)]
        mb0 = [0.5 * (box_b[0][i] + box_b[1][i]) for i in range(3)]
        cp_a0, cp_b0 = _closest_pair(box_a, box_b)

        # Immediate penetration check at centers
        ctr_pts = torch.tensor([ma0, mb0], dtype=torch.float64)
        with torch.no_grad():
            fa_c0 = self.eval_field(ctr_pts, params=params, dtype=torch.float64)
            fb_c0 = other.eval_field(ctr_pts, params=params, dtype=torch.float64)
        for idx, pt in enumerate((ma0, mb0)):
            if float(fa_c0[idx]) <= 0.0 and float(fb_c0[idx]) <= 0.0:
                w = (float(pt[0]), float(pt[1]), float(pt[2]))
                return DistanceResult(
                    value=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    solver_deflection=deflection,
                    witness_a=w,
                    witness_b=w,
                    containment=True,
                    guarantee_level=GuaranteeLevel.SOLVER_TOLERANCE,
                    status=Status.PASS,
                    is_done=True,
                )

        # Multi-start alternating surface projection for tight initial upper bound
        pa_init = torch.tensor([ma0, mb0, list(cp_a0)], dtype=torch.float64)
        pb_init = torch.tensor([mb0, ma0, list(cp_b0)], dtype=torch.float64)
        pa_p, pb_p = _alt_proj(pa_init, pb_init, rounds=8)
        with torch.no_grad():
            fa_p = torch.abs(self.eval_field(pa_p, params=params, dtype=torch.float64))
            fb_p = torch.abs(other.eval_field(pb_p, params=params, dtype=torch.float64))
            fa_on_b = self.eval_field(pb_p, params=params, dtype=torch.float64)
            fb_on_a = other.eval_field(pa_p, params=params, dtype=torch.float64)

        best_upper = float("inf")
        best_wa = (float(ma0[0]), float(ma0[1]), float(ma0[2]))
        best_wb = (float(mb0[0]), float(mb0[1]), float(mb0[2]))
        for idx in range(3):
            if float(fb_on_a[idx]) <= 0.0 and float(fa_p[idx]) <= 1e-4:
                w = tuple(float(v) for v in pa_p[idx].tolist())
                return DistanceResult(
                    value=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    solver_deflection=deflection,
                    witness_a=w,  # type: ignore[arg-type]
                    witness_b=w,  # type: ignore[arg-type]
                    containment=True,
                    guarantee_level=GuaranteeLevel.SOLVER_TOLERANCE,
                    status=Status.PASS,
                    is_done=True,
                )
            if float(fa_on_b[idx]) <= 0.0 and float(fb_p[idx]) <= 1e-4:
                w = tuple(float(v) for v in pb_p[idx].tolist())
                return DistanceResult(
                    value=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    solver_deflection=deflection,
                    witness_a=w,  # type: ignore[arg-type]
                    witness_b=w,  # type: ignore[arg-type]
                    containment=True,
                    guarantee_level=GuaranteeLevel.SOLVER_TOLERANCE,
                    status=Status.PASS,
                    is_done=True,
                )
            if float(fa_p[idx]) < 1e-4 and float(fb_p[idx]) < 1e-4:
                dc = float(torch.linalg.norm(pa_p[idx] - pb_p[idx]).item())
                if dc < best_upper:
                    best_upper = dc
                    best_wa = tuple(float(v) for v in pa_p[idx].tolist())  # type: ignore[assignment]
                    best_wb = tuple(float(v) for v in pb_p[idx].tolist())  # type: ignore[assignment]

        pen_witness: tuple[float, float, float] | None = None

        def _bnb_3d(
            q_own: GeometryQuery,
            q_other: GeometryQuery,
            box_own: tuple[tuple[float, float, float], tuple[float, float, float]],
            L_own: float,
            L_other: float,
            max_waves: int = 60,
            batch_size: int = 32,
        ) -> tuple[float, bool]:
            nonlocal best_upper, best_wa, best_wb, pen_witness
            lo0 = np.array(box_own[0], dtype=np.float64)
            hi0 = np.array(box_own[1], dtype=np.float64)
            pq: list[tuple[float, int, np.ndarray, np.ndarray]] = [(0.0, 0, lo0, hi0)]
            seq = 1
            waves = 0
            while pq and waves < max_waves:
                if pq[0][0] >= best_upper - deflection:
                    break
                if float(np.linalg.norm(pq[0][3] - pq[0][2])) < deflection * 0.25:
                    break
                waves += 1
                popped: list[tuple[float, int, np.ndarray, np.ndarray]] = []
                while pq and len(popped) < batch_size:
                    item = heapq.heappop(pq)
                    if item[0] >= best_upper - deflection:
                        heapq.heappush(pq, item)
                        break
                    popped.append(item)
                if not popped:
                    break

                c_lo_l: list[np.ndarray] = []
                c_hi_l: list[np.ndarray] = []
                for _, _, lo, hi in popped:
                    mid = 0.5 * (lo + hi)
                    for ox in (0, 1):
                        for oy in (0, 1):
                            for oz in (0, 1):
                                c_lo_l.append(
                                    np.array([lo[0] if ox == 0 else mid[0], lo[1] if oy == 0 else mid[1], lo[2] if oz == 0 else mid[2]])
                                )
                                c_hi_l.append(
                                    np.array([mid[0] if ox == 0 else hi[0], mid[1] if oy == 0 else hi[1], mid[2] if oz == 0 else hi[2]])
                                )
                c_lo = np.stack(c_lo_l, axis=0)
                c_hi = np.stack(c_hi_l, axis=0)
                m = 0.5 * (c_lo + c_hi)
                r = 0.5 * np.linalg.norm(c_hi - c_lo, axis=-1)
                pts = torch.from_numpy(m)
                with torch.no_grad():
                    f_own = q_own.eval_field(pts, params=params, dtype=torch.float64).numpy()
                    f_oth = q_other.eval_field(pts, params=params, dtype=torch.float64).numpy()

                pen_mask = (f_own <= 0.0) & (f_oth <= 0.0)
                if np.any(pen_mask):
                    pi = int(np.nonzero(pen_mask)[0][0])
                    pen_witness = (float(m[pi, 0]), float(m[pi, 1]), float(m[pi, 2]))
                    return 0.0, True

                keep = (f_own - L_own * r <= 0.0) & ~((f_own + L_own * r < 0.0) & (f_oth - L_other * r > 0.0))
                lb_c = np.maximum(0.0, f_oth / L_other - r)
                keep &= lb_c < best_upper - deflection
                idxs = np.nonzero(keep)[0]
                if len(idxs) > 0:
                    if (waves == 1 and math.isinf(best_upper)) or waves % 3 == 0:
                        bi = int(idxs[np.argmin(lb_c[idxs])])
                        seed = torch.tensor([m[bi].tolist()], dtype=torch.float64)
                        pa_c, pb_c = _alt_proj(seed, seed, rounds=4)
                        with torch.no_grad():
                            fac = abs(float(self.eval_field(pa_c, params=params, dtype=torch.float64)[0]))
                            fbc = abs(float(other.eval_field(pb_c, params=params, dtype=torch.float64)[0]))
                        if fac < 1e-4 and fbc < 1e-4:
                            dc = float(torch.linalg.norm(pa_c[0] - pb_c[0]))
                            if dc < best_upper:
                                best_upper = dc
                                best_wa = tuple(float(v) for v in pa_c[0].tolist())  # type: ignore[assignment]
                                best_wb = tuple(float(v) for v in pb_c[0].tolist())  # type: ignore[assignment]

                    for i in idxs:
                        clo_t = (float(c_lo[i, 0]), float(c_lo[i, 1]), float(c_lo[i, 2]))
                        chi_t = (float(c_hi[i, 0]), float(c_hi[i, 1]), float(c_hi[i, 2]))
                        af_own = eval_affine_form(q_own.root, clo_t, chi_t, params)
                        lo_own, hi_own = af_own.bounds()
                        if lo_own > 0.0:
                            continue
                        af_oth = eval_affine_form(q_other.root, clo_t, chi_t, params)
                        lo_oth, _ = af_oth.bounds()
                        if hi_own < 0.0 and lo_oth > 0.0:
                            continue
                        # Certified Lagrangian Affine Cancellation lower bound
                        denom = float(np.dot(af_own.coeffs, af_own.coeffs))
                        lam = max(0.0, -float(np.dot(af_oth.coeffs, af_own.coeffs)) / denom) if denom > 1e-18 else 0.0
                        comb_c = af_oth.center + lam * af_own.center
                        comb_g = af_oth.coeffs + lam * af_own.coeffs
                        comb_r = af_oth.radius + lam * af_own.radius
                        lo_lag = comb_c - float(np.sum(np.abs(comb_g))) - comb_r
                        lbi = max(float(lb_c[i]), lo_oth / L_other, lo_lag / L_other)
                        if lbi < best_upper - deflection:
                            heapq.heappush(pq, (lbi, seq, c_lo[i], c_hi[i]))
                            seq += 1

            lb_out = pq[0][0] if pq else best_upper
            return lb_out, False

        lb_b, pen = _bnb_3d(other, self, box_b, L_b, L_a)
        if pen and pen_witness is not None:
            return DistanceResult(
                value=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                solver_deflection=deflection,
                witness_a=pen_witness,
                witness_b=pen_witness,
                containment=True,
                guarantee_level=GuaranteeLevel.SOLVER_TOLERANCE,
                status=Status.PASS,
                is_done=True,
            )

        min_lb = lb_b
        if min_lb < best_upper - deflection:
            lb_a, pen = _bnb_3d(self, other, box_a, L_a, L_b)
            if pen and pen_witness is not None:
                return DistanceResult(
                    value=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    solver_deflection=deflection,
                    witness_a=pen_witness,
                    witness_b=pen_witness,
                    containment=True,
                    guarantee_level=GuaranteeLevel.SOLVER_TOLERANCE,
                    status=Status.PASS,
                    is_done=True,
                )
            min_lb = max(min_lb, lb_a)

        # Check containment / penetration at witness points
        wa_t = torch.tensor([list(best_wa)], dtype=torch.float64)
        wb_t = torch.tensor([list(best_wb)], dtype=torch.float64)
        containment = (
            other.classify_point(wa_t, params=params)[0] == PointClass.INSIDE
            or self.classify_point(wb_t, params=params)[0] == PointClass.INSIDE
            or best_upper <= deflection
        )
        if containment:
            min_lb = 0.0
            best_upper = 0.0

        eps = max(self.eps_num(params=params, dtype=torch.float64), other.eps_num(params=params, dtype=torch.float64))
        val = 0.5 * (min_lb + best_upper)
        return DistanceResult(
            value=val,
            lower_bound=max(0.0, min_lb - eps),
            upper_bound=best_upper + eps,
            solver_deflection=deflection,
            witness_a=best_wa,  # type: ignore[arg-type]
            witness_b=best_wb,  # type: ignore[arg-type]
            containment=containment,
            guarantee_level=GuaranteeLevel.SOLVER_TOLERANCE,
            status=Status.PASS,
            is_done=True,
        )

    def _project_to_implicit_surface(
        self, pts: torch.Tensor, params: dict[str, Any] | None, steps: int = 8
    ) -> torch.Tensor:
        """Newton-project points onto f(x)=0 using active-set batched finite-difference gradients."""
        x = pts.clone().to(torch.float64)
        if x.shape[0] == 0:
            return x
        h = 1e-5
        eye3 = torch.eye(3, dtype=torch.float64, device=x.device)
        for _ in range(steps):
            f = self.eval_field(x, params=params, dtype=torch.float64)
            active = torch.abs(f) > 1e-7
            if not torch.any(active):
                break
            x_act = x[active]
            f_act = f[active]
            k_act = x_act.shape[0]
            shifts = torch.cat([eye3 * h, -eye3 * h], dim=0)  # (6, 3)
            x_pert = (x_act.unsqueeze(0) + shifts.unsqueeze(1)).reshape(6 * k_act, 3)
            f_pert = self.eval_field(x_pert, params=params, dtype=torch.float64).reshape(6, k_act)
            grad = ((f_pert[:3] - f_pert[3:]) / (2.0 * h)).T
            gnorm2 = torch.sum(grad * grad, dim=-1, keepdim=True).clamp(min=1e-16)
            step = (f_act.unsqueeze(-1) / gnorm2) * grad
            step_norm = torch.linalg.norm(step, dim=-1, keepdim=True).clamp(min=1e-12)
            scale = torch.clamp(2.0 / step_norm, max=1.0)
            x[active] = x_act - step * scale
        return x

    def bidirectional_hausdorff_to_brep(
        self,
        brep_shape: Any,
        params: dict[str, Any] | None = None,
        sample_radius: float = 0.5,
    ) -> dict[str, float]:
        """Compute bidirectional Hausdorff distance between implicit zero-set and B-Rep surface."""
        # 1. B-Rep -> Implicit
        brep_pts_np = _tessellate_brep_surface_points(brep_shape, deflection=0.01)
        if brep_pts_np.shape[0] > 4000:
            idx = np.linspace(0, brep_pts_np.shape[0] - 1, 4000, dtype=int)
            brep_pts_np = brep_pts_np[idx]
        brep_pts = torch.from_numpy(brep_pts_np).to(torch.float64)
        proj_impl = self._project_to_implicit_surface(brep_pts, params=params, steps=6)
        brep_to_impl_dists = torch.linalg.norm(brep_pts - proj_impl, dim=-1)
        # Verify projection actually landed near f=0; fallback to |f|/L if needed
        f_rem = torch.abs(self.eval_field(proj_impl, params=params, dtype=torch.float64))
        L = self.lipschitz_bound(params)
        brep_to_impl = float(torch.amax(brep_to_impl_dists + f_rem / L).item())

        # 2. Implicit -> B-Rep
        cov = self.cover_surface(max_radius=sample_radius, params=params)
        impl_pts = self._project_to_implicit_surface(cov.centers, params=params, steps=6)
        # Fast prefilter against tessellated B-Rep surface points, then exact OCCT on worst top-K
        impl_np = impl_pts.detach().cpu().numpy()
        # Chunked pairwise min distance to tessellated B-Rep points for fast ranking
        approx_d = np.empty(impl_np.shape[0], dtype=np.float64)
        chunk = 512
        for start in range(0, impl_np.shape[0], chunk):
            sub = impl_np[start : start + chunk]
            d2 = np.sum((sub[:, None, :] - brep_pts_np[None, :, :]) ** 2, axis=-1)
            approx_d[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
        top_k = min(25, impl_np.shape[0])
        worst_idx = np.argsort(approx_d)[-top_k:]

        impl_to_brep = 0.0
        for i in worst_idx:
            p = impl_np[int(i)]
            vtx = BRepBuilderAPI_MakeVertex(gp_Pnt(float(p[0]), float(p[1]), float(p[2]))).Vertex()
            ext = BRepExtrema_DistShapeShape(vtx, brep_shape)
            if ext.IsDone() and ext.NbSolution() > 0:
                impl_to_brep = max(impl_to_brep, float(ext.Value()))

        return {
            "brep_to_implicit_mm": brep_to_impl,
            "implicit_to_brep_mm": impl_to_brep,
            "bidirectional_hausdorff_mm": max(brep_to_impl, impl_to_brep),
        }

    def freeze_check_world_shell(
        self,
        new_query: GeometryQuery,
        tol_mm: float = 1e-3,
        params_old: dict[str, Any] | None = None,
        params_new: dict[str, Any] | None = None,
        sample_radius: float = 0.5,
    ) -> VerificationReport:
        """Certify world_shell freeze strictly via bidirectional surface Hausdorff distance."""
        cov_old = self.cover_surface(max_radius=sample_radius, params=params_old)
        cov_new = new_query.cover_surface(max_radius=sample_radius, params=params_new)

        pts_old = self._project_to_implicit_surface(cov_old.centers, params=params_old, steps=8)
        pts_new = new_query._project_to_implicit_surface(cov_new.centers, params=params_new, steps=8)

        # Compute bidirectional Hausdorff between projected zero-set surfaces
        p1 = pts_old.detach().cpu().numpy()
        p2 = pts_new.detach().cpu().numpy()

        # Also refine worst distance via Newton projection into the opposing surface
        d12 = np.sqrt(np.min(np.sum((p1[:, None, :] - p2[None, :, :]) ** 2, axis=-1), axis=1))
        d21 = np.sqrt(np.min(np.sum((p2[:, None, :] - p1[None, :, :]) ** 2, axis=-1), axis=1))

        # Project top worst points directly onto opposing surface to eliminate discretization lag
        w1_idx = int(np.argmax(d12))
        w2_idx = int(np.argmax(d21))
        p1_worst = pts_old[w1_idx : w1_idx + 1]
        p2_worst = pts_new[w2_idx : w2_idx + 1]
        proj_1_to_2 = new_query._project_to_implicit_surface(p1_worst, params=params_new, steps=10)
        proj_2_to_1 = self._project_to_implicit_surface(p2_worst, params=params_old, steps=10)

        h12 = float(torch.linalg.norm(p1_worst - proj_1_to_2, dim=-1).item())
        h21 = float(torch.linalg.norm(p2_worst - proj_2_to_1, dim=-1).item())
        dh = max(h12, h21)

        if dh <= tol_mm:
            return VerificationReport(
                status=Status.PASS,
                metrics={"bidirectional_hausdorff_mm": dh, "h_old_to_new": h12, "h_new_to_old": h21},
                message="world_shell freeze satisfied via bidirectional Hausdorff distance.",
            )
        witness = p1[w1_idx].tolist() if h12 >= h21 else p2[w2_idx].tolist()
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.FREEZE_VIOLATION,
            witness_points=[(float(witness[0]), float(witness[1]), float(witness[2]))],
            metrics={"bidirectional_hausdorff_mm": dh, "h_old_to_new": h12, "h_new_to_old": h21},
            message=f"world_shell freeze violated: bidirectional Hausdorff {dh:.6f} mm > {tol_mm} mm.",
        )
