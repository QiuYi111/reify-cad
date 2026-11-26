"""Manifold Dual Contouring (Schaefer-Ju-Warren 2007) + Ju 2002 SVD-QEF + glTF Y-up exporter."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch
import trimesh

from reify.ast import Box, Cylinder, Intersect, Node, Part, SmoothUnion, Sphere, Subtract, Transform, Union, _val_mm_float
from reify.device import get_compute_device
from reify.features import CircularPattern, ExtrudeSketch, Frustum, LinearPattern, Offset, Shell, TaperedExtrude
from reify.finalize import BRepFinalized
from reify.types import VerificationReport

_PALETTE = ("#4a6fa5", "#5b8c5a", "#8e6c88", "#d4a373", "#3d5a80", "#e07a5f", "#6b9080", "#b5838d")

# ---------------------------------------------------------------------------
# Precomputed 256-entry Manifold Dual Contouring cube topology table
# ---------------------------------------------------------------------------
# Corners c in 0..7: c = ix + 2*iy + 4*iz
# Edges 0..11:
#   0..3 (X): (0,1), (2,3), (4,5), (6,7)
#   4..7 (Y): (0,2), (1,3), (4,6), (5,7)
#   8..11(Z): (0,4), (1,5), (2,6), (3,7)
_CUBE_EDGE_CORNERS = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# 6 faces, each given by 4 corners in canonical 2D (u,v) order [(0,0), (1,0), (1,1), (0,1)]
# and the 4 corresponding edges around the face loop: e0=(c0,c1), e1=(c1,c2), e2=(c2,c3), e3=(c3,c0)
_CUBE_FACES = (
    ((0, 2, 6, 4), (4, 10, 6, 8)),   # x = 0: (y,z)
    ((1, 3, 7, 5), (5, 11, 7, 9)),   # x = 1: (y,z)
    ((0, 1, 5, 4), (0, 9, 2, 8)),    # y = 0: (x,z)
    ((2, 3, 7, 6), (1, 11, 3, 10)),  # y = 1: (x,z)
    ((0, 1, 3, 2), (0, 5, 1, 4)),    # z = 0: (x,y)
    ((4, 5, 7, 6), (2, 7, 3, 6)),    # z = 1: (x,y)
)


def _build_mdc_table() -> tuple[np.ndarray, np.ndarray]:
    """Build (256, 12) edge->component_id table and (256,) component count table."""
    edge_comp = np.full((256, 12), -1, dtype=np.int8)
    comp_count = np.zeros(256, dtype=np.int8)

    for mask in range(256):
        neg = [(mask >> c) & 1 for c in range(8)]
        active_edges = [e for e, (c0, c1) in enumerate(_CUBE_EDGE_CORNERS) if neg[c0] != neg[c1]]
        if not active_edges:
            continue

        adj: dict[int, list[int]] = {e: [] for e in active_edges}
        for (c0, c1, c2, c3), (e0, e1, e2, e3) in _CUBE_FACES:
            f_edges = [e for e in (e0, e1, e2, e3) if e in adj]
            if len(f_edges) == 2:
                a, b = f_edges
                adj[a].append(b)
                adj[b].append(a)
            elif len(f_edges) == 4:
                # Canonical saddle disambiguation using corner (0,0) sign (identical on shared faces!)
                if neg[c0]:
                    pairs = ((e0, e1), (e2, e3))
                else:
                    pairs = ((e1, e2), (e3, e0))
                for a, b in pairs:
                    adj[a].append(b)
                    adj[b].append(a)

        cid = 0
        visited: set[int] = set()
        for e in active_edges:
            if e in visited:
                continue
            stack = [e]
            visited.add(e)
            while stack:
                cur = stack.pop()
                edge_comp[mask, cur] = cid
                for nxt in adj[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            cid += 1
        comp_count[mask] = cid

    return edge_comp, comp_count


_MDC_EDGE_COMP, _MDC_COMP_COUNT = _build_mdc_table()


def _hex_to_rgba(hex_color: str) -> list[int]:
    s = hex_color.lstrip("#")
    if len(s) == 6:
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255]
    if len(s) == 8:
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16)]
    return [74, 111, 165, 255]


def _min_feature_scale(node: Node, params: dict[str, Any] | None = None) -> float:
    """Estimate smallest feature diameter/thickness in AST for adaptive grid spacing."""
    if isinstance(node, Transform):
        return _min_feature_scale(node.child, params)
    if isinstance(node, (Union, Intersect)):
        return min(_min_feature_scale(ch, params) for ch in node.children)
    if isinstance(node, Subtract):
        return min(_min_feature_scale(node.base, params), _min_feature_scale(node.tool, params))
    if isinstance(node, SmoothUnion):
        return min(_min_feature_scale(node.a, params), _min_feature_scale(node.b, params))
    if isinstance(node, Box):
        return min(_val_mm_float(node.lx, params), _val_mm_float(node.ly, params), _val_mm_float(node.lz, params))
    if isinstance(node, Cylinder):
        return min(2.0 * _val_mm_float(node.radius, params), _val_mm_float(node.height, params))
    if isinstance(node, Sphere):
        return 2.0 * _val_mm_float(node.radius, params)
    if isinstance(node, Frustum):
        rmin = min(_val_mm_float(node.r_bottom, params), _val_mm_float(node.r_top, params))
        rmax = max(_val_mm_float(node.r_bottom, params), _val_mm_float(node.r_top, params))
        return min(2.0 * max(rmin, 0.8), 2.0 * rmax, _val_mm_float(node.height, params))
    if isinstance(node, Shell):
        return min(_val_mm_float(node.thickness, params), _min_feature_scale(node.child, params))
    if isinstance(node, (Offset, LinearPattern, CircularPattern)):
        return _min_feature_scale(node.child, params)
    if isinstance(node, (ExtrudeSketch, TaperedExtrude)):
        bmin2, bmax2 = node.sketch.bounds_2d(params)
        return min(bmax2[0] - bmin2[0], bmax2[1] - bmin2[1], _val_mm_float(node.height, params))
    bmin, bmax = node.local_bounds(params)
    return min(max(bmax[i] - bmin[i], 1.0) for i in range(3))


def _eval_sdf_batched(
    root: Node,
    pts_np: np.ndarray,
    params: dict[str, Any] | None,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Evaluate SDF on (N, 3) numpy points in float32 chunks with no autograd overhead."""
    dev = get_compute_device(torch.float32)
    eff_chunk = 131072 if dev.type != "cpu" else chunk_size
    n = pts_np.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, eff_chunk):
            end = min(n, start + eff_chunk)
            t = torch.from_numpy(pts_np[start:end].astype(np.float32, copy=False)).to(dev)
            vals = root.eval_sdf(t, params=params, dtype=torch.float32)
            if vals.ndim == 2:
                vals = vals[0]
            out[start:end] = vals.detach().cpu().numpy()
    return out


def _eval_grad_batched(
    root: Node,
    pts_np: np.ndarray,
    params: dict[str, Any] | None,
    h_fd: float,
    chunk_size: int = 16384,
) -> np.ndarray:
    """Normalized central-difference SDF normals at (M, 3) points."""
    m = pts_np.shape[0]
    if m == 0:
        return np.zeros((0, 3), dtype=np.float64)
    dev = get_compute_device(torch.float32)
    eff_chunk = 32768 if dev.type != "cpu" else chunk_size
    offsets = np.array(
        [
            [h_fd, 0.0, 0.0],
            [-h_fd, 0.0, 0.0],
            [0.0, h_fd, 0.0],
            [0.0, -h_fd, 0.0],
            [0.0, 0.0, h_fd],
            [0.0, 0.0, -h_fd],
        ],
        dtype=np.float32,
    )
    normals = np.empty((m, 3), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, m, eff_chunk):
            end = min(m, start + eff_chunk)
            sub = pts_np[start:end].astype(np.float32, copy=False)
            k = sub.shape[0]
            probe = (sub[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            vals = root.eval_sdf(torch.from_numpy(probe).to(dev), params=params, dtype=torch.float32)
            if vals.ndim == 2:
                vals = vals[0]
            v = vals.detach().cpu().numpy().reshape(k, 6).astype(np.float64)
            g = np.stack([v[:, 0] - v[:, 1], v[:, 2] - v[:, 3], v[:, 4] - v[:, 5]], axis=-1) / (2.0 * h_fd)
            nrm = np.linalg.norm(g, axis=-1, keepdims=True)
            g = np.where(nrm > 1e-9, g / np.maximum(nrm, 1e-9), np.array([0.0, 0.0, 1.0]))
            normals[start:end] = g
    return normals


def _split_creases(
    mesh: trimesh.Trimesh,
    crease_angle_deg: float,
    sdf_normals: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Split vertices across sharp dihedral edges (> crease_angle_deg) for crisp CAD normals."""
    verts = mesh.vertices
    faces = mesh.faces
    if len(faces) == 0:
        return mesh
    cos_thresh = float(np.cos(np.radians(crease_angle_deg)))
    fnorm = mesh.face_normals

    n_verts = len(verts)
    v_to_f: list[list[int]] = [[] for _ in range(n_verts)]
    for fi, (a, b, c) in enumerate(faces):
        v_to_f[a].append(fi)
        v_to_f[b].append(fi)
        v_to_f[c].append(fi)

    new_verts: list[np.ndarray] = []
    new_normals: list[np.ndarray] = []
    new_faces = np.full_like(faces, -1)

    for vi in range(n_verts):
        inc = v_to_f[vi]
        if not inc:
            continue
        if len(inc) == 1:
            fi = inc[0]
            new_vi = len(new_verts)
            new_verts.append(verts[vi])
            new_normals.append(sdf_normals[vi] if sdf_normals is not None else fnorm[fi])
            for k in range(3):
                if faces[fi, k] == vi:
                    new_faces[fi, k] = new_vi
            continue

        clusters: list[list[int]] = []
        cluster_norms: list[np.ndarray] = []
        for fi in inc:
            fn = fnorm[fi]
            placed = False
            for ci, cn in enumerate(cluster_norms):
                if float(np.dot(fn, cn)) >= cos_thresh:
                    clusters[ci].append(fi)
                    avg = np.sum(fnorm[clusters[ci]], axis=0)
                    ln = float(np.linalg.norm(avg))
                    cluster_norms[ci] = avg / ln if ln > 1e-9 else fn
                    placed = True
                    break
            if not placed:
                clusters.append([fi])
                cluster_norms.append(fn.copy())

        for ci, cl in enumerate(clusters):
            new_vi = len(new_verts)
            new_verts.append(verts[vi])
            use_n = sdf_normals[vi] if (len(clusters) == 1 and sdf_normals is not None) else cluster_norms[ci]
            new_normals.append(use_n)
            for fi in cl:
                for k in range(3):
                    if faces[fi, k] == vi:
                        new_faces[fi, k] = new_vi

    out_mesh = trimesh.Trimesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=new_faces,
        vertex_normals=np.asarray(new_normals, dtype=np.float64),
        process=False,
    )
    if len(out_mesh.faces) > 0:
        valid_f = out_mesh.area_faces > 1e-10
        if not np.all(valid_f):
            out_mesh.update_faces(valid_f)
    return out_mesh


def _tessellate_brep_finalized(obj: BRepFinalized, linear_deflection: float = 0.05) -> trimesh.Trimesh:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(obj.shape, linear_deflection, False, 0.25, True)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0
    exp = TopExp_Explorer(obj.shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            n_nodes = tri.NbNodes()
            n_tris = tri.NbTriangles()
            v_arr = np.empty((n_nodes, 3), dtype=np.float64)
            for i in range(1, n_nodes + 1):
                p = tri.Node(i).Transformed(trsf)
                v_arr[i - 1] = (p.X(), p.Y(), p.Z())
            f_arr = np.empty((n_tris, 3), dtype=np.int64)
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, n_tris + 1):
                t = tri.Triangle(i)
                n1, n2, n3 = t.Get()
                if rev:
                    f_arr[i - 1] = (offset + n1 - 1, offset + n3 - 1, offset + n2 - 1)
                else:
                    f_arr[i - 1] = (offset + n1 - 1, offset + n2 - 1, offset + n3 - 1)
            all_verts.append(v_arr)
            all_faces.append(f_arr)
            offset += n_nodes
        exp.Next()
    if not all_verts:
        return trimesh.Trimesh()
    return trimesh.Trimesh(vertices=np.vstack(all_verts), faces=np.vstack(all_faces), process=True)


def extract_sdf_mesh(
    obj: Node | Part | BRepFinalized,
    *,
    params: dict[str, Any] | None = None,
    resolution: int = 96,
    crease_angle_deg: float = 42.0,
    split_creases: bool = True,
) -> trimesh.Trimesh:
    """Extract a manifold, sharp-feature-preserving watertight mesh via Manifold Dual Contouring."""
    if isinstance(obj, BRepFinalized):
        mesh = _tessellate_brep_finalized(obj)
        if split_creases and len(mesh.faces) > 0:
            return _split_creases(mesh, crease_angle_deg)
        return mesh

    root = obj.root if isinstance(obj, Part) else obj
    bmin_t, bmax_t = root.local_bounds(params)
    bmin = np.asarray(bmin_t, dtype=np.float64)
    bmax = np.asarray(bmax_t, dtype=np.float64)
    extents = np.maximum(bmax - bmin, 1e-3)
    max_ext = float(np.max(extents))

    # Adaptive multi-scale grid spacing: resolve small features (e.g. 1.8mm horns) cleanly
    r_min = max(0.5, _min_feature_scale(root, params))
    dx_base = max_ext / max(16, int(resolution))
    dx = min(dx_base, max(r_min / 3.5, max_ext / 240.0))

    pad = 2.5 * dx
    origin = bmin - pad
    span = extents + 2.0 * pad

    stride = 4
    nc = np.maximum(4, np.ceil(span / (stride * dx)).astype(np.int64))
    nf = nc * stride
    nx, ny, nz = int(nf[0]), int(nf[1]), int(nf[2])
    dx_vec = span / nf.astype(np.float64)

    # 1. Coarse-to-fine narrow-band SDF evaluation
    cnx, cny, cnz = int(nc[0]), int(nc[1]), int(nc[2])
    cx = np.linspace(origin[0], origin[0] + span[0], cnx + 1, dtype=np.float64)
    cy = np.linspace(origin[1], origin[1] + span[1], cny + 1, dtype=np.float64)
    cz = np.linspace(origin[2], origin[2] + span[2], cnz + 1, dtype=np.float64)
    cgx, cgy, cgz = np.meshgrid(cx, cy, cz, indexing="ij")
    c_sdf = _eval_sdf_batched(root, np.stack([cgx.ravel(), cgy.ravel(), cgz.ravel()], axis=-1), params).reshape(
        cnx + 1, cny + 1, cnz + 1
    )

    c_corners = [c_sdf[a : a + cnx, b : b + cny, c : c + cnz] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    c_min = np.minimum.reduce(c_corners)
    c_max = np.maximum.reduce(c_corners)
    c_diag = float(np.linalg.norm(dx_vec * stride))
    c_active = (c_min <= 1.5 * c_diag) & (c_max >= -1.5 * c_diag)

    f_vert_active = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    ci, cj, ck = np.nonzero(c_active)
    for di in range(stride + 1):
        for dj in range(stride + 1):
            for dk in range(stride + 1):
                f_vert_active[ci * stride + di, cj * stride + dj, ck * stride + dk] = True

    fine_sdf = np.empty((nx + 1, ny + 1, nz + 1), dtype=np.float32)
    c_rep = np.repeat(np.repeat(np.repeat(c_sdf[:-1, :-1, :-1], stride, axis=0), stride, axis=1), stride, axis=2)
    fine_sdf[:-1, :-1, :-1] = np.where(c_rep < 0.0, -10.0 * dx, 10.0 * dx).astype(np.float32)
    fine_sdf[-1, :, :] = 10.0 * dx
    fine_sdf[:, -1, :] = 10.0 * dx
    fine_sdf[:, :, -1] = 10.0 * dx

    ai, aj, ak = np.nonzero(f_vert_active)
    if len(ai) == 0:
        return trimesh.Trimesh()

    active_pts = np.stack(
        [origin[0] + ai * dx_vec[0], origin[1] + aj * dx_vec[1], origin[2] + ak * dx_vec[2]],
        axis=-1,
    )
    fine_sdf[ai, aj, ak] = _eval_sdf_batched(root, active_pts, params)

    # 2. Manifold Dual Contouring: compute 8-bit corner sign mask per cell and allocate >=1 vertex per cell
    neg = fine_sdf < 0.0
    cell_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                bit = ix + 2 * iy + 4 * iz
                cell_mask |= (neg[ix : ix + nx, iy : iy + ny, iz : iz + nz].astype(np.uint8) << bit)

    active_cell_mask = (cell_mask != 0) & (cell_mask != 255)
    cell_i, cell_j, cell_k = np.nonzero(active_cell_mask)
    n_cells = len(cell_i)
    if n_cells == 0:
        return trimesh.Trimesh()

    active_masks = cell_mask[cell_i, cell_j, cell_k]
    n_comps = _MDC_COMP_COUNT[active_masks].astype(np.int64)
    cell_vert_base = np.zeros(n_cells, dtype=np.int64)
    if n_cells > 1:
        cell_vert_base[1:] = np.cumsum(n_comps[:-1])
    n_verts = int(cell_vert_base[-1] + n_comps[-1])

    cell_id_grid = np.full((nx, ny, nz), -1, dtype=np.int64)
    cell_id_grid[cell_i, cell_j, cell_k] = np.arange(n_cells, dtype=np.int64)

    # Map each vertex back to its owning cell (for smooth interior clamping)
    vert_cell = np.repeat(np.arange(n_cells, dtype=np.int64), n_comps)

    # 3. Crossing edges & normals
    ex_i, ex_j, ex_k = np.nonzero(neg[:-1, :, :] != neg[1:, :, :])
    ey_i, ey_j, ey_k = np.nonzero(neg[:, :-1, :] != neg[:, 1:, :])
    ez_i, ez_j, ez_k = np.nonzero(neg[:, :, :-1] != neg[:, :, 1:])

    def _edge_crossings(ei: np.ndarray, ej: np.ndarray, ek: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        if len(ei) == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        di, dj, dk = int(axis == 0), int(axis == 1), int(axis == 2)
        f0 = fine_sdf[ei, ej, ek].astype(np.float64)
        f1 = fine_sdf[ei + di, ej + dj, ek + dk].astype(np.float64)
        denom = f0 - f1
        alpha = np.clip(np.where(np.abs(denom) > 1e-12, f0 / denom, 0.5), 0.0, 1.0)
        p = np.stack(
            [
                origin[0] + (ei + di * alpha) * dx_vec[0],
                origin[1] + (ej + dj * alpha) * dx_vec[1],
                origin[2] + (ek + dk * alpha) * dx_vec[2],
            ],
            axis=-1,
        )
        h_fd = max(1e-4, 0.05 * dx)
        nrm = _eval_grad_batched(root, p, params, h_fd=h_fd)
        f_p = _eval_sdf_batched(root, p, params).astype(np.float64)
        p = p - np.clip(f_p, -0.45 * dx, 0.45 * dx)[:, None] * nrm
        nrm = _eval_grad_batched(root, p, params, h_fd=h_fd)
        return p, nrm

    px, nx_e = _edge_crossings(ex_i, ex_j, ex_k, 0)
    py, ny_e = _edge_crossings(ey_i, ey_j, ey_k, 1)
    pz, nz_e = _edge_crossings(ez_i, ez_j, ez_k, 2)

    # Helper: for edge (axis, ei, ej, ek), the 4 incident (cell_shift, local_edge_idx)
    if True:
        _SHIFTS_LOCAL = {
            0: ((0, 0, 0, 0), (0, -1, 0, 1), (0, 0, -1, 2), (0, -1, -1, 3)),
            1: ((0, 0, 0, 4), (-1, 0, 0, 5), (0, 0, -1, 6), (-1, 0, -1, 7)),
            2: ((0, 0, 0, 8), (-1, 0, 0, 9), (0, -1, 0, 10), (-1, -1, 0, 11)),
        }

    def _lookup_vertex_ids(ei: np.ndarray, ej: np.ndarray, ek: np.ndarray, si: int, sj: int, sk: int, le: int) -> np.ndarray:
        ci_s, cj_s, ck_s = ei + si, ej + sj, ek + sk
        in_b = (ci_s >= 0) & (ci_s < nx) & (cj_s >= 0) & (cj_s < ny) & (ck_s >= 0) & (ck_s < nz)
        vids = np.full(len(ei), -1, dtype=np.int64)
        if not np.any(in_b):
            return vids
        cids = cell_id_grid[ci_s[in_b], cj_s[in_b], ck_s[in_b]]
        ok = cids >= 0
        if np.any(ok):
            valid_cids = cids[ok]
            cm = cell_mask[ci_s[in_b][ok], cj_s[in_b][ok], ck_s[in_b][ok]]
            lcomp = _MDC_EDGE_COMP[cm, le].astype(np.int64)
            vids_in_b = vids[in_b]
            vids_in_b[ok] = np.where(lcomp >= 0, cell_vert_base[valid_cids] + lcomp, -1)
            vids[in_b] = vids_in_b
        return vids

    # 4. Ju 2002 SVD-truncated QEF strictly per manifold component vertex
    p_sum = np.zeros((n_verts, 3), dtype=np.float64)
    cnt = np.zeros(n_verts, dtype=np.float64)
    for ei, ej, ek, p_e, axis in ((ex_i, ex_j, ex_k, px, 0), (ey_i, ey_j, ey_k, py, 1), (ez_i, ez_j, ez_k, pz, 2)):
        if len(ei) == 0:
            continue
        for si, sj, sk, le in _SHIFTS_LOCAL[axis]:
            vids = _lookup_vertex_ids(ei, ej, ek, si, sj, sk, le)
            valid = vids >= 0
            if np.any(valid):
                np.add.at(p_sum, vids[valid], p_e[valid])
                np.add.at(cnt, vids[valid], 1.0)

    p_mean = p_sum / np.maximum(cnt, 1.0)[:, None]

    A = np.zeros((n_verts, 3, 3), dtype=np.float64)
    rhs = np.zeros((n_verts, 3), dtype=np.float64)
    for ei, ej, ek, p_e, n_e, axis in (
        (ex_i, ex_j, ex_k, px, nx_e, 0),
        (ey_i, ey_j, ey_k, py, ny_e, 1),
        (ez_i, ez_j, ez_k, pz, nz_e, 2),
    ):
        if len(ei) == 0:
            continue
        outer = n_e[:, :, None] * n_e[:, None, :]
        for si, sj, sk, le in _SHIFTS_LOCAL[axis]:
            vids = _lookup_vertex_ids(ei, ej, ek, si, sj, sk, le)
            valid = vids >= 0
            if np.any(valid):
                vv = vids[valid]
                dp = p_e[valid] - p_mean[vv]
                np.add.at(A, vv, outer[valid])
                np.add.at(rhs, vv, n_e[valid] * np.sum(n_e[valid] * dp, axis=-1, keepdims=True))

    evals, evecs = np.linalg.eigh(A)  # (V, 3), (V, 3, 3)
    emax = np.maximum(evals[:, -1:], 1e-12)
    keep = evals >= (0.25**2) * emax  # SVD truncation: ignore flat-surface curvature noise
    inv_e = np.where(keep, 1.0 / np.maximum(evals, 1e-12), 0.0)
    vt_rhs = np.einsum("nji,nj->ni", evecs, rhs)
    delta = np.einsum("nij,nj->ni", evecs, inv_e * vt_rhs)

    # Smooth interior clamping: never collide on shared cell faces or flip quads
    c_lo = origin[None, :] + np.stack([cell_i[vert_cell], cell_j[vert_cell], cell_k[vert_cell]], axis=-1) * dx_vec[None, :]
    c_hi = c_lo + dx_vec[None, :]
    pad_in = 0.04 * dx_vec[None, :]
    verts = np.clip(p_mean + delta, c_lo + pad_in, c_hi - pad_in)

    # Smooth Rank-1 Isosurface Newton Polish: project smooth-surface vertices onto f(v)=0
    # while leaving sharp-edge/corner vertices (rank >= 2) anchored at QEF intersections.
    smooth_mask = np.sum(keep, axis=-1) == 1
    if np.any(smooth_mask):
        sv = verts[smooth_mask]
        fv = _eval_sdf_batched(root, sv, params).astype(np.float64)
        gv = _eval_grad_batched(root, sv, params, h_fd=max(1e-4, 0.04 * dx))
        verts[smooth_mask] = np.clip(
            sv - np.clip(fv, -0.35 * dx, 0.35 * dx)[:, None] * gv,
            c_lo[smooth_mask] + pad_in,
            c_hi[smooth_mask] - pad_in,
        )

    # 5. Emit consistently oriented manifold quads with dihedral-optimal diagonal splitting
    tris: list[np.ndarray] = []

    def _emit_quads(ei: np.ndarray, ej: np.ndarray, ek: np.ndarray, axis: int) -> None:
        if len(ei) == 0:
            return
        if axis == 0:
            valid = (ej >= 1) & (ej < ny) & (ek >= 1) & (ek < nz)
            ei, ej, ek = ei[valid], ej[valid], ek[valid]
            # Ring around +X edge in CCW order looking from +X: (0,0), (0,-1), (-1,-1), (-1,0) in (j,k)
            s_list = ((0, -1, -1, 3), (0, 0, -1, 2), (0, 0, 0, 0), (0, -1, 0, 1))
        elif axis == 1:
            valid = (ei >= 1) & (ei < nx) & (ek >= 1) & (ek < nz)
            ei, ej, ek = ei[valid], ej[valid], ek[valid]
            s_list = ((-1, 0, -1, 7), (-1, 0, 0, 5), (0, 0, 0, 4), (0, 0, -1, 6))
        else:
            valid = (ei >= 1) & (ei < nx) & (ej >= 1) & (ej < ny)
            ei, ej, ek = ei[valid], ej[valid], ek[valid]
            s_list = ((-1, -1, 0, 11), (0, -1, 0, 10), (0, 0, 0, 8), (-1, 0, 0, 9))

        if len(ei) == 0:
            return
        v0 = _lookup_vertex_ids(ei, ej, ek, *s_list[0])
        v1 = _lookup_vertex_ids(ei, ej, ek, *s_list[1])
        v2 = _lookup_vertex_ids(ei, ej, ek, *s_list[2])
        v3 = _lookup_vertex_ids(ei, ej, ek, *s_list[3])
        ok_q = (v0 >= 0) & (v1 >= 0) & (v2 >= 0) & (v3 >= 0)
        if not np.any(ok_q):
            return
        v0, v1, v2, v3 = v0[ok_q], v1[ok_q], v2[ok_q], v3[ok_q]
        flip = neg[ei[ok_q], ej[ok_q], ek[ok_q]]

        # Choose diagonal (v0, v2) vs (v1, v3) maximizing min dihedral dot product
        p0, p1, p2, p3 = verts[v0], verts[v1], verts[v2], verts[v3]
        na1 = np.cross(p1 - p0, p2 - p0)
        na2 = np.cross(p2 - p0, p3 - p0)
        nb1 = np.cross(p1 - p0, p3 - p0)
        nb2 = np.cross(p2 - p1, p3 - p1)
        na1 /= np.maximum(np.linalg.norm(na1, axis=-1, keepdims=True), 1e-12)
        na2 /= np.maximum(np.linalg.norm(na2, axis=-1, keepdims=True), 1e-12)
        nb1 /= np.maximum(np.linalg.norm(nb1, axis=-1, keepdims=True), 1e-12)
        nb2 /= np.maximum(np.linalg.norm(nb2, axis=-1, keepdims=True), 1e-12)
        use_diag_b = np.sum(nb1 * nb2, axis=-1) > np.sum(na1 * na2, axis=-1)

        # Diagonal A: (v0, v1, v2) + (v0, v2, v3); Diagonal B: (v0, v1, v3) + (v1, v2, v3)
        t1_a = np.where(flip[:, None], np.stack([v0, v1, v2], axis=-1), np.stack([v0, v2, v1], axis=-1))
        t2_a = np.where(flip[:, None], np.stack([v0, v2, v3], axis=-1), np.stack([v0, v3, v2], axis=-1))
        t1_b = np.where(flip[:, None], np.stack([v0, v1, v3], axis=-1), np.stack([v0, v3, v1], axis=-1))
        t2_b = np.where(flip[:, None], np.stack([v1, v2, v3], axis=-1), np.stack([v1, v3, v2], axis=-1))

        tris.append(np.where(use_diag_b[:, None], t1_b, t1_a))
        tris.append(np.where(use_diag_b[:, None], t2_b, t2_a))

    _emit_quads(ex_i, ex_j, ex_k, 0)
    _emit_quads(ey_i, ey_j, ey_k, 1)
    _emit_quads(ez_i, ez_j, ez_k, 2)

    if not tris:
        return trimesh.Trimesh()

    faces = np.vstack(tris)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if len(mesh.faces) > 0 and mesh.volume < 0.0:
        mesh.invert()
    if split_creases and len(mesh.faces) > 0:
        sdf_nrms = _eval_grad_batched(root, mesh.vertices, params, h_fd=max(1e-4, 0.04 * dx))
        return _split_creases(mesh, crease_angle_deg, sdf_normals=sdf_nrms)
    return mesh


def _zup_to_gltf_yup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Convert native Reify Z-up mesh to glTF 2.0 Y-up (-90 deg around +X: [x,y,z] -> [x,z,-y])."""
    if len(mesh.vertices) == 0:
        return mesh
    v = mesh.vertices
    v_new = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
    n_new = None
    if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) == len(v):
        n = mesh.vertex_normals
        n_new = np.column_stack([n[:, 0], n[:, 2], -n[:, 1]])
    return trimesh.Trimesh(vertices=v_new, faces=mesh.faces.copy(), vertex_normals=n_new, process=False)


def _make_colored_mesh(mesh: trimesh.Trimesh, hex_color: str) -> trimesh.Trimesh:
    rgba = _hex_to_rgba(hex_color)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=rgba,
        metallicFactor=0.15,
        roughnessFactor=0.45,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh


def export_sdf_glb_with_stats(
    obj: Node | Part | BRepFinalized | dict[str, Node | Part | BRepFinalized],
    glb_path: str | pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    color: str | None = None,
    resolution: int = 96,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Export Reify Node/Part/BRepFinalized or assembly dict to glTF 2.0 Y-up (.glb) and return raw mesh stats."""
    out = pathlib.Path(glb_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    items = obj.items() if isinstance(obj, dict) else [(obj.name if isinstance(obj, Part) else "part", obj)]
    total_verts = 0
    total_faces = 0
    all_watertight = True
    for idx, (name, item) in enumerate(items):
        m_raw = extract_sdf_mesh(item, params=params, resolution=resolution, split_creases=False)
        total_verts += len(m_raw.vertices)
        total_faces += len(m_raw.faces)
        all_watertight = all_watertight and bool(m_raw.is_watertight)
        if len(m_raw.faces) > 0:
            if isinstance(item, BRepFinalized):
                m_vis = _split_creases(m_raw, 42.0)
            else:
                root_item = item.root if isinstance(item, Part) else item
                bmin_t, bmax_t = root_item.local_bounds(params)
                max_ext = float(np.max(np.maximum(np.asarray(bmax_t) - np.asarray(bmin_t), 1e-3)))
                r_min = max(0.5, _min_feature_scale(root_item, params))
                dx = min(max_ext / max(16, int(resolution)), max(r_min / 3.5, max_ext / 240.0))
                sdf_nrms = _eval_grad_batched(root_item, m_raw.vertices, params, h_fd=max(1e-4, 0.04 * dx))
                m_vis = _split_creases(m_raw, 42.0, sdf_normals=sdf_nrms)
        else:
            m_vis = m_raw
        m_vis = _zup_to_gltf_yup(m_vis)
        c = color or _PALETTE[idx % len(_PALETTE)]
        _make_colored_mesh(m_vis, c)
        scene.add_geometry(m_vis, node_name=str(name), geom_name=str(name))

    scene.export(str(out))
    return out, {"verts": total_verts, "faces": total_faces, "watertight": all_watertight}


def export_sdf_glb(
    obj: Node | Part | BRepFinalized | dict[str, Node | Part | BRepFinalized],
    glb_path: str | pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    color: str | None = None,
    resolution: int = 96,
) -> pathlib.Path:
    """Export Reify Node/Part/BRepFinalized or assembly dict to glTF 2.0 Y-up (.glb) via Manifold DC."""
    out, _ = export_sdf_glb_with_stats(obj, glb_path, params=params, color=color, resolution=resolution)
    return out


def export_violation_overlay_glb(
    part: Node | Part | BRepFinalized,
    spec: Any,
    report: VerificationReport,
    glb_path: str | pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    resolution: int = 84,
) -> pathlib.Path:
    """Export multi-color Y-up GLB highlighting part (blue), obstacles (orange), and witness spheres (red)."""
    out = pathlib.Path(glb_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()

    m_part = _zup_to_gltf_yup(extract_sdf_mesh(part, params=params, resolution=resolution, split_creases=True))
    _make_colored_mesh(m_part, "#4a6fa5")
    scene.add_geometry(m_part, node_name="part", geom_name="part")

    for idx, rule in enumerate(getattr(spec, "obstacles", []) or []):
        m_obs = _zup_to_gltf_yup(
            extract_sdf_mesh(rule.obstacle, params=params, resolution=max(48, resolution // 2), split_creases=True)
        )
        _make_colored_mesh(m_obs, "#e67e22")
        scene.add_geometry(m_obs, node_name=f"obstacle_{idx}", geom_name=f"obstacle_{idx}")

    for idx, pt in enumerate(getattr(report, "witness_points", []) or []):
        if len(pt) == 3:
            sph = trimesh.creation.icosphere(subdivisions=3, radius=1.25)
            sph.apply_translation([float(pt[0]), float(pt[1]), float(pt[2])])
            sph = _zup_to_gltf_yup(sph)
            _make_colored_mesh(sph, "#e74c3c")
            scene.add_geometry(sph, node_name=f"witness_{idx}", geom_name=f"witness_{idx}")

    scene.export(str(out))
    return out
