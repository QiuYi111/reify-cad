"""Pure-SDF sharp-feature Dual Contouring mesher and GLB/mesh exporter (zero OCCT in hot path)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch
import trimesh

from reify.ast import Node, Part
from reify.finalize import BRepFinalized
from reify.types import VerificationReport

_PALETTE = ("#4a6fa5", "#5b8c5a", "#8e6c88", "#d4a373", "#3d5a80", "#e07a5f")


def _hex_to_rgba(hex_color: str) -> list[int]:
    s = hex_color.lstrip("#")
    if len(s) == 6:
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255]
    if len(s) == 8:
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16)]
    return [74, 111, 165, 255]


def _eval_sdf_batched(
    root: Node,
    pts_np: np.ndarray,
    params: dict[str, Any] | None,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Evaluate SDF on (N, 3) numpy points in float32 chunks with no autograd overhead."""
    n = pts_np.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            t = torch.from_numpy(pts_np[start:end].astype(np.float32, copy=False))
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
        for start in range(0, m, chunk_size):
            end = min(m, start + chunk_size)
            sub = pts_np[start:end].astype(np.float32, copy=False)  # (K, 3)
            k = sub.shape[0]
            probe = (sub[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            vals = root.eval_sdf(torch.from_numpy(probe), params=params, dtype=torch.float32)
            if vals.ndim == 2:
                vals = vals[0]
            v = vals.detach().cpu().numpy().reshape(k, 6).astype(np.float64)
            g = np.stack([v[:, 0] - v[:, 1], v[:, 2] - v[:, 3], v[:, 4] - v[:, 5]], axis=-1) / (2.0 * h_fd)
            nrm = np.linalg.norm(g, axis=-1, keepdims=True)
            g = np.where(nrm > 1e-9, g / np.maximum(nrm, 1e-9), np.array([0.0, 0.0, 1.0]))
            normals[start:end] = g
    return normals


def _split_creases(mesh: trimesh.Trimesh, crease_angle_deg: float) -> trimesh.Trimesh:
    """Split vertices across sharp dihedral edges (> crease_angle_deg) for crisp CAD normals."""
    verts = mesh.vertices
    faces = mesh.faces
    if len(faces) == 0:
        return mesh
    cos_thresh = float(np.cos(np.radians(crease_angle_deg)))
    fnorm = mesh.face_normals  # (F, 3)

    # Build vertex -> list of incident face indices
    n_verts = len(verts)
    v_to_f: list[list[int]] = [[] for _ in range(n_verts)]
    for fi, (a, b, c) in enumerate(faces):
        v_to_f[a].append(fi)
        v_to_f[b].append(fi)
        v_to_f[c].append(fi)

    new_verts: list[np.ndarray] = []
    new_normals: list[np.ndarray] = []
    new_faces = faces.copy()

    # Map (face_idx, local_corner) -> new vertex index
    for vi in range(n_verts):
        inc = v_to_f[vi]
        if not inc:
            continue
        if len(inc) == 1:
            fi = inc[0]
            new_vi = len(new_verts)
            new_verts.append(verts[vi])
            new_normals.append(fnorm[fi])
            for k in range(3):
                if new_faces[fi, k] == vi:
                    new_faces[fi, k] = new_vi
            continue

        # Greedy clustering by face normal angle
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
            new_normals.append(cluster_norms[ci])
            for fi in cl:
                for k in range(3):
                    if new_faces[fi, k] == vi:
                        new_faces[fi, k] = new_vi

    out = trimesh.Trimesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=new_faces,
        vertex_normals=np.asarray(new_normals, dtype=np.float64),
        process=False,
    )
    return out


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
    crease_angle_deg: float = 28.0,
    split_creases: bool = True,
) -> trimesh.Trimesh:
    """Extract a sharp-feature-preserving watertight mesh directly from Reify SDF (Dual Contouring)."""
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
    dx = float(np.max(extents) / max(16, int(resolution)))

    # Pad bounds so boundary faces always have exterior (+) samples
    pad = 2.5 * dx
    origin = bmin - pad
    span = extents + 2.0 * pad

    # Choose grid cell counts along each axis (multiple of 4 for clean 4x coarse-to-fine)
    stride = 4
    nc = np.maximum(4, np.ceil(span / (stride * dx)).astype(np.int64))
    nf = nc * stride  # fine cells (nx, ny, nz)
    nx, ny, nz = int(nf[0]), int(nf[1]), int(nf[2])
    dx_vec = span / nf.astype(np.float64)

    # 1. Coarse SDF grid
    cnx, cny, cnz = int(nc[0]), int(nc[1]), int(nc[2])
    cx = np.linspace(origin[0], origin[0] + span[0], cnx + 1, dtype=np.float64)
    cy = np.linspace(origin[1], origin[1] + span[1], cny + 1, dtype=np.float64)
    cz = np.linspace(origin[2], origin[2] + span[2], cnz + 1, dtype=np.float64)
    cgx, cgy, cgz = np.meshgrid(cx, cy, cz, indexing="ij")
    c_pts = np.stack([cgx.ravel(), cgy.ravel(), cgz.ravel()], axis=-1)
    c_sdf = _eval_sdf_batched(root, c_pts, params).reshape(cnx + 1, cny + 1, cnz + 1)

    c000 = c_sdf[:-1, :-1, :-1]
    c100 = c_sdf[1:, :-1, :-1]
    c010 = c_sdf[:-1, 1:, :-1]
    c110 = c_sdf[1:, 1:, :-1]
    c001 = c_sdf[:-1, :-1, 1:]
    c101 = c_sdf[1:, :-1, 1:]
    c011 = c_sdf[:-1, 1:, 1:]
    c111 = c_sdf[1:, 1:, 1:]
    c_min = np.minimum(
        np.minimum(np.minimum(c000, c100), np.minimum(c010, c110)),
        np.minimum(np.minimum(c001, c101), np.minimum(c011, c111)),
    )
    c_max = np.maximum(
        np.maximum(np.maximum(c000, c100), np.maximum(c010, c110)),
        np.maximum(np.maximum(c001, c101), np.maximum(c011, c111)),
    )
    c_diag = float(np.linalg.norm(dx_vec * stride))
    c_active = (c_min <= 1.5 * c_diag) & (c_max >= -1.5 * c_diag)

    # Expand active coarse cells to fine vertex mask
    f_vert_active = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    ci, cj, ck = np.nonzero(c_active)
    for di in range(stride + 1):
        for dj in range(stride + 1):
            for dk in range(stride + 1):
                f_vert_active[ci * stride + di, cj * stride + dj, ck * stride + dk] = True

    # Initialize fine SDF with coarse sign-preserving far-field values
    fine_sdf = np.empty((nx + 1, ny + 1, nz + 1), dtype=np.float32)
    c_rep = np.repeat(np.repeat(np.repeat(c000, stride, axis=0), stride, axis=1), stride, axis=2)
    fine_sdf[:-1, :-1, :-1] = np.where(c_rep < 0.0, -10.0 * dx, 10.0 * dx).astype(np.float32)
    fine_sdf[-1, :, :] = 10.0 * dx
    fine_sdf[:, -1, :] = 10.0 * dx
    fine_sdf[:, :, -1] = 10.0 * dx

    ai, aj, ak = np.nonzero(f_vert_active)
    if len(ai) == 0:
        return trimesh.Trimesh()

    active_pts = np.stack(
        [
            origin[0] + ai * dx_vec[0],
            origin[1] + aj * dx_vec[1],
            origin[2] + ak * dx_vec[2],
        ],
        axis=-1,
    )
    fine_sdf[ai, aj, ak] = _eval_sdf_batched(root, active_pts, params)

    # 2. Identify sign-changing fine cells
    neg = fine_sdf < 0.0
    c_neg = (
        neg[:-1, :-1, :-1].astype(np.int8)
        + neg[1:, :-1, :-1]
        + neg[:-1, 1:, :-1]
        + neg[1:, 1:, :-1]
        + neg[:-1, :-1, 1:]
        + neg[1:, :-1, 1:]
        + neg[:-1, 1:, 1:]
        + neg[1:, 1:, 1:]
    )
    active_cell_mask = (c_neg > 0) & (c_neg < 8)
    cell_i, cell_j, cell_k = np.nonzero(active_cell_mask)
    n_cells = len(cell_i)
    if n_cells == 0:
        return trimesh.Trimesh()

    cell_id_grid = np.full((nx, ny, nz), -1, dtype=np.int64)
    cell_id_grid[cell_i, cell_j, cell_k] = np.arange(n_cells, dtype=np.int64)

    # 3. Find all crossing edges along X, Y, Z and compute intersection + normal
    ex_mask = neg[:-1, :, :] != neg[1:, :, :]
    ey_mask = neg[:, :-1, :] != neg[:, 1:, :]
    ez_mask = neg[:, :, :-1] != neg[:, :, 1:]

    ex_i, ex_j, ex_k = np.nonzero(ex_mask)
    ey_i, ey_j, ey_k = np.nonzero(ey_mask)
    ez_i, ez_j, ez_k = np.nonzero(ez_mask)

    def _edge_crossings(ei: np.ndarray, ej: np.ndarray, ek: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        if len(ei) == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        di, dj, dk = int(axis == 0), int(axis == 1), int(axis == 2)
        f0 = fine_sdf[ei, ej, ek].astype(np.float64)
        f1 = fine_sdf[ei + di, ej + dj, ek + dk].astype(np.float64)
        denom = f0 - f1
        alpha = np.where(np.abs(denom) > 1e-12, f0 / denom, 0.5)
        alpha = np.clip(alpha, 0.0, 1.0)
        p = np.stack(
            [
                origin[0] + (ei + di * alpha) * dx_vec[0],
                origin[1] + (ej + dj * alpha) * dx_vec[1],
                origin[2] + (ek + dk * alpha) * dx_vec[2],
            ],
            axis=-1,
        )
        # 1 damped Newton step onto f = 0 for sub-grid precision on curves
        h_fd = max(1e-4, 0.05 * dx)
        nrm = _eval_grad_batched(root, p, params, h_fd=h_fd)
        f_p = _eval_sdf_batched(root, p, params).astype(np.float64)
        step = np.clip(f_p, -0.45 * dx, 0.45 * dx)[:, None]
        p = p - step * nrm
        nrm = _eval_grad_batched(root, p, params, h_fd=h_fd)
        return p, nrm

    px, nx_e = _edge_crossings(ex_i, ex_j, ex_k, 0)
    py, ny_e = _edge_crossings(ey_i, ey_j, ey_k, 1)
    pz, nz_e = _edge_crossings(ez_i, ez_j, ez_k, 2)

    # 4. Accumulate batched QEF per active cell: A (n_cells, 3, 3), b (n_cells, 3)
    A = np.zeros((n_cells, 3, 3), dtype=np.float64)
    b = np.zeros((n_cells, 3), dtype=np.float64)
    p_sum = np.zeros((n_cells, 3), dtype=np.float64)
    cnt = np.zeros(n_cells, dtype=np.float64)

    def _accum_edges(ei: np.ndarray, ej: np.ndarray, ek: np.ndarray, p_e: np.ndarray, n_e: np.ndarray, axis: int) -> None:
        if len(ei) == 0:
            return
        outer = n_e[:, :, None] * n_e[:, None, :]  # (E, 3, 3)
        rhs = n_e * np.sum(n_e * p_e, axis=-1, keepdims=True)  # (E, 3)
        # An interior edge along `axis` touches 4 cells in the two orthogonal axes
        if axis == 0:
            shifts = ((0, 0, 0), (0, -1, 0), (0, 0, -1), (0, -1, -1))
            limits = (nx, ny, nz)
        elif axis == 1:
            shifts = ((0, 0, 0), (-1, 0, 0), (0, 0, -1), (-1, 0, -1))
            limits = (nx, ny, nz)
        else:
            shifts = ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0))
            limits = (nx, ny, nz)

        for si, sj, sk in shifts:
            ci_s = ei + si
            cj_s = ej + sj
            ck_s = ek + sk
            in_b = (
                (ci_s >= 0) & (ci_s < limits[0]) &
                (cj_s >= 0) & (cj_s < limits[1]) &
                (ck_s >= 0) & (ck_s < limits[2])
            )
            cids = np.full(len(ei), -1, dtype=np.int64)
            cids[in_b] = cell_id_grid[ci_s[in_b], cj_s[in_b], ck_s[in_b]]
            valid = cids >= 0
            if np.any(valid):
                vc = cids[valid]
                np.add.at(A, vc, outer[valid])
                np.add.at(b, vc, rhs[valid])
                np.add.at(p_sum, vc, p_e[valid])
                np.add.at(cnt, vc, 1.0)

    _accum_edges(ex_i, ex_j, ex_k, px, nx_e, 0)
    _accum_edges(ey_i, ey_j, ey_k, py, ny_e, 1)
    _accum_edges(ez_i, ez_j, ez_k, pz, nz_e, 2)

    cnt_safe = np.maximum(cnt, 1.0)[:, None]
    p_mean = p_sum / cnt_safe
    lam = 1e-3
    A += lam * np.eye(3, dtype=np.float64)[None, :, :]
    b += lam * p_mean

    verts = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    c_lo = origin[None, :] + np.stack([cell_i, cell_j, cell_k], axis=-1) * dx_vec[None, :]
    c_hi = c_lo + dx_vec[None, :]
    out_of_cell = np.any((verts < c_lo - 0.15 * dx_vec[None, :]) | (verts > c_hi + 0.15 * dx_vec[None, :]), axis=-1)
    verts = np.where(out_of_cell[:, None], p_mean, np.clip(verts, c_lo, c_hi))

    # 5. Emit consistently oriented quads (2 triangles each) for interior crossing edges
    tris: list[np.ndarray] = []

    def _emit_quads(ei: np.ndarray, ej: np.ndarray, ek: np.ndarray, axis: int) -> None:
        if len(ei) == 0:
            return
        if axis == 0:
            valid = (ej >= 1) & (ej < ny) & (ek >= 1) & (ek < nz)
            ei, ej, ek = ei[valid], ej[valid], ek[valid]
            c0 = cell_id_grid[ei, ej - 1, ek - 1]
            c1 = cell_id_grid[ei, ej, ek - 1]
            c2 = cell_id_grid[ei, ej, ek]
            c3 = cell_id_grid[ei, ej - 1, ek]
            flip = neg[ei, ej, ek]  # True => f0 < 0, f1 >= 0 => outward +X
        elif axis == 1:
            valid = (ei >= 1) & (ei < nx) & (ek >= 1) & (ek < nz)
            ei, ej, ek = ei[valid], ej[valid], ek[valid]
            c0 = cell_id_grid[ei - 1, ej, ek - 1]
            c1 = cell_id_grid[ei - 1, ej, ek]
            c2 = cell_id_grid[ei, ej, ek]
            c3 = cell_id_grid[ei, ej, ek - 1]
            flip = neg[ei, ej, ek]  # True => f0 < 0, f1 >= 0 => outward +Y
        else:
            valid = (ei >= 1) & (ei < nx) & (ej >= 1) & (ej < ny)
            ei, ej, ek = ei[valid], ej[valid], ek[valid]
            c0 = cell_id_grid[ei - 1, ej - 1, ek]
            c1 = cell_id_grid[ei, ej - 1, ek]
            c2 = cell_id_grid[ei, ej, ek]
            c3 = cell_id_grid[ei - 1, ej, ek]
            flip = neg[ei, ej, ek]  # True => f0 < 0, f1 >= 0 => outward +Z

        ok_q = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
        if not np.any(ok_q):
            return
        c0, c1, c2, c3, flip = c0[ok_q], c1[ok_q], c2[ok_q], c3[ok_q], flip[ok_q]
        # When flip=True, outward normal is +axis => (c0, c1, c2) & (c0, c2, c3)
        # When flip=False, outward normal is -axis => (c0, c2, c1) & (c0, c3, c2)
        t1 = np.where(flip[:, None], np.stack([c0, c1, c2], axis=-1), np.stack([c0, c2, c1], axis=-1))
        t2 = np.where(flip[:, None], np.stack([c0, c2, c3], axis=-1), np.stack([c0, c3, c2], axis=-1))
        tris.append(t1)
        tris.append(t2)

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
        return _split_creases(mesh, crease_angle_deg)
    return mesh


def _make_colored_mesh(mesh: trimesh.Trimesh, hex_color: str) -> trimesh.Trimesh:
    rgba = _hex_to_rgba(hex_color)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=rgba,
        metallicFactor=0.15,
        roughnessFactor=0.45,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh


def export_sdf_glb(
    obj: Node | Part | BRepFinalized | dict[str, Node | Part | BRepFinalized],
    glb_path: str | pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    color: str | None = None,
    resolution: int = 96,
) -> pathlib.Path:
    """Export Reify Node/Part/BRepFinalized or assembly dict to binary glTF (.glb) via pure SDF."""
    out = pathlib.Path(glb_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    items = obj.items() if isinstance(obj, dict) else [(obj.name if isinstance(obj, Part) else "part", obj)]
    for idx, (name, item) in enumerate(items):
        m = extract_sdf_mesh(item, params=params, resolution=resolution, split_creases=True)
        c = color or _PALETTE[idx % len(_PALETTE)]
        _make_colored_mesh(m, c)
        scene.add_geometry(m, node_name=str(name), geom_name=str(name))

    scene.export(str(out))
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
    """Export multi-color GLB highlighting part (blue), obstacles (orange), and witness spheres (red)."""
    out = pathlib.Path(glb_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()

    m_part = extract_sdf_mesh(part, params=params, resolution=resolution, split_creases=True)
    _make_colored_mesh(m_part, "#4a6fa5")
    scene.add_geometry(m_part, node_name="part", geom_name="part")

    for idx, rule in enumerate(getattr(spec, "obstacles", []) or []):
        m_obs = extract_sdf_mesh(rule.obstacle, params=params, resolution=max(48, resolution // 2), split_creases=True)
        _make_colored_mesh(m_obs, "#e67e22")
        scene.add_geometry(m_obs, node_name=f"obstacle_{idx}", geom_name=f"obstacle_{idx}")

    for idx, pt in enumerate(getattr(report, "witness_points", []) or []):
        if len(pt) == 3:
            sph = trimesh.creation.icosphere(subdivisions=3, radius=1.25)
            sph.apply_translation([float(pt[0]), float(pt[1]), float(pt[2])])
            _make_colored_mesh(sph, "#e74c3c")
            scene.add_geometry(sph, node_name=f"witness_{idx}", geom_name=f"witness_{idx}")

    scene.export(str(out))
    return out
