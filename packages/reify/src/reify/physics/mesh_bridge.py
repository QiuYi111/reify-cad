"""Dual-channel 3D tetrahedral mesh bridge (Gmsh OCCT Channel A + BCC Isosurface Stuffing Channel B) & CalculiX .inp (M3)."""

from __future__ import annotations

import math
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any

import gmsh
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from OCP.BRepTools import BRepTools

from reify.ast import Node, Part
from reify.brep_compiler import compile_brep
from reify.query import GeometryQuery
from reify.types import Status, VerificationReport


def compute_tet_quality(nodes: np.ndarray, tets: np.ndarray) -> dict[str, Any]:
    """Compute signed volumes, dihedral angle bounds [deg], and scaled Jacobians for C3D4 tets."""
    # Use first 4 corner vertices even for C3D10 (10-node quadratic tets)
    t4 = tets[:, :4]
    p0 = nodes[t4[:, 0]]
    p1 = nodes[t4[:, 1]]
    p2 = nodes[t4[:, 2]]
    p3 = nodes[t4[:, 3]]

    e01 = p1 - p0
    e02 = p2 - p0
    e03 = p3 - p0
    vols = np.einsum("ij,ij->i", e01, np.cross(e02, e03)) / 6.0

    # Face outward-oriented normals for the 4 faces
    # f0: (p1, p3, p2), f1: (p0, p2, p3), f2: (p0, p3, p1), f3: (p0, p1, p2)
    n0 = np.cross(p3 - p1, p2 - p1)
    n1 = np.cross(p2 - p0, p3 - p0)
    n2 = np.cross(p3 - p0, p1 - p0)
    n3 = np.cross(p1 - p0, p2 - p0)

    def _norm(a: np.ndarray) -> np.ndarray:
        return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-18)

    un = [_norm(n0), _norm(n1), _norm(n2), _norm(n3)]
    dihedrals = []
    for i in range(4):
        for j in range(i + 1, 4):
            cos_ij = np.clip(-np.sum(un[i] * un[j], axis=-1), -1.0, 1.0)
            dihedrals.append(np.degrees(np.arccos(cos_ij)))
    all_dih = np.stack(dihedrals, axis=-1)

    # Scaled Jacobian: 6 * sqrt(2) * V / l_rms^3
    edges = [p1 - p0, p2 - p0, p3 - p0, p2 - p1, p3 - p1, p3 - p2]
    l_rms = np.sqrt(np.mean(np.stack([np.sum(e * e, axis=-1) for e in edges], axis=-1), axis=-1))
    scaled_jac = 6.0 * math.sqrt(2.0) * vols / np.maximum(l_rms**3, 1e-18)

    return {
        "n_nodes": int(nodes.shape[0]),
        "n_tets": int(tets.shape[0]),
        "min_volume": float(np.min(vols)) if vols.size else 0.0,
        "total_volume": float(np.sum(vols)) if vols.size else 0.0,
        "min_dihedral_deg": float(np.min(all_dih)) if all_dih.size else 0.0,
        "max_dihedral_deg": float(np.max(all_dih)) if all_dih.size else 0.0,
        "min_scaled_jacobian": float(np.min(scaled_jac)) if scaled_jac.size else 0.0,
    }


def _orient_tets_positive(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    t = tets.copy()
    p0 = nodes[t[:, 0]]
    p1 = nodes[t[:, 1]]
    p2 = nodes[t[:, 2]]
    p3 = nodes[t[:, 3]]
    vols = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    neg = vols < 0.0
    if np.any(neg):
        t[neg, 1], t[neg, 2] = t[neg, 2].copy(), t[neg, 1].copy()
    return t


def mesh_brep_gmsh_channel_a(
    node: Node | Part,
    mesh_size_mm: float = 2.0,
    order: int = 1,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Channel A: sharp-feature-preserving 3D tetrahedral mesh via Gmsh OpenCASCADE kernel."""
    root = node.root if isinstance(node, Part) else node
    st, shape = compile_brep(root, params=params)
    if st != Status.PASS or shape is None:
        return {"status": Status.OUT_OF_DOMAIN, "message": "AST is not B-Rep exportable for Channel A."}

    with tempfile.TemporaryDirectory() as td:
        brep_path = pathlib.Path(td) / "part.brep"
        BRepTools.Write_s(shape, str(brep_path))

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("reify_ch_a")
            gmsh.model.occ.importShapes(str(brep_path))
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.25 * mesh_size_mm)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_mm)
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            gmsh.model.mesh.generate(3)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            # Map gmsh 1-based arbitrary node tags to dense 0-based indices
            tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
            nodes = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

            el_type = 4 if order == 1 else 11  # 4=4-node tet, 11=10-node second-order tet
            elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
            tets_list: list[np.ndarray] = []
            for et, nt in zip(elem_types, elem_node_tags):
                if int(et) == el_type:
                    stride = 4 if order == 1 else 10
                    raw = np.array([tag_to_idx[int(t)] for t in nt], dtype=np.int64).reshape(-1, stride)
                    tets_list.append(raw)
            if not tets_list:
                return {"status": Status.FAIL, "message": "Gmsh generated zero 3D tetrahedral elements."}
            tets = np.concatenate(tets_list, axis=0)
        finally:
            gmsh.finalize()

    if order == 1:
        tets = _orient_tets_positive(nodes, tets)
    elif order == 2:
        # Gmsh 10-node tet swaps edge mid-nodes 8 and 9 relative to Abaqus/CalculiX C3D10
        tets = tets[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    # Compact unused nodes
    used = np.unique(tets.ravel())
    remap = np.full(nodes.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    nodes = nodes[used]
    tets = remap[tets]

    qual = compute_tet_quality(nodes, tets)
    return {
        "status": Status.PASS if qual["min_volume"] > 0.0 else Status.FAIL,
        "channel": "A_GMSH_OCCT",
        "elem_type": "C3D4" if order == 1 else "C3D10",
        "nodes": nodes,
        "tets": tets,
        "quality": qual,
    }


def mesh_implicit_isosurface_stuffing_channel_b(
    node: Node | Part,
    cell_size_mm: float = 1.5,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Channel B: Labelle & Shewchuk (2007) BCC Isosurface Stuffing for implicit-only geometry."""
    root = node.root if isinstance(node, Part) else node
    gq = GeometryQuery(root)
    bmin, bmax = root.local_bounds(params)
    h = float(cell_size_mm)
    pad = 1.5 * h
    ox, oy, oz = bmin[0] - pad, bmin[1] - pad, bmin[2] - pad
    nx = max(2, int(math.ceil((bmax[0] - bmin[0] + 2.0 * pad) / h)))
    ny = max(2, int(math.ceil((bmax[1] - bmin[1] + 2.0 * pad) / h)))
    nz = max(2, int(math.ceil((bmax[2] - bmin[2] + 2.0 * pad) / h)))

    # Black grid: (nx+1) x (ny+1) x (nz+1)
    bx = np.arange(nx + 1, dtype=np.float64) * h + ox
    by = np.arange(ny + 1, dtype=np.float64) * h + oy
    bz = np.arange(nz + 1, dtype=np.float64) * h + oz
    gbx, gby, gbz = np.meshgrid(bx, by, bz, indexing="ij")
    b_pts = np.column_stack([gbx.ravel(), gby.ravel(), gbz.ravel()])
    n_black = b_pts.shape[0]

    # Red grid: nx x ny x nz at cube centers (plus 1-cell border so every black-black edge has 4 red neighbors)
    rx = (np.arange(-1, nx + 1, dtype=np.float64) + 0.5) * h + ox
    ry = (np.arange(-1, ny + 1, dtype=np.float64) + 0.5) * h + oy
    rz = (np.arange(-1, nz + 1, dtype=np.float64) + 0.5) * h + oz
    grx, gry, grz = np.meshgrid(rx, ry, rz, indexing="ij")
    r_pts = np.column_stack([grx.ravel(), gry.ravel(), grz.ravel()])

    verts = np.vstack([b_pts, r_pts])
    f_vals = gq.eval_field(torch.from_numpy(verts).to(torch.float64), params=params, dtype=torch.float64).detach().cpu().numpy()

    def _b_idx(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    r_ny = ny + 2
    r_nz = nz + 2

    def _r_idx(i: int, j: int, k: int) -> int:
        # i, j, k in [-1..nx], [-1..ny], [-1..nz] -> shift by +1
        return n_black + ((i + 1) * r_ny + (j + 1)) * r_nz + (k + 1)

    # Build BCC background tets around every interior black-black edge
    bg_tets: list[tuple[int, int, int, int]] = []
    # X-edges
    for i in range(nx):
        for j in range(1, ny):
            for k in range(1, nz):
                b0, b1 = _b_idx(i, j, k), _b_idx(i + 1, j, k)
                r0 = _r_idx(i, j - 1, k - 1)
                r1 = _r_idx(i, j, k - 1)
                r2 = _r_idx(i, j, k)
                r3 = _r_idx(i, j - 1, k)
                bg_tets.extend([(b0, b1, r0, r1), (b0, b1, r1, r2), (b0, b1, r2, r3), (b0, b1, r3, r0)])
    # Y-edges
    for i in range(1, nx):
        for j in range(ny):
            for k in range(1, nz):
                b0, b1 = _b_idx(i, j, k), _b_idx(i, j + 1, k)
                r0 = _r_idx(i - 1, j, k - 1)
                r1 = _r_idx(i, j, k - 1)
                r2 = _r_idx(i, j, k)
                r3 = _r_idx(i - 1, j, k)
                bg_tets.extend([(b0, b1, r0, r1), (b0, b1, r1, r2), (b0, b1, r2, r3), (b0, b1, r3, r0)])
    # Z-edges
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(nz):
                b0, b1 = _b_idx(i, j, k), _b_idx(i, j, k + 1)
                r0 = _r_idx(i - 1, j - 1, k)
                r1 = _r_idx(i, j - 1, k)
                r2 = _r_idx(i, j, k)
                r3 = _r_idx(i - 1, j, k)
                bg_tets.extend([(b0, b1, r0, r1), (b0, b1, r1, r2), (b0, b1, r2, r3), (b0, b1, r3, r0)])

    bg_arr = np.array(bg_tets, dtype=np.int64)
    # Keep tets with at least one inside vertex (f <= 0)
    f_tet = f_vals[bg_arr]
    has_in = np.any(f_tet <= 0.0, axis=1)
    cand_tets = bg_arr[has_in]

    # Labelle & Shewchuk vertex snapping & cut-edge bisection
    alpha_long = 0.24999
    alpha_short = 0.42978
    warped_pos = verts.copy()
    is_black = np.zeros(verts.shape[0], dtype=bool)
    is_black[:n_black] = True

    def _bisect_edges_batch(
        p_a: np.ndarray, p_b: np.ndarray, fa: np.ndarray, fb: np.ndarray, steps: int = 10
    ) -> np.ndarray:
        if len(p_a) == 0:
            return np.zeros(0, dtype=np.float64)
        lo = np.zeros(len(p_a), dtype=np.float64)
        hi = np.ones(len(p_a), dtype=np.float64)
        fa_in = fa <= 0.0
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            pm = (1.0 - mid[:, None]) * p_a + mid[:, None] * p_b
            fm = gq.eval_field(torch.from_numpy(pm).to(torch.float64), params=params, dtype=torch.float64).detach().cpu().numpy()
            same_side = (fm <= 0.0) == fa_in
            lo = np.where(same_side, mid, lo)
            hi = np.where(same_side, hi, mid)
        return 0.5 * (lo + hi)

    cut_cache: dict[tuple[int, int], int] = {}
    node_list = list(warped_pos)
    out_tets: list[tuple[int, int, int, int]] = []

    # First pass: batch-bisect all unique cut edges of cand_tets, then warp close vertices
    edge_pairs = cand_tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
    edge_pairs = np.sort(edge_pairs, axis=1)
    uniq_edges = np.unique(edge_pairs, axis=0)
    u_all, v_all = uniq_edges[:, 0], uniq_edges[:, 1]
    fu_all, fv_all = f_vals[u_all], f_vals[v_all]
    cut_mask = (fu_all <= 0.0) != (fv_all <= 0.0)
    cut_edges = uniq_edges[cut_mask]

    t_pass1 = _bisect_edges_batch(
        verts[cut_edges[:, 0]], verts[cut_edges[:, 1]], f_vals[cut_edges[:, 0]], f_vals[cut_edges[:, 1]]
    )
    snap_targets: dict[int, tuple[float, np.ndarray]] = {}
    for idx_e in range(len(cut_edges)):
        u = int(cut_edges[idx_e, 0])
        v = int(cut_edges[idx_e, 1])
        t = float(t_pass1[idx_e])
        alpha = alpha_long if (is_black[u] == is_black[v]) else alpha_short
        p_cut = (1.0 - t) * verts[u] + t * verts[v]
        if t < alpha:
            if u not in snap_targets or t < snap_targets[u][0]:
                snap_targets[u] = (t, p_cut)
        elif (1.0 - t) < alpha:
            if v not in snap_targets or (1.0 - t) < snap_targets[v][0]:
                snap_targets[v] = (1.0 - t, p_cut)

    for vid, (_, p_c) in snap_targets.items():
        node_list[vid] = p_c
        f_vals[vid] = 0.0  # snapped onto boundary

    # Pass 2: batch-bisect all remaining unsnapped cut edges once
    rem_mask = (
        (f_vals[u_all] != 0.0)
        & (f_vals[v_all] != 0.0)
        & ((f_vals[u_all] <= 0.0) != (f_vals[v_all] <= 0.0))
    )
    rem_edges = uniq_edges[rem_mask]
    node_arr_tmp = np.array(node_list, dtype=np.float64)
    t_pass2 = _bisect_edges_batch(
        node_arr_tmp[rem_edges[:, 0]],
        node_arr_tmp[rem_edges[:, 1]],
        f_vals[rem_edges[:, 0]],
        f_vals[rem_edges[:, 1]],
    )
    p_cuts2 = (1.0 - t_pass2[:, None]) * node_arr_tmp[rem_edges[:, 0]] + t_pass2[:, None] * node_arr_tmp[rem_edges[:, 1]]
    for idx_e in range(len(rem_edges)):
        u = int(rem_edges[idx_e, 0])
        v = int(rem_edges[idx_e, 1])
        new_idx = len(node_list)
        node_list.append(p_cuts2[idx_e])
        cut_cache[(u, v)] = new_idx

    def _get_cut_node(u: int, v: int) -> int:
        if f_vals[u] == 0.0:
            return u
        if f_vals[v] == 0.0:
            return v
        key = (u, v) if u < v else (v, u)
        return cut_cache[key]

    # Second pass: emit interior & clipped tetrahedra
    for tet in cand_tets:
        ins = [int(v) for v in tet if f_vals[int(v)] <= 0.0]
        outs = [int(v) for v in tet if f_vals[int(v)] > 0.0]
        if len(ins) == 4:
            out_tets.append((ins[0], ins[1], ins[2], ins[3]))
        elif len(ins) == 1:
            v0 = ins[0]
            c0 = _get_cut_node(v0, outs[0])
            c1 = _get_cut_node(v0, outs[1])
            c2 = _get_cut_node(v0, outs[2])
            out_tets.append((v0, c0, c1, c2))
        elif len(ins) == 2:
            a, b = ins[0], ins[1]
            o0, o1 = outs[0], outs[1]
            ca0 = _get_cut_node(a, o0)
            ca1 = _get_cut_node(a, o1)
            cb0 = _get_cut_node(b, o0)
            cb1 = _get_cut_node(b, o1)
            out_tets.extend([(a, b, ca0, ca1), (b, ca0, ca1, cb0), (b, ca1, cb0, cb1)])
        elif len(ins) == 3:
            a, b, c = ins[0], ins[1], ins[2]
            o0 = outs[0]
            ca = _get_cut_node(a, o0)
            cb = _get_cut_node(b, o0)
            cc = _get_cut_node(c, o0)
            out_tets.extend([(a, b, c, ca), (b, c, ca, cb), (c, ca, cb, cc)])

    # Deduplicate collapsed tets (where snapped vertex == cut node)
    valid_unique = [t for t in out_tets if len({t[0], t[1], t[2], t[3]}) == 4]
    nodes_arr = np.array(node_list, dtype=np.float64)
    tets_arr = np.array(valid_unique, dtype=np.int64)
    tets_arr = _orient_tets_positive(nodes_arr, tets_arr)
    # Filter degenerate zero-volume boundary slivers from exact snap
    p0 = nodes_arr[tets_arr[:, 0]]
    p1 = nodes_arr[tets_arr[:, 1]]
    p2 = nodes_arr[tets_arr[:, 2]]
    p3 = nodes_arr[tets_arr[:, 3]]
    vols = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0
    tets_arr = tets_arr[vols > 1e-9]

    used = np.unique(tets_arr.ravel())
    remap = np.full(nodes_arr.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    nodes_arr = nodes_arr[used]
    tets_arr = remap[tets_arr]

    qual = compute_tet_quality(nodes_arr, tets_arr)
    return {
        "status": Status.PASS if qual["min_volume"] > 0.0 else Status.FAIL,
        "channel": "B_BCC_ISOSURFACE_STUFFING",
        "elem_type": "C3D4",
        "nodes": nodes_arr,
        "tets": tets_arr,
        "quality": qual,
    }


def export_calculix_inp(
    mesh_result: dict[str, Any],
    inp_path: str | pathlib.Path,
    E: float,
    nu: float,
    fixed_nodes: list[int],
    load_nodes: list[int],
    nodal_force: tuple[float, float, float],
) -> pathlib.Path:
    """Write Abaqus/CalculiX .inp file and verify readability via meshio."""
    nodes = mesh_result["nodes"]
    tets = mesh_result["tets"]
    el_type = mesh_result["elem_type"]
    path = pathlib.Path(inp_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["*HEADING", "Reify CalculiX Export"]
    lines.append("*NODE, NSET=NALL")
    for i, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"{i}, {x:.10g}, {y:.10g}, {z:.10g}")

    lines.append(f"*ELEMENT, TYPE={el_type}, ELSET=EALL")
    for eid, tet in enumerate(tets, start=1):
        idx_str = ", ".join(str(int(n) + 1) for n in tet)
        lines.append(f"{eid}, {idx_str}")

    lines.append("*NSET, NSET=FIXED")
    for idx in fixed_nodes:
        lines.append(f"{int(idx) + 1}")

    lines.append("*NSET, NSET=LOAD")
    for idx in load_nodes:
        lines.append(f"{int(idx) + 1}")

    lines.append("*MATERIAL, NAME=MAT1")
    lines.append("*ELASTIC")
    lines.append(f"{E:.8g}, {nu:.8g}")
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1")
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*BOUNDARY")
    lines.append("FIXED, 1, 3, 0.0")
    lines.append("*CLOAD")
    fx, fy, fz = nodal_force
    n_load = max(len(load_nodes), 1)
    for idx in load_nodes:
        nid = int(idx) + 1
        if abs(fx) > 0:
            lines.append(f"{nid}, 1, {fx / n_load:.10g}")
        if abs(fy) > 0:
            lines.append(f"{nid}, 2, {fy / n_load:.10g}")
        if abs(fz) > 0:
            lines.append(f"{nid}, 3, {fz / n_load:.10g}")
    lines.append("*NODE PRINT, NSET=NALL")
    lines.append("U")
    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*END STEP")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify readability with meshio
    _ = meshio.read(str(path))
    return path


def _find_ccx_binary() -> str | None:
    """Locate CalculiX ccx binary on PATH or Homebrew prefixes."""
    found = shutil.which("ccx")
    if found is not None:
        return found
    for pattern_dir in ("/opt/homebrew/bin", "/opt/homebrew/opt/calculix-ccx/bin", "/usr/local/bin"):
        p_dir = pathlib.Path(pattern_dir)
        if p_dir.is_dir():
            for cand in sorted(p_dir.glob("ccx*")):
                if cand.is_file():
                    return str(cand)
    return None


def _parse_calculix_dat_displacements(dat_path: pathlib.Path, n_nodes: int) -> np.ndarray:
    """Parse nodal displacements (vx, vy, vz) from CalculiX .dat output."""
    u = np.zeros((n_nodes, 3), dtype=np.float64)
    if not dat_path.exists():
        raise FileNotFoundError(f"CalculiX .dat file not found: {dat_path}")
    lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_disp = False
    parsed_count = 0
    for line in lines:
        low = line.strip().lower()
        if low.startswith("displacements"):
            in_disp = True
            continue
        if in_disp:
            if not low:
                if parsed_count > 0:
                    break
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[0]) - 1
                    if 0 <= nid < n_nodes:
                        u[nid, 0] = float(parts[1])
                        u[nid, 1] = float(parts[2])
                        u[nid, 2] = float(parts[3])
                        parsed_count += 1
                except ValueError:
                    if parsed_count > 0:
                        break
    if parsed_count == 0:
        raise ValueError(f"No displacement rows parsed from CalculiX .dat: {dat_path}")
    return u


def run_calculix_or_conformal_fem(
    inp_path: str | pathlib.Path,
    mesh_result: dict[str, Any],
    E: float,
    nu: float,
    fixed_nodes: list[int],
    load_nodes: list[int],
    nodal_force: tuple[float, float, float],
    require_ccx: bool = False,
) -> VerificationReport:
    """Run external CalculiX ccx if available, or fall back to conformal P1 tet FEM unless require_ccx=True."""
    ccx_bin = _find_ccx_binary()
    p = pathlib.Path(inp_path)
    if ccx_bin is not None:
        proc = subprocess.run(
            [ccx_bin, "-i", p.stem], cwd=str(p.parent), capture_output=True, text=True, timeout=60
        )
        if proc.returncode == 0:
            dat_path = p.with_suffix(".dat")
            u_vec = _parse_calculix_dat_displacements(dat_path, mesh_result["nodes"].shape[0])
            return VerificationReport(
                status=Status.PASS if np.all(np.isfinite(u_vec)) else Status.FAIL,
                metrics={
                    "solver_backend": "ccx",
                    "ccx_binary": ccx_bin,
                    "max_disp_mm": float(np.max(np.linalg.norm(u_vec, axis=-1))),
                    "load_mean_disp_xyz": (
                        np.mean(u_vec[load_nodes], axis=0).tolist() if load_nodes else [0.0, 0.0, 0.0]
                    ),
                    "u": u_vec,
                    "stdout": proc.stdout[-500:],
                },
                message="Solved via external CalculiX ccx binary.",
            )
        if require_ccx:
            return VerificationReport(
                status=Status.FAIL,
                metrics={"solver_backend": "ccx", "returncode": proc.returncode, "stderr": proc.stderr[-500:], "stdout": proc.stdout[-500:]},
                message=f"CalculiX ccx failed with exit code {proc.returncode}.",
            )
    elif require_ccx:
        return VerificationReport(
            status=Status.FAIL,
            message="CalculiX ccx binary not found while require_ccx=True.",
        )

    # Conformal 3D P1 tetrahedral FEM directly on the exported .inp mesh
    m = meshio.read(str(inp_path))
    nodes = np.asarray(m.points, dtype=np.float64)
    tets4 = None
    for cb in m.cells:
        if "tetra" in cb.type:
            tets4 = np.asarray(cb.data[:, :4], dtype=np.int64)
            break
    if tets4 is None:
        tets4 = mesh_result["tets"][:, :4]

    n_nodes = nodes.shape[0]
    n_dofs = 3 * n_nodes
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.array(
        [
            [lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for tet in tets4:
        coords = nodes[tet]
        mat = np.column_stack([np.ones(4), coords])
        det_j = float(np.linalg.det(mat))
        vol = abs(det_j) / 6.0
        if vol <= 1e-12:
            continue
        inv_mat = np.linalg.inv(mat)
        # dN/dx, dN/dy, dN/dz are rows 1, 2, 3 of inv_mat
        gx, gy, gz = inv_mat[1], inv_mat[2], inv_mat[3]
        B = np.zeros((6, 12), dtype=np.float64)
        for a in range(4):
            B[0, 3 * a] = gx[a]
            B[1, 3 * a + 1] = gy[a]
            B[2, 3 * a + 2] = gz[a]
            B[3, 3 * a] = gy[a]
            B[3, 3 * a + 1] = gx[a]
            B[4, 3 * a + 1] = gz[a]
            B[4, 3 * a + 2] = gy[a]
            B[5, 3 * a] = gz[a]
            B[5, 3 * a + 2] = gx[a]
        Ke = (B.T @ D @ B) * vol
        dofs = np.array([3 * n + c for n in tet for c in range(3)], dtype=np.int64)
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        vals.append(Ke.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dofs, n_dofs),
    ).tocsr()

    F = np.zeros(n_dofs, dtype=np.float64)
    n_load = max(len(load_nodes), 1)
    for nid in load_nodes:
        F[3 * nid] += nodal_force[0] / n_load
        F[3 * nid + 1] += nodal_force[1] / n_load
        F[3 * nid + 2] += nodal_force[2] / n_load

    fixed_dofs = np.array([3 * nid + c for nid in fixed_nodes for c in range(3)], dtype=np.int64)
    free_mask = np.ones(n_dofs, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = np.nonzero(free_mask)[0]

    u = np.zeros(n_dofs, dtype=np.float64)
    u[free_dofs] = spla.spsolve(K[free_dofs, :][:, free_dofs].tocsc(), F[free_dofs])
    u_vec = u.reshape(-1, 3)
    return VerificationReport(
        status=Status.PASS if np.all(np.isfinite(u_vec)) else Status.FAIL,
        metrics={
            "solver_backend": "conformal_tet_fem",
            "max_disp_mm": float(np.max(np.linalg.norm(u_vec, axis=-1))),
            "load_mean_disp_xyz": np.mean(u_vec[load_nodes], axis=0).tolist() if load_nodes else [0, 0, 0],
            "u": u_vec,
        },
        message="Exported .inp mesh solved via conformal 3D tetrahedral FEM.",
    )
