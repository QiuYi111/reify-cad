"""Stock-aware Local-ROI 3D FFT accessibility, directional wall thickness, and structural thickness (M2)."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import scipy.fft
import scipy.ndimage
import torch

from reify.ast import Node, Part
from reify.query import GeometryQuery
from reify.types import FailReason, Status, VerificationReport


def _voxelize_node(
    node: Node,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    q = GeometryQuery(node)
    pts_t = torch.from_numpy(pts).to(torch.float64)
    vals = q.eval_field(pts_t, params=params, dtype=torch.float64).detach().cpu().numpy()
    return (vals <= 0.0).reshape(len(xs), len(ys), len(zs))


def _resolve_fft_device(device: str, n_fft: tuple[int, int, int]) -> str:
    n_elems = int(n_fft[0]) * int(n_fft[1]) * int(n_fft[2])
    if device == "auto":
        if n_elems >= 64**3:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fft_linear_conv_bool(
    vol: np.ndarray,
    kernel: np.ndarray,
    origin: tuple[int, int, int],
    n_fft: tuple[int, int, int],
    correlate: bool,
    device: str = "cpu",
) -> np.ndarray:
    k_arr = kernel[::-1, ::-1, ::-1].copy() if correlate else kernel
    nx, ny, nz = vol.shape
    kx, ky, kz = kernel.shape
    ox, oy, oz = origin
    sx = (kx - 1 - ox) if correlate else ox
    sy = (ky - 1 - oy) if correlate else oy
    sz = (kz - 1 - oz) if correlate else oz

    if device in ("cuda", "mps"):
        t_vol = torch.from_numpy(vol.astype(np.float32)).to(device=device)
        t_ker = torch.from_numpy(k_arr.astype(np.float32)).to(device=device)
        f_vol = torch.fft.rfftn(t_vol, s=n_fft)
        f_ker = torch.fft.rfftn(t_ker, s=n_fft)
        conv = torch.fft.irfftn(f_vol * f_ker, s=n_fft)
        out = conv[sx : sx + nx, sy : sy + ny, sz : sz + nz] > 0.5
        return out.detach().cpu().numpy()

    f_vol = scipy.fft.rfftn(vol.astype(np.float32), s=n_fft)
    f_ker = scipy.fft.rfftn(k_arr.astype(np.float32), s=n_fft)
    conv = scipy.fft.irfftn(f_vol * f_ker, s=n_fft)
    return conv[sx : sx + nx, sy : sy + ny, sz : sz + nz] > 0.5


def brute_force_binary_morphology(
    vol: np.ndarray,
    kernel: np.ndarray,
    origin: tuple[int, int, int],
    correlate: bool,
) -> np.ndarray:
    """Exact spatial brute-force binary correlation/dilation with zero boundary conditions."""
    nx, ny, nz = vol.shape
    ox, oy, oz = origin
    out = np.zeros_like(vol, dtype=bool)
    k_pts = np.argwhere(kernel)
    for kx, ky, kz in k_pts:
        dx = int(kx) - ox
        dy = int(ky) - oy
        dz = int(kz) - oz
        if not correlate:
            dx, dy, dz = -dx, -dy, -dz
        # out[x, y, z] |= vol[x + dx, y + dy, z + dz]
        x0_dst = max(0, -dx)
        x1_dst = min(nx, nx - dx)
        y0_dst = max(0, -dy)
        y1_dst = min(ny, ny - dy)
        z0_dst = max(0, -dz)
        z1_dst = min(nz, nz - dz)
        if x0_dst < x1_dst and y0_dst < y1_dst and z0_dst < z1_dst:
            out[x0_dst:x1_dst, y0_dst:y1_dst, z0_dst:z1_dst] |= vol[
                x0_dst + dx : x1_dst + dx,
                y0_dst + dy : y1_dst + dy,
                z0_dst + dz : z1_dst + dz,
            ]
    return out


def check_stock_fft_accessibility(
    part_node: Node | Part,
    stock_node: Node,
    fixture_node: Node | None,
    cutting_flute_node: Node,
    non_cutting_shank_node: Node,
    approach_axis: str = "+z",
    voxel_mm: float = 0.5,
    roi_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    max_memory_bytes: int = 2 * 1024**3,
    params: dict[str, Any] | None = None,
    device: str = "auto",
) -> VerificationReport:
    """Conservative single-setup stock-aware Local-ROI 3D FFT accessibility check."""
    t0 = time.perf_counter()
    p_root = part_node.root if isinstance(part_node, Part) else part_node

    if roi_bounds is None:
        smin, smax = stock_node.local_bounds(params)
        pmin, pmax = p_root.local_bounds(params)
        roi_bounds = (
            tuple(min(smin[i], pmin[i]) - voxel_mm for i in range(3)),  # type: ignore[assignment]
            tuple(max(smax[i], pmax[i]) + voxel_mm for i in range(3)),  # type: ignore[assignment]
        )
    rmin, rmax = roi_bounds
    xs = np.arange(rmin[0] + 0.5 * voxel_mm, rmax[0], voxel_mm, dtype=np.float64)
    ys = np.arange(rmin[1] + 0.5 * voxel_mm, rmax[1], voxel_mm, dtype=np.float64)
    zs = np.arange(rmin[2] + 0.5 * voxel_mm, rmax[2], voxel_mm, dtype=np.float64)
    roi_shape = (len(xs), len(ys), len(zs))

    cmin, cmax = cutting_flute_node.local_bounds(params)
    tmin, tmax = non_cutting_shank_node.local_bounds(params)
    kmin = (min(cmin[0], tmin[0], 0.0), min(cmin[1], tmin[1], 0.0), min(cmin[2], tmin[2], 0.0))
    kmax = (max(cmax[0], tmax[0], 0.0), max(cmax[1], tmax[1], 0.0), max(cmax[2], tmax[2], 0.0))

    # Build kernel grid aligned to exact integer multiples of voxel_mm around (0, 0, 0)
    ix_lo = int(math.floor(kmin[0] / voxel_mm))
    ix_hi = int(math.ceil(kmax[0] / voxel_mm))
    iy_lo = int(math.floor(kmin[1] / voxel_mm))
    iy_hi = int(math.ceil(kmax[1] / voxel_mm))
    iz_lo = int(math.floor(kmin[2] / voxel_mm))
    iz_hi = int(math.ceil(kmax[2] / voxel_mm))

    kxs = np.arange(ix_lo, ix_hi + 1, dtype=np.float64) * voxel_mm
    kys = np.arange(iy_lo, iy_hi + 1, dtype=np.float64) * voxel_mm
    kzs = np.arange(iz_lo, iz_hi + 1, dtype=np.float64) * voxel_mm
    tool_shape = (len(kxs), len(kys), len(kzs))
    origin = (-ix_lo, -iy_lo, -iz_lo)

    n_fft = tuple(
        int(scipy.fft.next_fast_len(n + k - 1)) for n, k in zip(roi_shape, tool_shape)
    )
    # Peak memory: ~6 arrays of float32/complex64 equivalents
    est_bytes = 6 * int(n_fft[0]) * int(n_fft[1]) * int(n_fft[2]) * 4
    if est_bytes > max_memory_bytes:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            metrics={
                "roi_shape": roi_shape,
                "tool_shape": tool_shape,
                "n_fft": n_fft,
                "estimated_bytes": est_bytes,
                "max_memory_bytes": max_memory_bytes,
            },
            message=(
                f"Estimated FFT memory {est_bytes / (1024**2):.1f} MiB exceeds "
                f"budget {max_memory_bytes / (1024**2):.1f} MiB."
            ),
        )

    v_part = _voxelize_node(p_root, xs, ys, zs, params=params)
    v_stock = _voxelize_node(stock_node, xs, ys, zs, params=params)
    if fixture_node is not None:
        v_fix = _voxelize_node(fixture_node, xs, ys, zs, params=params)
    else:
        v_fix = np.zeros_like(v_part, dtype=bool)

    k_cut = _voxelize_node(cutting_flute_node, kxs, kys, kzs, params=params)
    k_shank = _voxelize_node(non_cutting_shank_node, kxs, kys, kzs, params=params)

    eff_dev = _resolve_fft_device(device, n_fft)

    # F_cons = ~( (P_part ⊕ C_cut^) ∪ ((P_stock ∪ O_fixture) ⊕ T_nc^) )
    obs_shank = v_stock | v_fix
    col_cut = _fft_linear_conv_bool(v_part, k_cut, origin, n_fft, correlate=True, device=eff_dev)
    col_shank = _fft_linear_conv_bool(obs_shank, k_shank, origin, n_fft, correlate=True, device=eff_dev)
    f_cons = ~(col_cut | col_shank)

    # 6-connected flood fill from approach boundary plane
    labels, _ = scipy.ndimage.label(f_cons)
    nx, ny, nz = roi_shape
    if approach_axis == "+z":
        seed_labels = np.unique(labels[:, :, nz - 1])
    elif approach_axis == "-z":
        seed_labels = np.unique(labels[:, :, 0])
    elif approach_axis == "+x":
        seed_labels = np.unique(labels[nx - 1, :, :])
    elif approach_axis == "-x":
        seed_labels = np.unique(labels[0, :, :])
    elif approach_axis == "+y":
        seed_labels = np.unique(labels[:, ny - 1, :])
    elif approach_axis == "-y":
        seed_labels = np.unique(labels[:, 0, :])
    else:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message=f"Unsupported approach_axis: {approach_axis!r}",
        )
    seed_labels = seed_labels[seed_labels > 0]
    f_reach = np.isin(labels, seed_labels)

    # M_removable = (P_stock \ P_part) ∩ (F_reach ⊕ C_cut)
    swept_cut = _fft_linear_conv_bool(f_reach, k_cut, origin, n_fft, correlate=False, device=eff_dev)
    excess_stock = v_stock & (~v_part)
    m_removable = excess_stock & swept_cut
    unremovable = excess_stock & (~m_removable)
    n_unrem = int(unremovable.sum())
    elapsed = time.perf_counter() - t0

    metrics = {
        "roi_shape": roi_shape,
        "tool_shape": tool_shape,
        "n_fft": n_fft,
        "fft_backend": f"torch_{eff_dev}" if eff_dev != "cpu" else "scipy_cpu",
        "estimated_bytes": est_bytes,
        "voxel_half_diag_mm": 0.5 * math.sqrt(3.0) * voxel_mm,
        "excess_voxels": int(excess_stock.sum()),
        "removable_voxels": int(m_removable.sum()),
        "unremovable_voxels": n_unrem,
        "elapsed_s": elapsed,
        # Arrays preserved for exact brute-force equivalence auditing
        "v_part": v_part,
        "v_stock": v_stock,
        "v_fix": v_fix,
        "k_cut": k_cut,
        "k_shank": k_shank,
        "kernel_origin": origin,
        "f_cons_mask": f_cons,
        "f_reach_mask": f_reach,
        "m_removable_mask": m_removable,
    }

    if n_unrem == 0:
        return VerificationReport(
            status=Status.PASS,
            metrics=metrics,
            message="All excess stock is conservatively reachable without shank/part collision.",
        )

    bad_idx = np.argwhere(unremovable)[0]
    w_pt = (float(xs[bad_idx[0]]), float(ys[bad_idx[1]]), float(zs[bad_idx[2]]))
    return VerificationReport(
        status=Status.FAIL,
        reason=FailReason.TOOL_COLLISION,
        witness_points=[w_pt],
        metrics=metrics,
        message=f"{n_unrem} excess stock voxels are unreachable under conservative shank/flute rules.",
    )


def _finite_diff_normals(
    q: GeometryQuery, pts: torch.Tensor, params: dict[str, Any] | None, h: float = 1e-4
) -> torch.Tensor:
    grad = torch.zeros_like(pts)
    for d in range(3):
        dx = torch.zeros_like(pts)
        dx[:, d] = h
        fp = q.eval_field(pts + dx, params=params, dtype=torch.float64)
        fm = q.eval_field(pts - dx, params=params, dtype=torch.float64)
        grad[:, d] = (fp - fm) / (2.0 * h)
    norms = torch.linalg.norm(grad, dim=-1, keepdim=True).clamp(min=1e-12)
    return grad / norms


def check_directional_wall_thickness(
    part_node: Node | Part,
    min_wall_mm: float,
    cone_half_angle_deg: float = 15.0,
    sample_radius: float = 0.6,
    ray_step_mm: float = 0.05,
    params: dict[str, Any] | None = None,
) -> VerificationReport:
    """Screen directional wall thickness within inward normal cone to non-adjacent opposing faces."""
    root = part_node.root if isinstance(part_node, Part) else part_node
    q = GeometryQuery(root)
    cov = q.cover_surface(max_radius=sample_radius, params=params)
    if cov.centers.shape[0] == 0:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message="Empty surface cover for directional wall thickness check.",
        )

    p_surf = q._project_to_implicit_surface(cov.centers, params=params, steps=8)
    n_out = _finite_diff_normals(q, p_surf, params)
    bmin, bmax = root.local_bounds(params)
    max_diag = math.sqrt(sum((bmax[i] - bmin[i]) ** 2 for i in range(3))) + 1.0

    # Build central inward ray + 6 tilted cone rays
    n_np = n_out.detach().cpu().numpy()
    p_np = p_surf.detach().cpu().numpy()
    alpha = math.radians(cone_half_angle_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)

    ray_dirs_list: list[np.ndarray] = [-n_np]
    if cone_half_angle_deg > 1e-3:
        # Tangent vectors
        ref = np.zeros_like(n_np)
        use_x = np.abs(n_np[:, 0]) < 0.8
        ref[use_x, 0] = 1.0
        ref[~use_x, 1] = 1.0
        t1 = np.cross(n_np, ref)
        t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
        t2 = np.cross(n_np, t1)
        for k in range(6):
            phi = k * (math.pi / 3.0)
            tang = math.cos(phi) * t1 + math.sin(phi) * t2
            ray_dirs_list.append(-ca * n_np + sa * tang)

    n_pts = p_np.shape[0]
    L_part = max(1.0, q.lipschitz_bound(params))
    min_step = max(0.5 * ray_step_mm, 1e-3)
    t_init = 2.0 * ray_step_mm

    min_valid_thick = float("inf")
    best_pair: list[tuple[float, float, float]] = []

    for r_dirs in ray_dirs_list:
        # Lipschitz-safe interior Sphere Tracing (O(N) memory instead of dense ray grid)
        t_cur = np.full(n_pts, t_init, dtype=np.float64)
        t_prev = np.full(n_pts, t_init, dtype=np.float64)
        p0_t = torch.from_numpy(p_np + t_cur[:, None] * r_dirs).to(torch.float64)
        v0 = q.eval_field(p0_t, params=params, dtype=torch.float64).detach().cpu().numpy()
        active = v0 < -1e-6
        exited = np.zeros(n_pts, dtype=bool)
        t_exit = np.zeros(n_pts, dtype=np.float64)

        for _ in range(128):
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            pts_m = torch.from_numpy(p_np[idx] + t_cur[idx, None] * r_dirs[idx]).to(torch.float64)
            vals_m = q.eval_field(pts_m, params=params, dtype=torch.float64).detach().cpu().numpy()
            hit_m = vals_m >= 0.0
            if np.any(hit_m):
                h_idx = idx[hit_m]
                exited[h_idx] = True
                t_exit[h_idx] = t_cur[h_idx]
                active[h_idx] = False
            cont = ~hit_m
            if np.any(cont):
                c_idx = idx[cont]
                t_prev[c_idx] = t_cur[c_idx]
                step = np.maximum(np.abs(vals_m[cont]) / L_part, min_step)
                t_cur[c_idx] += step
                active[c_idx] = t_cur[c_idx] <= max_diag

        if not np.any(exited):
            continue
        cand_i = np.nonzero(exited)[0]
        t_lo = t_prev[cand_i]
        t_hi = t_exit[cand_i]
        p_sub = p_np[cand_i]
        d_sub = r_dirs[cand_i]
        for _ in range(8):
            t_mid = 0.5 * (t_lo + t_hi)
            pm = torch.from_numpy(p_sub + t_mid[:, None] * d_sub).to(torch.float64)
            vm = q.eval_field(pm, params=params, dtype=torch.float64).detach().cpu().numpy()
            inside_m = vm < 0.0
            t_lo = np.where(inside_m, t_mid, t_lo)
            t_hi = np.where(inside_m, t_hi, t_mid)

        t_hit = 0.5 * (t_lo + t_hi)
        q_hit = p_sub + t_hit[:, None] * d_sub
        n_hit = _finite_diff_normals(q, torch.from_numpy(q_hit).to(torch.float64), params).detach().cpu().numpy()
        # Non-adjacent opposing surface filter: dot(n_start, n_hit) < -0.5 (angle > 120 deg)
        dots = np.sum(n_np[cand_i] * n_hit, axis=-1)
        opposing = (dots < -0.5) & (t_hit > 2.5 * ray_step_mm)
        if np.any(opposing):
            valid_t = t_hit[opposing]
            loc_min = int(np.argmin(valid_t))
            if float(valid_t[loc_min]) < min_valid_thick:
                min_valid_thick = float(valid_t[loc_min])
                orig_idx = int(cand_i[opposing][loc_min])
                p0 = tuple(float(v) for v in p_np[orig_idx])
                q0 = tuple(float(v) for v in q_hit[opposing][loc_min])
                best_pair = [p0, q0]  # type: ignore[list-item]

    if not math.isfinite(min_valid_thick):
        return VerificationReport(
            status=Status.UNKNOWN,
            metrics={"cone_half_angle_deg": cone_half_angle_deg, "ray_step_mm": ray_step_mm},
            message="No opposing non-adjacent surface hits found within directional cone.",
        )

    err_band = ray_step_mm
    metrics = {
        "min_directional_thickness_mm": min_valid_thick,
        "min_wall_mm": min_wall_mm,
        "discretization_band_mm": err_band,
        "cone_half_angle_deg": cone_half_angle_deg,
    }
    if min_valid_thick + err_band < min_wall_mm:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.WALL_THICKNESS,
            witness_points=best_pair,
            metrics=metrics,
            message=(
                f"Directional wall thickness {min_valid_thick:.4f} mm "
                f"< {min_wall_mm:.4f} mm."
            ),
        )
    if min_valid_thick - err_band > min_wall_mm:
        return VerificationReport(
            status=Status.PASS,
            metrics=metrics,
            message=f"Directional wall thickness {min_valid_thick:.4f} mm >= {min_wall_mm:.4f} mm.",
        )
    return VerificationReport(
        status=Status.UNKNOWN,
        witness_points=best_pair,
        metrics=metrics,
        message="Directional wall thickness falls within ray discretization uncertainty band.",
    )


def compute_structural_thickness(
    part_node: Node | Part,
    voxel_mm: float = 0.25,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hildebrand (1997) maximal-inscribed-ball local structural thickness via Meijster/SciPy EDT."""
    root = part_node.root if isinstance(part_node, Part) else part_node
    bmin, bmax = root.local_bounds(params)
    pad = 2.0 * voxel_mm
    xs = np.arange(bmin[0] - pad, bmax[0] + pad, voxel_mm, dtype=np.float64)
    ys = np.arange(bmin[1] - pad, bmax[1] + pad, voxel_mm, dtype=np.float64)
    zs = np.arange(bmin[2] - pad, bmax[2] + pad, voxel_mm, dtype=np.float64)
    solid = _voxelize_node(root, xs, ys, zs, params=params)
    edt_mm = scipy.ndimage.distance_transform_edt(solid) * voxel_mm

    # Propagate maximal inscribed ball diameter 2 * r(y) to covered interior voxels
    thick = np.zeros_like(edt_mm, dtype=np.float64)
    inside_pts = np.argwhere(solid)
    # Sort descending by radius so larger balls dominate early
    radii_vox = edt_mm[solid] / voxel_mm
    order = np.argsort(-radii_vox)
    nx, ny, nz = solid.shape
    for idx in order:
        cx, cy, cz = inside_pts[idx]
        r_v = float(radii_vox[idx])
        diam = 2.0 * r_v * voxel_mm
        if thick[cx, cy, cz] >= diam:
            continue
        ir = int(math.ceil(r_v))
        x0, x1 = max(0, cx - ir), min(nx, cx + ir + 1)
        y0, y1 = max(0, cy - ir), min(ny, cy + ir + 1)
        z0, z1 = max(0, cz - ir), min(nz, cz + ir + 1)
        gx, gy, gz = np.ogrid[x0:x1, y0:y1, z0:z1]
        mask = ((gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2) <= (r_v * r_v + 1e-9)
        sub = thick[x0:x1, y0:y1, z0:z1]
        sub[mask] = np.maximum(sub[mask], diam)

    vals = thick[solid]
    voxel_err = math.sqrt(3.0) * voxel_mm
    return {
        "min_structural_thickness_mm": float(np.min(vals)) if vals.size else 0.0,
        "max_structural_thickness_mm": float(np.max(vals)) if vals.size else 0.0,
        "mean_structural_thickness_mm": float(np.mean(vals)) if vals.size else 0.0,
        "voxel_error_mm": voxel_err,
        "thickness_field": thick,
    }


def check_draft_and_overhang(
    part_node: Node | Part,
    pull_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
    min_draft_deg: float = 1.0,
    max_overhang_deg: float = 45.0,
    sample_radius: float = 0.8,
    params: dict[str, Any] | None = None,
) -> VerificationReport:
    """Pure-SDF analytic surface normal screening for molding draft angle and 3D-printing overhang."""
    root = part_node.root if isinstance(part_node, Part) else part_node
    q = GeometryQuery(root)
    cov = q.cover_surface(max_radius=sample_radius, params=params)
    if cov.centers.shape[0] == 0:
        return VerificationReport(status=Status.OUT_OF_DOMAIN, message="Empty surface cover.")

    p_surf = q._project_to_implicit_surface(cov.centers, params=params, steps=8)
    n_out = _finite_diff_normals(q, p_surf, params).detach().cpu().numpy()
    p_np = p_surf.detach().cpu().numpy()

    d_vec = np.asarray(pull_dir, dtype=np.float64)
    d_vec = d_vec / max(float(np.linalg.norm(d_vec)), 1e-12)
    dots = np.clip(n_out @ d_vec, -1.0, 1.0)

    # Side walls: |dot(n, pull)| < 0.7 -> draft_deg = arcsin(dot(n, pull))
    side_mask = np.abs(dots) < 0.7
    draft_angles = np.degrees(np.arcsin(np.abs(dots[side_mask]))) if np.any(side_mask) else np.array([90.0])
    min_draft_seen = float(np.min(draft_angles))

    # Downward-facing overhang faces: dot(n, pull) < -0.05 (exclude bottom-most base plate if flat on z_min)
    bmin, _ = root.local_bounds(params)
    not_floor = (p_np @ d_vec) > (float(np.dot(bmin, d_vec)) + 0.5)
    down_mask = (dots < -0.05) & not_floor
    if np.any(down_mask):
        # Overhang angle from vertical: 0 deg = vertical wall, 90 deg = horizontal ceiling
        overhang_angles = np.degrees(np.arcsin(np.abs(dots[down_mask])))
        max_overhang_seen = float(np.max(overhang_angles))
        bad_ov = overhang_angles > max_overhang_deg
        ov_frac = float(np.sum(bad_ov) / len(dots))
        worst_idx = int(np.nonzero(down_mask)[0][int(np.argmax(overhang_angles))])
    else:
        max_overhang_seen = 0.0
        ov_frac = 0.0
        worst_idx = 0

    metrics = {
        "min_draft_deg": min_draft_seen,
        "max_overhang_deg": max_overhang_seen,
        "overhang_fraction": ov_frac,
        "n_surface_samples": int(len(dots)),
    }
    if min_draft_seen < min_draft_deg or max_overhang_seen > max_overhang_deg:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.DFM_VIOLATION if hasattr(FailReason, "DFM_VIOLATION") else FailReason.WALL_THICKNESS,
            witness_points=[tuple(float(v) for v in p_np[worst_idx])],  # type: ignore[list-item]
            metrics=metrics,
            message=(
                f"Draft/overhang violation: min_draft={min_draft_seen:.2f}° (req >= {min_draft_deg}°), "
                f"max_overhang={max_overhang_seen:.2f}° (req <= {max_overhang_deg}°)."
            ),
        )
    return VerificationReport(
        status=Status.PASS,
        metrics=metrics,
        message=f"Draft/overhang PASS (min_draft={min_draft_seen:.2f}°, max_overhang={max_overhang_seen:.2f}°).",
    )
