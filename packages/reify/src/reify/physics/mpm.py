"""Forward-only quasi-static 2D MLS-MPM with ADR damping and plane-stress F_zz solve (M3 / R4)."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from reify.types import Status, VerificationReport


def mpm_solve_quasistatic_2d(
    domain_fn: Callable[[np.ndarray], np.ndarray],
    bbox_2d: tuple[tuple[float, float], tuple[float, float]],
    E: float,
    nu: float,
    rho: float,
    mode: str = "plane_stress",
    fixed_region_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    particle_force_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    dx: float = 0.5,
    ppc_1d: int = 2,
    max_steps: int = 2500,
    tol_residual: float = 1e-3,
    tol_ke_ratio: float = 1e-3,
    damping_zeta: float = 0.08,
    require_friction_grad: bool = False,
) -> VerificationReport:
    """Forward quasi-static 2D MLS-MPM with Neo-Hookean plane stress (sigma_zz=0) or plane strain."""
    if require_friction_grad or mode not in ("plane_stress", "plane_strain"):
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message=(
                f"M3 MPM supports forward plane_stress/plane_strain only "
                f"(got mode={mode!r}, require_friction_grad={require_friction_grad})."
            ),
        )

    (xmin, ymin), (xmax, ymax) = bbox_2d
    pad = 2.0 * dx
    gx0 = xmin - pad
    gy0 = ymin - pad
    nx = max(6, int(math.ceil((xmax - xmin + 2.0 * pad) / dx)))
    ny = max(6, int(math.ceil((ymax - ymin + 2.0 * pad) / dx)))

    # Seed particles inside domain_fn <= 0
    sub_dx = dx / ppc_1d
    pxs = np.arange(xmin + 0.5 * sub_dx, xmax, sub_dx, dtype=np.float64)
    pys = np.arange(ymin + 0.5 * sub_dx, ymax, sub_dx, dtype=np.float64)
    gx, gy = np.meshgrid(pxs, pys, indexing="ij")
    cand = np.column_stack([gx.ravel(), gy.ravel()])
    inside = domain_fn(cand) <= 0.0
    x0 = cand[inside].copy()
    n_p = x0.shape[0]
    if n_p == 0:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message="No material points inside MPM domain.",
        )

    vol0 = sub_dx * sub_dx
    mass_p = np.full(n_p, rho * vol0, dtype=np.float64)
    x = x0.copy()
    v = np.zeros((n_p, 2), dtype=np.float64)
    C = np.zeros((n_p, 2, 2), dtype=np.float64)
    F = np.tile(np.eye(2, dtype=np.float64), (n_p, 1, 1))
    lam3 = np.ones(n_p, dtype=np.float64)

    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
    c_wave = math.sqrt((K_bulk + 4.0 * mu / 3.0) / rho)
    dt = 0.25 * dx / c_wave

    # Grid node coordinates and Dirichlet fixed mask
    gnx = np.arange(nx + 1, dtype=np.float64) * dx + gx0
    gny = np.arange(ny + 1, dtype=np.float64) * dx + gy0
    ggx, ggy = np.meshgrid(gnx, gny, indexing="ij")
    grid_coords = np.column_stack([ggx.ravel(), ggy.ravel()])
    fixed_nodes = (
        fixed_region_fn(grid_coords).reshape(nx + 1, ny + 1)
        if fixed_region_fn is not None
        else np.zeros((nx + 1, ny + 1), dtype=bool)
    )

    f_ext_p = (
        particle_force_fn(x0)
        if particle_force_fn is not None
        else np.zeros((n_p, 2), dtype=np.float64)
    )
    f0_scale = max(float(np.linalg.norm(f_ext_p)) * 1e-3, 1e-9)

    inv_dx = 1.0 / dx
    d_inv = 4.0 * inv_dx * inv_dx

    def _compute_stress(F_arr: np.ndarray, l3: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        det2 = F_arr[:, 0, 0] * F_arr[:, 1, 1] - F_arr[:, 0, 1] * F_arr[:, 1, 0]
        det2 = np.maximum(det2, 1e-6)
        inv_FT = np.zeros_like(F_arr)
        inv_FT[:, 0, 0] = F_arr[:, 1, 1] / det2
        inv_FT[:, 0, 1] = -F_arr[:, 1, 0] / det2
        inv_FT[:, 1, 0] = -F_arr[:, 0, 1] / det2
        inv_FT[:, 1, 1] = F_arr[:, 0, 0] / det2

        if mode == "plane_strain":
            l3_out = np.ones_like(l3)
            J = det2
        else:
            l3_out = l3.copy()
            for _ in range(6):
                g_val = mu * (l3_out * l3_out - 1.0) + lam * np.log(det2 * l3_out)
                g_der = 2.0 * mu * l3_out + lam / np.maximum(l3_out, 1e-6)
                l3_out = np.maximum(l3_out - g_val / g_der, 0.05)
            J = det2 * l3_out

        logJ = np.log(np.maximum(J, 1e-6))
        P = mu * (F_arr - inv_FT) + lam * logJ[:, None, None] * inv_FT
        # Cauchy stress sigma_zz = (mu*(l3^2 - 1) + lam*log(J)) / J
        sigma_zz = (mu * (l3_out * l3_out - 1.0) + lam * logJ) / J
        # Strain energy density Psi = 0.5*mu*(tr(F^T F) + l3^2 - 3 - 2*log(J)) + 0.5*lam*(log(J))^2
        tr_c = np.sum(F_arr * F_arr, axis=(1, 2)) + l3_out * l3_out
        psi = 0.5 * mu * (tr_c - 3.0 - 2.0 * logJ) + 0.5 * lam * logJ * logJ
        e_int = float(np.sum(psi) * vol0)
        # Relative sigma_zz vs Cauchy 2D norm
        cauchy_2d = np.einsum("pij,pkj->pik", P, F_arr) / J[:, None, None]
        norm_2d = np.sqrt(np.sum(cauchy_2d * cauchy_2d, axis=(1, 2)) + sigma_zz * sigma_zz)
        max_szz_rel = float(np.max(np.abs(sigma_zz) / np.maximum(norm_2d, 1e-12)))
        return P, l3_out, e_int, max_szz_rel

    converged = False
    r_F = float("inf")
    ke_ratio = float("inf")
    max_szz_rel = 0.0
    prev_ke = 0.0
    steps_done = 0

    for step in range(1, max_steps + 1):
        steps_done = step
        P, lam3, e_int, max_szz_rel = _compute_stress(F, lam3)
        stress_F_T = np.einsum("pij,pkj->pik", P, F) * (-vol0 * d_inv)

        # Quadratic B-spline weights
        rel_x = (x[:, 0] - gx0) * inv_dx
        rel_y = (x[:, 1] - gy0) * inv_dx
        base_x = np.clip(np.floor(rel_x - 0.5).astype(np.int64), 0, nx - 2)
        base_y = np.clip(np.floor(rel_y - 0.5).astype(np.int64), 0, ny - 2)
        fx = rel_x - base_x
        fy = rel_y - base_y
        wx = np.column_stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
        wy = np.column_stack([0.5 * (1.5 - fy) ** 2, 0.75 - (fy - 1.0) ** 2, 0.5 * (fy - 0.5) ** 2])

        grid_m = np.zeros((nx + 1, ny + 1), dtype=np.float64)
        grid_mv = np.zeros((nx + 1, ny + 1, 2), dtype=np.float64)
        grid_f_int = np.zeros((nx + 1, ny + 1, 2), dtype=np.float64)
        grid_f_ext = np.zeros((nx + 1, ny + 1, 2), dtype=np.float64)

        for ox in range(3):
            ix_n = base_x + ox
            for oy in range(3):
                iy_n = base_y + oy
                w = wx[:, ox] * wy[:, oy]
                dpos_x = (ix_n * dx + gx0) - x[:, 0]
                dpos_y = (iy_n * dx + gy0) - x[:, 1]
                dpos = np.column_stack([dpos_x, dpos_y])
                affine_v = v + np.einsum("pij,pj->pi", C, dpos)

                np.add.at(grid_m, (ix_n, iy_n), w * mass_p)
                np.add.at(grid_mv, (ix_n, iy_n), (w * mass_p)[:, None] * affine_v)
                f_int_c = w[:, None] * np.einsum("pij,pj->pi", stress_F_T, dpos)
                np.add.at(grid_f_int, (ix_n, iy_n), f_int_c)
                np.add.at(grid_f_ext, (ix_n, iy_n), w[:, None] * f_ext_p)

        active_g = grid_m > 1e-16
        grid_v = np.zeros((nx + 1, ny + 1, 2), dtype=np.float64)
        grid_v[active_g] = grid_mv[active_g] / grid_m[active_g, None]

        R_grid = grid_f_int + grid_f_ext
        R_free = R_grid.copy()
        R_free[fixed_nodes] = 0.0
        f_int_free = grid_f_int.copy()
        f_int_free[fixed_nodes] = 0.0
        f_ext_free = grid_f_ext.copy()
        f_ext_free[fixed_nodes] = 0.0

        r_norm = float(np.linalg.norm(R_free))
        denom = float(np.linalg.norm(f_int_free)) + float(np.linalg.norm(f_ext_free)) + f0_scale
        r_F = r_norm / denom

        # ADR damped grid velocity update
        grid_v[active_g] = (1.0 - damping_zeta) * grid_v[active_g] + dt * (
            R_grid[active_g] / grid_m[active_g, None]
        )
        grid_v[fixed_nodes] = 0.0

        ke = 0.5 * float(np.sum(grid_m[active_g] * np.sum(grid_v[active_g] ** 2, axis=-1)))
        ke_ratio = ke / (abs(e_int) + 1e-12)

        if step > 50 and r_F <= tol_residual and ke_ratio <= tol_ke_ratio:
            converged = True
            break
        if step > 20 and ke < prev_ke and r_F > tol_residual:
            # Mild velocity peak damping for faster quasi-static settling
            grid_v[active_g] *= 0.92
        prev_ke = ke

        # G2P
        v_new = np.zeros((n_p, 2), dtype=np.float64)
        C_new = np.zeros((n_p, 2, 2), dtype=np.float64)
        for ox in range(3):
            ix_n = base_x + ox
            for oy in range(3):
                iy_n = base_y + oy
                w = wx[:, ox] * wy[:, oy]
                gv = grid_v[ix_n, iy_n]
                dpos_x = (ix_n * dx + gx0) - x[:, 0]
                dpos_y = (iy_n * dx + gy0) - x[:, 1]
                dpos = np.column_stack([dpos_x, dpos_y])
                v_new += w[:, None] * gv
                C_new += d_inv * w[:, None, None] * np.einsum("pi,pj->pij", gv, dpos)

        v = v_new
        C = C_new
        x = x + dt * v
        F = np.einsum("pij,pjk->pik", np.eye(2)[None, :, :] + dt * C, F)

    disp = x - x0
    metrics = {
        "mode": mode,
        "n_particles": n_p,
        "steps": steps_done,
        "force_residual": r_F,
        "ke_ratio": ke_ratio,
        "max_sigma_zz_rel": max_szz_rel,
        "mean_disp": np.mean(disp, axis=0).tolist(),
        "max_disp": np.max(np.abs(disp), axis=0).tolist(),
        "particle_positions": x,
        "particle_displacements": disp,
    }
    if converged:
        return VerificationReport(
            status=Status.PASS,
            metrics=metrics,
            message=f"Quasi-static {mode} MLS-MPM converged in {steps_done} steps (r_F={r_F:.2e}).",
        )
    return VerificationReport(
        status=Status.UNKNOWN,
        metrics=metrics,
        message=f"Quasi-static {mode} MLS-MPM did not converge within {max_steps} steps (r_F={r_F:.2e}).",
    )
