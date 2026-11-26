"""Finite Cell Method (FCM): adaptive cut-cell quadrature, Nitsche BC, and modal energy filter (M3)."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from reify.types import Status, VerificationReport


def _elasticity_matrix_2d(E: float, nu: float, plane_mode: str = "plane_stress") -> np.ndarray:
    if plane_mode == "plane_stress":
        c = E / (1.0 - nu * nu)
        return c * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
            dtype=np.float64,
        )
    if plane_mode == "plane_strain":
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported plane_mode: {plane_mode!r}")


def _q1_shape_and_grad(xi: float, eta: float, hx: float, hy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q1 bilinear shape functions N(4,) and physical derivatives dN/dx, dN/dy on [-1,1]^2."""
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)], dtype=np.float64)
    dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)], dtype=np.float64)
    return N, dN_dxi * (2.0 / hx), dN_deta * (2.0 / hy)


def _assemble_fcm_2d(
    domain_fn: Callable[[np.ndarray], np.ndarray],
    bbox_2d: tuple[tuple[float, float], tuple[float, float]],
    E: float,
    nu: float,
    thickness: float,
    dirichlet_segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    tractions: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    h: float,
    rho: float = 7.85e-9,
    quad_depth: int = 3,
    nitsche_gamma: float = 20.0,
    plane_mode: str = "plane_stress",
):
    """Assemble separate physical and void K, M matrices plus Nitsche BC and external load."""
    (xmin, ymin), (xmax, ymax) = bbox_2d
    nx = max(2, int(math.ceil((xmax - xmin) / h)))
    ny = max(2, int(math.ceil((ymax - ymin) / h)))
    hx = (xmax - xmin) / nx
    hy = (ymax - ymin) / ny
    h_eff = max(hx, hy)

    D = _elasticity_matrix_2d(E, nu, plane_mode=plane_mode)
    n_nodes = (nx + 1) * (ny + 1)
    n_dofs = 2 * n_nodes

    gp_1d = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))

    rows_p: list[np.ndarray] = []
    cols_p: list[np.ndarray] = []
    vals_kp: list[np.ndarray] = []
    vals_mp: list[np.ndarray] = []

    rows_v: list[np.ndarray] = []
    cols_v: list[np.ndarray] = []
    vals_kv: list[np.ndarray] = []
    vals_mv: list[np.ndarray] = []

    def _node_idx(ix: int, iy: int) -> int:
        return ix * (ny + 1) + iy

    min_cut_frac = 1.0
    n_phys_cells = 0
    n_cut_cells = 0

    # Precompute sub-cell Gauss point local offsets for depth=0 (2x2) and depth=quad_depth
    def _subcell_points(depth: int) -> tuple[np.ndarray, np.ndarray, float]:
        m = 2**depth
        sub_w = 1.0 / (m * m)
        xis = []
        etas = []
        for si in range(m):
            c_xi = -1.0 + (2.0 * si + 1.0) / m
            for sj in range(m):
                c_eta = -1.0 + (2.0 * sj + 1.0) / m
                for g1 in gp_1d:
                    for g2 in gp_1d:
                        xis.append(c_xi + g1 / m)
                        etas.append(c_eta + g2 / m)
        return np.array(xis, dtype=np.float64), np.array(etas, dtype=np.float64), sub_w * (0.25 * hx * hy)

    xi_coarse, eta_coarse, w_coarse = _subcell_points(0)
    xi_fine, eta_fine, w_fine = _subcell_points(quad_depth)

    for ix in range(nx):
        x0 = xmin + ix * hx
        x_mid = x0 + 0.5 * hx
        for iy in range(ny):
            y0 = ymin + iy * hy
            y_mid = y0 + 0.5 * hy
            # Quick cell classification via 4 corners + center
            probe = np.array(
                [
                    [x0, y0],
                    [x0 + hx, y0],
                    [x0 + hx, y0 + hy],
                    [x0, y0 + hy],
                    [x_mid, y_mid],
                ],
                dtype=np.float64,
            )
            f_probe = domain_fn(probe)
            diag = math.hypot(hx, hy)
            if np.all(f_probe < -diag):
                xis, etas, w_gp = xi_coarse, eta_coarse, w_coarse
                is_phys = np.ones(len(xis), dtype=bool)
                n_phys_cells += 1
            elif np.all(f_probe > diag):
                xis, etas, w_gp = xi_coarse, eta_coarse, w_coarse
                is_phys = np.zeros(len(xis), dtype=bool)
            else:
                xis, etas, w_gp = xi_fine, eta_fine, w_fine
                pts_gp = np.column_stack([x_mid + 0.5 * hx * xis, y_mid + 0.5 * hy * etas])
                is_phys = domain_fn(pts_gp) <= 0.0
                frac = float(np.mean(is_phys))
                if 0.0 < frac < 1.0:
                    n_cut_cells += 1
                    min_cut_frac = min(min_cut_frac, frac)
                elif frac == 1.0:
                    n_phys_cells += 1

            n0 = _node_idx(ix, iy)
            n1 = _node_idx(ix + 1, iy)
            n2 = _node_idx(ix + 1, iy + 1)
            n3 = _node_idx(ix, iy + 1)
            dofs = np.array(
                [2 * n0, 2 * n0 + 1, 2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1],
                dtype=np.int64,
            )
            rr = np.repeat(dofs, 8)
            cc = np.tile(dofs, 8)

            Ke_p = np.zeros((8, 8), dtype=np.float64)
            Me_p = np.zeros((8, 8), dtype=np.float64)
            Ke_v = np.zeros((8, 8), dtype=np.float64)
            Me_v = np.zeros((8, 8), dtype=np.float64)

            for k_gp in range(len(xis)):
                N, dNx, dNy = _q1_shape_and_grad(xis[k_gp], etas[k_gp], hx, hy)
                B = np.zeros((3, 8), dtype=np.float64)
                B[0, 0::2] = dNx
                B[1, 1::2] = dNy
                B[2, 0::2] = dNy
                B[2, 1::2] = dNx
                Nu = np.zeros((2, 8), dtype=np.float64)
                Nu[0, 0::2] = N
                Nu[1, 1::2] = N
                dK = (B.T @ D @ B) * (thickness * w_gp)
                dM = (Nu.T @ Nu) * (rho * thickness * w_gp)
                if is_phys[k_gp]:
                    Ke_p += dK
                    Me_p += dM
                else:
                    Ke_v += dK
                    Me_v += dM

            if np.any(Ke_p):
                rows_p.append(rr)
                cols_p.append(cc)
                vals_kp.append(Ke_p.ravel())
                vals_mp.append(Me_p.ravel())
            if np.any(Ke_v):
                rows_v.append(rr)
                cols_v.append(cc)
                vals_kv.append(Ke_v.ravel())
                vals_mv.append(Me_v.ravel())

    K_phys = sp.coo_matrix(
        (np.concatenate(vals_kp), (np.concatenate(rows_p), np.concatenate(cols_p))),
        shape=(n_dofs, n_dofs),
    ).tocsr()
    M_phys = sp.coo_matrix(
        (np.concatenate(vals_mp), (np.concatenate(rows_p), np.concatenate(cols_p))),
        shape=(n_dofs, n_dofs),
    ).tocsr()
    if rows_v:
        K_void = sp.coo_matrix(
            (np.concatenate(vals_kv), (np.concatenate(rows_v), np.concatenate(cols_v))),
            shape=(n_dofs, n_dofs),
        ).tocsr()
        M_void = sp.coo_matrix(
            (np.concatenate(vals_mv), (np.concatenate(rows_v), np.concatenate(cols_v))),
            shape=(n_dofs, n_dofs),
        ).tocsr()
    else:
        K_void = sp.csr_matrix((n_dofs, n_dofs), dtype=np.float64)
        M_void = sp.csr_matrix((n_dofs, n_dofs), dtype=np.float64)

    # Assemble Symmetric Stabilized Nitsche Dirichlet BC & External Tractions
    F_ext = np.zeros(n_dofs, dtype=np.float64)
    nit_rows: list[np.ndarray] = []
    nit_cols: list[np.ndarray] = []
    nit_vals: list[np.ndarray] = []

    def _locate_cell(x: float, y: float) -> tuple[int, int, float, float]:
        ix = min(max(int((x - xmin) / hx), 0), nx - 1)
        iy = min(max(int((y - ymin) / hy), 0), ny - 1)
        xi = 2.0 * (x - (xmin + ix * hx)) / hx - 1.0
        eta = 2.0 * (y - (ymin + iy * hy)) / hy - 1.0
        return ix, iy, min(max(xi, -1.0), 1.0), min(max(eta, -1.0), 1.0)

    gp_1d_4 = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
    gw_1d_4 = (0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538)

    for (p0, p1, u_pres) in dirichlet_segments:
        seg_dx = p1[0] - p0[0]
        seg_dy = p1[1] - p0[1]
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len < 1e-12:
            continue
        # Outward normal assuming segment is traversed with material on left, or estimate via domain_fn
        nx_s, ny_s = seg_dy / seg_len, -seg_dx / seg_len
        mid_pt = np.array([[0.5 * (p0[0] + p1[0]) + 1e-3 * nx_s, 0.5 * (p0[1] + p1[1]) + 1e-3 * ny_s]])
        if domain_fn(mid_pt)[0] < domain_fn(np.array([[0.5 * (p0[0] + p1[0]) - 1e-3 * nx_s, 0.5 * (p0[1] + p1[1]) - 1e-3 * ny_s]]))[0]:
            nx_s, ny_s = -nx_s, -ny_s
        N_mat_n = np.array([[nx_s, 0.0, ny_s], [0.0, ny_s, nx_s]], dtype=np.float64)

        n_sub = max(2, int(math.ceil(seg_len / (0.5 * h_eff))))
        sub_len = seg_len / n_sub
        u_vec = np.array(u_pres, dtype=np.float64)
        for s_i in range(n_sub):
            t0_s = s_i / n_sub
            t1_s = (s_i + 1) / n_sub
            for gp, gw in zip(gp_1d_4, gw_1d_4):
                t_g = 0.5 * (t0_s + t1_s) + 0.5 * (t1_s - t0_s) * gp
                xg = p0[0] + t_g * seg_dx
                yg = p0[1] + t_g * seg_dy
                w_g = 0.5 * sub_len * gw * thickness
                ix, iy, xi, eta = _locate_cell(xg, yg)
                N, dNx, dNy = _q1_shape_and_grad(xi, eta, hx, hy)
                B = np.zeros((3, 8), dtype=np.float64)
                B[0, 0::2] = dNx
                B[1, 1::2] = dNy
                B[2, 0::2] = dNy
                B[2, 1::2] = dNx
                Nu = np.zeros((2, 8), dtype=np.float64)
                Nu[0, 0::2] = N
                Nu[1, 1::2] = N
                TracOp = N_mat_n @ D @ B  # 2 x 8
                pen = (nitsche_gamma * E / h_eff) * (Nu.T @ Nu)
                sym_flux = Nu.T @ TracOp + TracOp.T @ Nu
                Ke_n = (pen - sym_flux) * w_g
                Fe_n = (((nitsche_gamma * E / h_eff) * Nu.T - TracOp.T) @ u_vec) * w_g

                n0 = _node_idx(ix, iy)
                n1 = _node_idx(ix + 1, iy)
                n2 = _node_idx(ix + 1, iy + 1)
                n3 = _node_idx(ix, iy + 1)
                dofs = np.array(
                    [2 * n0, 2 * n0 + 1, 2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1],
                    dtype=np.int64,
                )
                nit_rows.append(np.repeat(dofs, 8))
                nit_cols.append(np.tile(dofs, 8))
                nit_vals.append(Ke_n.ravel())
                F_ext[dofs] += Fe_n

    for (p0, p1, trac) in tractions:
        seg_dx = p1[0] - p0[0]
        seg_dy = p1[1] - p0[1]
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len < 1e-12:
            continue
        n_sub = max(2, int(math.ceil(seg_len / (0.5 * h_eff))))
        sub_len = seg_len / n_sub
        t_vec = np.array(trac, dtype=np.float64)
        for s_i in range(n_sub):
            t0_s = s_i / n_sub
            t1_s = (s_i + 1) / n_sub
            for gp, gw in zip(gp_1d_4, gw_1d_4):
                t_g = 0.5 * (t0_s + t1_s) + 0.5 * (t1_s - t0_s) * gp
                xg = p0[0] + t_g * seg_dx
                yg = p0[1] + t_g * seg_dy
                w_g = 0.5 * sub_len * gw * thickness
                ix, iy, xi, eta = _locate_cell(xg, yg)
                N, _, _ = _q1_shape_and_grad(xi, eta, hx, hy)
                Nu = np.zeros((2, 8), dtype=np.float64)
                Nu[0, 0::2] = N
                Nu[1, 1::2] = N
                n0 = _node_idx(ix, iy)
                n1 = _node_idx(ix + 1, iy)
                n2 = _node_idx(ix + 1, iy + 1)
                n3 = _node_idx(ix, iy + 1)
                dofs = np.array(
                    [2 * n0, 2 * n0 + 1, 2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1],
                    dtype=np.int64,
                )
                F_ext[dofs] += (Nu.T @ t_vec) * w_g

    if nit_rows:
        K_nit = sp.coo_matrix(
            (np.concatenate(nit_vals), (np.concatenate(nit_rows), np.concatenate(nit_cols))),
            shape=(n_dofs, n_dofs),
        ).tocsr()
        K_phys = K_phys + K_nit

    return {
        "K_phys": K_phys,
        "M_phys": M_phys,
        "K_void": K_void,
        "M_void": M_void,
        "F_ext": F_ext,
        "nx": nx,
        "ny": ny,
        "hx": hx,
        "hy": hy,
        "xmin": xmin,
        "ymin": ymin,
        "min_cut_fraction": min_cut_frac,
        "n_phys_cells": n_phys_cells,
        "n_cut_cells": n_cut_cells,
    }


def eval_fcm_displacement_2d(
    asm: dict[str, Any], u: np.ndarray, pts_xy: np.ndarray
) -> np.ndarray:
    """Interpolate Q1 FCM displacement field at arbitrary 2D points."""
    xmin, ymin = asm["xmin"], asm["ymin"]
    hx, hy = asm["hx"], asm["hy"]
    nx, ny = asm["nx"], asm["ny"]
    out = np.zeros((len(pts_xy), 2), dtype=np.float64)
    for k, (x, y) in enumerate(pts_xy):
        ix = min(max(int((x - xmin) / hx), 0), nx - 1)
        iy = min(max(int((y - ymin) / hy), 0), ny - 1)
        xi = min(max(2.0 * (x - (xmin + ix * hx)) / hx - 1.0, -1.0), 1.0)
        eta = min(max(2.0 * (y - (ymin + iy * hy)) / hy - 1.0, -1.0), 1.0)
        N, _, _ = _q1_shape_and_grad(xi, eta, hx, hy)
        n0 = ix * (ny + 1) + iy
        n1 = (ix + 1) * (ny + 1) + iy
        n2 = (ix + 1) * (ny + 1) + (iy + 1)
        n3 = ix * (ny + 1) + (iy + 1)
        nodes = [n0, n1, n2, n3]
        out[k, 0] = sum(N[i] * u[2 * nodes[i]] for i in range(4))
        out[k, 1] = sum(N[i] * u[2 * nodes[i] + 1] for i in range(4))
    return out


def fcm_solve_elasticity_2d(
    domain_fn: Callable[[np.ndarray], np.ndarray],
    bbox_2d: tuple[tuple[float, float], tuple[float, float]],
    E: float,
    nu: float,
    thickness: float,
    dirichlet_segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    tractions: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    h: float,
    alpha_k: float = 1e-6,
    quad_depth: int = 3,
    nitsche_gamma: float = 20.0,
    plane_mode: str = "plane_stress",
    probe_points: np.ndarray | None = None,
) -> dict[str, Any]:
    """Forward 2D FCM static elasticity solve with stabilized Nitsche BC."""
    asm = _assemble_fcm_2d(
        domain_fn,
        bbox_2d,
        E,
        nu,
        thickness,
        dirichlet_segments,
        tractions,
        h,
        quad_depth=quad_depth,
        nitsche_gamma=nitsche_gamma,
        plane_mode=plane_mode,
    )
    K_tot = (asm["K_phys"] + alpha_k * asm["K_void"]).tocsc()
    u = spla.spsolve(K_tot, asm["F_ext"])
    if not np.all(np.isfinite(u)):
        return {"status": Status.FAIL, "u": u, "compliance": float("nan"), **asm}

    compliance = float(np.dot(asm["F_ext"], u))
    probe_disp = (
        eval_fcm_displacement_2d(asm, u, probe_points) if probe_points is not None else None
    )
    return {
        "status": Status.PASS,
        "u": u,
        "compliance": compliance,
        "probe_disp": probe_disp,
        "n_dofs": len(u),
        "is_screening": True,
        **asm,
    }


def fcm_verify_3level_gci(
    solve_at_h_fn: Callable[[float], float],
    h_coarse: float,
    gci_tol: float = 0.05,
) -> VerificationReport:
    """Offline 3-level mesh refinement (h, h/2, h/4) Richardson extrapolation and GCI check."""
    phi1 = float(solve_at_h_fn(h_coarse))
    phi2 = float(solve_at_h_fn(0.5 * h_coarse))
    phi3 = float(solve_at_h_fn(0.25 * h_coarse))

    e12 = abs(phi1 - phi2)
    e23 = abs(phi2 - phi3)
    if e23 < 1e-14:
        p_obs = 2.0
    elif e12 <= e23:
        p_obs = 0.0
    else:
        p_obs = math.log2(e12 / e23)

    if p_obs <= 0.2:
        return VerificationReport(
            status=Status.FAIL,
            metrics={"phi_h": phi1, "phi_h2": phi2, "phi_h4": phi3, "p_obs": p_obs},
            message=f"3-level refinement failed to converge monotonically (p_obs={p_obs:.3f}).",
        )

    r_p = 2.0**p_obs
    phi_ext = phi3 + (phi3 - phi2) / max(r_p - 1.0, 1e-9)
    gci_fine = 1.25 * e23 / (max(abs(phi3), 1e-12) * max(r_p - 1.0, 1e-9))
    status = Status.PASS if gci_fine <= gci_tol else Status.UNKNOWN
    return VerificationReport(
        status=status,
        metrics={
            "phi_h": phi1,
            "phi_h2": phi2,
            "phi_h4": phi3,
            "p_obs": p_obs,
            "phi_richardson": phi_ext,
            "gci_fine": gci_fine,
        },
        message=f"3-level GCI: p_obs={p_obs:.2f}, GCI={gci_fine:.4f}, Richardson={phi_ext:.6g}.",
    )


def compute_mac_matrix(modes_a: np.ndarray, modes_b: np.ndarray) -> np.ndarray:
    """Modal Assurance Criterion matrix MAC[i, j] between mode sets (n_dofs, m)."""
    na = modes_a.shape[1]
    nb = modes_b.shape[1]
    mac = np.zeros((na, nb), dtype=np.float64)
    for i in range(na):
        va = modes_a[:, i]
        na2 = float(np.dot(va, va))
        for j in range(nb):
            vb = modes_b[:, j]
            nb2 = float(np.dot(vb, vb))
            if na2 > 1e-30 and nb2 > 1e-30:
                mac[i, j] = (float(np.dot(va, vb)) ** 2) / (na2 * nb2)
    return mac


def fcm_solve_modal_2d(
    domain_fn: Callable[[np.ndarray], np.ndarray],
    bbox_2d: tuple[tuple[float, float], tuple[float, float]],
    E: float,
    nu: float,
    rho: float,
    thickness: float,
    dirichlet_segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    h: float,
    n_modes: int = 6,
    alpha_k: float = 1e-6,
    alpha_m: float = 1e-8,
    phys_participation_min: float = 0.85,
    quad_depth: int = 3,
    nitsche_gamma: float = 20.0,
    plane_mode: str = "plane_stress",
) -> dict[str, Any]:
    """FCM modal analysis with physical-domain energy participation filter for void modes."""
    asm = _assemble_fcm_2d(
        domain_fn,
        bbox_2d,
        E,
        nu,
        thickness,
        dirichlet_segments,
        tractions=[],
        h=h,
        rho=rho,
        quad_depth=quad_depth,
        nitsche_gamma=nitsche_gamma,
        plane_mode=plane_mode,
    )
    K_tot = (asm["K_phys"] + alpha_k * asm["K_void"]).tocsc()
    M_tot = (asm["M_phys"] + alpha_m * asm["M_void"]).tocsc()
    M_phys = asm["M_phys"]

    k_ask = min(n_modes * 3 + 6, K_tot.shape[0] - 2)
    eigvals, eigvecs = spla.eigsh(K_tot, k=k_ask, M=M_tot, sigma=0.0, which="LM")
    order = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[order])
    eigvecs = np.real(eigvecs[:, order])

    phys_freqs: list[float] = []
    phys_modes: list[np.ndarray] = []
    participations: list[float] = []
    n_spurious = 0

    for j in range(len(eigvals)):
        lam = float(eigvals[j])
        if lam <= 1e-6:
            continue
        phi = eigvecs[:, j]
        m_p = float(phi @ (M_phys @ phi))
        m_t = max(float(phi @ (M_tot @ phi)), 1e-30)
        eta = m_p / m_t
        if eta >= phys_participation_min:
            phys_freqs.append(math.sqrt(lam) / (2.0 * math.pi))
            phys_modes.append(phi)
            participations.append(eta)
            if len(phys_freqs) >= n_modes:
                break
        else:
            n_spurious += 1

    return {
        "status": Status.PASS if len(phys_freqs) > 0 else Status.UNKNOWN,
        "frequencies_hz": np.array(phys_freqs, dtype=np.float64),
        "modes": np.column_stack(phys_modes) if phys_modes else np.zeros((K_tot.shape[0], 0)),
        "participations": participations,
        "n_spurious_filtered": n_spurious,
        "asm": asm,
    }
