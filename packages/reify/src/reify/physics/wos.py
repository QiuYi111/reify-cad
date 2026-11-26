"""Homogeneous Dirichlet Walk-on-Spheres (WoS) point probe with normal-approx 95% CI (M3)."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from reify.device import get_compute_device
from reify.query import GeometryQuery
from reify.types import FailReason, PointClass, Status, VerificationReport


def wos_solve_point(
    query: GeometryQuery,
    point_xyz: tuple[float, float, float],
    boundary_value_fn: Callable[[np.ndarray], np.ndarray],
    source_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_samples: int = 400,
    eps_shell_mm: float = 1e-3,
    max_steps: int = 512,
    medium: str = "homogeneous",
    bc_type: str = "dirichlet",
    limit_max: float | None = None,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> VerificationReport:
    """IID Monte Carlo WoS for 3D homogeneous Laplace/Poisson with Dirichlet BC."""
    if medium != "homogeneous" or bc_type != "dirichlet":
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message=(
                f"WoS supports only medium='homogeneous' and bc_type='dirichlet'; "
                f"got medium={medium!r}, bc_type={bc_type!r}."
            ),
        )

    dev = torch.device(device) if device is not None else get_compute_device(torch.float32)
    eval_dt = torch.float32 if dev.type == "mps" else torch.float64

    p0 = torch.tensor([list(point_xyz)], dtype=torch.float64)
    if query.classify_point(p0, params=params)[0] != PointClass.INSIDE:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message="WoS query point is not certified INSIDE the domain.",
        )

    rng = np.random.Generator(np.random.PCG64(seed))
    x = np.tile(np.array(point_xyz, dtype=np.float64), (n_samples, 1))
    active = np.ones(n_samples, dtype=bool)
    accum_source = np.zeros(n_samples, dtype=np.float64)

    for _ in range(max_steps):
        if not np.any(active):
            break
        idx = np.nonzero(active)[0]
        x_act = torch.from_numpy(x[idx]).to(device=dev, dtype=eval_dt)
        R = query.unsigned_boundary_distance_lower(x_act, params=params).detach().cpu().numpy()

        done_mask = R <= eps_shell_mm
        if np.any(done_mask):
            active[idx[done_mask]] = False
        cont_idx = idx[~done_mask]
        R_cont = R[~done_mask]
        if cont_idx.size == 0:
            break

        if source_fn is not None:
            # Uniform 3D ball sample for Poisson source integral E[f(Y) * G_R(0, Y)]
            u_r = rng.random(cont_idx.size) ** (1.0 / 3.0)
            z_s = 2.0 * rng.random(cont_idx.size) - 1.0
            phi_s = 2.0 * math.pi * rng.random(cont_idx.size)
            r_xy = np.sqrt(np.maximum(1.0 - z_s * z_s, 0.0))
            dir_s = np.column_stack([r_xy * np.cos(phi_s), r_xy * np.sin(phi_s), z_s])
            r_samp = np.maximum(u_r * R_cont, 1e-12)
            y_samp = x[cont_idx] + r_samp[:, None] * dir_s
            # 3D ball Green's function integrated over ball: (R^2 / 6) * (3 * (1/r - 1/R) * r^2 / R^2)...
            # Exact unbiased estimator with uniform ball sampling (volume = 4/3 pi R^3):
            # G_3d(0, y) * Vol(B_R) = (1/(4*pi) * (1/r - 1/R)) * (4/3 pi R^3) = (R^3 / 3) * (1/r - 1/R)
            g_vol = (R_cont**3 / 3.0) * (1.0 / r_samp - 1.0 / R_cont)
            accum_source[cont_idx] += source_fn(y_samp) * g_vol

        # Jump to uniformly sampled point on sphere S^2 of radius R_cont
        z_u = 2.0 * rng.random(cont_idx.size) - 1.0
        th_u = 2.0 * math.pi * rng.random(cont_idx.size)
        r_xy_u = np.sqrt(np.maximum(1.0 - z_u * z_u, 0.0))
        step_dir = np.column_stack([r_xy_u * np.cos(th_u), r_xy_u * np.sin(th_u), z_u])
        x[cont_idx] = x[cont_idx] + R_cont[:, None] * step_dir

    # Evaluate boundary values at converged walks (if any walks didn't terminate, mark UNKNOWN)
    n_unconverged = int(np.sum(active))
    if n_unconverged > 0:
        return VerificationReport(
            status=Status.UNKNOWN,
            metrics={"n_samples": n_samples, "n_unconverged": n_unconverged},
            message=f"{n_unconverged}/{n_samples} WoS walks did not reach eps_shell within {max_steps} steps.",
        )

    b_vals = boundary_value_fn(x)
    samples = b_vals + accum_source
    mean_val = float(np.mean(samples))
    sample_var = float(np.var(samples, ddof=1)) if n_samples > 1 else 0.0
    mean_var = sample_var / float(n_samples)
    se = math.sqrt(mean_var)
    ci_low = mean_val - 1.96 * se
    ci_high = mean_val + 1.96 * se

    metrics = {
        "mean": mean_val,
        "sample_variance": sample_var,
        "mean_variance": mean_var,
        "standard_error": se,
        "ci_95": (ci_low, ci_high),
        "n_samples": n_samples,
        "sampling_mode": "IID_PCG64",
        "ci_kind": "normal_approximation_95",
    }

    if limit_max is not None:
        if ci_high < limit_max:
            return VerificationReport(
                status=Status.PASS,
                metrics=metrics,
                message=f"WoS 95% CI [{ci_low:.4f}, {ci_high:.4f}] < limit_max={limit_max:.4f}.",
            )
        if ci_low > limit_max:
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.PHYSICS_LIMIT,
                metrics=metrics,
                message=f"WoS 95% CI [{ci_low:.4f}, {ci_high:.4f}] > limit_max={limit_max:.4f}.",
            )
        return VerificationReport(
            status=Status.UNKNOWN,
            metrics=metrics,
            message=f"Threshold limit_max={limit_max:.4f} lies inside WoS 95% CI [{ci_low:.4f}, {ci_high:.4f}].",
        )

    return VerificationReport(
        status=Status.PASS,
        metrics=metrics,
        message=f"WoS estimate {mean_val:.4f} +/- {1.96 * se:.4f} (95% normal-approx CI).",
    )


def wost_solve_point(
    query: GeometryQuery,
    point_xyz: tuple[float, float, float],
    dirichlet_dist_fn: Callable[[np.ndarray], np.ndarray],
    dirichlet_value_fn: Callable[[np.ndarray], np.ndarray],
    neumann_flux_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    source_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_samples: int = 400,
    eps_shell_mm: float = 1e-2,
    max_steps: int = 256,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> VerificationReport:
    """Walk-on-Stars (Sawhney et al., SIGGRAPH 2023) for 3D Laplace/Poisson with mixed Dirichlet + Neumann BCs."""
    dev = torch.device(device) if device is not None else get_compute_device(torch.float32)
    eval_dt = torch.float32 if dev.type == "mps" else torch.float64
    p0 = torch.tensor([list(point_xyz)], dtype=torch.float64)
    if query.classify_point(p0, params=params)[0] != PointClass.INSIDE:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message="WoSt query point is not certified INSIDE the domain.",
        )

    rng = np.random.Generator(np.random.PCG64(seed))
    x = np.tile(np.array(point_xyz, dtype=np.float64), (n_samples, 1))
    active = np.ones(n_samples, dtype=bool)
    on_neumann = np.zeros(n_samples, dtype=bool)
    n_neumann = np.zeros((n_samples, 3), dtype=np.float64)
    accum = np.zeros(n_samples, dtype=np.float64)
    L_dom = max(1.0, query.lipschitz_bound(params))

    def _sdf_grad(pts_np: np.ndarray, h: float = 1e-3) -> np.ndarray:
        offs = np.array(
            [[h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]],
            dtype=np.float64,
        )
        probe = (pts_np[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        v = query.eval_field(torch.from_numpy(probe).to(device=dev, dtype=eval_dt), params=params, dtype=eval_dt).detach().cpu().numpy().reshape(-1, 6)
        g = np.stack([v[:, 0] - v[:, 1], v[:, 2] - v[:, 3], v[:, 4] - v[:, 5]], axis=-1) / (2.0 * h)
        return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)

    for _ in range(max_steps):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break

        d_dir = np.maximum(0.0, dirichlet_dist_fn(x[idx]))
        done = d_dir <= eps_shell_mm
        if np.any(done):
            active[idx[done]] = False
        c_idx = idx[~done]
        if c_idx.size == 0:
            break

        # Star radius: closest Dirichlet boundary (for convex/star-shaped regions, whole domain up to Dirichlet!)
        R_star = d_dir[~done]

        # Sample random ray direction: full S^2 in interior; inward hemisphere when on Neumann boundary!
        z_u = 2.0 * rng.random(c_idx.size) - 1.0
        th_u = 2.0 * math.pi * rng.random(c_idx.size)
        r_xy = np.sqrt(np.maximum(1.0 - z_u * z_u, 0.0))
        ray_dir = np.column_stack([r_xy * np.cos(th_u), r_xy * np.sin(th_u), z_u])

        neu_mask = on_neumann[c_idx]
        if np.any(neu_mask):
            # Flip outward rays into interior hemisphere (dot(ray_dir, n_out) < 0)
            n_out = n_neumann[c_idx[neu_mask]]
            dots = np.sum(ray_dir[neu_mask] * n_out, axis=-1)
            flip = dots > 0.0
            if np.any(flip):
                rd_n = ray_dir[neu_mask]
                rd_n[flip] = rd_n[flip] - 2.0 * dots[flip, None] * n_out[flip]
                ray_dir[neu_mask] = rd_n

        # March ray from x[c_idx] up to R_star via SDF Sphere Tracing to find nearest Neumann boundary hit
        t = np.zeros(c_idx.size, dtype=np.float64)
        # Step slightly off Neumann face to avoid immediate self-hit
        t[neu_mask] = 2.0 * eps_shell_mm
        ray_active = np.ones(c_idx.size, dtype=bool)
        hit_neu = np.zeros(c_idx.size, dtype=bool)

        for _m in range(48):
            r_idx = np.nonzero(ray_active)[0]
            if r_idx.size == 0:
                break
            p_m = x[c_idx[r_idx]] + t[r_idx, None] * ray_dir[r_idx]
            f_m = query.eval_field(torch.from_numpy(p_m).to(device=dev, dtype=eval_dt), params=params, dtype=eval_dt).detach().cpu().numpy()
            # Inside -> f_m < 0; distance to boundary is |f_m| / L_dom
            dist_b = np.maximum(-f_m, 0.0) / L_dom
            reached_bnd = (f_m >= -eps_shell_mm) & (t[r_idx] > 2.5 * eps_shell_mm)
            if np.any(reached_bnd):
                hit_neu[r_idx[reached_bnd]] = True
                ray_active[r_idx[reached_bnd]] = False
            cont_r = ~reached_bnd
            if np.any(cont_r):
                cr_idx = r_idx[cont_r]
                step = np.maximum(dist_b[cont_r], 0.5 * eps_shell_mm)
                t_next = t[cr_idx] + step
                past_star = t_next >= R_star[cr_idx]
                t[cr_idx[past_star]] = R_star[cr_idx[past_star]]
                ray_active[cr_idx[past_star]] = False
                t[cr_idx[~past_star]] = t_next[~past_star]

        # Project onto boundary if hit_neu, else jump to R_star
        x_next = x[c_idx] + t[:, None] * ray_dir
        if neumann_flux_fn is not None and np.any(hit_neu):
            h_pts = x_next[hit_neu]
            n_h = _sdf_grad(h_pts)
            # In a star ball of radius R, Neumann boundary reflection contributes g(y) * G_3d(x, y) * ...
            # For pure insulated Neumann (flux=0), contrib=0; otherwise include Green's boundary weight
            g_flux = neumann_flux_fn(h_pts, n_h)
            r_hit = np.maximum(t[hit_neu], eps_shell_mm)
            alpha_geom = np.maximum(np.abs(np.sum(ray_dir[hit_neu] * n_h, axis=-1)), 0.1)
            accum[c_idx[hit_neu]] += g_flux * (R_star[hit_neu] - r_hit) / (2.0 * alpha_geom)

        if source_fn is not None:
            u_r = rng.random(c_idx.size) ** (1.0 / 3.0)
            z_s = 2.0 * rng.random(c_idx.size) - 1.0
            phi_s = 2.0 * math.pi * rng.random(c_idx.size)
            r_xy_s = np.sqrt(np.maximum(1.0 - z_s * z_s, 0.0))
            dir_s = np.column_stack([r_xy_s * np.cos(phi_s), r_xy_s * np.sin(phi_s), z_s])
            r_samp = np.maximum(u_r * t, 1e-9)
            y_samp = x[c_idx] + r_samp[:, None] * dir_s
            g_vol = (t**3 / 3.0) * (1.0 / r_samp - 1.0 / np.maximum(t, 1e-9))
            accum[c_idx] += source_fn(y_samp) * g_vol

        x[c_idx] = x_next
        on_neumann[c_idx] = hit_neu
        if np.any(hit_neu):
            n_neumann[c_idx[hit_neu]] = _sdf_grad(x_next[hit_neu])

    n_unconv = int(np.sum(active))
    if n_unconv > int(0.05 * n_samples):
        return VerificationReport(
            status=Status.UNKNOWN,
            metrics={"n_samples": n_samples, "n_unconverged": n_unconv},
            message=f"{n_unconv}/{n_samples} WoSt walks did not reach Dirichlet boundary.",
        )

    conv_mask = ~active
    b_vals = dirichlet_value_fn(x[conv_mask])
    samples = b_vals + accum[conv_mask]
    mean_val = float(np.mean(samples))
    sample_var = float(np.var(samples, ddof=1)) if len(samples) > 1 else 0.0
    se = math.sqrt(sample_var / max(1, len(samples)))
    ci_low = mean_val - 1.96 * se
    ci_high = mean_val + 1.96 * se

    return VerificationReport(
        status=Status.PASS,
        metrics={
            "mean": mean_val,
            "standard_error": se,
            "ci_95": (ci_low, ci_high),
            "n_samples": int(len(samples)),
            "n_unconverged": n_unconv,
            "solver": "Walk-on-Stars (Sawhney et al. 2023)",
        },
        message=f"WoSt estimate {mean_val:.4f} +/- {1.96 * se:.4f} (95% CI).",
    )
