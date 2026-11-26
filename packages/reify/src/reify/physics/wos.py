"""Homogeneous Dirichlet Walk-on-Spheres (WoS) point probe with normal-approx 95% CI (M3)."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

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
        x_act = torch.from_numpy(x[idx]).to(torch.float64)
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
