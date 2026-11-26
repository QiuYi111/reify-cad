"""Geometric ray/FOV occlusion checking via safe sphere-tracing (M3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from reify.ast import Node, Part
from reify.query import GeometryQuery
from reify.types import FailReason, PointClass, Status, VerificationReport


def check_ray_occlusion(
    part_node: Node | Part,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    max_dist_mm: float,
    hit_tol_mm: float = 1e-3,
    max_steps: int = 128,
    params: dict[str, Any] | None = None,
) -> VerificationReport:
    """Check geometric occlusion along rays (no wave-optics claims)."""
    root = part_node.root if isinstance(part_node, Part) else part_node
    q = GeometryQuery(root)

    origins = np.asarray(ray_origins, dtype=np.float64)
    dirs = np.asarray(ray_dirs, dtype=np.float64)
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-12)
    n_rays = origins.shape[0]
    t = np.zeros(n_rays, dtype=np.float64)
    active = np.ones(n_rays, dtype=bool)

    for _ in range(max_steps):
        if not np.any(active):
            break
        idx = np.nonzero(active)[0]
        pts = origins[idx] + t[idx, None] * dirs[idx]
        pts_t = torch.from_numpy(pts).to(torch.float64)
        cls = q.classify_point(pts_t, params=params)
        f_vals = q.eval_field(pts_t, params=params, dtype=torch.float64).detach().cpu().numpy()
        for k, c in enumerate(cls):
            if c == PointClass.INSIDE or f_vals[k] <= hit_tol_mm:
                hit_pt = tuple(float(v) for v in pts[k])
                return VerificationReport(
                    status=Status.FAIL,
                    reason=FailReason.KINEMATIC_COLLISION,
                    witness_points=[hit_pt],  # type: ignore[list-item]
                    metrics={"hit_ray_idx": int(idx[k]), "hit_dist_mm": float(t[idx[k]])},
                    message=f"Ray {int(idx[k])} occluded at t={float(t[idx[k]]):.4f} mm.",
                )
        step_d = q.unsigned_boundary_distance_lower(pts_t, params=params).detach().cpu().numpy()
        t[idx] += np.maximum(step_d, hit_tol_mm)
        done = t[idx] >= max_dist_mm
        active[idx[done]] = False

    if np.any(active):
        return VerificationReport(
            status=Status.UNKNOWN,
            metrics={"remaining_rays": int(np.sum(active))},
            message="Ray occlusion check exhausted max_steps before clearing max_dist_mm.",
        )

    return VerificationReport(
        status=Status.PASS,
        metrics={"n_rays": n_rays, "max_dist_mm": max_dist_mm},
        message="All optical rays certified geometrically unoccluded.",
    )
