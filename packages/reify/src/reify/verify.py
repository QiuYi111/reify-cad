"""Certified bidirectional clearance verification and scoped freeze checking (M1)."""

from __future__ import annotations

import math
from typing import Any

import torch

from reify.ast import (
    Cylinder,
    FreezeScope,
    FreezeSpec,
    Intersect,
    Node,
    Part,
    SmoothUnion,
    Subtract,
    Transform,
    Union,
    _val_mm_float,
)
from reify.query import GeometryQuery
from reify.spec import Spec
from reify.types import (
    FailReason,
    OutOfDomainError,
    PointClass,
    Status,
    VerificationReport,
)


def _tangent_basis(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute two orthonormal tangent vectors per normal (N, 3)."""
    n = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-12)
    ref = torch.zeros_like(n)
    use_x = torch.abs(n[:, 0]) < 0.8
    ref[use_x, 0] = 1.0
    ref[~use_x, 1] = 1.0
    t1 = torch.linalg.cross(n, ref, dim=-1)
    t1 = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True).clamp(min=1e-12)
    t2 = torch.linalg.cross(n, t1, dim=-1)
    t2 = t2 / torch.linalg.norm(t2, dim=-1, keepdim=True).clamp(min=1e-12)
    return t1, t2


def _collect_interior_candidates(
    query: GeometryQuery, params: dict[str, Any] | None
) -> torch.Tensor:
    """Sample candidate points inside bounding box and return those certified INSIDE."""
    bmin, bmax = query.root.local_bounds(params)
    xs = torch.linspace(bmin[0], bmax[0], 7, dtype=torch.float64)
    ys = torch.linspace(bmin[1], bmax[1], 7, dtype=torch.float64)
    zs = torch.linspace(bmin[2], bmax[2], 7, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid = torch.stack([gx.ravel(), gy.ravel(), gz.ravel()], dim=-1)

    # Also include transformed leaf centers (catches small isolated components!)
    from reify.query import _iter_transformed_leaves

    eye = torch.eye(3, dtype=torch.float64)
    zero = torch.zeros(3, dtype=torch.float64)
    leaf_centers = [t_w for _, _, t_w in _iter_transformed_leaves(query.root, params, eye, zero)]
    if leaf_centers:
        cands = torch.cat([grid, torch.stack(leaf_centers, dim=0)], dim=0)
    else:
        cands = grid

    classes = query.classify_point(cands, params=params)
    inside_idx = [i for i, c in enumerate(classes) if c == PointClass.INSIDE]
    if not inside_idx:
        return torch.zeros((0, 3), dtype=torch.float64)
    return cands[inside_idx]


def _check_one_direction_step(
    centers: torch.Tensor,
    radii: torch.Tensor,
    normals: torch.Tensor,
    q_own: GeometryQuery,
    q_opp: GeometryQuery,
    min_clearance_mm: float,
    params_own: dict[str, Any] | None,
    params_opp: dict[str, Any] | None,
) -> tuple[VerificationReport | None, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Evaluate active patches of own against opponent; return (early_fail, rem_c, rem_r, rem_n, min_lb)."""
    if centers.shape[0] == 0:
        return None, centers, radii, normals, float("inf")

    classes_opp = q_opp.classify_point(centers, params=params_opp)
    inside_indices = [i for i, c in enumerate(classes_opp) if c == PointClass.INSIDE]
    if inside_indices:
        # Project candidate onto own surface to confirm actual surface point is INSIDE opponent
        cand = centers[inside_indices[:16]]
        proj_own = q_own._project_to_implicit_surface(cand, params=params_own, steps=8)
        f_own = torch.abs(q_own.eval_field(proj_own, params=params_own, dtype=torch.float64))
        proj_cls = q_opp.classify_point(proj_own, params=params_opp)
        for k, cls in enumerate(proj_cls):
            if cls == PointClass.INSIDE and float(f_own[k].item()) < 1e-4:
                pt = tuple(float(v) for v in proj_own[k].tolist())
                return (
                    VerificationReport(
                        status=Status.FAIL,
                        reason=FailReason.PENETRATION,
                        witness_points=[pt],  # type: ignore[list-item]
                        message="Certified penetration: surface point is INSIDE opponent.",
                    ),
                    centers,
                    radii,
                    normals,
                    0.0,
                )

    out_idx = [i for i, c in enumerate(classes_opp) if c == PointClass.OUTSIDE]
    unc_idx = [i for i, c in enumerate(classes_opp) if c != PointClass.OUTSIDE]

    keep_mask_indices: list[int] = list(unc_idx)
    min_lb = float("inf")
    if out_idx:
        out_t = torch.tensor(out_idx, dtype=torch.long)
        d_lb = q_opp.external_clearance_lower(centers[out_t], params=params_opp)
        min_lb = min(min_lb, float(torch.amin(d_lb).item()))
        safe = d_lb > (radii[out_t] + min_clearance_mm)
        unsafe_rel = torch.nonzero(~safe, as_tuple=False).squeeze(-1).tolist()
        for r_i in unsafe_rel:
            keep_mask_indices.append(out_idx[int(r_i)])

    if not keep_mask_indices:
        return None, centers[:0], radii[:0], normals[:0], min_lb

    keep_t = torch.tensor(keep_mask_indices, dtype=torch.long)
    rem_c = centers[keep_t]
    rem_r = radii[keep_t]
    rem_n = normals[keep_t]

    # Upper-bound witness check on closest K active patches (only valid way to prove CLEARANCE_VIOLATION)
    f_opp = q_opp.eval_field(rem_c, params=params_opp, dtype=torch.float64)
    k_top = min(8, rem_c.shape[0])
    top_idx = torch.topk(-f_opp, k=k_top).indices
    p_surf = q_own._project_to_implicit_surface(rem_c[top_idx], params=params_own, steps=8)
    f_p_own = torch.abs(q_own.eval_field(p_surf, params=params_own, dtype=torch.float64))

    # Check if projected surface point is INSIDE opponent
    p_cls = q_opp.classify_point(p_surf, params=params_opp)
    for idx_k in range(k_top):
        if p_cls[idx_k] == PointClass.INSIDE and float(f_p_own[idx_k].item()) < 1e-4:
            pt = tuple(float(v) for v in p_surf[idx_k].tolist())
            return (
                VerificationReport(
                    status=Status.FAIL,
                    reason=FailReason.PENETRATION,
                    witness_points=[pt],  # type: ignore[list-item]
                    message="Certified penetration witness found on surface.",
                ),
                rem_c,
                rem_r,
                rem_n,
                0.0,
            )

    q_surf = q_opp._project_to_implicit_surface(p_surf, params=params_opp, steps=8)
    f_q_opp = torch.abs(q_opp.eval_field(q_surf, params=params_opp, dtype=torch.float64))
    eps = max(
        q_own.eps_num(pts=p_surf, params=params_own, dtype=torch.float64),
        q_opp.eps_num(pts=q_surf, params=params_opp, dtype=torch.float64),
    )
    dists = torch.linalg.norm(p_surf - q_surf, dim=-1)
    for idx_k in range(k_top):
        if float(f_p_own[idx_k].item()) < 1e-5 and float(f_q_opp[idx_k].item()) < 1e-5:
            d_val = float(dists[idx_k].item())
            if d_val - eps <= min_clearance_mm:
                pa = tuple(float(v) for v in p_surf[idx_k].tolist())
                pb = tuple(float(v) for v in q_surf[idx_k].tolist())
                return (
                    VerificationReport(
                        status=Status.FAIL,
                        reason=FailReason.CLEARANCE_VIOLATION,
                        witness_points=[pa, pb],  # type: ignore[list-item]
                        metrics={"witness_distance_mm": d_val, "min_clearance_mm": min_clearance_mm},
                        message=(
                            f"Clearance violation: witness distance {d_val:.6f} mm "
                            f"<= {min_clearance_mm:.6f} mm."
                        ),
                    ),
                    rem_c,
                    rem_r,
                    rem_n,
                    min_lb,
                )

    return None, rem_c, rem_r, rem_n, min_lb


def _subdivide_patches(
    centers: torch.Tensor,
    radii: torch.Tensor,
    normals: torch.Tensor,
    q_own: GeometryQuery,
    params_own: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if centers.shape[0] == 0:
        return centers, radii, normals
    t1, t2 = _tangent_basis(normals)
    half_r = 0.5 * radii.unsqueeze(-1)
    offsets = [
        half_r * (0.5 * t1 + 0.5 * t2),
        half_r * (0.5 * t1 - 0.5 * t2),
        half_r * (-0.5 * t1 + 0.5 * t2),
        half_r * (-0.5 * t1 - 0.5 * t2),
    ]
    c_new = torch.cat([centers + off for off in offsets], dim=0)
    r_new = torch.cat([0.55 * radii] * 4, dim=0)
    n_new = torch.cat([normals] * 4, dim=0)

    L = q_own.lipschitz_bound(params_own)
    eps = q_own.eps_num(pts=c_new, params=params_own, dtype=torch.float64)
    vals = q_own.eval_field(c_new, params=params_own, dtype=torch.float64)
    keep = (torch.abs(vals) / L) <= (r_new + eps)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    return c_new[idx], r_new[idx], n_new[idx]


def check_clearance(
    query_a: GeometryQuery,
    query_b: GeometryQuery,
    min_clearance_mm: float = 0.0,
    max_depth: int = 4,
    initial_radius: float = 1.0,
    params_a: dict[str, Any] | None = None,
    params_b: dict[str, Any] | None = None,
) -> VerificationReport:
    """Three-branch certified clearance check: PASS (lower>m), FAIL (penetration/upper<=m), UNKNOWN."""
    try:
        bmin_a, bmax_a = query_a.root.local_bounds(params_a)
        bmin_b, bmax_b = query_b.root.local_bounds(params_b)
        for bmin, bmax in [(bmin_a, bmax_a), (bmin_b, bmax_b)]:
            if any(not math.isfinite(v) for v in (*bmin, *bmax)) or any(
                bmax[i] <= bmin[i] for i in range(3)
            ):
                return VerificationReport(
                    status=Status.OUT_OF_DOMAIN,
                    message="Input geometry has non-finite or degenerate bounding box.",
                )
        cov_a = query_a.cover_surface(max_radius=initial_radius, params=params_a)
        cov_b = query_b.cover_surface(max_radius=initial_radius, params=params_b)
    except OutOfDomainError as exc:
        return VerificationReport(status=Status.OUT_OF_DOMAIN, message=str(exc))

    if cov_a.centers.shape[0] == 0 or cov_b.centers.shape[0] == 0:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message="Empty surface cover: solid appears empty or degenerate.",
        )

    # 1. Interior containment check (eliminates small-inside-large false PASS)
    in_a = _collect_interior_candidates(query_a, params_a)
    if in_a.shape[0] > 0:
        cls_in_b = query_b.classify_point(in_a, params=params_b)
        for i, c in enumerate(cls_in_b):
            if c == PointClass.INSIDE:
                pt = tuple(float(v) for v in in_a[i].tolist())
                return VerificationReport(
                    status=Status.FAIL,
                    reason=FailReason.PENETRATION,
                    witness_points=[pt],  # type: ignore[list-item]
                    message="Certified interior containment/penetration: point INSIDE both A and B.",
                )

    in_b = _collect_interior_candidates(query_b, params_b)
    if in_b.shape[0] > 0:
        cls_in_a = query_a.classify_point(in_b, params=params_a)
        for i, c in enumerate(cls_in_a):
            if c == PointClass.INSIDE:
                pt = tuple(float(v) for v in in_b[i].tolist())
                return VerificationReport(
                    status=Status.FAIL,
                    reason=FailReason.PENETRATION,
                    witness_points=[pt],  # type: ignore[list-item]
                    message="Certified interior containment/penetration: point INSIDE both B and A.",
                )

    # 2. Adaptive bidirectional surface clearance
    ca, ra, na = cov_a.centers, cov_a.radii, cov_a.normals
    cb, rb, nb = cov_b.centers, cov_b.radii, cov_b.normals
    min_lb_overall = float("inf")

    for depth in range(max_depth + 1):
        rep_a, ca, ra, na, lba = _check_one_direction_step(
            ca, ra, na, query_a, query_b, min_clearance_mm, params_a, params_b
        )
        if rep_a is not None:
            return rep_a
        rep_b, cb, rb, nb, lbb = _check_one_direction_step(
            cb, rb, nb, query_b, query_a, min_clearance_mm, params_b, params_a
        )
        if rep_b is not None:
            return rep_b

        min_lb_overall = min(min_lb_overall, lba, lbb)
        if ca.shape[0] == 0 and cb.shape[0] == 0:
            return VerificationReport(
                status=Status.PASS,
                metrics={
                    "min_clearance_mm": min_clearance_mm,
                    "certified_min_lower_bound_mm": min_lb_overall,
                    "subdivision_depth_used": depth,
                },
                message="Bidirectional external clearance certified PASS.",
            )

        if depth == max_depth:
            return VerificationReport(
                status=Status.UNKNOWN,
                metrics={
                    "min_clearance_mm": min_clearance_mm,
                    "min_lower_bound_mm": min_lb_overall,
                    "remaining_patches_a": int(ca.shape[0]),
                    "remaining_patches_b": int(cb.shape[0]),
                },
                message=(
                    f"Undecided at max_depth={max_depth}: lower bound did not clear threshold "
                    f"and no upper-bound violation witness was found."
                ),
            )

        ca, ra, na = _subdivide_patches(ca, ra, na, query_a, params_a)
        cb, rb, nb = _subdivide_patches(cb, rb, nb, query_b, params_b)

    return VerificationReport(status=Status.UNKNOWN, message="Clearance check terminated undecided.")


def _extract_world_subtree(
    node: Node,
    target_id: str,
    params: dict[str, Any] | None,
    R_acc: torch.Tensor,
    t_acc: torch.Tensor,
) -> tuple[Node, torch.Tensor, torch.Tensor] | None:
    if node.node_id == target_id:
        return node, R_acc, t_acc
    if isinstance(node, Transform):
        Rn, tn = node.world_matrix_tensors(params, dtype=torch.float64)
        return _extract_world_subtree(node.child, target_id, params, R_acc @ Rn, R_acc @ tn + t_acc)
    if isinstance(node, (Union, Intersect)):
        for ch in node.children:
            res = _extract_world_subtree(ch, target_id, params, R_acc, t_acc)
            if res is not None:
                return res
    elif isinstance(node, Subtract):
        res = _extract_world_subtree(node.base, target_id, params, R_acc, t_acc)
        if res is not None:
            return res
        return _extract_world_subtree(node.tool, target_id, params, R_acc, t_acc)
    elif isinstance(node, SmoothUnion):
        res = _extract_world_subtree(node.a, target_id, params, R_acc, t_acc)
        if res is not None:
            return res
        return _extract_world_subtree(node.b, target_id, params, R_acc, t_acc)
    return None


class _RigidMatrixNode(Node):
    """Internal wrapper applying a fixed world (R, t) to an extracted subtree."""

    def __init__(self, child: Node, R: torch.Tensor, t: torch.Tensor) -> None:
        super().__init__(node_id=f"world_{child.node_id}")
        self.child = child
        self.R = R.to(torch.float64)
        self.t = t.to(torch.float64)

    def lipschitz_bound(self, params: dict[str, Any] | None = None) -> float:
        return self.child.lipschitz_bound(params)

    def ast_depth(self) -> int:
        return 1 + self.child.ast_depth()

    def local_bounds(self, params: dict[str, Any] | None = None):
        bmin, bmax = self.child.local_bounds(params)
        corners = torch.tensor(
            [
                [x, y, z]
                for x in (bmin[0], bmax[0])
                for y in (bmin[1], bmax[1])
                for z in (bmin[2], bmax[2])
            ],
            dtype=torch.float64,
        )
        wc = corners @ self.R.T + self.t
        return tuple(wc.min(dim=0).values.tolist()), tuple(wc.max(dim=0).values.tolist())

    def eval_sdf(
        self,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        p = pts.to(dtype=dtype)
        R = self.R.to(dtype=dtype, device=p.device)
        t = self.t.to(dtype=dtype, device=p.device)
        return self.child.eval_sdf((p - t) @ R, params=params, dtype=dtype)


def check_freeze(
    old_part: Part | Node,
    new_part: Part | Node,
    freeze_spec: FreezeSpec,
    params_old: dict[str, Any] | None = None,
    params_new: dict[str, Any] | None = None,
) -> VerificationReport:
    """Verify WORLD_SHELL (bidirectional Hausdorff) or DATUM_AXIS (axis & radius) freeze."""
    old_root = old_part.root if isinstance(old_part, Part) else old_part
    new_root = new_part.root if isinstance(new_part, Part) else new_part
    eye = torch.eye(3, dtype=torch.float64)
    zero = torch.zeros(3, dtype=torch.float64)

    ext_old = _extract_world_subtree(old_root, freeze_spec.node_id, params_old, eye, zero)
    ext_new = _extract_world_subtree(new_root, freeze_spec.node_id, params_new, eye, zero)
    if ext_old is None or ext_new is None:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.FREEZE_VIOLATION,
            message=f"Frozen node {freeze_spec.node_id!r} missing in part.",
        )

    n_old, R_old, t_old = ext_old
    n_new, R_new, t_new = ext_new

    if freeze_spec.scope == FreezeScope.WORLD_SHELL:
        q_old = GeometryQuery(_RigidMatrixNode(n_old, R_old, t_old))
        q_new = GeometryQuery(_RigidMatrixNode(n_new, R_new, t_new))
        return q_old.freeze_check_world_shell(
            q_new, tol_mm=freeze_spec.tol_mm, params_old=params_old, params_new=params_new
        )

    if freeze_spec.scope == FreezeScope.DATUM_AXIS:
        if not isinstance(n_old, Cylinder) or not isinstance(n_new, Cylinder):
            return VerificationReport(
                status=Status.OUT_OF_DOMAIN,
                message=f"DATUM_AXIS freeze expects Cylinder node, got {type(n_old).__name__}.",
            )
        r_old = _val_mm_float(n_old.radius, params_old)
        r_new = _val_mm_float(n_new.radius, params_new)
        ax_old = R_old[:, 2] / torch.linalg.norm(R_old[:, 2])
        ax_new = R_new[:, 2] / torch.linalg.norm(R_new[:, 2])
        cos_ang = abs(float(torch.dot(ax_old, ax_new).item()))
        # Perpendicular offset of t_new from old axis line (t_old + s * ax_old)
        dt = t_new - t_old
        perp = dt - torch.dot(dt, ax_old) * ax_old
        axis_offset = float(torch.linalg.norm(perp).item())
        rad_diff = abs(r_old - r_new)
        if cos_ang < 1.0 - 1e-6 or axis_offset > freeze_spec.tol_mm or rad_diff > freeze_spec.tol_mm:
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.FREEZE_VIOLATION,
                metrics={
                    "axis_cos_angle": cos_ang,
                    "axis_offset_mm": axis_offset,
                    "radius_diff_mm": rad_diff,
                },
                message=(
                    f"DATUM_AXIS freeze violated: offset={axis_offset:.5f} mm, "
                    f"radius_diff={rad_diff:.5f} mm."
                ),
            )
        return VerificationReport(
            status=Status.PASS,
            metrics={
                "axis_cos_angle": cos_ang,
                "axis_offset_mm": axis_offset,
                "radius_diff_mm": rad_diff,
            },
            message="DATUM_AXIS freeze satisfied (axis & diameter locked; thickness free).",
        )

    return VerificationReport(
        status=Status.OUT_OF_DOMAIN,
        message=f"Unsupported freeze scope: {freeze_spec.scope}",
    )


def verify(
    part: Part | Node,
    spec: Spec,
    reference_part: Part | Node | None = None,
    params: dict[str, Any] | None = None,
    reference_params: dict[str, Any] | None = None,
    max_depth: int = 4,
    initial_radius: float = 1.0,
) -> VerificationReport:
    """Run all clearance and freeze rules in Spec on hard geometry."""
    root = part.root if isinstance(part, Part) else part
    q_part = GeometryQuery(root, budget=spec.budget)

    obstacle_metrics: dict[str, Any] = {}
    for rule in spec.obstacles:
        q_obs = GeometryQuery(rule.obstacle, budget=spec.budget)
        rep = check_clearance(
            q_part,
            q_obs,
            min_clearance_mm=rule.min_clearance_mm,
            max_depth=max_depth,
            initial_radius=initial_radius,
            params_a=params,
        )
        obstacle_metrics[rule.name] = rep.status.value
        if rep.status != Status.PASS:
            return rep

    freeze_metrics: dict[str, Any] = {}
    all_freezes = list(spec.freezes)
    if isinstance(part, Part):
        for f in part.freezes:
            if f not in all_freezes:
                all_freezes.append(f)

    if all_freezes:
        ref = reference_part if reference_part is not None else part
        for fspec in all_freezes:
            frep = check_freeze(ref, part, fspec, params_old=reference_params, params_new=params)
            freeze_metrics[fspec.node_id] = frep.status.value
            if frep.status != Status.PASS:
                return frep

    return VerificationReport(
        status=Status.PASS,
        metrics={"obstacles": obstacle_metrics, "freezes": freeze_metrics},
        message="All Spec clearance and freeze constraints verified PASS.",
    )
