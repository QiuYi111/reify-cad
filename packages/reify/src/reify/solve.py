"""Smooth polynomial boolean proxies and batched multi-start solve() with hard-geometry verify()."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch

from reify.ast import (
    Intersect,
    Node,
    Param,
    Part,
    SmoothUnion,
    Subtract,
    Transform,
    Union,
    _val_deg_float,
    _val_mm_float,
    _val_mm_tensor,
    _val_rad_tensor,
)
from reify.features import CircularPattern, LinearPattern, Offset, Shell, TaperedExtrude
from reify.query import GeometryQuery
from reify.spec import Spec
from reify.types import Status, VerificationReport
from reify.verify import verify


class SolveStatus(str, Enum):
    FEASIBLE_FOUND = "FEASIBLE_FOUND"
    NO_FEASIBLE_FOUND_IN_BUDGET = "NO_FEASIBLE_FOUND_IN_BUDGET"


@dataclass(frozen=True)
class SolveResult:
    """Result of two-stage heuristic multi-start solve() followed by hard-geometry verify()."""

    status: SolveStatus
    params: dict[str, float]
    verify_report: VerificationReport
    best_soft_loss: float
    n_starts: int
    n_feasible_starts: int
    history: list[dict[str, Any]] = field(default_factory=list)


class SmoothBooleanProxy:
    """Polynomial smooth min/max boolean proxy (C1 quad_smin or C2 cubic_smin)."""

    def __init__(self, k_mm: float = 1.0, kind: str = "quad_smin") -> None:
        if kind not in ("quad_smin", "cubic_smin"):
            raise ValueError(f"Unsupported SmoothBooleanProxy kind: {kind!r}")
        self.k_mm = float(k_mm)
        self.kind = kind
        self.smoothness_class = "C1" if kind == "quad_smin" else "C2"

    def smooth_min(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        k = max(self.k_mm, 1e-12)
        if self.kind == "quad_smin":
            h = torch.clamp(0.5 + 0.5 * (fb - fa) / k, 0.0, 1.0)
            return fb * (1.0 - h) + fa * h - k * h * (1.0 - h)
        h = torch.clamp(k - torch.abs(fa - fb), min=0.0) / k
        return torch.minimum(fa, fb) - (1.0 / 6.0) * k * h * h * h

    def smooth_max(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        return -self.smooth_min(-fa, -fb)

    def eval_field(
        self,
        node: Node,
        pts: torch.Tensor,
        params: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if isinstance(node, Transform):
            p = pts.to(dtype=dtype)
            R, t = node.world_matrix_tensors(params, dtype=dtype)
            R = R.to(device=p.device)
            t = t.to(device=p.device)
            if R.ndim == 3:
                if p.ndim == 2:
                    p = p.unsqueeze(0)
                local_p = torch.matmul(p - t[:, None, :], R)
            else:
                local_p = (p - t) @ R
            return self.eval_field(node.child, local_p, params=params, dtype=dtype)
        if isinstance(node, Union):
            acc = self.eval_field(node.children[0], pts, params=params, dtype=dtype)
            for ch in node.children[1:]:
                acc = self.smooth_min(acc, self.eval_field(ch, pts, params=params, dtype=dtype))
            return acc
        if isinstance(node, Intersect):
            acc = self.eval_field(node.children[0], pts, params=params, dtype=dtype)
            for ch in node.children[1:]:
                acc = self.smooth_max(acc, self.eval_field(ch, pts, params=params, dtype=dtype))
            return acc
        if isinstance(node, Subtract):
            fb = self.eval_field(node.base, pts, params=params, dtype=dtype)
            ft = self.eval_field(node.tool, pts, params=params, dtype=dtype)
            return self.smooth_max(fb, -ft)
        if isinstance(node, SmoothUnion):
            return node.eval_sdf(pts, params=params, dtype=dtype)
        if isinstance(node, Offset):
            return self.eval_field(node.child, pts, params=params, dtype=dtype) - node.radius.to_tensor(params, dtype=dtype).to(device=pts.device) if isinstance(node.radius, Param) else self.eval_field(node.child, pts, params=params, dtype=dtype) - node.radius.as_mm()
        if isinstance(node, Shell) and not node.open_top:
            f = self.eval_field(node.child, pts, params=params, dtype=dtype)
            t = node.thickness.to_tensor(params, dtype=dtype).to(device=pts.device) if isinstance(node.thickness, Param) else node.thickness.as_mm()
            if isinstance(t, torch.Tensor) and t.ndim == 1 and f.ndim == 2:
                t = t[:, None]
            return self.smooth_max(f, -(f + t))
        return node.eval_sdf(pts, params=params, dtype=dtype)


def _degeneracy_barrier_loss(
    root: Node,
    p_dict: dict[str, torch.Tensor],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Differentiable barrier keeping TaperedExtrude and Shell(TaperedExtrude) away from wall breakthrough."""
    loss = torch.zeros(batch_size, dtype=dtype, device=device)

    def _walk(node: Node) -> None:
        nonlocal loss
        if isinstance(node, Transform):
            _walk(node.child)
        elif isinstance(node, (Union, Intersect)):
            for ch in node.children:
                _walk(ch)
        elif isinstance(node, Subtract):
            _walk(node.base)
            _walk(node.tool)
        elif isinstance(node, SmoothUnion):
            _walk(node.a)
            _walk(node.b)
        elif isinstance(node, (Offset, LinearPattern, CircularPattern)):
            _walk(node.child)
        elif isinstance(node, TaperedExtrude):
            bmin2, bmax2 = node.sketch.bounds_2d(None)
            half_span = 0.5 * min(bmax2[0] - bmin2[0], bmax2[1] - bmin2[1])
            h_t = _val_mm_tensor(node.height, p_dict, dtype).to(device=device)
            th_t = _val_rad_tensor(node.draft_angle, p_dict, dtype).to(device=device)
            inset = h_t * torch.tan(th_t)
            viol = torch.relu(inset - (half_span - 0.5))
            loss = loss + viol * viol
        elif isinstance(node, Shell):
            if isinstance(node.child, TaperedExtrude):
                te = node.child
                bmin2, bmax2 = te.sketch.bounds_2d(None)
                half_span = 0.5 * min(bmax2[0] - bmin2[0], bmax2[1] - bmin2[1])
                h_t = _val_mm_tensor(te.height, p_dict, dtype).to(device=device)
                th_t = _val_rad_tensor(te.draft_angle, p_dict, dtype).to(device=device)
                t_w = _val_mm_tensor(node.thickness, p_dict, dtype).to(device=device)
                total_inset = h_t * torch.tan(th_t) + t_w / torch.cos(th_t).clamp(min=1e-4)
                viol = torch.relu(total_inset - (half_span - 0.5))
                loss = loss + viol * viol
            else:
                _walk(node.child)

    _walk(root)
    return loss


def _resolve_solve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def solve(
    part: Part | Node,
    params_list: list[Param],
    spec: Spec,
    reference_part: Part | Node | None = None,
    reference_params: dict[str, Any] | None = None,
    soft_objective: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None,
    initial_batch_size: int = 64,
    steps_per_start: int = 80,
    lr: float = 0.05,
    proxy_k_mm: float = 1.0,
    proxy_kind: str = "quad_smin",
    verify_max_depth: int = 4,
    seed: int = 0,
    device: str = "auto",
) -> SolveResult:
    """Two-stage feasibility-first batched multi-start solver with mandatory hard-geometry verify()."""
    if type(part).__name__ == "BRepFinalized":
        raise TypeError("Cannot run solve() on irreversible BRepFinalized terminal object.")
    if not params_list:
        rep = verify(
            part,
            spec,
            reference_part=reference_part,
            reference_params=reference_params,
            max_depth=verify_max_depth,
        )
        return SolveResult(
            status=(
                SolveStatus.FEASIBLE_FOUND
                if rep.status == Status.PASS
                else SolveStatus.NO_FEASIBLE_FOUND_IN_BUDGET
            ),
            params={},
            verify_report=rep,
            best_soft_loss=0.0,
            n_starts=1,
            n_feasible_starts=1 if rep.status == Status.PASS else 0,
        )

    dev = _resolve_solve_device(device)
    part_obj = part if isinstance(part, Part) else Part(root=part)
    root = part_obj.root
    proxy = SmoothBooleanProxy(k_mm=proxy_k_mm, kind=proxy_kind)
    gq_part = GeometryQuery(root, budget=spec.budget)

    # Pre-sample static obstacle surface covers once on target device
    obs_data: list[tuple[GeometryQuery, torch.Tensor, float, float]] = []
    for rule in spec.obstacles:
        q_obs = GeometryQuery(rule.obstacle, budget=spec.budget)
        cov_obs = q_obs.cover_surface(max_radius=1.0)
        obs_pts = cov_obs.centers.to(dtype=torch.float32, device=dev)
        obs_data.append((q_obs, obs_pts, float(rule.min_clearance_mm), q_obs.lipschitz_bound()))

    dim = len(params_list)
    lows = torch.tensor([p.low for p in params_list], dtype=torch.float32, device=dev)
    highs = torch.tensor([p.high for p in params_list], dtype=torch.float32, device=dev)
    nom = torch.tensor([p.value for p in params_list], dtype=torch.float32, device=dev)
    u_nom = torch.clamp((nom - lows) / (highs - lows).clamp(min=1e-9), 0.0, 1.0)

    if initial_batch_size > 1:
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
        u_sobol = sobol.draw(initial_batch_size - 1).to(dtype=torch.float32, device=dev)
        u_starts = torch.cat([u_nom.unsqueeze(0), u_sobol], dim=0)
    else:
        u_starts = u_nom.unsqueeze(0)

    B = int(u_starts.shape[0])
    u = u_starts.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, steps_per_start), eta_min=0.15 * lr
    )

    stage1_steps = int(0.35 * steps_per_start)
    stage2_steps = int(0.75 * steps_per_start)
    margin_pad = 0.25
    L_part = gq_part.lipschitz_bound()

    final_hard_vec = torch.zeros(B, dtype=torch.float32, device=dev)
    final_soft_vec = torch.zeros(B, dtype=torch.float32, device=dev)

    for step in range(steps_per_start):
        # Homotopy / continuation annealing of smooth boolean proxy k_mm
        if step < stage1_steps:
            proxy.k_mm = 2.5 * float(proxy_k_mm)
        elif step < stage2_steps:
            proxy.k_mm = float(proxy_k_mm)
        else:
            proxy.k_mm = max(0.2, 0.25 * float(proxy_k_mm))

        opt.zero_grad()
        theta = lows.unsqueeze(0) + u * (highs - lows).unsqueeze(0)  # (B, P)
        p_dict: dict[str, torch.Tensor] = {p.name: theta[:, i] for i, p in enumerate(params_list)}

        # Batched parametric points: (B, N_part, 3)
        part_pts = gq_part.sample_parametric_points(
            n_per_primitive=64, params=p_dict, dtype=torch.float32
        )

        hard_loss = _degeneracy_barrier_loss(root, p_dict, B, torch.float32, dev)
        for q_obs, obs_pts, m_req, L_obs in obs_data:
            f_part_in_obs = q_obs.eval_field(part_pts, dtype=torch.float32) / L_obs  # (B, N_part)
            f_obs_in_part = proxy.eval_field(root, obs_pts, params=p_dict, dtype=torch.float32) / L_part  # (B, N_obs)
            v1 = torch.relu((m_req + margin_pad) - f_part_in_obs)
            v2 = torch.relu((m_req + margin_pad) - f_obs_in_part)
            hard_loss = hard_loss + torch.sum(v1 * v1, dim=-1) + torch.sum(v2 * v2, dim=-1)

        if soft_objective is not None:
            soft_loss = soft_objective(p_dict).to(dtype=torch.float32, device=dev)
        else:
            soft_loss = torch.sum((u - u_nom.unsqueeze(0)) ** 2, dim=-1)

        if step < stage1_steps:
            total = torch.sum(100.0 * hard_loss + 0.01 * soft_loss)
        elif step < stage2_steps:
            total = torch.sum(150.0 * hard_loss + 0.25 * soft_loss)
        else:
            total = torch.sum(200.0 * hard_loss + 1.0 * soft_loss)

        total.backward()
        opt.step()
        scheduler.step()
        with torch.no_grad():
            u.clamp_(0.0, 1.0)
            final_hard_vec = hard_loss.detach()
            final_soft_vec = soft_loss.detach()

    with torch.no_grad():
        theta_end = (lows.unsqueeze(0) + u * (highs - lows).unsqueeze(0)).detach().cpu()
        h_cpu = final_hard_vec.cpu().tolist()
        s_cpu = final_soft_vec.cpu().tolist()

    candidates: list[tuple[float, float, dict[str, float]]] = []
    history: list[dict[str, Any]] = []
    for s_idx in range(B):
        cand_dict = {p.name: float(theta_end[s_idx, i].item()) for i, p in enumerate(params_list)}
        candidates.append((float(h_cpu[s_idx]), float(s_cpu[s_idx]), cand_dict))
        history.append(
            {
                "start_idx": s_idx,
                "final_hard_loss": float(h_cpu[s_idx]),
                "final_soft_loss": float(s_cpu[s_idx]),
                "params": cand_dict,
            }
        )

    candidates.sort(key=lambda item: (item[0] > 1e-4, item[0], item[1]))

    # Hard-geometry final verification gate in float64 on CPU
    best_verified: tuple[float, dict[str, float], VerificationReport] | None = None
    last_rep = VerificationReport(status=Status.UNKNOWN, message="No candidates evaluated.")
    n_feasible = 0

    for _, soft_val, cand_params in candidates[:10]:
        rep = verify(
            part_obj,
            spec,
            reference_part=reference_part,
            params=cand_params,
            reference_params=reference_params,
            max_depth=verify_max_depth,
        )
        last_rep = rep
        if rep.status == Status.PASS:
            n_feasible += 1
            if best_verified is None or soft_val < best_verified[0]:
                best_verified = (soft_val, cand_params, rep)

    if best_verified is not None:
        return SolveResult(
            status=SolveStatus.FEASIBLE_FOUND,
            params=best_verified[1],
            verify_report=best_verified[2],
            best_soft_loss=best_verified[0],
            n_starts=B,
            n_feasible_starts=n_feasible,
            history=history,
        )

    return SolveResult(
        status=SolveStatus.NO_FEASIBLE_FOUND_IN_BUDGET,
        params=candidates[0][2],
        verify_report=last_rep,
        best_soft_loss=candidates[0][1],
        n_starts=B,
        n_feasible_starts=0,
        history=history,
    )
