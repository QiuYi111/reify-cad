"""Native PyTorch SDF Sphere Tracing renderer for pixel-exact CAD snapshots (zero mesh/GLB)."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from reify.ast import Box, Node, Part, Sphere, Transform
from reify.finalize import BRepFinalized
from reify.types import VerificationReport

_PALETTE_HEX = ("#4a6fa5", "#5b8c5a", "#8e6c88", "#d4a373", "#3d5a80", "#e07a5f", "#6b9080", "#b5838d")


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    s = hex_color.lstrip("#")
    if len(s) >= 6:
        return (
            int(s[0:2], 16) / 255.0,
            int(s[2:4], 16) / 255.0,
            int(s[4:6], 16) / 255.0,
        )
    return (0.29, 0.44, 0.65)


@dataclass
class _ScenePart:
    name: str
    root: Node
    bmin: torch.Tensor  # (3,) float32
    bmax: torch.Tensor  # (3,) float32
    lip: float
    color: tuple[float, float, float]


def _normalize_parts(
    obj: Node | Part | dict[str, Node | Part],
    *,
    params: dict[str, Any] | None = None,
    color: str | None = None,
) -> list[_ScenePart]:
    if isinstance(obj, BRepFinalized):
        raise NotImplementedError("render_sdf_snapshot does not render Class 2 BRepFinalized; use cad glb snapshot.")
    items = list(obj.items()) if isinstance(obj, dict) else [(obj.name if isinstance(obj, Part) else "part", obj)]
    parts: list[_ScenePart] = []
    for idx, (name, item) in enumerate(items):
        if isinstance(item, BRepFinalized):
            raise NotImplementedError("render_sdf_snapshot does not render Class 2 BRepFinalized; use cad glb snapshot.")
        root = item.root if isinstance(item, Part) else item
        bmin_t, bmax_t = root.local_bounds(params)
        lip = max(1.0, float(root.lipschitz_bound(params)))
        c_hex = color or _PALETTE_HEX[idx % len(_PALETTE_HEX)]
        parts.append(
            _ScenePart(
                name=str(name),
                root=root,
                bmin=torch.tensor(bmin_t, dtype=torch.float32),
                bmax=torch.tensor(bmax_t, dtype=torch.float32),
                lip=lip,
                color=_hex_to_rgb01(c_hex),
            )
        )
    return parts


def _parse_camera_dir(camera: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit view direction (from scene toward camera) and world up hint (Z-up)."""
    cam = (camera or "iso").strip().lower()
    if ":" in cam:
        az_s, el_s = cam.split(":", 1)
        az = math.radians(float(az_s))
        el = math.radians(float(el_s))
        v = torch.tensor(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            dtype=torch.float32,
        )
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        return v / torch.linalg.norm(v), up

    presets = {
        "iso": (torch.tensor([1.0, -1.0, 0.85], dtype=torch.float32), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)),
        "top": (torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)),
        "bottom": (torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)),
        "front": (torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)),
        "back": (torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)),
        "left": (torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)),
        "right": (torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)),
    }
    v, up = presets.get(cam, presets["iso"])
    return v / torch.linalg.norm(v), up


def _parse_clip(display: str | dict[str, Any] | None) -> tuple[bool, int, float, float]:
    """Return (enabled, axis_idx, at, sign) from display dict or JSON string."""
    if not display:
        return False, 2, 0.0, 1.0
    obj: Any = json.loads(display) if isinstance(display, str) else display
    if not isinstance(obj, dict):
        return False, 2, 0.0, 1.0
    clip = obj.get("clip")
    if not isinstance(clip, dict) or not clip.get("enabled", True):
        return False, 2, 0.0, 1.0
    axis_str = str(clip.get("axis", "z")).lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis_str, 2)
    at = float(clip.get("at", clip.get("offset", 0.0)))
    sign = -1.0 if bool(clip.get("invert", False)) else 1.0
    return True, axis_idx, at, sign


def _ray_aabb_intersect(
    origins: torch.Tensor,
    ray_dir: torch.Tensor,
    bmin: torch.Tensor,
    bmax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized slab intersection for parallel orthographic rays (N, 3) against AABB."""
    inv_d = torch.where(
        torch.abs(ray_dir) > 1e-8,
        1.0 / torch.where(torch.abs(ray_dir) > 1e-8, ray_dir, torch.ones_like(ray_dir)),
        torch.full_like(ray_dir, 1e12),
    )
    t0 = (bmin[None, :] - origins) * inv_d[None, :]
    t1 = (bmax[None, :] - origins) * inv_d[None, :]
    tmin_v = torch.minimum(t0, t1)
    tmax_v = torch.maximum(t0, t1)
    # Handle near-zero ray_dir axes safely
    for ax in range(3):
        if abs(float(ray_dir[ax])) <= 1e-8:
            inside_ax = (origins[:, ax] >= bmin[ax]) & (origins[:, ax] <= bmax[ax])
            tmin_v[:, ax] = torch.where(inside_ax, -1e9, 1e9)
            tmax_v[:, ax] = torch.where(inside_ax, 1e9, -1e9)
    t_near = torch.amax(tmin_v, dim=-1)
    t_far = torch.amin(tmax_v, dim=-1)
    hit = t_far >= torch.maximum(t_near, torch.zeros_like(t_near))
    return hit, torch.clamp(t_near, min=0.0), t_far


def _eval_scene(
    parts: list[_ScenePart],
    pts: torch.Tensor,
    params: dict[str, Any] | None,
    clip_cfg: tuple[bool, int, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate conservative scene SDF step, winning part_id, and clip-cap mask on (N, 3) pts."""
    n = pts.shape[0]
    if n == 0:
        return (
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
        )

    best_d = torch.full((n,), 1e9, dtype=torch.float32)
    best_pid = torch.zeros((n,), dtype=torch.long)

    for k, part in enumerate(parts):
        if len(parts) == 1:
            fk = part.root.eval_sdf(pts, params=params, dtype=torch.float32)
            if fk.ndim == 2:
                fk = fk[0]
            dk = fk / part.lip
        else:
            # Conservative AABB distance lower bound: skip eval_sdf when point is strictly farther than best_d
            d_box = torch.linalg.norm(
                torch.clamp(part.bmin[None, :] - pts, min=0.0) + torch.clamp(pts - part.bmax[None, :], min=0.0),
                dim=-1,
            )
            mask = d_box < best_d + 1e-3
            dk = torch.full((n,), 1e9, dtype=torch.float32)
            if torch.any(mask):
                sub_pts = pts[mask]
                fk = part.root.eval_sdf(sub_pts, params=params, dtype=torch.float32)
                if fk.ndim == 2:
                    fk = fk[0]
                dk[mask] = fk / part.lip
        better = dk < best_d
        best_d = torch.where(better, dk, best_d)
        best_pid = torch.where(better, k, best_pid)

    clip_on, clip_ax, clip_at, clip_sign = clip_cfg
    if clip_on:
        d_clip = (pts[:, clip_ax] - clip_at) * clip_sign
        is_cap = d_clip > best_d
        d_step = torch.maximum(best_d, d_clip)
    else:
        is_cap = torch.zeros((n,), dtype=torch.bool)
        d_step = best_d

    return d_step, best_pid, is_cap


def _sphere_trace(
    parts: list[_ScenePart],
    origins: torch.Tensor,
    ray_dir: torch.Tensor,
    t_init: torch.Tensor,
    t_far: torch.Tensor,
    active_mask: torch.Tensor,
    params: dict[str, Any] | None,
    clip_cfg: tuple[bool, int, float, float],
    *,
    eps: float,
    max_steps: int = 96,
    polish: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lipschitz-safe batched Sphere Tracing with active-ray compaction."""
    n = origins.shape[0]
    t = t_init.clone()
    hit = torch.zeros((n,), dtype=torch.bool)
    hit_pid = torch.zeros((n,), dtype=torch.long)
    hit_cap = torch.zeros((n,), dtype=torch.bool)

    idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    for _ in range(max_steps):
        if idx.numel() == 0:
            break
        pts = origins[idx] + t[idx, None] * ray_dir[None, :]
        d, pid, is_cap = _eval_scene(parts, pts, params, clip_cfg)

        converged = d <= eps
        if torch.any(converged):
            c_idx = idx[converged]
            hit[c_idx] = True
            hit_pid[c_idx] = pid[converged]
            hit_cap[c_idx] = is_cap[converged]

        keep = (~converged) & ((t[idx] + torch.clamp(d, min=eps)) <= t_far[idx])
        if not torch.any(keep):
            break
        idx = idx[keep]
        t[idx] = t[idx] + torch.clamp(d[keep], min=eps)

    if polish and torch.any(hit):
        h_idx = torch.nonzero(hit, as_tuple=False).squeeze(-1)
        pts = origins[h_idx] + t[h_idx, None] * ray_dir[None, :]
        d, pid, is_cap = _eval_scene(parts, pts, params, clip_cfg)
        t[h_idx] = torch.clamp(t[h_idx] + d, min=0.0)
        hit_pid[h_idx] = pid
        hit_cap[h_idx] = is_cap

    return hit, t, hit_pid, hit_cap


def _compute_part_normals(
    parts: list[_ScenePart],
    hit_pts: torch.Tensor,
    hit_pid: torch.Tensor,
    hit_cap: torch.Tensor,
    clip_cfg: tuple[bool, int, float, float],
    params: dict[str, Any] | None,
    h_fd: float,
) -> torch.Tensor:
    """Compute central-difference SDF normals per winning part (zero cross-part normal bleed)."""
    m = hit_pts.shape[0]
    normals = torch.zeros((m, 3), dtype=torch.float32)
    if m == 0:
        return normals

    offsets = torch.tensor(
        [
            [h_fd, 0.0, 0.0],
            [-h_fd, 0.0, 0.0],
            [0.0, h_fd, 0.0],
            [0.0, -h_fd, 0.0],
            [0.0, 0.0, h_fd],
            [0.0, 0.0, -h_fd],
        ],
        dtype=torch.float32,
    )

    non_cap = ~hit_cap
    for k, part in enumerate(parts):
        mask = non_cap & (hit_pid == k)
        if not torch.any(mask):
            continue
        sub = hit_pts[mask]
        probe = (sub[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        v = part.root.eval_sdf(probe, params=params, dtype=torch.float32)
        if v.ndim == 2:
            v = v[0]
        v = v.reshape(sub.shape[0], 6)
        g = torch.stack([v[:, 0] - v[:, 1], v[:, 2] - v[:, 3], v[:, 4] - v[:, 5]], dim=-1) / (2.0 * h_fd)
        nrm = torch.linalg.norm(g, dim=-1, keepdim=True).clamp(min=1e-8)
        normals[mask] = g / nrm

    if torch.any(hit_cap):
        _, clip_ax, _, clip_sign = clip_cfg
        cap_n = torch.zeros((3,), dtype=torch.float32)
        cap_n[clip_ax] = clip_sign
        normals[hit_cap] = cap_n

    return normals


def _compute_ao(
    parts: list[_ScenePart],
    hit_pts: torch.Tensor,
    normals: torch.Tensor,
    params: dict[str, Any] | None,
    clip_cfg: tuple[bool, int, float, float],
    scene_rad: float,
) -> torch.Tensor:
    """5-tap SDF ambient occlusion along surface normal."""
    m = hit_pts.shape[0]
    if m == 0:
        return torch.zeros((0,), dtype=torch.float32)
    ao = torch.zeros((m,), dtype=torch.float32)
    taps = (0.008, 0.025, 0.070)
    weights = (0.45, 0.35, 0.20)
    for frac, w in zip(taps, weights):
        dist = max(0.05, frac * scene_rad)
        d, _, _ = _eval_scene(parts, hit_pts + dist * normals, params, clip_cfg)
        occ = torch.clamp(1.0 - d / dist, 0.0, 1.0)
        ao += w * occ
    return torch.clamp(1.0 - 0.85 * ao, 0.22, 1.0)


def _render_parts_to_png(
    parts: list[_ScenePart],
    out_png: pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    camera: str = "iso",
    display: str | dict[str, Any] | None = None,
    width: int = 800,
    height: int = 600,
) -> pathlib.Path:
    with torch.no_grad():
        scene_bmin = torch.amin(torch.stack([p.bmin for p in parts], dim=0), dim=0)
        scene_bmax = torch.amax(torch.stack([p.bmax for p in parts], dim=0), dim=0)
        center = 0.5 * (scene_bmin + scene_bmax)
        extents = torch.clamp(scene_bmax - scene_bmin, min=1e-2)
        scene_diag = float(torch.linalg.norm(extents))
        scene_rad = 0.5 * scene_diag

        view_from_dir, up_hint = _parse_camera_dir(camera)
        ray_dir = -view_from_dir
        right = torch.linalg.cross(ray_dir, up_hint)
        if float(torch.linalg.norm(right)) < 1e-5:
            right = torch.linalg.cross(ray_dir, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
        right = right / torch.linalg.norm(right)
        cam_up = torch.linalg.cross(right, ray_dir)
        cam_up = cam_up / torch.linalg.norm(cam_up)

        # Project 8 corners of scene AABB onto (right, cam_up, view_from_dir) for tight framing
        corners = torch.tensor(
            [[scene_bmin[0] if (i & 1) == 0 else scene_bmax[0],
              scene_bmin[1] if (i & 2) == 0 else scene_bmax[1],
              scene_bmin[2] if (i & 4) == 0 else scene_bmax[2]] for i in range(8)],
            dtype=torch.float32,
        )
        rel = corners - center[None, :]
        proj_u = rel @ right
        proj_v = rel @ cam_up
        span_u = float(torch.amax(proj_u) - torch.amin(proj_u))
        span_v = float(torch.amax(proj_v) - torch.amin(proj_v))
        u_mid = 0.5 * float(torch.amax(proj_u) + torch.amin(proj_u))
        v_mid = 0.5 * float(torch.amax(proj_v) + torch.amin(proj_v))

        aspect = width / max(1, height)
        half_h = 0.56 * max(span_v, span_u / aspect, 1.0)
        half_w = half_h * aspect
        pixel_world = (2.0 * half_h) / max(1, height)

        cam_dist = scene_diag + 4.0
        cam_center = center + u_mid * right + v_mid * cam_up + cam_dist * view_from_dir
        clip_cfg = _parse_clip(display)

        # Expand part AABBs slightly by 1 pixel for conservative culling
        pad_box = max(0.5, 2.0 * pixel_world)
        scene_bmin_pad = scene_bmin - pad_box
        scene_bmax_pad = scene_bmax + pad_box

        # 1. 4x Coarse pre-pass
        cw, ch = max(16, width // 4), max(16, height // 4)
        cu = torch.linspace(-half_w, half_w, cw, dtype=torch.float32)
        cv = torch.linspace(half_h, -half_h, ch, dtype=torch.float32)
        cug, cvg = torch.meshgrid(cu, cv, indexing="xy")  # (ch, cw)
        c_origins = (
            cam_center[None, :]
            + cug.reshape(-1, 1) * right[None, :]
            + cvg.reshape(-1, 1) * cam_up[None, :]
        )
        c_active, c_tnear, c_tfar = _ray_aabb_intersect(c_origins, ray_dir, scene_bmin_pad, scene_bmax_pad)
        c_hit, c_t, _, _ = _sphere_trace(
            parts,
            c_origins,
            ray_dir,
            c_tnear,
            c_tfar,
            c_active,
            params,
            clip_cfg,
            eps=max(1e-3, 1.5 * pixel_world),
            max_steps=80,
            polish=False,
        )

        # Upsample 3x3 minimum coarse hit depth to fine grid
        c_depth_img = torch.where(c_hit, c_t, torch.full_like(c_t, 1e9)).reshape(1, 1, ch, cw)
        # 3x3 min-pool via -max_pool2d(-x)
        c_min_3x3 = -torch.nn.functional.max_pool2d(-c_depth_img, kernel_size=3, stride=1, padding=1)
        # Also detect if ANY 3x3 neighbor hit (so silhouette edges start safely at t_near)
        c_any_hit = torch.nn.functional.max_pool2d(c_hit.float().reshape(1, 1, ch, cw), kernel_size=3, stride=1, padding=1) > 0.5
        c_all_hit = (-torch.nn.functional.max_pool2d(-c_hit.float().reshape(1, 1, ch, cw), kernel_size=3, stride=1, padding=1)) > 0.5

        f_u = torch.linspace(-half_w, half_w, width, dtype=torch.float32)
        f_v = torch.linspace(half_h, -half_h, height, dtype=torch.float32)
        fug, fvg = torch.meshgrid(f_u, f_v, indexing="xy")  # (H, W)
        f_origins = (
            cam_center[None, :]
            + fug.reshape(-1, 1) * right[None, :]
            + fvg.reshape(-1, 1) * cam_up[None, :]
        )
        f_active, f_tnear, f_tfar = _ray_aabb_intersect(f_origins, ray_dir, scene_bmin_pad, scene_bmax_pad)

        # Map each fine pixel to nearest coarse cell for safe t_start
        ci_y = torch.clamp((torch.arange(height) * ch) // height, 0, ch - 1)
        ci_x = torch.clamp((torch.arange(width) * cw) // width, 0, cw - 1)
        gy_idx, gx_idx = torch.meshgrid(ci_y, ci_x, indexing="ij")
        fine_min_d = c_min_3x3[0, 0, gy_idx, gx_idx].reshape(-1)
        fine_all_hit = c_all_hit[0, 0, gy_idx, gx_idx].reshape(-1)
        fine_any_hit = c_any_hit[0, 0, gy_idx, gx_idx].reshape(-1)

        # If all 3x3 coarse neighbors hit, start safely just before min_d; on boundaries/thin features, start at t_near!
        t_start = torch.where(
            fine_all_hit,
            torch.maximum(f_tnear, fine_min_d - 4.0 * pixel_world),
            f_tnear,
        )
        # Only skip rays if NO 3x3 coarse neighbor hit AND ray misses every individual tight part AABB!
        if len(parts) > 1:
            any_part_hit = torch.zeros_like(f_active)
            for p in parts:
                ph, _, _ = _ray_aabb_intersect(f_origins, ray_dir, p.bmin - pad_box, p.bmax + pad_box)
                any_part_hit = any_part_hit | ph
            f_active = f_active & (fine_any_hit | any_part_hit)

        # 2. Fine Sphere Tracing
        f_hit, f_t, f_pid, f_cap = _sphere_trace(
            parts,
            f_origins,
            ray_dir,
            t_start,
            f_tfar,
            f_active,
            params,
            clip_cfg,
            eps=max(2e-3, 0.04 * pixel_world),
            max_steps=56,
            polish=True,
        )

        # 3. Shading on hit pixels
        bg_rgb = torch.tensor([0.945, 0.960, 0.980], dtype=torch.float32)  # #f1f5f9 studio background
        img_rgb = bg_rgb[None, None, :].expand(height, width, 3).clone()
        normals_img = torch.zeros((height, width, 3), dtype=torch.float32)
        depth_img = torch.full((height, width), float("inf"), dtype=torch.float32)
        pid_img = torch.full((height, width), -1, dtype=torch.long)
        cap_img = torch.zeros((height, width), dtype=torch.bool)

        h_idx = torch.nonzero(f_hit, as_tuple=False).squeeze(-1)
        if h_idx.numel() > 0:
            hit_pts = f_origins[h_idx] + f_t[h_idx, None] * ray_dir[None, :]
            hit_pid = f_pid[h_idx]
            hit_cap = f_cap[h_idx]

            h_fd = max(1e-4, 0.25 * pixel_world)
            nrm = _compute_part_normals(parts, hit_pts, hit_pid, hit_cap, clip_cfg, params, h_fd)
            ao = _compute_ao(parts, hit_pts, nrm, params, clip_cfg, scene_rad)

            # Key, fill, rim lights relative to camera frame
            key_dir = 0.55 * right + 0.70 * cam_up + 0.45 * view_from_dir
            key_dir = key_dir / torch.linalg.norm(key_dir)
            fill_dir = -0.65 * right - 0.35 * cam_up + 0.65 * view_from_dir
            fill_dir = fill_dir / torch.linalg.norm(fill_dir)

            # Soft shadow march toward key light (12 steps)
            sh_o = hit_pts + max(0.05, 1.5 * pixel_world) * nrm
            sh_t = torch.full((h_idx.numel(),), 0.5 * pixel_world, dtype=torch.float32)
            sh_active = torch.ones((h_idx.numel(),), dtype=torch.bool)
            shadow = torch.ones((h_idx.numel(),), dtype=torch.float32)
            s_idx = torch.nonzero(sh_active, as_tuple=False).squeeze(-1)
            for _ in range(6):
                if s_idx.numel() == 0:
                    break
                sd, _, _ = _eval_scene(parts, sh_o[s_idx] + sh_t[s_idx, None] * key_dir[None, :], params, clip_cfg)
                hit_sh = sd <= max(1e-3, 0.35 * pixel_world)
                if torch.any(hit_sh):
                    shadow[s_idx[hit_sh]] = 0.55
                # Soft penumbra factor
                shadow[s_idx] = torch.minimum(shadow[s_idx], torch.clamp(0.55 + 2.5 * sd / (sh_t[s_idx] + 1e-3), 0.55, 1.0))
                keep_s = (~hit_sh) & (sh_t[s_idx] < 0.6 * scene_diag)
                if not torch.any(keep_s):
                    break
                s_idx = s_idx[keep_s]
                sh_t[s_idx] = sh_t[s_idx] + torch.clamp(sd[keep_s], min=0.5 * pixel_world)

            palette_t = torch.tensor([p.color for p in parts], dtype=torch.float32)
            base_col = palette_t[hit_pid]
            # Distinct darker slate/warm tint on section-cut cap faces
            cap_tint = 0.72 * base_col + 0.28 * torch.tensor([0.22, 0.26, 0.32], dtype=torch.float32)
            base_col = torch.where(hit_cap[:, None], cap_tint, base_col)

            ndotl_k = torch.clamp(torch.sum(nrm * key_dir[None, :], dim=-1), 0.0, 1.0)
            ndotl_f = torch.clamp(torch.sum(nrm * fill_dir[None, :], dim=-1), 0.0, 1.0)
            half_v = key_dir + view_from_dir
            half_v = half_v / torch.linalg.norm(half_v)
            spec = torch.pow(torch.clamp(torch.sum(nrm * half_v[None, :], dim=-1), 0.0, 1.0), 36.0) * 0.22

            hemi = 0.5 + 0.5 * torch.sum(nrm * cam_up[None, :], dim=-1)
            lit = (
                (0.28 + 0.12 * hemi) * ao
                + 0.58 * ndotl_k * shadow
                + 0.22 * ndotl_f * ao
            )
            col = torch.clamp(base_col * lit[:, None] + spec[:, None] * shadow[:, None], 0.0, 1.0)

            flat_rgb = img_rgb.reshape(-1, 3)
            flat_rgb[h_idx] = col
            flat_nrm = normals_img.reshape(-1, 3)
            flat_nrm[h_idx] = nrm
            depth_img.reshape(-1)[h_idx] = f_t[h_idx]
            pid_img.reshape(-1)[h_idx] = hit_pid
            cap_img.reshape(-1)[h_idx] = hit_cap

        # 4. Crisp screen-space technical CAD edge ink
        hit_2d = f_hit.reshape(height, width)
        edge = torch.zeros((height, width), dtype=torch.bool)
        # Compare right & down neighbors
        for dy, dx_i in ((0, 1), (1, 0)):
            h0 = hit_2d[: height - dy, : width - dx_i]
            h1 = hit_2d[dy:, dx_i:]
            both = h0 & h1
            n0 = normals_img[: height - dy, : width - dx_i]
            n1 = normals_img[dy:, dx_i:]
            n_jump = (1.0 - torch.sum(n0 * n1, dim=-1)) > 0.24
            d0 = depth_img[: height - dy, : width - dx_i]
            d1 = depth_img[dy:, dx_i:]
            d_jump = torch.abs(d0 - d1) > 3.5 * pixel_world
            p_jump = pid_img[: height - dy, : width - dx_i] != pid_img[dy:, dx_i:]
            c_jump = cap_img[: height - dy, : width - dx_i] != cap_img[dy:, dx_i:]
            sil_jump = h0 ^ h1
            e = (both & (n_jump | d_jump | p_jump | c_jump)) | sil_jump
            edge[: height - dy, : width - dx_i] |= e

        ink_col = torch.tensor([0.12, 0.16, 0.23], dtype=torch.float32)
        img_rgb = torch.where(edge[:, :, None], 0.45 * img_rgb + 0.55 * ink_col[None, None, :], img_rgb)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        rgb_u8 = (torch.clamp(img_rgb, 0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(rgb_u8, mode="RGB").save(str(out_png))
        return out_png


def render_sdf_snapshot(
    obj: Node | Part | dict[str, Node | Part],
    out_png: str | pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    camera: str = "iso",
    display: str | dict[str, Any] | None = None,
    width: int = 800,
    height: int = 600,
    color: str | None = None,
) -> pathlib.Path:
    """Render a Reify Node/Part or multi-part dict directly from SDF via Sphere Tracing."""
    parts = _normalize_parts(obj, params=params, color=color)
    return _render_parts_to_png(
        parts,
        pathlib.Path(out_png),
        params=params,
        camera=camera,
        display=display,
        width=width,
        height=height,
    )


def render_violation_overlay_png(
    part: Node | Part,
    spec: Any,
    report: VerificationReport,
    out_png: str | pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    camera: str = "iso",
    width: int = 800,
    height: int = 600,
) -> pathlib.Path:
    """Render pure-SDF violation overlay (blue part, orange obstacles, red witness spheres)."""
    parts = _normalize_parts(part, params=params, color="#4a6fa5")
    for idx, rule in enumerate(getattr(spec, "obstacles", []) or []):
        obs_parts = _normalize_parts(rule.obstacle, params=params, color="#e67e22")
        for p in obs_parts:
            p.name = f"obstacle_{idx}"
        parts.extend(obs_parts)
    for idx, pt in enumerate(getattr(report, "witness_points", []) or []):
        if len(pt) == 3:
            from reify.units import mm

            sph = Transform(Sphere(mm(1.35)), translation=(mm(float(pt[0])), mm(float(pt[1])), mm(float(pt[2]))))
            w_parts = _normalize_parts(sph, params=params, color="#e74c3c")
            for p in w_parts:
                p.name = f"witness_{idx}"
            parts.extend(w_parts)

    return _render_parts_to_png(
        parts,
        pathlib.Path(out_png),
        params=params,
        camera=camera,
        width=width,
        height=height,
    )
