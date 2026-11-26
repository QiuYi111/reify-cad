"""M6 E2E Verification: 3D RMF spline sweep, parametric loft, and differentiable certified NURBS shell."""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch
from OCP.BRepCheck import BRepCheck_Analyzer

from reify.ast import Box, JointGraph, Param, Part, PrismaticJoint, Transform
from reify.brep_compiler import compile_brep, export_step
from reify.kinematics import KinematicChain, LinearMotionPath, check_kinematic_sweep
from reify.nurbs import (
    BSplineCurve3D,
    BSplineSurface,
    BSplineSurfaceShell,
    LoftSketches,
    SweepAlongSpline,
)
from reify.physics import mesh_brep_gmsh_channel_a
from reify.sketch import Loop2D, Sketch2D
from reify.solve import SolveStatus, solve
from reify.spec import ClearanceRule, Spec
from reify.types import Status
from reify.units import mm


def test_m6_sweep_and_loft_e2e(tmp_path: pathlib.Path) -> None:
    """I-M6-1: S-curved 3D RMF spline sweep + multi-section LoftSketches -> STEP, kinematics, Gmsh."""
    spine = BSplineCurve3D(
        [
            (mm(0.0), mm(0.0), mm(0.0)),
            (mm(15.0), mm(10.0), mm(8.0)),
            (mm(30.0), mm(-10.0), mm(16.0)),
            (mm(45.0), mm(0.0), mm(24.0)),
        ],
        degree=3,
    )
    pipe = Part(root=SweepAlongSpline(spine, profile=mm(3.0)), name="s_pipe")
    rep = export_step(pipe, tmp_path / "s_pipe.step", hausdorff_tol_mm=0.05)
    assert rep.status == Status.PASS, rep.message

    # M2 kinematic sweep along Z away from an obstacle
    obs = Transform(Box(mm(10.0), mm(10.0), mm(6.0)), translation=(mm(22.0), mm(0.0), mm(-12.0)))
    chain = KinematicChain(
        joint_graph=JointGraph([PrismaticJoint("slide", "base", "pipe", (0.0, 0.0, 1.0), 0.0, 10.0)]),
        link_shapes={"pipe": pipe.root},
    )
    kin_rep = check_kinematic_sweep(
        chain=chain,
        moving_link="pipe",
        obstacle_node=obs,
        path=LinearMotionPath(q_start={"slide": 0.0}, q_end={"slide": 10.0}),
        min_clearance_mm=2.0,
    )
    assert kin_rep.status in (Status.PASS, Status.UNKNOWN)

    # M3 Gmsh mesh export
    mesh = mesh_brep_gmsh_channel_a(pipe.root, mesh_size_mm=3.0, order=1)
    assert mesh["tets"].shape[0] > 50

    # Multi-section LoftSketches (compatible 8-segment rounded_rect topology)
    s0 = Sketch2D(Loop2D.rounded_rect(mm(20.0), mm(14.0), mm(3.0)))
    s1 = Sketch2D(Loop2D.rounded_rect(mm(14.0), mm(18.0), mm(4.0)))
    s2 = Sketch2D(Loop2D.rounded_rect(mm(12.0), mm(12.0), mm(5.99)))
    loft = LoftSketches([s0, s1, s2], z_levels=[mm(0.0), mm(10.0), mm(22.0)])
    rep_loft = export_step(loft, tmp_path / "loft.step", hausdorff_tol_mm=0.25)
    assert rep_loft.status == Status.PASS, rep_loft.message


def test_m6_nurbs_shell_bounds_autograd_and_solve(tmp_path: pathlib.Path) -> None:
    """I-M6-2 & I-M6-3: NURBS interval lower bounds (5k boxes), Envelope autograd, STEP & solve() bulge."""
    z_mid = Param("z_mid", 0.0, 0.0, 16.0, "mm")

    # 4x4 bicubic B-Spline control net over [-20, 20] x [-20, 20]
    xs = [-20.0, -7.0, 7.0, 20.0]
    ys = [-20.0, -7.0, 7.0, 20.0]
    net = []
    for i, xv in enumerate(xs):
        row = []
        for j, yv in enumerate(ys):
            if i in (1, 2) and j in (1, 2):
                row.append((mm(xv), mm(yv), z_mid))
            else:
                row.append((mm(xv), mm(yv), mm(0.0)))
        net.append(row)

    surf = BSplineSurface(net, degree_u=3, degree_v=3)
    canopy = Part(root=BSplineSurfaceShell(surf, thickness=mm(2.0)), name="nurbs_canopy")

    # 1. Native NURBS STEP round-trip at z_mid = 6.0 mm
    rep_step = export_step(canopy, tmp_path / "canopy.step", params={"z_mid": 6.0}, hausdorff_tol_mm=0.35)
    assert rep_step.status == Status.PASS, rep_step.message

    # 2. 5,000 random 3D box interval lower bound falsification (0 violations)
    rng = np.random.default_rng(123)
    b_lo = rng.uniform([-25.0, -25.0, -8.0], [20.0, 20.0, 12.0], size=(5000, 3))
    b_w = rng.uniform([0.4, 0.4, 0.4], [3.5, 3.5, 3.5], size=(5000, 3))
    b_hi = b_lo + b_w
    offsets = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.25, 0.75, 0.5]], dtype=np.float64)
    sample_pts = (b_lo[:, None, :] + b_w[:, None, :] * offsets[None, :, :]).reshape(-1, 3)
    sample_d = surf.eval_unsigned_distance(torch.from_numpy(sample_pts), params={"z_mid": 6.0}, dtype=torch.float64).detach().numpy()
    sample_d = sample_d.reshape(5000, len(offsets))

    violations = 0
    for i in range(5000):
        d_lb = surf.interval_bound_distance(
            (float(b_lo[i, 0]), float(b_lo[i, 1]), float(b_lo[i, 2])),
            (float(b_hi[i, 0]), float(b_hi[i, 1]), float(b_hi[i, 2])),
            params={"z_mid": 6.0},
        )
        if d_lb > float(np.min(sample_d[i])) + 1e-9:
            violations += 1
    assert violations == 0

    # 3. Envelope Theorem autograd vs float64 central finite difference
    q = torch.tensor([[0.0, 0.0, 10.0], [5.0, -4.0, -3.0], [-8.0, 6.0, 8.0]], dtype=torch.float64)
    tz = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
    loss = surf.eval_unsigned_distance(q, params={"z_mid": tz}, dtype=torch.float64).sum()
    loss.backward()
    g_ad = float(tz.grad.item())

    h = 1e-6
    with torch.no_grad():
        fp = float(surf.eval_unsigned_distance(q, params={"z_mid": 6.0 + h}, dtype=torch.float64).sum().item())
        fm = float(surf.eval_unsigned_distance(q, params={"z_mid": 6.0 - h}, dtype=torch.float64).sum().item())
    g_fd = (fp - fm) / (2.0 * h)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-12) < 1e-4

    # 4. solve(): obstacle at (0, 0, 2.0) of size 10x10x6 -> canopy bulges upward in z_mid to clear it!
    obs = Transform(Box(mm(10.0), mm(10.0), mm(6.0)), translation=(mm(0.0), mm(0.0), mm(1.0)))
    spec = Spec(obstacles=[ClearanceRule(obstacle=obs, min_clearance_mm=1.5)])
    t0 = time.perf_counter()
    res = solve(canopy, params_list=[z_mid], spec=spec, initial_batch_size=12, steps_per_start=35, lr=0.12)
    dt = time.perf_counter() - t0
    assert res.status == SolveStatus.FEASIBLE_FOUND
    assert res.verify_report.status == Status.PASS
    assert res.params["z_mid"] > 7.5
    assert dt < 30.0


def test_m6_sweep_along_spline_sketch2d_profile(tmp_path: pathlib.Path) -> None:
    """Verify SweepAlongSpline with non-circular Sketch2D profile (gp_Ax3 SetDisplacement + SDF mesh)."""
    from reify.sdf_mesh import extract_sdf_mesh

    path = BSplineCurve3D(
        control_points=[
            (mm(0.0), mm(0.0), mm(0.0)),
            (mm(20.0), mm(0.0), mm(8.0)),
            (mm(45.0), mm(15.0), mm(20.0)),
            (mm(70.0), mm(20.0), mm(26.0)),
        ],
        degree=3,
    )
    prof = Sketch2D(Loop2D.rounded_rect(mm(8.0), mm(5.0), mm(1.2)))
    arm = SweepAlongSpline(path=path, profile=prof)

    mesh = extract_sdf_mesh(arm, resolution=64, split_creases=False)
    assert mesh.is_watertight
    assert len(mesh.faces) > 100

    rep = export_step(arm, tmp_path / "sweep_sketch2d.step", hausdorff_tol_mm=0.2)
    assert rep.status == Status.PASS, rep.message
