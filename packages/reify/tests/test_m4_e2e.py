"""M4 E2E Verification: 2D Sketch & Spline Kernel (distance, interval bounds, autograd, OCCT face)."""

from __future__ import annotations

import math

import numpy as np
import torch
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from scipy.interpolate import BSpline as SciPyBSpline

from reify.ast import Param
from reify.sketch import (
    Arc2D,
    BSpline2D,
    CubicBezier2D,
    LineSeg2D,
    Loop2D,
    QuadBezier2D,
    Sketch2D,
)
from reify.units import deg, mm


def test_m4_sketch_distance_and_interval_bounds() -> None:
    """I-M4-1 & I-M4-2: Exact 1-Lipschitz SDF vs dense brute-force oracle and 10k-box interval falsification."""
    # Complex outer loop: straight edges + arc + cubic bezier
    outer = Loop2D(
        [
            LineSeg2D((mm(-10.0), mm(-10.0)), (mm(10.0), mm(-10.0))),
            Arc2D((mm(10.0), mm(0.0)), mm(10.0), deg(-90.0), deg(90.0)),
            CubicBezier2D(
                (mm(10.0), mm(10.0)),
                (mm(3.0), mm(18.0)),
                (mm(-3.0), mm(4.0)),
                (mm(-10.0), mm(10.0)),
            ),
            LineSeg2D((mm(-10.0), mm(10.0)), (mm(-10.0), mm(-10.0))),
        ]
    )
    hole = Loop2D.circle((mm(0.0), mm(0.0)), mm(3.0))
    sketch = Sketch2D(outer=outer, holes=[hole])

    # Verify OCCT 2D face compilation & area
    face = sketch.to_occt_face()
    assert BRepCheck_Analyzer(face).IsValid()
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    assert props.Mass() > 300.0

    # Dense brute-force boundary oracle (10,000 points)
    bnd = sketch.sample_boundary_points(n_per_seg=2500, dtype=torch.float64).detach().numpy()
    rng = np.random.default_rng(42)
    q_pts_np = rng.uniform([-25.0, -20.0], [28.0, 22.0], size=(400, 2))
    q_pts = torch.from_numpy(q_pts_np).to(torch.float64)
    sdf = sketch.eval_sdf_2d(q_pts, dtype=torch.float64).detach().numpy()

    d2 = np.sum((q_pts_np[:, None, :] - bnd[None, :, :]) ** 2, axis=-1)
    true_unsigned = np.sqrt(np.min(d2, axis=1))
    # Outside points far from segments must match dense oracle tightly
    assert float(np.max(np.abs(np.abs(sdf) - true_unsigned))) < 2e-3

    # 10,000 random 2D box interval bound falsification
    rng2 = np.random.default_rng(7)
    b_lo = rng2.uniform([-24.0, -18.0], [24.0, 18.0], size=(10000, 2))
    b_w = rng2.uniform([0.2, 0.2], [3.5, 3.5], size=(10000, 2))
    b_hi = b_lo + b_w
    violations = 0
    # Sample 9 points (center + 8 box points) per box to attempt falsification
    offsets = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.25, 0.75],
            [0.75, 0.25],
        ],
        dtype=np.float64,
    )
    all_sample_pts = (b_lo[:, None, :] + b_w[:, None, :] * offsets[None, :, :]).reshape(-1, 2)
    all_sdfs = sketch.eval_sdf_2d(torch.from_numpy(all_sample_pts), dtype=torch.float64).detach().numpy()
    all_sdfs = all_sdfs.reshape(10000, len(offsets))

    for i in range(10000):
        lo, hi = sketch.interval_bound_2d(
            (float(b_lo[i, 0]), float(b_lo[i, 1])),
            (float(b_hi[i, 0]), float(b_hi[i, 1])),
        )
        smin = float(np.min(all_sdfs[i]))
        smax = float(np.max(all_sdfs[i]))
        if lo > smin + 1e-9 or hi < smax - 1e-9:
            violations += 1
    assert violations == 0


def test_m4_bspline_boehm_and_envelope_autograd() -> None:
    """I-M4-3: Boehm extraction matches SciPy BSpline (<1e-12) and Envelope Theorem matches FD (<1e-5)."""
    ctrl_vals = [(0.0, 0.0), (5.0, 12.0), (14.0, 14.0), (22.0, -3.0), (30.0, 8.0), (36.0, 0.0)]
    p_x = Param("cx2", 14.0, 8.0, 20.0, "mm")
    p_y = Param("cy2", 14.0, 6.0, 22.0, "mm")
    ctrl_pts = [
        (mm(ctrl_vals[0][0]), mm(ctrl_vals[0][1])),
        (mm(ctrl_vals[1][0]), mm(ctrl_vals[1][1])),
        (p_x, p_y),
        (mm(ctrl_vals[3][0]), mm(ctrl_vals[3][1])),
        (mm(ctrl_vals[4][0]), mm(ctrl_vals[4][1])),
        (mm(ctrl_vals[5][0]), mm(ctrl_vals[5][1])),
    ]
    bs = BSpline2D(ctrl_pts, degree=3)

    # 1. Verify Boehm extraction against SciPy BSpline
    n = len(ctrl_vals)
    deg_k = 3
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0])
    sp_x = SciPyBSpline(knots, [c[0] for c in ctrl_vals], deg_k)
    sp_y = SciPyBSpline(knots, [c[1] for c in ctrl_vals], deg_k)
    poly = bs.sample_polyline_tensor(50, dtype=torch.float64).detach().numpy()
    ts = np.linspace(0.0, 1.0, poly.shape[0])
    # Evaluate SciPy at span-uniform samples
    ts_spans = np.concatenate([
        np.linspace(0.0, 1.0 / 3.0, 51)[:-1],
        np.linspace(1.0 / 3.0, 2.0 / 3.0, 51)[:-1],
        np.linspace(2.0 / 3.0, 1.0, 51),
    ])
    ref_xy = np.stack([sp_x(ts_spans), sp_y(ts_spans)], axis=-1)
    assert float(np.max(np.abs(poly - ref_xy))) < 1e-12

    # 2. Envelope Theorem gradient vs float64 central finite difference
    q = torch.tensor([[15.0, 2.0], [10.0, 20.0], [28.0, -5.0]], dtype=torch.float64)
    tx = torch.tensor(14.0, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(14.0, dtype=torch.float64, requires_grad=True)
    loss = bs.eval_unsigned_distance(q, params={"cx2": tx, "cy2": ty}, dtype=torch.float64).sum()
    loss.backward()
    grad_ad = np.array([float(tx.grad.item()), float(ty.grad.item())])

    h = 1e-6
    grad_fd = np.zeros(2, dtype=np.float64)
    with torch.no_grad():
        for idx, name in enumerate(["cx2", "cy2"]):
            p_plus = {"cx2": 14.0 + (h if idx == 0 else 0.0), "cy2": 14.0 + (h if idx == 1 else 0.0)}
            p_minus = {"cx2": 14.0 - (h if idx == 0 else 0.0), "cy2": 14.0 - (h if idx == 1 else 0.0)}
            fp = float(bs.eval_unsigned_distance(q, params=p_plus, dtype=torch.float64).sum().item())
            fm = float(bs.eval_unsigned_distance(q, params=p_minus, dtype=torch.float64).sum().item())
            grad_fd[idx] = (fp - fm) / (2.0 * h)

    rel_err = np.linalg.norm(grad_ad - grad_fd) / max(np.linalg.norm(grad_fd), 1e-12)
    assert rel_err < 1e-5

    # 3. Also verify QuadBezier2D & closed periodic BSpline OCCT face
    qb = QuadBezier2D((mm(-5.0), mm(0.0)), (mm(0.0), mm(10.0)), (mm(5.0), mm(0.0)))
    assert float(qb.eval_unsigned_distance(torch.tensor([[0.0, 5.0]], dtype=torch.float64))[0].item()) < 1e-6

    closed_loop = Loop2D.closed_bspline(
        [(mm(10.0), mm(0.0)), (mm(0.0), mm(10.0)), (mm(-10.0), mm(0.0)), (mm(0.0), mm(-10.0))],
        degree=3,
    )
    closed_sketch = Sketch2D(closed_loop)
    cface = closed_sketch.to_occt_face()
    assert BRepCheck_Analyzer(cface).IsValid()


def test_m4_spline_extrude_solve_e2e(tmp_path) -> None:
    """I-M4-4: Optimize BSpline2D control point Param via solve(), certify PASS, and export STEP."""
    from reify.ast import Box, Part, Transform
    from reify.brep_compiler import export_step
    from reify.features import ExtrudeSketch
    from reify.solve import SolveStatus, solve
    from reify.spec import ClearanceRule, Spec
    from reify.types import Status

    # Spline cam whose top bump y-param starts too high (colliding with obstacle at y=16..26)
    bump_y = Param("bump_y", 18.0, 6.0, 20.0, "mm")
    loop = Loop2D(
        [
            LineSeg2D((mm(-12.0), mm(-8.0)), (mm(12.0), mm(-8.0))),
            LineSeg2D((mm(12.0), mm(-8.0)), (mm(12.0), mm(6.0))),
            CubicBezier2D(
                (mm(12.0), mm(6.0)),
                (mm(4.0), bump_y),
                (mm(-4.0), bump_y),
                (mm(-12.0), mm(6.0)),
            ),
            LineSeg2D((mm(-12.0), mm(6.0)), (mm(-12.0), mm(-8.0))),
        ]
    )
    cam = Part(root=ExtrudeSketch(Sketch2D(loop), height=mm(10.0)), name="spline_cam")
    obs = Transform(Box(mm(16.0), mm(10.0), mm(16.0)), translation=(mm(0.0), mm(19.0), mm(0.0)))
    spec = Spec(obstacles=[ClearanceRule(obstacle=obs, min_clearance_mm=2.0)])

    res = solve(
        cam,
        params_list=[bump_y],
        spec=spec,
        initial_batch_size=16,
        steps_per_start=40,
        lr=0.1,
    )
    assert res.status == SolveStatus.FEASIBLE_FOUND
    assert res.verify_report.status == Status.PASS
    assert res.params["bump_y"] < 14.0

    rep = export_step(cam, tmp_path / "spline_cam.step", params=res.params, hausdorff_tol_mm=0.05)
    assert rep.status == Status.PASS
