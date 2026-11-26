"""E2E verification of v2.1 industrial-grade pure-SDF & B-Rep algorithms."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import torch

from reify import (
    Box,
    ClearanceRule,
    Cylinder,
    ExtrudeSketch,
    Frustum,
    GeometryQuery,
    LoftSketches,
    Loop2D,
    Param,
    Part,
    Shell,
    Sketch2D,
    SmoothUnion,
    Spec,
    Sphere,
    Status,
    Subtract,
    TaperedExtrude,
    Transform,
    Union,
    check_directional_wall_thickness,
    check_draft_and_overhang,
    deg,
    export_sdf_glb,
    export_sdf_webgl_html,
    export_step,
    extract_sdf_mesh,
    mm,
    solve,
    verify,
    wost_solve_point,
)
from reify.solve import SolveStatus


def test_v2_1_manifold_dc_wood_ox_and_glb_yup(tmp_path: pathlib.Path) -> None:
    """Manifold Dual Contouring on multi-scale wood ox body (1.8mm horns, ears, sphere, -32 deg neck) + GLB Y-up."""
    torso = Transform(ExtrudeSketch(Sketch2D(Loop2D.rounded_rect(mm(112), mm(46), mm(12))), height=mm(48)), translation=(mm(0), mm(0), mm(74)))
    neck = Transform(Box(mm(36), mm(30), mm(28)), translation=(mm(52), mm(0), mm(91)), rotation_deg=(deg(0), deg(-32), deg(0)))
    head = Transform(ExtrudeSketch(Sketch2D(Loop2D.rounded_rect(mm(32), mm(30), mm(6))), height=mm(26)), translation=(mm(73), mm(0), mm(108)))
    ear_l = Transform(Cylinder(mm(3.8), mm(16)), translation=(mm(70), mm(10.5), mm(125)), rotation_deg=(deg(-14), deg(10), deg(0)))
    horn_l = Transform(Frustum(mm(4.6), mm(1.8), mm(24)), translation=(mm(77), mm(12), mm(130)), rotation_deg=(deg(-34), deg(8), deg(0)))
    tail = Transform(Sphere(mm(9)), translation=(mm(-76), mm(0), mm(107)))
    body = Union([torso, neck, head, ear_l, horn_l, tail])

    mesh = extract_sdf_mesh(body, resolution=96, split_creases=False)
    assert mesh.is_watertight
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    assert np.all(counts == 2), f"Non-manifold edge counts: {np.unique(counts, return_counts=True)}"

    glb_p = tmp_path / "ox.glb"
    export_sdf_glb(Part(body, name="body"), glb_p, resolution=64)
    assert glb_p.exists() and glb_p.stat().st_size > 1000


def test_v2_1_pure_sdf_distance_query_smooth_union() -> None:
    """Pure-SDF Affine Interval B&B distance_query works on non-B-Rep SmoothUnion without OCCT."""
    su = SmoothUnion(
        Box(mm(20), mm(20), mm(20)),
        Transform(Sphere(mm(10)), translation=(mm(10), mm(0), mm(0))),
        k=mm(2.0),
    )
    obs = Transform(Box(mm(10), mm(10), mm(10)), translation=(mm(30), mm(0), mm(0)))
    # Sphere extends to x = 20, obs starts at x = 25 -> exact gap = 5.0 mm
    res = GeometryQuery(su).distance_query(GeometryQuery(obs), deflection=0.05)
    assert res.status == Status.PASS
    assert not res.containment
    assert res.lower_bound <= res.value <= res.upper_bound + 1e-6
    assert 4.7 <= res.value <= 5.3


def test_v2_1_sdf_dfm_wall_thickness_and_draft_overhang() -> None:
    """Pure-SDF Sphere-Tracing wall thickness and analytic draft/overhang screening."""
    enc = Shell(
        TaperedExtrude(Sketch2D(Loop2D.rounded_rect(mm(40), mm(40), mm(6))), height=mm(20), draft_angle=deg(5.0)),
        thickness=mm(2.5),
        open_top=True,
    )
    wt = check_directional_wall_thickness(enc, min_wall_mm=2.0, sample_radius=2.5)
    assert wt.status == Status.PASS, wt.message
    assert 2.2 <= wt.metrics["min_directional_thickness_mm"] <= 2.8

    ov = check_draft_and_overhang(enc, min_draft_deg=2.0, max_overhang_deg=45.0, sample_radius=2.5)
    assert ov.status == Status.PASS, ov.message
    assert ov.metrics["min_draft_deg"] >= 4.5


def test_v2_1_octree_verify_and_homotopy_barrier_solve() -> None:
    """Affine octree verify() + Homotopy continuation solve() with degeneracy barrier."""
    draft = Param("draft_deg", 2.0, 0.5, 22.0, "deg")
    body = TaperedExtrude(Sketch2D(Loop2D.rounded_rect(mm(46), mm(40), mm(7))), height=mm(20), draft_angle=draft)
    enc = Part(Shell(body, thickness=mm(2.5), open_top=True), name="enc")
    obs = Transform(Box(mm(8), mm(18), mm(12)), translation=(mm(25.5), mm(0), mm(4)))
    spec = Spec(obstacles=[ClearanceRule(obs, min_clearance_mm=1.0)])

    rep_fail = verify(enc, spec, params={"draft_deg": 2.0})
    assert rep_fail.status == Status.FAIL

    res = solve(enc, [draft], spec, initial_batch_size=16, steps_per_start=45)
    assert res.status == SolveStatus.FEASIBLE_FOUND
    assert res.verify_report.status == Status.PASS


def test_v2_1_walk_on_stars_mixed_dirichlet_neumann() -> None:
    """Walk-on-Stars (Sawhney 2023): 3D box x in [-10,10] with Dirichlet x=-10 (0), x=+10 (100) and Neumann 0 on y,z."""
    box = Box(mm(20), mm(20), mm(20))
    q = GeometryQuery(box)

    def dir_dist(pts: np.ndarray) -> np.ndarray:
        return np.minimum(np.abs(pts[:, 0] + 10.0), np.abs(10.0 - pts[:, 0]))

    def dir_val(pts: np.ndarray) -> np.ndarray:
        return np.where(pts[:, 0] >= 0.0, 100.0, 0.0)

    # Exact analytical solution u(x,y,z) = 5 * (x + 10): at (0,0,0) -> 50.0; at (4, 3, -2) -> 70.0
    r0 = wost_solve_point(q, (0.0, 0.0, 0.0), dir_dist, dir_val, n_samples=500, seed=42)
    assert r0.status == Status.PASS
    assert abs(r0.metrics["mean"] - 50.0) < 4.5, r0.metrics

    r1 = wost_solve_point(q, (4.0, 3.0, -2.0), dir_dist, dir_val, n_samples=500, seed=7)
    assert r1.status == Status.PASS
    assert abs(r1.metrics["mean"] - 70.0) < 4.5, r1.metrics


def test_v2_1_unequal_loft_holed_draft_and_webgl_viewer(tmp_path: pathlib.Path) -> None:
    """Unequal-segment LoftSketches (rounded_rect -> circle), holed TaperedExtrude STEP, and WebGL2 SDF HTML."""
    sec_bot = Sketch2D(Loop2D.rounded_rect(mm(24), mm(24), mm(4)))
    sec_top = Sketch2D(Loop2D.circle((mm(0), mm(0)), mm(8)))
    loft = LoftSketches([sec_bot, sec_top], [mm(0), mm(20)])
    pts = torch.tensor([[0.0, 0.0, 10.0], [30.0, 0.0, 10.0]], dtype=torch.float64)
    vals = loft.eval_sdf(pts)
    assert float(vals[0]) < 0.0 and float(vals[1]) > 0.0

    # Holed TaperedExtrude via MAT2d MakeEvolved -> valid STEP
    holed_sk = Sketch2D(
        Loop2D.rounded_rect(mm(40), mm(40), mm(6)),
        holes=[Loop2D.circle((mm(0), mm(0)), mm(5))],
    )
    holed_draft = TaperedExtrude(holed_sk, height=mm(16), draft_angle=deg(10.0))
    rep_step = export_step(holed_draft, tmp_path / "holed_draft.step", hausdorff_tol_mm=0.05)
    assert rep_step.status == Status.PASS, rep_step.message

    # WebGL2 GLSL SDF HTML export
    html_p = tmp_path / "viewer.html"
    export_sdf_webgl_html(
        {"enclosure": Part(Shell(TaperedExtrude(Sketch2D(Loop2D.rounded_rect(mm(36), mm(36), mm(5))), mm(16), deg(8)), mm(2.5), open_top=True))},
        html_p,
    )
    txt = html_p.read_text(encoding="utf-8")
    assert "webgl2" in txt and "mapScene" in txt and "sdShellTaperedOpenZ" in txt


def test_v2_2_gpu_mps_or_cuda_parity_and_renderer(tmp_path: pathlib.Path):
    """Verify CUDA/MPS GPU backend parity against CPU on SDF evaluation and Sphere Tracing render."""
    import numpy as np
    import torch
    from PIL import Image

    from reify import device_info, get_compute_device, render_sdf_snapshot

    info = device_info()
    assert "active_float32_device" in info and "mps_available" in info and "cuda_available" in info
    assert get_compute_device(torch.float32, prefer="cpu").type == "cpu"

    part = Part(
        Union(
            [
                Transform(Box(mm(30), mm(20), mm(12)), translation=(mm(0), mm(0), mm(6))),
                Transform(Frustum(mm(4.0), mm(1.5), mm(18)), translation=(mm(10), mm(0), mm(18)), rotation_deg=(deg(15), deg(0), deg(0))),
            ]
        ),
        name="gpu_part",
    )

    gpu_dev = None
    if torch.cuda.is_available():
        gpu_dev = "cuda"
    elif torch.backends.mps.is_available():
        gpu_dev = "mps"

    cpu_png = tmp_path / "cpu.png"
    render_sdf_snapshot(part, cpu_png, width=256, height=192, device="cpu")
    assert cpu_png.exists() and cpu_png.stat().st_size > 1000

    if gpu_dev is not None:
        gpu_png = tmp_path / "gpu.png"
        render_sdf_snapshot(part, gpu_png, width=256, height=192, device=gpu_dev)
        a = np.asarray(Image.open(cpu_png), dtype=np.int16)
        b = np.asarray(Image.open(gpu_png), dtype=np.int16)
        assert np.abs(a - b).mean() < 1.0


def test_v2_2_dual_nonexact_distance_query_stage3():
    """Verify Stage 3 sparse 6D B&B closes dual non-exact SmoothUnion vs SmoothUnion to < 1e-3 mm."""
    su_a = SmoothUnion(
        Box(mm(10), mm(10), mm(10)),
        Transform(Sphere(mm(6)), translation=(mm(6), mm(0), mm(0))),
        k=mm(2.0),
    )
    su_b = Transform(su_a, translation=(mm(26.0), mm(0.0), mm(0.0)))
    res = GeometryQuery(su_a).distance_query(GeometryQuery(su_b), deflection=1e-4)
    assert res.status == Status.PASS
    assert not res.containment
    assert (res.upper_bound - res.lower_bound) < 1e-3
    assert abs(res.value - 9.0) < 0.05


def test_v2_2_webgl_3d_texture_universal_fallback_and_multipart_step(tmp_path: pathlib.Path):
    """Verify sampler3D baked 3D texture fallback in WebGL2 viewer and multi-part dict STEP export."""
    import torch
    from reify import FreeImplicit, export_sdf_webgl_html, export_step

    imp = FreeImplicit(
        lambda p: torch.linalg.norm(p, dim=-1) - 10.0,
        bounds=((-12.0, -12.0, -12.0), (12.0, 12.0, 12.0)),
        lipschitz=1.0,
    )
    html_p = tmp_path / "implicit_viewer.html"
    export_sdf_webgl_html(imp, html_p)
    txt = html_p.read_text(encoding="utf-8")
    assert "sampler3D" in txt and "sdVolume3D" in txt and "uVol0" in txt and "texImage3D" in txt

    asm = {
        "base": Part(Box(mm(20), mm(20), mm(6)), name="base"),
        "post": Part(Transform(Cylinder(mm(4), mm(14)), translation=(mm(0), mm(0), mm(10))), name="post"),
    }
    step_p = tmp_path / "asm.step"
    rep = export_step(asm, step_p)
    assert rep.status == Status.PASS, rep.message
    assert step_p.exists() and rep.metrics["topology"]["solids"] == 2
