"""M5 E2E Verification: Drafted open-top shelled enclosure + revolved boss + circular bolt pattern + CalculiX."""

from __future__ import annotations

import pathlib

from reify.ast import Cylinder, Param, Part, Subtract, Transform, Union
from reify.brep_compiler import compile_brep, export_step
from reify.dfm import compute_structural_thickness
from reify.features import (
    CircularPattern,
    ExtrudeSketch,
    Frustum,
    LinearPattern,
    Offset,
    RevolveSketch,
    Shell,
    TaperedExtrude,
)
from reify.physics import export_calculix_inp, mesh_brep_gmsh_channel_a, run_calculix_or_conformal_fem
from reify.sketch import Loop2D, Sketch2D
from reify.solve import SolveStatus, solve
from reify.spec import ClearanceRule, Spec
from reify.types import Status
from reify.units import deg, mm


def test_m5_molded_enclosure_step_dfm_and_calculix(tmp_path: pathlib.Path) -> None:
    """I-M5-1 & I-M5-3: Drafted open-top shell + revolved boss + circular pattern -> STEP, DFM, CalculiX."""
    outer_loop = Loop2D.rounded_rect(mm(60.0), mm(40.0), mm(6.0))
    body_solid = TaperedExtrude(Sketch2D(outer_loop), height=mm(20.0), draft_angle=deg(3.0))
    enclosure = Shell(body_solid, thickness=mm(2.5), open_top=True)

    # Revolved boss in (r, z) placed at bottom center
    boss_profile = Loop2D.polygon([(mm(3.0), mm(-10.0)), (mm(7.0), mm(-10.0)), (mm(6.0), mm(-3.0)), (mm(3.0), mm(-3.0))])
    boss = RevolveSketch(Sketch2D(boss_profile), angle_deg=deg(360.0))

    # Circular bolt-hole pattern
    holes = CircularPattern(Cylinder(mm(1.5), mm(8.0)), count=4, radius=mm(14.0))
    holes_shifted = Transform(holes, translation=(mm(0.0), mm(0.0), mm(-10.0)))

    part = Part(
        root=Subtract(Union([enclosure, boss]), holes_shifted),
        name="molded_enclosure",
    )

    rep = export_step(part, tmp_path / "enclosure.step", hausdorff_tol_mm=0.25)
    assert rep.status == Status.PASS, rep.message

    # Verify M2 structural thickness on the open-top drafted shell
    thick = compute_structural_thickness(enclosure, voxel_mm=1.0)
    assert thick["max_structural_thickness_mm"] > 1.5

    # Verify M3 Channel A Gmsh + CalculiX ccx_2.23 on the enclosure
    mesh = mesh_brep_gmsh_channel_a(enclosure, mesh_size_mm=2.0, order=1)
    nodes = mesh["nodes"]
    fixed_nodes = [i for i, pt in enumerate(nodes) if pt[2] <= -9.5]
    load_nodes = [i for i, pt in enumerate(nodes) if pt[2] >= 9.5]
    assert fixed_nodes and load_nodes
    inp_p = tmp_path / "enclosure.inp"
    export_calculix_inp(
        mesh,
        inp_p,
        E=2500.0,
        nu=0.35,
        fixed_nodes=fixed_nodes,
        load_nodes=load_nodes,
        nodal_force=(0.0, 0.0, -0.5),
    )
    fem_rep = run_calculix_or_conformal_fem(
        inp_p,
        mesh,
        E=2500.0,
        nu=0.35,
        fixed_nodes=fixed_nodes,
        load_nodes=load_nodes,
        nodal_force=(0.0, 0.0, -0.5),
        require_ccx=True,
    )
    assert fem_rep.status == Status.PASS, fem_rep.message
    assert fem_rep.metrics["max_disp_mm"] > 0.0


def test_m5_solve_draft_and_thickness_and_patterns(tmp_path: pathlib.Path) -> None:
    """I-M5-2: solve() optimizes draft_angle & thickness against obstacle + Frustum/Offset/LinearPattern STEP."""
    draft_deg = Param("draft_deg", 1.0, 0.5, 10.0, "deg")
    wall_t = Param("wall_t", 2.0, 1.5, 4.0, "mm")

    base = TaperedExtrude(Sketch2D(Loop2D.rounded_rect(mm(30.0), mm(30.0), mm(4.0))), height=mm(16.0), draft_angle=draft_deg)
    tray = Shell(base, thickness=wall_t, open_top=True)
    part = Part(root=tray, name="draft_tray")

    # Obstacle near top outer edge at x = 16.5..22.5, z = 4..12; draft pulls top inward to clear it
    obs = Transform(Box_or := None if False else __import__("reify.ast", fromlist=["Box"]).Box(mm(6.0), mm(12.0), mm(8.0)), translation=(mm(17.5), mm(0.0), mm(6.0)))
    spec = Spec(obstacles=[ClearanceRule(obstacle=obs, min_clearance_mm=1.0)])

    res = solve(part, params_list=[draft_deg], spec=spec, initial_batch_size=16, steps_per_start=40, lr=0.1)
    assert res.status == SolveStatus.FEASIBLE_FOUND
    assert res.verify_report.status == Status.PASS
    assert res.params["draft_deg"] > 4.0

    # Also verify Frustum, Offset, and LinearPattern compile to valid STEP
    combo = Union([
        Frustum(mm(8.0), mm(4.0), mm(10.0)),
        Transform(LinearPattern(Offset(ExtrudeSketch(Sketch2D(Loop2D.circle((mm(0.0), mm(0.0)), mm(3.0))), mm(6.0)), mm(0.5)), count=3, spacing=(mm(10.0), mm(0.0), mm(0.0))), translation=(mm(15.0), mm(0.0), mm(0.0))),
    ])
    rep2 = export_step(combo, tmp_path / "m5_combo.step", hausdorff_tol_mm=0.05)
    assert rep2.status == Status.PASS, rep2.message
