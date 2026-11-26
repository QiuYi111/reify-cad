"""M1 End-to-End Verification Suite (ACT-T-M1): three-branch clearance, dual freeze, and bracket solve()."""

from __future__ import annotations

import math
import pathlib

import torch
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from reify import (
    Box,
    ClearanceRule,
    Cylinder,
    FailReason,
    FreeImplicit,
    FreezeScope,
    FreezeSpec,
    GeometryQuery,
    Param,
    ParamKind,
    Part,
    SolveStatus,
    Spec,
    Status,
    Subtract,
    Transform,
    Union,
    check_clearance,
    check_freeze,
    compile_brep,
    export_step,
    extract_topology_invariants,
    mm,
    solve,
)


def test_m1_three_branch_clearance_traps():
    """I-M1-1: Needle trap, full containment trap, loose lower-bound UNKNOWN trap, and OCCT truth."""
    # 1. Needle trap: 0.2 mm diameter needle piercing a 20x20x4 mm plate with coarse initial_radius=1.0
    plate = Box(mm(20.0), mm(20.0), mm(4.0), node_id="plate")
    needle = Cylinder(mm(0.1), mm(12.0), node_id="needle")
    rep_needle = check_clearance(
        GeometryQuery(plate),
        GeometryQuery(needle),
        min_clearance_mm=0.0,
        max_depth=4,
        initial_radius=1.0,
    )
    assert rep_needle.status == Status.FAIL
    assert rep_needle.reason == FailReason.PENETRATION
    assert len(rep_needle.witness_points) >= 1

    # 2. Full containment trap: 5 mm cube centered inside 50 mm cube
    small_cube = Box(mm(5.0), mm(5.0), mm(5.0), node_id="small")
    big_cube = Box(mm(50.0), mm(50.0), mm(50.0), node_id="big")
    rep_contain = check_clearance(
        GeometryQuery(small_cube),
        GeometryQuery(big_cube),
        min_clearance_mm=0.0,
        max_depth=2,
        initial_radius=1.0,
    )
    assert rep_contain.status == Status.FAIL
    assert rep_contain.reason == FailReason.PENETRATION

    # 3. Loose lower-bound trap: true gap is 0.5 mm > m=0.2 mm, but Lipschitz L=5.0 makes d_lower=0.1 <= m
    # With max_depth=0 (and initial_radius=0.5), MUST return UNKNOWN — NEVER false FAIL!
    box_a = Box(mm(10.0), mm(10.0), mm(10.0), node_id="box_a")
    obs_true = Transform(Box(mm(10.0), mm(10.0), mm(10.0)), translation=(mm(10.5), mm(0.0), mm(0.0)))
    obs_scaled_field = FreeImplicit(
        fn=lambda p: obs_true.eval_sdf(p),
        lipschitz=5.0,
        bounds=((5.5, -5.0, -5.0), (15.5, 5.0, 5.0)),
        node_id="obs_loose_L5",
    )
    rep_loose = check_clearance(
        GeometryQuery(box_a),
        GeometryQuery(obs_scaled_field),
        min_clearance_mm=0.2,
        max_depth=0,
        initial_radius=0.5,
    )
    assert rep_loose.status == Status.UNKNOWN

    # 4. Disjoint clearance violation: gap = 0.10 mm, required = 0.20 mm -> FAIL (CLEARANCE_VIOLATION)
    box_close = Transform(
        Box(mm(10.0), mm(10.0), mm(10.0), node_id="box_close"),
        translation=(mm(10.10), mm(0.0), mm(0.0)),
    )
    rep_viol = check_clearance(
        GeometryQuery(box_a),
        GeometryQuery(box_close),
        min_clearance_mm=0.20,
        max_depth=4,
        initial_radius=1.0,
    )
    assert rep_viol.status == Status.FAIL
    assert rep_viol.reason == FailReason.CLEARANCE_VIOLATION
    assert len(rep_viol.witness_points) == 2
    pa, pb = rep_viol.witness_points
    witness_d = math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))
    assert witness_d <= 0.20 + 1e-5

    # 5. Certified PASS: gap = 1.50 mm, required = 0.50 mm -> PASS, confirmed by OCCT
    box_far = Transform(
        Box(mm(10.0), mm(10.0), mm(10.0), node_id="box_far"),
        translation=(mm(11.50), mm(0.0), mm(0.0)),
    )
    rep_pass = check_clearance(
        GeometryQuery(box_a),
        GeometryQuery(box_far),
        min_clearance_mm=0.50,
        max_depth=4,
        initial_radius=1.0,
    )
    assert rep_pass.status == Status.PASS

    _, shp_a = compile_brep(box_a)
    _, shp_far = compile_brep(box_far)
    common = BRepAlgoAPI_Common(shp_a, shp_far)
    common.Build()
    assert extract_topology_invariants(common.Shape())["volume_mm3"] == 0.0
    ext = BRepExtrema_DistShapeShape(shp_a, shp_far)
    ext.Perform()
    assert ext.Value() >= 0.50 - 1e-6


def test_m1_dual_freeze_scopes():
    """Verify WORLD_SHELL (bidirectional Hausdorff) and DATUM_AXIS (thickness free, axis/dia locked)."""
    p_thick = Param("thick", 6.0, 4.0, 16.0, unit="mm", kind=ParamKind.SMOOTH)
    p_hole_x = Param("hole_x", 0.0, -5.0, 5.0, unit="mm", kind=ParamKind.SMOOTH)

    def _make_part(base_lx: float = 20.0) -> Part:
        base_flange = Transform(
            Box(mm(base_lx), mm(20.0), mm(6.0), node_id="base_flange"),
            translation=(mm(-15.0), mm(0.0), mm(0.0)),
        )
        plate = Box(mm(20.0), mm(20.0), p_thick, node_id="main_plate")
        hole = Transform(
            Cylinder(mm(1.6), mm(30.0), node_id="m3_hole"),
            translation=(p_hole_x, mm(0.0), mm(0.0)),
        )
        return Part(root=Subtract(Union([base_flange, plate]), hole), name="dual_freeze_test")

    ref_part = _make_part(20.0)
    ref_params = {"thick": 6.0, "hole_x": 0.0}

    # DATUM_AXIS allows plate thickness 6 -> 12 mm when hole axis & diameter are unchanged
    rep_datum_ok = check_freeze(
        ref_part,
        ref_part,
        FreezeSpec("m3_hole", FreezeScope.DATUM_AXIS, tol_mm=1e-3),
        params_old=ref_params,
        params_new={"thick": 12.0, "hole_x": 0.0},
    )
    assert rep_datum_ok.status == Status.PASS

    # DATUM_AXIS rejects 0.2 mm hole center drift
    rep_datum_bad = check_freeze(
        ref_part,
        ref_part,
        FreezeSpec("m3_hole", FreezeScope.DATUM_AXIS, tol_mm=1e-3),
        params_old=ref_params,
        params_new={"thick": 6.0, "hole_x": 0.2},
    )
    assert rep_datum_bad.status == Status.FAIL
    assert rep_datum_bad.reason == FailReason.FREEZE_VIOLATION

    # WORLD_SHELL accepts unchanged base_flange, rejects 0.05 mm expansion
    rep_ws_ok = check_freeze(
        ref_part,
        ref_part,
        FreezeSpec("base_flange", FreezeScope.WORLD_SHELL, tol_mm=1e-3),
        params_old=ref_params,
        params_new={"thick": 12.0, "hole_x": 0.0},
    )
    assert rep_ws_ok.status == Status.PASS

    exp_part = _make_part(20.10)
    rep_ws_bad = check_freeze(
        ref_part,
        exp_part,
        FreezeSpec("base_flange", FreezeScope.WORLD_SHELL, tol_mm=1e-3),
        params_old=ref_params,
        params_new=ref_params,
    )
    assert rep_ws_bad.status == Status.FAIL
    assert rep_ws_bad.reason == FailReason.FREEZE_VIOLATION


def test_m1_bracket_solve_hard_verify_and_step_e2e(tmp_path: pathlib.Path):
    """I-M1-2 E2E: Mounting bracket with 3 PCB obstacles, dual freezes, solve(), OCCT & STEP check."""
    # Coupled continuous parameters starting in an infeasible colliding state
    arm_y = Param("arm_y", 0.0, -25.0, 25.0, unit="mm", kind=ParamKind.SMOOTH)
    arm_len = Param("arm_len", 22.0, 12.0, 35.0, unit="mm", kind=ParamKind.SMOOTH)
    arm_w = Param("arm_w", 10.0, 6.0, 18.0, unit="mm", kind=ParamKind.SMOOTH)
    plate_thick = Param("plate_thick", 6.0, 4.0, 12.0, unit="mm", kind=ParamKind.SMOOTH)

    # Frozen mounting base flange (world_shell) at x = -20
    base_flange = Transform(
        Box(mm(12.0), mm(24.0), mm(6.0), node_id="mount_flange"),
        translation=(mm(-20.0), mm(0.0), mm(0.0)),
    )
    # Movable arm body attached near origin, thickness controlled by plate_thick
    arm_body = Transform(
        Box(arm_len, arm_w, plate_thick, node_id="arm_body"),
        translation=(mm(0.0), arm_y, mm(0.0)),
    )
    # M3 mounting hole on base_flange (datum_axis frozen)
    m3_hole = Transform(
        Cylinder(mm(1.6), mm(20.0), node_id="m3_mount_hole"),
        translation=(mm(-20.0), mm(0.0), mm(0.0)),
    )
    bracket = Part(
        root=Subtract(Union([base_flange, arm_body]), m3_hole),
        name="pcb_bracket",
    )

    # 3 PCB capacitor keep-out zones that collide with initial arm_y=0.0, arm_len=22.0
    obs1 = Transform(Cylinder(mm(4.0), mm(14.0), node_id="pcb_cap_1"), translation=(mm(0.0), mm(0.0), mm(0.0)))
    obs2 = Transform(Cylinder(mm(4.0), mm(14.0), node_id="pcb_cap_2"), translation=(mm(8.0), mm(-8.0), mm(0.0)))
    obs3 = Transform(Box(mm(8.0), mm(8.0), mm(14.0), node_id="pcb_ic_3"), translation=(mm(-5.0), mm(-12.0), mm(0.0)))

    spec = Spec(
        obstacles=(
            ClearanceRule(obs1, min_clearance_mm=1.0, name="cap1"),
            ClearanceRule(obs2, min_clearance_mm=1.0, name="cap2"),
            ClearanceRule(obs3, min_clearance_mm=1.0, name="ic3"),
        ),
        freezes=(
            FreezeSpec("mount_flange", FreezeScope.WORLD_SHELL, tol_mm=1e-3),
            FreezeSpec("m3_mount_hole", FreezeScope.DATUM_AXIS, tol_mm=1e-3),
        ),
    )

    def _soft_vol(p: dict[str, torch.Tensor]) -> torch.Tensor:
        return 0.01 * (p["arm_len"] * p["arm_w"] * p["plate_thick"])

    res = solve(
        bracket,
        [arm_y, arm_len, arm_w, plate_thick],
        spec,
        reference_part=bracket,
        soft_objective=_soft_vol,
        initial_batch_size=32,
        steps_per_start=70,
        lr=0.08,
        proxy_k_mm=1.0,
        proxy_kind="quad_smin",
        verify_max_depth=4,
        seed=7,
    )

    assert res.status == SolveStatus.FEASIBLE_FOUND
    assert res.verify_report.status == Status.PASS
    assert res.n_feasible_starts >= 1

    # Independent OCCT BRep check against all 3 obstacles
    st_b, shp_bracket = compile_brep(bracket, params=res.params)
    assert st_b == Status.PASS and shp_bracket is not None
    for rule in spec.obstacles:
        _, shp_obs = compile_brep(rule.obstacle)
        common = BRepAlgoAPI_Common(shp_bracket, shp_obs)
        common.Build()
        assert extract_topology_invariants(common.Shape())["volume_mm3"] == 0.0
        ext = BRepExtrema_DistShapeShape(shp_bracket, shp_obs)
        ext.Perform()
        assert ext.Value() >= rule.min_clearance_mm - 1e-3

    # Export verified STEP of solved bracket
    step_rep = export_step(bracket, tmp_path / "solved_bracket.step", params=res.params)
    assert step_rep.status == Status.PASS
    assert step_rep.metrics["bidirectional_hausdorff_mm"] < 1e-3
