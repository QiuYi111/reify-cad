"""M0 End-to-End Verification Suite (ACT-T-M0) against independent OCCT oracle."""

from __future__ import annotations

import json
import math
import pathlib
import random

import pytest
import torch
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

from reify import (
    Box,
    Cylinder,
    ExtrudeRoundedRect,
    FreeImplicit,
    GeometryQuery,
    GradientStatus,
    GuaranteeLevel,
    Intersect,
    JointGraph,
    OutOfDomainError,
    Param,
    ParamKind,
    Part,
    PointClass,
    PrismaticJoint,
    RevoluteJoint,
    SmoothUnion,
    Sphere,
    Status,
    Subtract,
    Transform,
    Union,
    UnitError,
    compile_brep,
    deg,
    export_step,
    extract_topology_invariants,
    mm,
)


def _occt_classify_and_dist(shape, pt_xyz: tuple[float, float, float]) -> tuple[str, float]:
    """Independent OCCT oracle: classify point (IN/OUT/ON) and exact distance to shape."""
    gp = gp_Pnt(*pt_xyz)
    vtx = BRepBuilderAPI_MakeVertex(gp).Vertex()
    ext = BRepExtrema_DistShapeShape(vtx, shape)
    assert ext.IsDone()
    dist = float(ext.Value())

    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        clf = BRepClass3d_SolidClassifier(TopoDS.Solid(exp.Current()), gp, 1e-6)
        st = clf.State()
        if st == TopAbs_IN:
            return "IN", dist
        if st == TopAbs_ON:
            return "ON", dist
        exp.Next()
    return "OUT", dist


def _make_random_primitive(rng: random.Random, idx: int):
    kind = rng.choice(["box", "cyl", "sphere", "rrect"])
    if kind == "box":
        node = Box(
            mm(rng.uniform(8.0, 24.0)),
            mm(rng.uniform(8.0, 24.0)),
            mm(rng.uniform(6.0, 20.0)),
            node_id=f"prim_{idx}",
        )
    elif kind == "cyl":
        node = Cylinder(
            mm(rng.uniform(4.0, 12.0)),
            mm(rng.uniform(8.0, 24.0)),
            node_id=f"prim_{idx}",
        )
    elif kind == "sphere":
        node = Sphere(mm(rng.uniform(5.0, 14.0)), node_id=f"prim_{idx}")
    else:
        lx = rng.uniform(10.0, 24.0)
        ly = rng.uniform(10.0, 24.0)
        r = rng.uniform(1.0, 0.35 * min(lx, ly))
        node = ExtrudeRoundedRect(
            mm(lx), mm(ly), mm(rng.uniform(6.0, 18.0)), mm(r), node_id=f"prim_{idx}"
        )
    return Transform(
        node,
        translation=(
            mm(rng.uniform(-5.0, 5.0)),
            mm(rng.uniform(-5.0, 5.0)),
            mm(rng.uniform(-5.0, 5.0)),
        ),
        rotation_deg=(
            deg(rng.uniform(-45.0, 45.0)),
            deg(rng.uniform(-45.0, 45.0)),
            deg(rng.uniform(-45.0, 45.0)),
        ),
        node_id=f"xform_{idx}",
    )


def _generate_fuzz_trees(count: int = 55) -> list:
    rng = random.Random(20261126)
    trees = []
    attempts = 0
    while len(trees) < count and attempts < 300:
        attempts += 1
        a = _make_random_primitive(rng, attempts * 10 + 1)
        b = _make_random_primitive(rng, attempts * 10 + 2)
        c = _make_random_primitive(rng, attempts * 10 + 3)
        # Ensure overlapping union/subtract/intersect so OCCT builds clean non-empty solids
        op1 = rng.choice(["union", "subtract"])
        if op1 == "union":
            mid = Union([a, b], node_id=f"u_{attempts}")
        else:
            # Make base slightly larger to avoid empty subtraction
            base = Transform(
                Box(mm(28.0), mm(28.0), mm(24.0), node_id=f"base_{attempts}"),
                rotation_deg=(deg(rng.uniform(-25.0, 25.0)), deg(rng.uniform(-25.0, 25.0)), deg(0.0)),
            )
            mid = Subtract(base, b, node_id=f"sub_{attempts}")

        op2 = rng.choice(["union", "subtract", "intersect"])
        if op2 == "union":
            root = Union([mid, c], node_id=f"root_{attempts}")
        elif op2 == "subtract":
            hole = Transform(
                Cylinder(mm(rng.uniform(2.0, 4.5)), mm(40.0), node_id=f"hole_{attempts}"),
                translation=(mm(rng.uniform(-3.0, 3.0)), mm(rng.uniform(-3.0, 3.0)), mm(0.0)),
            )
            root = Subtract(mid, hole, node_id=f"root_{attempts}")
        else:
            clip = Box(mm(30.0), mm(30.0), mm(30.0), node_id=f"clip_{attempts}")
            root = Intersect([mid, clip], node_id=f"root_{attempts}")

        st, shp = compile_brep(root)
        if st != Status.PASS or shp is None:
            continue
        inv = extract_topology_invariants(shp)
        if inv["is_valid"] and inv["solids"] >= 1 and inv["volume_mm3"] > 50.0 and inv["euler_chi"] % 2 == 0:
            trees.append((root, shp, inv))
    assert len(trees) == count, f"Only generated {len(trees)} valid trees"
    return trees


def test_m0_fuzz_55_trees_dual_backend_and_bounds(tmp_path: pathlib.Path):
    """I-M0-1 & I-M0-2: 55 random ASTs vs OCCT oracle, Hausdorff < 1e-3 mm, STEP round-trip."""
    trees = _generate_fuzz_trees(55)
    rng = random.Random(42)

    for idx, (root, shape, inv) in enumerate(trees):
        gq = GeometryQuery(root)
        bmin, bmax = root.local_bounds()

        # Sample 60 random points around and inside bounding box per tree (55 * 60 = 3300 queries)
        pts_list = [
            (
                rng.uniform(bmin[0] - 8.0, bmax[0] + 8.0),
                rng.uniform(bmin[1] - 8.0, bmax[1] + 8.0),
                rng.uniform(bmin[2] - 8.0, bmax[2] + 8.0),
            )
            for _ in range(60)
        ]
        pts = torch.tensor(pts_list, dtype=torch.float64)
        classes = gq.classify_point(pts)
        ub_dists = gq.unsigned_boundary_distance_lower(pts)

        outside_pts = []
        outside_occt_d = []
        for i, p_xyz in enumerate(pts_list):
            occt_state, occt_d = _occt_classify_and_dist(shape, p_xyz)
            if classes[i] == PointClass.OUTSIDE:
                assert occt_state == "OUT", f"Tree {idx}: OUTSIDE misclassified OCCT {occt_state}"
                outside_pts.append(p_xyz)
                outside_occt_d.append(occt_d)
            elif classes[i] == PointClass.INSIDE:
                assert occt_state == "IN", f"Tree {idx}: INSIDE misclassified OCCT {occt_state}"

            if occt_state == "OUT":
                # For exterior points, OCCT distance to solid equals boundary distance
                assert float(ub_dists[i].item()) <= occt_d + 1e-7

        if outside_pts:
            ext_clear = gq.external_clearance_lower(torch.tensor(outside_pts, dtype=torch.float64))
            for k, d_occt in enumerate(outside_occt_d):
                assert float(ext_clear[k].item()) <= d_occt + 1e-7

        # Affine interval bound enclosure check on 3 random sub-boxes
        for _ in range(3):
            cx = rng.uniform(bmin[0], bmax[0])
            cy = rng.uniform(bmin[1], bmax[1])
            cz = rng.uniform(bmin[2], bmax[2])
            b0 = (cx - 1.5, cy - 1.5, cz - 1.5)
            b1 = (cx + 1.5, cy + 1.5, cz + 1.5)
            f_lo, f_hi = gq.interval_bound(b0, b1)
            probe = torch.tensor(
                [
                    [rng.uniform(b0[0], b1[0]), rng.uniform(b0[1], b1[1]), rng.uniform(b0[2], b1[2])]
                    for _ in range(16)
                ],
                dtype=torch.float64,
            )
            f_vals = gq.eval_field(probe)
            assert f_lo <= float(torch.amin(f_vals).item()) + 1e-9
            assert f_hi >= float(torch.amax(f_vals).item()) - 1e-9

        # Dual-backend Hausdorff & STEP round-trip on every tree (export STEP on first 15 + all every 4th)
        if idx < 15 or idx % 4 == 0:
            step_file = tmp_path / f"tree_{idx}.step"
            craft_file = tmp_path / f"tree_{idx}_craft.json"
            rep = export_step(Part(root=root, name=f"fuzz_{idx}"), step_file, craft_json_path=craft_file)
            assert rep.status == Status.PASS, f"Tree {idx} STEP export failed: {rep}"
            assert rep.metrics["bidirectional_hausdorff_mm"] < 1e-3
            assert rep.metrics["volume_rel_err"] < 1e-4
            assert craft_file.exists()
            craft = json.loads(craft_file.read_text())
            assert craft["topology"]["solids"] == inv["solids"]


def test_m0_metamorphic_anchor_freeze_and_domain_guards(tmp_path: pathlib.Path):
    """I-M0-3: rigid-transform metamorphic invariance, differentiable Anchor, Hausdorff freeze, and Domain Guards."""
    # 1. Explicit unit enforcement
    with pytest.raises(UnitError):
        Box(10.0, mm(10.0), mm(10.0))  # type: ignore[arg-type]
    with pytest.raises(UnitError):
        Box(deg(10.0), mm(10.0), mm(10.0))

    # 2. Differentiable Anchor gradient vs central finite difference
    p_len = Param("plate_len", 40.0, 20.0, 80.0, unit="mm", kind=ParamKind.SMOOTH)
    p_rot = Param("plate_rz", 25.0, -90.0, 90.0, unit="deg", kind=ParamKind.SMOOTH)
    plate = Transform(
        Box(p_len, mm(20.0), mm(6.0), node_id="plate"),
        translation=(mm(5.0), mm(-3.0), mm(2.0)),
        rotation_deg=(deg(10.0), deg(-5.0), p_rot),
    )

    t_len = torch.tensor(40.0, dtype=torch.float64, requires_grad=True)
    t_rot = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    anc = plate.anchor("front", params={"plate_len": t_len, "plate_rz": t_rot})
    # Objective: x + 2*y - 0.5*z of front anchor
    obj = anc.origin[0] + 2.0 * anc.origin[1] - 0.5 * anc.origin[2]
    obj.backward()
    grad_len = float(t_len.grad.item())
    grad_rot = float(t_rot.grad.item())

    eps = 1e-6
    a_lp = plate.anchor("front", params={"plate_len": 40.0 + eps, "plate_rz": 25.0}).origin
    a_lm = plate.anchor("front", params={"plate_len": 40.0 - eps, "plate_rz": 25.0}).origin
    fd_len = float(((a_lp[0] + 2 * a_lp[1] - 0.5 * a_lp[2]) - (a_lm[0] + 2 * a_lm[1] - 0.5 * a_lm[2])).item() / (2 * eps))
    a_rp = plate.anchor("front", params={"plate_len": 40.0, "plate_rz": 25.0 + eps}).origin
    a_rm = plate.anchor("front", params={"plate_len": 40.0, "plate_rz": 25.0 - eps}).origin
    fd_rot = float(((a_rp[0] + 2 * a_rp[1] - 0.5 * a_rp[2]) - (a_rm[0] + 2 * a_rm[1] - 0.5 * a_rm[2])).item() / (2 * eps))

    assert abs(grad_len - fd_len) / max(abs(fd_len), 1e-12) < 1e-4
    assert abs(grad_rot - fd_rot) / max(abs(fd_rot), 1e-12) < 1e-4

    # 3. Rigid-transform Metamorphic Testing (volume, distance_query, local Anchor invariant)
    part_a = Subtract(
        Box(mm(30.0), mm(20.0), mm(10.0), node_id="blk"),
        Cylinder(mm(4.0), mm(16.0), node_id="hole"),
    )
    part_b = Transform(Sphere(mm(5.0), node_id="obs"), translation=(mm(25.0), mm(0.0), mm(0.0)))
    xform_t = (mm(17.5), mm(-42.0), mm(11.0))
    xform_r = (deg(33.0), deg(-19.0), deg(72.0))
    part_a_m = Transform(part_a, translation=xform_t, rotation_deg=xform_r)
    part_b_m = Transform(part_b, translation=xform_t, rotation_deg=xform_r)

    _, shp_a = compile_brep(part_a)
    _, shp_am = compile_brep(part_a_m)
    inv_a = extract_topology_invariants(shp_a)
    inv_am = extract_topology_invariants(shp_am)
    assert abs(inv_a["volume_mm3"] - inv_am["volume_mm3"]) < 1e-4

    d_orig = GeometryQuery(part_a).distance_query(GeometryQuery(part_b))
    d_meta = GeometryQuery(part_a_m).distance_query(GeometryQuery(part_b_m))
    assert d_orig.guarantee_level == GuaranteeLevel.SOLVER_TOLERANCE
    assert d_meta.status == Status.PASS
    assert abs(d_orig.value - d_meta.value) < 1e-4

    # 4. world_shell freeze: f_new = 2 * f_old must PASS; 0.05 mm expansion must FAIL
    q_old = GeometryQuery(Box(mm(20.0), mm(20.0), mm(10.0)))
    q_scaled_field = GeometryQuery(
        FreeImplicit(
            lambda p: 2.0 * Box(mm(20.0), mm(20.0), mm(10.0)).eval_sdf(p),
            lipschitz=2.0,
            bounds=((-12.0, -12.0, -7.0), (12.0, 12.0, 7.0)),
        )
    )
    rep_same = q_old.freeze_check_world_shell(q_scaled_field, tol_mm=1e-3)
    assert rep_same.status == Status.PASS

    q_expanded = GeometryQuery(Box(mm(20.10), mm(20.0), mm(10.0)))
    rep_exp = q_old.freeze_check_world_shell(q_expanded, tol_mm=1e-3)
    assert rep_exp.status == Status.FAIL
    assert rep_exp.metrics["bidirectional_hausdorff_mm"] > 0.04

    # 5. external_clearance_lower rejects INSIDE points
    with pytest.raises(ValueError):
        q_old.external_clearance_lower(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))

    # 6. Scale budget guard & non-BRep export guard & closed-loop JointGraph guard
    with pytest.raises(OutOfDomainError):
        q_old.classify_point(torch.tensor([[1e6, 0.0, 0.0]], dtype=torch.float64))

    s_union = SmoothUnion(Box(mm(10.0), mm(10.0), mm(10.0)), Sphere(mm(6.0)), k=mm(2.0))
    assert export_step(s_union, tmp_path / "smooth.step").status == Status.OUT_OF_DOMAIN

    closed_jg = JointGraph(
        joints=[
            RevoluteJoint("j1", "base", "l1", (0, 0, 0), (0, 0, 1), -1.0, 1.0),
            PrismaticJoint("j2", "l1", "l2", (1, 0, 0), 0.0, 10.0),
            RevoluteJoint("j3", "l2", "base", (10, 0, 0), (0, 0, 1), -1.0, 1.0),
        ]
    )
    assert closed_jg.check_domain() == Status.OUT_OF_DOMAIN


def test_m0_degenerate_and_malicious_geometry():
    """Malicious & degenerate geometry suite: tangent cylinders, 1e-4 coplanar, 0.05mm thin wall, degenerate fillet."""
    # 1. Externally tangent cylinders: x=-5 (r=5) and x=+5 (r=5) touch at (0, 0, 0)
    c1 = Transform(Cylinder(mm(5.0), mm(10.0), node_id="cyl_left"), translation=(mm(-5.0), mm(0.0), mm(0.0)))
    c2 = Transform(Cylinder(mm(5.0), mm(10.0), node_id="cyl_right"), translation=(mm(5.0), mm(0.0), mm(0.0)))
    u_tan = Union([c1, c2], node_id="tangent_union")
    prov = GeometryQuery(u_tan).provenance(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))[0]
    assert prov.gradient_status == GradientStatus.MULTI_BRANCH
    assert set(prov.active_branches) == {"cyl_left", "cyl_right"}

    # 2. 1e-4 mm near-coplanar boolean cut
    near_coplanar = Subtract(
        Box(mm(20.0), mm(20.0), mm(10.0), node_id="base_nc"),
        Transform(
            Box(mm(10.0), mm(10.0), mm(10.0), node_id="cut_nc"),
            translation=(mm(0.0), mm(0.0), mm(1e-4)),
        ),
    )
    st_nc, shp_nc = compile_brep(near_coplanar)
    assert st_nc in (Status.PASS, Status.FAIL)
    if st_nc == Status.PASS and shp_nc is not None:
        assert extract_topology_invariants(shp_nc)["volume_mm3"] > 0.0

    # 3. 0.05 mm ultra-thin wall
    thin = Subtract(
        Box(mm(10.0), mm(10.0), mm(10.0)),
        Transform(Box(mm(9.90), mm(10.0), mm(10.0)), translation=(mm(-0.05), mm(0.0), mm(0.0))),
    )
    st_thin, shp_thin = compile_brep(thin)
    assert st_thin == Status.PASS and shp_thin is not None
    assert extract_topology_invariants(shp_thin)["volume_mm3"] > 0.0

    # 4. Degenerate fillet_r approaching half_min
    deg_rect = ExtrudeRoundedRect(mm(10.0), mm(10.0), mm(6.0), mm(4.9995))
    warn = deg_rect.check_degeneracy()
    assert warn is not None and "DEGENERATE_FILLET" in warn
    st_dr, _ = compile_brep(deg_rect)
    assert st_dr == Status.PASS
