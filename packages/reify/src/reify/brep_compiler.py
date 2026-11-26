"""1:1 AST-to-OCCT B-Rep compiler, topology invariant extractor, and STEP round-trip exporter."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

from build123d import (
    Box as BBox,
    Cylinder as BCyl,
    Pos,
    RectangleRounded,
    Rotation,
    Shape as BShape,
    Sphere as BSphere,
    extrude,
)
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFill import BRepFill
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepOffsetAPI import (
    BRepOffsetAPI_DraftAngle,
    BRepOffsetAPI_MakeEvolved,
    BRepOffsetAPI_MakeOffsetShape,
    BRepOffsetAPI_MakePipeShell,
    BRepOffsetAPI_MakeThickSolid,
    BRepOffsetAPI_ThruSections,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone, BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
from OCP.collections import List_TopoDS_Shape
from OCP.Geom import Geom_OffsetSurface
from OCP.GeomAbs import GeomAbs_Arc, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane
from OCP.gp import gp_Ax1, gp_Ax2, gp_Ax3, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from reify.ast import (
    Box,
    Cylinder,
    ExtrudeRoundedRect,
    Intersect,
    Node,
    Part,
    Sphere,
    Subtract,
    Transform,
    Union,
    _val_deg_float,
    _val_mm_float,
)
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
from reify.nurbs import BSplineSurfaceShell, LoftSketches, SweepAlongSpline
from reify.sketch import Sketch2D
from reify.types import FailReason, Status, VerificationReport


def _supports_mat2d_evolved(sketch: Any, *, allow_holes: bool = False) -> bool:
    from reify.sketch import Arc2D, LineSeg2D

    holes = getattr(sketch, "holes", None) or []
    if holes and not allow_holes:
        return False
    if not all(isinstance(seg, (LineSeg2D, Arc2D)) for seg in sketch.outer.segments):
        return False
    return all(all(isinstance(seg, (LineSeg2D, Arc2D)) for seg in h.segments) for h in holes)


def _fix_shape_if_needed(shp: TopoDS_Shape) -> TopoDS_Shape:
    if BRepCheck_Analyzer(shp).IsValid():
        return shp
    fixer = ShapeFix_Shape(shp)
    fixer.Perform()
    fixed = fixer.Shape()
    return fixed if not fixed.IsNull() else shp


def _compile_evolved_tapered(
    sketch: Any,
    h: float,
    ang_deg: float,
    z_bot: float,
    extra_inset: float = 0.0,
) -> TopoDS_Shape:
    """Compile drafted solid using OCCT MAT2d BRepOffsetAPI_MakeEvolved (handles vanishing fillets & holes cleanly)."""
    from reify.sketch import Sketch2D

    ang_rad = math.radians(ang_deg)
    inset = h * math.tan(ang_rad)
    outer_face = Sketch2D(sketch.outer).to_occt_face(z=0.0)
    p0 = gp_Pnt(0.0, extra_inset, 0.0)
    p1 = gp_Pnt(0.0, extra_inset + inset, h)
    prof = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(p0, p1).Edge()).Wire()
    ev = BRepOffsetAPI_MakeEvolved(outer_face, prof, GeomAbs_Arc, True, True, True, 1e-4)
    ev.Build()
    if not ev.IsDone() or ev.Shape().IsNull():
        raise RuntimeError("OCCT BRepOffsetAPI_MakeEvolved failed for TaperedExtrude")
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(0.0, 0.0, z_bot))
    shp = BRepBuilderAPI_Transform(ev.Shape(), tr, True).Shape()

    for hole in getattr(sketch, "holes", None) or []:
        # Inner hole shrinks outward with draft (negative inset along +Y in profile)
        h_face = Sketch2D(hole).to_occt_face(z=0.0)
        hp0 = gp_Pnt(0.0, -extra_inset, 0.0)
        hp1 = gp_Pnt(0.0, -extra_inset - inset, h)
        h_prof = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(hp0, hp1).Edge()).Wire()
        h_ev = BRepOffsetAPI_MakeEvolved(h_face, h_prof, GeomAbs_Arc, True, True, True, 1e-4)
        h_ev.Build()
        if not h_ev.IsDone() or h_ev.Shape().IsNull():
            raise RuntimeError("OCCT BRepOffsetAPI_MakeEvolved failed for TaperedExtrude hole")
        h_shp = BRepBuilderAPI_Transform(h_ev.Shape(), tr, True).Shape()
        cut = BRepAlgoAPI_Cut(shp, h_shp)
        cut.Build()
        if not cut.IsDone() or cut.Shape().IsNull():
            raise RuntimeError("OCCT TaperedExtrude hole cut failed")
        shp = cut.Shape()

    return _fix_shape_if_needed(shp)


def _compile_b123d(node: Node, params: dict[str, Any] | None = None):
    if isinstance(node, Box):
        lx = _val_mm_float(node.lx, params)
        ly = _val_mm_float(node.ly, params)
        lz = _val_mm_float(node.lz, params)
        return BBox(lx, ly, lz)

    if isinstance(node, Cylinder):
        r = _val_mm_float(node.radius, params)
        h = _val_mm_float(node.height, params)
        return BCyl(r, h)

    if isinstance(node, Sphere):
        r = _val_mm_float(node.radius, params)
        return BSphere(r)

    if isinstance(node, ExtrudeRoundedRect):
        lx = _val_mm_float(node.lx, params)
        ly = _val_mm_float(node.ly, params)
        h = _val_mm_float(node.height, params)
        r = _val_mm_float(node.fillet_r, params)
        node.check_degeneracy(params)
        if r <= 1e-9:
            return BBox(lx, ly, h)
        half_min = 0.5 * min(lx, ly)
        if half_min - r < 1e-6:
            r = half_min - 1e-6
        rect = RectangleRounded(lx, ly, r)
        return Pos(0.0, 0.0, -0.5 * h) * extrude(rect, amount=h)

    if isinstance(node, ExtrudeSketch):
        h = _val_mm_float(node.height, params)
        face = node.sketch.to_occt_face(params=params, z=-0.5 * h)
        prism = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, h))
        prism.Build()
        if not prism.IsDone() or prism.Shape().IsNull():
            raise RuntimeError("OCCT BRepPrimAPI_MakePrism failed for ExtrudeSketch")
        return BShape(prism.Shape())

    if isinstance(node, TaperedExtrude):
        node.check_degeneracy(params)
        h = _val_mm_float(node.height, params)
        ang_deg = _val_deg_float(node.draft_angle, params)
        ang_rad = math.radians(ang_deg)
        if abs(ang_rad) <= 1e-9:
            face = node.sketch.to_occt_face(params=params, z=-0.5 * h)
            prism_op = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, h))
            prism_op.Build()
            if not prism_op.IsDone() or prism_op.Shape().IsNull():
                raise RuntimeError("OCCT prism failed for TaperedExtrude")
            return BShape(prism_op.Shape())
        if _supports_mat2d_evolved(node.sketch, allow_holes=True):
            return BShape(_compile_evolved_tapered(node.sketch, h, ang_deg, -0.5 * h, 0.0))
        face = node.sketch.to_occt_face(params=params, z=-0.5 * h)
        prism_op = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, h))
        prism_op.Build()
        if not prism_op.IsDone() or prism_op.Shape().IsNull():
            raise RuntimeError("OCCT prism failed for TaperedExtrude")
        prism = prism_op.Shape()
        draft = BRepOffsetAPI_DraftAngle(prism)
        pull_dir = gp_Dir(0.0, 0.0, 1.0)
        neutral = gp_Pln(gp_Pnt(0.0, 0.0, -0.5 * h), pull_dir)
        exp = TopExp_Explorer(prism, TopAbs_FACE)
        while exp.More():
            f = TopoDS.Face(exp.Current())
            surf = BRepAdaptor_Surface(f)
            u0, u1, v0, v1 = surf.FirstUParameter(), surf.LastUParameter(), surf.FirstVParameter(), surf.LastVParameter()
            props = BRepLProp_SLProps(surf, 0.5 * (u0 + u1), 0.5 * (v0 + v1), 1, 1e-6)
            if props.IsNormalDefined():
                nz = abs(props.Normal().Z())
                if nz < 1e-4 and surf.GetType() in (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone):
                    draft.Add(f, pull_dir, ang_rad, neutral)
            exp.Next()
        draft.Build()
        if not draft.IsDone() or draft.Shape().IsNull():
            raise RuntimeError("OCCT BRepOffsetAPI_DraftAngle failed")
        return BShape(draft.Shape())

    if isinstance(node, Frustum):
        r1 = _val_mm_float(node.r_bottom, params)
        r2 = _val_mm_float(node.r_top, params)
        h = _val_mm_float(node.height, params)
        cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(0.0, 0.0, -0.5 * h), gp_Dir(0.0, 0.0, 1.0)), r1, r2, h)
        cone.Build()
        if not cone.IsDone() or cone.Shape().IsNull():
            raise RuntimeError("OCCT BRepPrimAPI_MakeCone failed")
        return BShape(cone.Shape())

    if isinstance(node, RevolveSketch):
        ang_deg = _val_deg_float(node.angle_deg, params)
        face_xy = node.sketch.to_occt_face(params=params, z=0.0)
        tr = gp_Trsf()
        tr.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), 0.5 * math.pi)
        face_xz = BRepBuilderAPI_Transform(face_xy, tr, True).Shape()
        ax_z = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
        if ang_deg >= 360.0 - 1e-6:
            rev = BRepPrimAPI_MakeRevol(face_xz, ax_z, True)
        else:
            rev = BRepPrimAPI_MakeRevol(face_xz, ax_z, math.radians(ang_deg), True)
        rev.Build()
        if not rev.IsDone() or rev.Shape().IsNull():
            raise RuntimeError("OCCT BRepPrimAPI_MakeRevol failed")
        return BShape(rev.Shape())

    if isinstance(node, Offset):
        r = _val_mm_float(node.radius, params)
        child_shp = _compile_b123d(node.child, params).wrapped
        if abs(r) <= 1e-9:
            return BShape(child_shp)
        op = BRepOffsetAPI_MakeOffsetShape()
        op.PerformByJoin(child_shp, r, 1e-4)
        op.Build()
        if not op.IsDone() or op.Shape().IsNull():
            raise RuntimeError("OCCT BRepOffsetAPI_MakeOffsetShape failed")
        return BShape(op.Shape())

    if isinstance(node, Shell):
        node.check_degeneracy(params)
        t = _val_mm_float(node.thickness, params)
        if node.open_top and isinstance(node.child, TaperedExtrude) and _supports_mat2d_evolved(node.child.sketch):
            h_c = _val_mm_float(node.child.height, params)
            ang_deg = _val_deg_float(node.child.draft_angle, params)
            ang_rad = math.radians(ang_deg)
            if abs(ang_rad) > 1e-9:
                outer = _compile_evolved_tapered(node.child.sketch, h_c, ang_deg, -0.5 * h_c, 0.0)
                inner_base_inset = t * math.tan(ang_rad) + t / math.cos(ang_rad)
                inner = _compile_evolved_tapered(
                    node.child.sketch,
                    h_c - t + 1.0,
                    ang_deg,
                    -0.5 * h_c + t,
                    inner_base_inset,
                )
                cut = BRepAlgoAPI_Cut(outer, inner)
                cut.Build()
                if not cut.IsDone() or cut.Shape().IsNull():
                    raise RuntimeError("OCCT MakeEvolved open-top Shell cut failed")
                return BShape(_fix_shape_if_needed(cut.Shape()))
        child_shp = _compile_b123d(node.child, params).wrapped
        if not node.open_top:
            op = BRepOffsetAPI_MakeOffsetShape()
            op.PerformByJoin(child_shp, -t, 1e-4)
            op.Build()
            if not op.IsDone() or op.Shape().IsNull():
                raise RuntimeError("OCCT closed Shell inner offset failed")
            cut = BRepAlgoAPI_Cut(child_shp, op.Shape())
            cut.Build()
            if not cut.IsDone() or cut.Shape().IsNull():
                raise RuntimeError("OCCT closed Shell cut failed")
            return BShape(cut.Shape())
        # Open-top shell: collect top +Z faces at max Z
        _, bmax = node.child.local_bounds(params)
        z_top = bmax[2]
        closing = List_TopoDS_Shape()
        exp = TopExp_Explorer(child_shp, TopAbs_FACE)
        while exp.More():
            f = TopoDS.Face(exp.Current())
            surf = BRepAdaptor_Surface(f)
            u0, u1, v0, v1 = surf.FirstUParameter(), surf.LastUParameter(), surf.FirstVParameter(), surf.LastVParameter()
            props = BRepLProp_SLProps(surf, 0.5 * (u0 + u1), 0.5 * (v0 + v1), 1, 1e-6)
            if props.IsNormalDefined() and props.Normal().Z() > 0.99 and abs(props.Value().Z() - z_top) < 1e-3:
                closing.Append(f)
            exp.Next()
        if closing.Size() == 0:
            raise RuntimeError("Shell(open_top=True) found no +Z top face to remove")
        op = BRepOffsetAPI_MakeThickSolid()
        op.MakeThickSolidByJoin(child_shp, closing, -t, 1e-4)
        op.Build()
        if not op.IsDone() or op.Shape().IsNull():
            raise RuntimeError("OCCT BRepOffsetAPI_MakeThickSolid failed")
        return BShape(op.Shape())

    if isinstance(node, LinearPattern):
        child_shp = _compile_b123d(node.child, params).wrapped
        sx = _val_mm_float(node.spacing[0], params)
        sy = _val_mm_float(node.spacing[1], params)
        sz = _val_mm_float(node.spacing[2], params)
        acc = child_shp
        for k in range(1, node.count):
            tr = gp_Trsf()
            tr.SetTranslation(gp_Vec(k * sx, k * sy, k * sz))
            inst = BRepBuilderAPI_Transform(child_shp, tr, True).Shape()
            op = BRepAlgoAPI_Fuse(acc, inst)
            op.Build()
            if not op.IsDone() or op.Shape().IsNull():
                raise RuntimeError("OCCT LinearPattern fuse failed")
            acc = op.Shape()
        return BShape(acc)

    if isinstance(node, CircularPattern):
        child_shp = _compile_b123d(node.child, params).wrapped
        rad = _val_mm_float(node.radius, params)
        a0 = math.radians(_val_deg_float(node.start_angle_deg, params))
        ax_z = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
        acc: TopoDS_Shape | None = None
        for k in range(node.count):
            ang = a0 + (2.0 * math.pi * k / node.count)
            tr_shift = gp_Trsf()
            tr_shift.SetTranslation(gp_Vec(rad, 0.0, 0.0))
            tr_rot = gp_Trsf()
            tr_rot.SetRotation(ax_z, ang)
            tr_tot = tr_rot * tr_shift
            inst = BRepBuilderAPI_Transform(child_shp, tr_tot, True).Shape()
            if acc is None:
                acc = inst
            else:
                op = BRepAlgoAPI_Fuse(acc, inst)
                op.Build()
                if not op.IsDone() or op.Shape().IsNull():
                    raise RuntimeError("OCCT CircularPattern fuse failed")
                acc = op.Shape()
        assert acc is not None
        return BShape(acc)

    if isinstance(node, SweepAlongSpline):
        spine = node.path.to_occt_wire(params=params)
        X, T, N_rmf, _, _ = node.path.compute_rmf_stations(n_per_span=24, params=params)
        p0 = gp_Pnt(float(X[0, 0]), float(X[0, 1]), float(X[0, 2]))
        d_t = gp_Dir(float(T[0, 0]), float(T[0, 1]), float(T[0, 2]))
        d_n = gp_Dir(float(N_rmf[0, 0]), float(N_rmf[0, 1]), float(N_rmf[0, 2]))
        if isinstance(node.profile, Sketch2D):
            prof_wire_xy = node.profile.outer.to_occt_wire(params=params, z=0.0)
            tr = gp_Trsf()
            tr.SetDisplacement(
                gp_Ax3(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0), gp_Dir(1.0, 0.0, 0.0)),
                gp_Ax3(p0, d_t, d_n),
            )
            prof_wire = TopoDS.Wire(BRepBuilderAPI_Transform(prof_wire_xy, tr, True).Shape())
        else:
            from OCP.gp import gp_Circ

            r = _val_mm_float(node.profile, params)
            circ = gp_Circ(gp_Ax2(p0, d_t, d_n), r)
            prof_wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circ).Edge()).Wire()
        pipe = BRepOffsetAPI_MakePipeShell(spine)
        pipe.SetMode(False)
        pipe.Add(prof_wire)
        pipe.Build()
        if not pipe.IsDone() or not pipe.MakeSolid():
            raise RuntimeError("OCCT BRepOffsetAPI_MakePipeShell failed")
        return BShape(pipe.Shape())

    if isinstance(node, BSplineSurfaceShell):
        t_val = _val_mm_float(node.thickness, params)
        surf = node.surface.to_occt_surface(params=params)
        s_pos = Geom_OffsetSurface(surf, 0.5 * t_val)
        s_neg = Geom_OffsetSurface(surf, -0.5 * t_val)
        f_pos = BRepBuilderAPI_MakeFace(s_pos, 1e-6).Face()
        f_neg = BRepBuilderAPI_MakeFace(s_neg, 1e-6).Face()
        u0, u1, v0, v1 = surf.Bounds()
        side_faces = [
            BRepFill.Face_s(
                BRepBuilderAPI_MakeEdge(s_pos.UIso(u0)).Edge(),
                BRepBuilderAPI_MakeEdge(s_neg.UIso(u0)).Edge(),
            ),
            BRepFill.Face_s(
                BRepBuilderAPI_MakeEdge(s_pos.UIso(u1)).Edge(),
                BRepBuilderAPI_MakeEdge(s_neg.UIso(u1)).Edge(),
            ),
            BRepFill.Face_s(
                BRepBuilderAPI_MakeEdge(s_pos.VIso(v0)).Edge(),
                BRepBuilderAPI_MakeEdge(s_neg.VIso(v0)).Edge(),
            ),
            BRepFill.Face_s(
                BRepBuilderAPI_MakeEdge(s_pos.VIso(v1)).Edge(),
                BRepBuilderAPI_MakeEdge(s_neg.VIso(v1)).Edge(),
            ),
        ]
        sew = BRepBuilderAPI_Sewing(1e-4)
        sew.Add(f_pos)
        sew.Add(f_neg)
        for sf in side_faces:
            sew.Add(sf)
        sew.Perform()
        sewed = sew.SewedShape()
        exp = TopExp_Explorer(sewed, TopAbs_SHELL)
        if not exp.More():
            raise RuntimeError("BSplineSurfaceShell sewing produced no shell")
        solid_op = BRepBuilderAPI_MakeSolid(TopoDS.Shell(exp.Current()))
        solid = solid_op.Solid()
        BRepLib.OrientClosedSolid_s(solid)
        return BShape(solid)

    if isinstance(node, LoftSketches):
        thru = BRepOffsetAPI_ThruSections(True, True, 1e-6)
        for sec, zq in zip(node.sections, node.z_levels):
            zv = _val_mm_float(zq, params)
            thru.AddWire(sec.outer.to_occt_wire(params=params, z=zv))
        thru.Build()
        if not thru.IsDone() or thru.Shape().IsNull():
            raise RuntimeError("OCCT BRepOffsetAPI_ThruSections failed for LoftSketches")
        return BShape(thru.Shape())

    if isinstance(node, Transform):
        child = _compile_b123d(node.child, params)
        tx = _val_mm_float(node.translation[0], params)
        ty = _val_mm_float(node.translation[1], params)
        tz = _val_mm_float(node.translation[2], params)
        rx = _val_deg_float(node.rotation[0], params)
        ry = _val_deg_float(node.rotation[1], params)
        rz = _val_deg_float(node.rotation[2], params)
        return Pos(tx, ty, tz) * Rotation(rx, ry, rz) * child

    if isinstance(node, Union):
        acc = _compile_b123d(node.children[0], params).wrapped
        for ch in node.children[1:]:
            rhs = _compile_b123d(ch, params).wrapped
            op = BRepAlgoAPI_Fuse(acc, rhs)
            op.Build()
            if not op.IsDone() or op.Shape().IsNull():
                raise RuntimeError("OCCT BRepAlgoAPI_Fuse failed")
            acc = op.Shape()
        return BShape(acc)

    if isinstance(node, Intersect):
        acc = _compile_b123d(node.children[0], params).wrapped
        for ch in node.children[1:]:
            rhs = _compile_b123d(ch, params).wrapped
            op = BRepAlgoAPI_Common(acc, rhs)
            op.Build()
            if not op.IsDone() or op.Shape().IsNull():
                raise RuntimeError("OCCT BRepAlgoAPI_Common failed")
            acc = op.Shape()
        return BShape(acc)

    if isinstance(node, Subtract):
        base = _compile_b123d(node.base, params).wrapped
        tool = _compile_b123d(node.tool, params).wrapped
        op = BRepAlgoAPI_Cut(base, tool)
        op.Build()
        if not op.IsDone() or op.Shape().IsNull():
            raise RuntimeError("OCCT BRepAlgoAPI_Cut failed")
        return BShape(op.Shape())

    raise TypeError(f"Node {type(node).__name__} is not B-Rep exportable")


_LAST_COMPILE_ERROR: str | None = None


def compile_brep_error() -> str | None:
    """Return the last exception/validation message from compile_brep()."""
    return _LAST_COMPILE_ERROR


def compile_brep(
    node: Node | Part, params: dict[str, Any] | None = None
) -> tuple[Status, TopoDS_Shape | None]:
    """Compile a standard AST to a BRepCheck-validated OCCT TopoDS_Shape, or return OUT_OF_DOMAIN / FAIL."""
    global _LAST_COMPILE_ERROR
    _LAST_COMPILE_ERROR = None
    root = node.root if isinstance(node, Part) else node
    if not root.is_brep_exportable():
        _LAST_COMPILE_ERROR = "AST contains non-B-Rep-exportable implicit nodes."
        return Status.OUT_OF_DOMAIN, None
    try:
        shape_obj = _compile_b123d(root, params)
        shp = _fix_shape_if_needed(shape_obj.wrapped)
        if not BRepCheck_Analyzer(shp).IsValid():
            _LAST_COMPILE_ERROR = "Compiled OCCT B-Rep failed BRepCheck_Analyzer topology validation."
            return Status.FAIL, None
        return Status.PASS, shp
    except Exception as exc:
        _LAST_COMPILE_ERROR = f"{type(exc).__name__}: {exc}"
        return Status.FAIL, None


def extract_topology_invariants(shape: TopoDS_Shape) -> dict[str, Any]:
    """Extract BRepCheck validity, volume, bounding box, counts, and Euler characteristic."""
    is_valid = bool(BRepCheck_Analyzer(shape).IsValid())
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = float(props.Mass())

    bshp = BShape(shape)
    bb = bshp.bounding_box()
    bbox = ((float(bb.min.X), float(bb.min.Y), float(bb.min.Z)), (float(bb.max.X), float(bb.max.Y), float(bb.max.Z)))
    n_solids = len(bshp.solids())
    n_shells = len(bshp.shells())
    n_faces = len(bshp.faces())
    n_edges = len(bshp.edges())
    n_vertices = len(bshp.vertices())
    euler_chi = n_vertices - n_edges + n_faces

    return {
        "is_valid": is_valid,
        "volume_mm3": volume,
        "bbox": bbox,
        "solids": n_solids,
        "shells": n_shells,
        "faces": n_faces,
        "edges": n_edges,
        "vertices": n_vertices,
        "euler_chi": euler_chi,
    }


def export_step(
    part_or_node: Node | Part | dict[str, Any],
    step_path: str | pathlib.Path,
    params: dict[str, Any] | None = None,
    craft_json_path: str | pathlib.Path | None = None,
    hausdorff_tol_mm: float = 1e-3,
    volume_rel_tol: float = 1e-4,
) -> VerificationReport:
    """Export AST (or multi-part assembly dict) to STEP, re-import for round-trip verification, and optionally write craft JSON."""
    from reify.finalize import BRepFinalized
    from reify.query import GeometryQuery

    if isinstance(part_or_node, dict):
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound

        if not part_or_node:
            return VerificationReport(status=Status.OUT_OF_DOMAIN, message="Empty assembly dict for STEP export.")
        builder = BRep_Builder()
        comp = TopoDS_Compound()
        builder.MakeCompound(comp)
        for name, item in part_or_node.items():
            if isinstance(item, BRepFinalized):
                shp = item.shape
            else:
                r_node = item.root if isinstance(item, Part) else item
                if not r_node.is_brep_exportable():
                    return VerificationReport(
                        status=Status.OUT_OF_DOMAIN,
                        message=f"Part {name!r} contains non-B-Rep-exportable implicit nodes; exact STEP export refused.",
                    )
                st, shp = compile_brep(r_node, params=params)
                if st != Status.PASS or shp is None:
                    return VerificationReport(
                        status=Status.FAIL,
                        reason=FailReason.TOPOLOGY_INVALID,
                        message=f"B-Rep compilation failed for part {name!r}.",
                    )
            p_inv = extract_topology_invariants(shp)
            if not p_inv["is_valid"] or p_inv["volume_mm3"] <= 1e-9:
                return VerificationReport(
                    status=Status.FAIL,
                    reason=FailReason.TOPOLOGY_INVALID,
                    metrics=p_inv,
                    message=f"Compiled part {name!r} failed BRepCheck_Analyzer or has zero volume.",
                )
            builder.Add(comp, shp)

        orig_inv = extract_topology_invariants(comp)
        if not orig_inv["is_valid"] or orig_inv["volume_mm3"] <= 1e-9:
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.TOPOLOGY_INVALID,
                metrics=orig_inv,
                message="Assembled compound failed BRepCheck_Analyzer.",
            )
        step_file = pathlib.Path(step_path)
        step_file.parent.mkdir(parents=True, exist_ok=True)
        writer = STEPControl_Writer()
        writer.Transfer(comp, STEPControl_AsIs)
        if writer.Write(str(step_file)) != IFSelect_RetDone:
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.TOPOLOGY_INVALID,
                message=f"Failed to write multi-part STEP file: {step_file}",
            )
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_file)) != IFSelect_RetDone:
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.TOPOLOGY_INVALID,
                message=f"Failed to re-import multi-part STEP file: {step_file}",
            )
        reader.TransferRoots()
        re_inv = extract_topology_invariants(reader.OneShape())
        vol_rel_err = abs(re_inv["volume_mm3"] - orig_inv["volume_mm3"]) / max(orig_inv["volume_mm3"], 1e-12)
        if not re_inv["is_valid"] or re_inv["solids"] != orig_inv["solids"] or vol_rel_err > volume_rel_tol:
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.TOPOLOGY_INVALID,
                metrics={"orig": orig_inv, "reimported": re_inv, "volume_rel_err": vol_rel_err},
                message="Multi-part STEP round-trip topology or volume consistency check failed.",
            )
        return VerificationReport(
            status=Status.PASS,
            metrics={"topology": re_inv, "volume_rel_err": vol_rel_err},
            message=f"Multi-part STEP export ({re_inv['solids']} solids) and round-trip verified.",
        )

    part = part_or_node if isinstance(part_or_node, Part) else Part(root=part_or_node)
    root = part.root
    if not root.is_brep_exportable():
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message="AST contains non-B-Rep-exportable implicit nodes; exact STEP export refused.",
        )

    status, shape = compile_brep(root, params=params)
    if status != Status.PASS or shape is None:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            message="B-Rep compilation failed.",
        )

    orig_inv = extract_topology_invariants(shape)
    if not orig_inv["is_valid"] or orig_inv["volume_mm3"] <= 1e-9:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            metrics=orig_inv,
            message="Compiled B-Rep failed BRepCheck_Analyzer or has zero volume.",
        )

    step_file = pathlib.Path(step_path)
    step_file.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(step_file)) != IFSelect_RetDone:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            message=f"Failed to write STEP file: {step_file}",
        )

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_file)) != IFSelect_RetDone:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            message=f"Failed to re-import STEP file: {step_file}",
        )
    reader.TransferRoots()
    reimported = reader.OneShape()
    re_inv = extract_topology_invariants(reimported)

    vol_rel_err = abs(re_inv["volume_mm3"] - orig_inv["volume_mm3"]) / max(
        orig_inv["volume_mm3"], 1e-12
    )
    if (
        not re_inv["is_valid"]
        or re_inv["solids"] != orig_inv["solids"]
        or vol_rel_err > volume_rel_tol
    ):
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            metrics={
                "orig": orig_inv,
                "reimported": re_inv,
                "volume_rel_err": vol_rel_err,
            },
            message="STEP round-trip topology or volume consistency check failed.",
        )

    gq = GeometryQuery(root)
    hd = gq.bidirectional_hausdorff_to_brep(reimported, params=params)
    if hd["bidirectional_hausdorff_mm"] > hausdorff_tol_mm:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            metrics={**hd, "volume_rel_err": vol_rel_err, "topology": re_inv},
            message=(
                f"Dual-backend bidirectional Hausdorff {hd['bidirectional_hausdorff_mm']:.6f} mm "
                f"> {hausdorff_tol_mm} mm"
            ),
        )

    if craft_json_path is not None:
        craft_file = pathlib.Path(craft_json_path)
        craft_file.parent.mkdir(parents=True, exist_ok=True)
        craft_data = {
            "part_name": part.name,
            "step_path": str(step_file),
            "topology": re_inv,
            "hausdorff_mm": hd["bidirectional_hausdorff_mm"],
            "volume_rel_err": vol_rel_err,
            "freezes": [
                {"node_id": f.node_id, "scope": f.scope.value, "tol_mm": f.tol_mm}
                for f in part.freezes
            ],
            "holes": part.holes,
        }
        craft_file.write_text(json.dumps(craft_data, indent=2), encoding="utf-8")

    return VerificationReport(
        status=Status.PASS,
        metrics={
            "topology": re_inv,
            "volume_rel_err": vol_rel_err,
            **hd,
        },
        message="STEP export, round-trip re-import, and dual-backend Hausdorff verified.",
    )
