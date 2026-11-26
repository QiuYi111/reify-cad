"""M7: Type-isolated irreversible terminal B-Rep escape hatch (BRepFinalized, BREP_ESCAPED)."""

from __future__ import annotations

import json
import math
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
from OCP.collections import IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_SOLID
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from reify.ast import Node, Part
from reify.brep_compiler import compile_brep, extract_topology_invariants
from reify.types import (
    CertificationTier,
    FailReason,
    GuaranteeLevel,
    Status,
    VerificationReport,
)
from reify.units import Quantity, require_angle, require_length


@dataclass
class BRepFinalized:
    """Irreversible terminal B-Rep container (DOES NOT inherit from Node; isolated from AST/solve)."""

    shape: TopoDS_Shape
    irreversible_reason: str
    tier: CertificationTier = CertificationTier.BREP_ESCAPED
    source_node: Node | None = None
    applied_ops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LocalFilletEdges:
    """Terminal local edge fillet operation (topological selection)."""

    radius: Quantity
    edge_filter: Callable[[Any], bool] | None = None

    def __post_init__(self) -> None:
        require_length(self.radius, "radius")


@dataclass(frozen=True)
class LocalChamferEdges:
    """Terminal local edge chamfer operation (topological selection)."""

    distance: Quantity
    edge_filter: Callable[[Any], bool] | None = None

    def __post_init__(self) -> None:
        require_length(self.distance, "distance")


@dataclass(frozen=True)
class LocalDraftFaces:
    """Terminal local face draft operation (topological selection)."""

    angle: Quantity
    neutral_z: Quantity
    face_filter: Callable[[Any], bool]

    def __post_init__(self) -> None:
        require_angle(self.angle, "angle")
        require_length(self.neutral_z, "neutral_z")


TerminalOp = LocalFilletEdges | LocalChamferEdges | LocalDraftFaces


def _require_consent(irreversible: bool, reason: str, fn_name: str) -> str:
    if irreversible is not True:
        raise ValueError(
            f"{fn_name}() is irreversible and strips Class 1 analytic/gradient guarantees; "
            "pass irreversible=True explicitly."
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{fn_name}() requires a non-empty reason explaining why Class 1 could not express this.")
    return reason.strip()


def finalize_brep(
    source: Node | Part,
    ops: Sequence[TerminalOp],
    *,
    irreversible: bool,
    reason: str,
    params: dict[str, Any] | None = None,
) -> BRepFinalized:
    """Compile a Class 1 Node/Part to OCCT B-Rep and apply irreversible terminal ops -> BRepFinalized."""
    clean_reason = _require_consent(irreversible, reason, "finalize_brep")
    if isinstance(source, BRepFinalized):
        raise TypeError("source is already a BRepFinalized terminal object")
    root = source.root if isinstance(source, Part) else source
    status, shape = compile_brep(root, params=params)
    if status != Status.PASS or shape is None:
        raise RuntimeError("Failed to compile source Class 1 AST to OCCT B-Rep before terminal ops")

    msg = (
        f"[BREP_ESCAPED / IRREVERSIBLE] Entering terminal B-Rep escape hatch ({clean_reason}); "
        "analytic SDF, Lipschitz interval bounds, and solve() gradients are disabled."
    )
    warnings.warn(msg, UserWarning, stacklevel=2)

    cur = shape
    applied: list[str] = []
    for op in ops:
        if isinstance(op, LocalFilletEdges):
            r = op.radius.as_mm()
            mk = BRepFilletAPI_MakeFillet(cur)
            exp = TopExp_Explorer(cur, TopAbs_EDGE)
            cnt = 0
            while exp.More():
                e = TopoDS.Edge(exp.Current())
                if op.edge_filter is None or op.edge_filter(e):
                    mk.Add(r, e)
                    cnt += 1
                exp.Next()
            if cnt == 0:
                raise ValueError("LocalFilletEdges matched 0 edges")
            mk.Build()
            if not mk.IsDone() or mk.Shape().IsNull():
                raise RuntimeError("OCCT BRepFilletAPI_MakeFillet failed")
            cur = mk.Shape()
            applied.append(f"LocalFilletEdges(r={r}mm, edges={cnt})")

        elif isinstance(op, LocalChamferEdges):
            d = op.distance.as_mm()
            mk_c = BRepFilletAPI_MakeChamfer(cur)
            ef_map = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
            TopExp.MapShapesAndAncestors_s(cur, TopAbs_EDGE, TopAbs_FACE, ef_map)
            cnt = 0
            for idx in range(1, ef_map.Extent() + 1):
                e = TopoDS.Edge(ef_map.FindKey(idx))
                if op.edge_filter is None or op.edge_filter(e):
                    faces = ef_map.FindFromIndex(idx)
                    if faces.Extent() >= 1:
                        f0 = TopoDS.Face(faces.First())
                        mk_c.Add(d, d, e, f0)
                        cnt += 1
            if cnt == 0:
                raise ValueError("LocalChamferEdges matched 0 edges")
            mk_c.Build()
            if not mk_c.IsDone() or mk_c.Shape().IsNull():
                raise RuntimeError("OCCT BRepFilletAPI_MakeChamfer failed")
            cur = mk_c.Shape()
            applied.append(f"LocalChamferEdges(d={d}mm, edges={cnt})")

        elif isinstance(op, LocalDraftFaces):
            ang = math.radians(op.angle.as_deg())
            nz = op.neutral_z.as_mm()
            mk_d = BRepOffsetAPI_DraftAngle(cur)
            pull = gp_Dir(0.0, 0.0, 1.0)
            neutral = gp_Pln(gp_Pnt(0.0, 0.0, nz), pull)
            exp = TopExp_Explorer(cur, TopAbs_FACE)
            cnt = 0
            while exp.More():
                f = TopoDS.Face(exp.Current())
                if op.face_filter(f):
                    mk_d.Add(f, pull, ang, neutral)
                    cnt += 1
                exp.Next()
            if cnt == 0:
                raise ValueError("LocalDraftFaces matched 0 faces")
            mk_d.Build()
            if not mk_d.IsDone() or mk_d.Shape().IsNull():
                raise RuntimeError("OCCT BRepOffsetAPI_DraftAngle failed")
            cur = mk_d.Shape()
            applied.append(f"LocalDraftFaces(angle={op.angle.as_deg()}deg, faces={cnt})")
        else:
            raise TypeError(f"Unsupported terminal op: {type(op).__name__}")

    if not BRepCheck_Analyzer(cur).IsValid():
        raise RuntimeError("Finalized B-Rep failed BRepCheck_Analyzer")

    return BRepFinalized(
        shape=cur,
        irreversible_reason=clean_reason,
        tier=CertificationTier.BREP_ESCAPED,
        source_node=root,
        applied_ops=applied,
        warnings=[msg],
    )


def import_step_terminal(
    step_path: str | pathlib.Path,
    *,
    irreversible: bool,
    reason: str,
) -> BRepFinalized:
    """Import an external vendor STEP file as a type-isolated BREP_ESCAPED terminal object."""
    clean_reason = _require_consent(irreversible, reason, "import_step_terminal")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    shp = reader.OneShape()
    if shp.IsNull() or not BRepCheck_Analyzer(shp).IsValid():
        raise RuntimeError(f"Imported STEP shape invalid: {step_path}")
    msg = f"[BREP_ESCAPED / IRREVERSIBLE] Imported external STEP ({step_path}): {clean_reason}"
    warnings.warn(msg, UserWarning, stacklevel=2)
    return BRepFinalized(
        shape=shp,
        irreversible_reason=clean_reason,
        tier=CertificationTier.BREP_ESCAPED,
        source_node=None,
        applied_ops=[f"import_step_terminal({step_path})"],
        warnings=[msg],
    )


def _to_occt_shape(obj: BRepFinalized | Node | Part, params: dict[str, Any] | None) -> TopoDS_Shape:
    if isinstance(obj, BRepFinalized):
        return obj.shape
    root = obj.root if isinstance(obj, Part) else obj
    st, shp = compile_brep(root, params=params)
    if st != Status.PASS or shp is None:
        raise RuntimeError("Failed to compile Node/Part to OCCT shape for B-Rep clearance check")
    return shp


def verify_brep_clearance(
    a: BRepFinalized | Node | Part,
    b: BRepFinalized | Node | Part,
    min_clearance_mm: float,
    params: dict[str, Any] | None = None,
    deflection: float = 1e-4,
) -> VerificationReport:
    """Post-finish OCCT B-Rep clearance check returning honest four-state SOLVER_TOLERANCE report."""
    sa = _to_occt_shape(a, params)
    sb = _to_occt_shape(b, params)
    ext = BRepExtrema_DistShapeShape()
    ext.LoadS1(sa)
    ext.LoadS2(sb)
    ext.SetDeflection(deflection)
    ext.Perform()
    if not ext.IsDone() or ext.NbSolution() == 0:
        return VerificationReport(
            status=Status.UNKNOWN,
            metrics={"certification_tier": CertificationTier.BREP_ESCAPED.value},
            message="OCCT BRepExtrema_DistShapeShape did not converge.",
        )

    val = float(ext.Value())
    pa = ext.PointOnShape1(1)
    pb = ext.PointOnShape2(1)
    wa = (float(pa.X()), float(pa.Y()), float(pa.Z()))
    wb = (float(pb.X()), float(pb.Y()), float(pb.Z()))

    containment = bool(ext.InnerSolution()) or (val <= deflection)
    if not containment:
        for shp_target, pt in ((sb, pa), (sa, pb)):
            exp = TopExp_Explorer(shp_target, TopAbs_SOLID)
            while exp.More():
                clf = BRepClass3d_SolidClassifier(TopoDS.Solid(exp.Current()), pt, 1e-6)
                if clf.State() in (TopAbs_IN, TopAbs_ON):
                    containment = True
                    break
                exp.Next()

    if containment:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.PENETRATION,
            witness_points=[wa, wb],  # type: ignore[list-item]
            metrics={
                "distance_mm": val,
                "guarantee_level": GuaranteeLevel.SOLVER_TOLERANCE.value,
                "certification_tier": CertificationTier.BREP_ESCAPED.value,
            },
            message="BREP_ESCAPED clearance check detected penetration/containment.",
        )

    if val - deflection > min_clearance_mm:
        return VerificationReport(
            status=Status.PASS,
            witness_points=[wa, wb],  # type: ignore[list-item]
            metrics={
                "distance_mm": val,
                "deflection_mm": deflection,
                "min_clearance_mm": min_clearance_mm,
                "guarantee_level": GuaranteeLevel.SOLVER_TOLERANCE.value,
                "certification_tier": CertificationTier.BREP_ESCAPED.value,
            },
            message=f"BREP_ESCAPED clearance {val:.4f} mm > {min_clearance_mm:.4f} mm (SOLVER_TOLERANCE).",
        )

    if val + deflection <= min_clearance_mm:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.CLEARANCE_VIOLATION,
            witness_points=[wa, wb],  # type: ignore[list-item]
            metrics={
                "distance_mm": val,
                "deflection_mm": deflection,
                "min_clearance_mm": min_clearance_mm,
                "guarantee_level": GuaranteeLevel.SOLVER_TOLERANCE.value,
                "certification_tier": CertificationTier.BREP_ESCAPED.value,
            },
            message=f"BREP_ESCAPED clearance violation: {val:.4f} mm <= {min_clearance_mm:.4f} mm.",
        )

    return VerificationReport(
        status=Status.UNKNOWN,
        witness_points=[wa, wb],  # type: ignore[list-item]
        metrics={
            "distance_mm": val,
            "deflection_mm": deflection,
            "min_clearance_mm": min_clearance_mm,
            "guarantee_level": GuaranteeLevel.SOLVER_TOLERANCE.value,
            "certification_tier": CertificationTier.BREP_ESCAPED.value,
        },
        message="BREP_ESCAPED distance lies within solver deflection band.",
    )


def export_finalized_step(
    finalized: BRepFinalized,
    step_path: str | pathlib.Path,
    craft_json_path: str | pathlib.Path | None = None,
    volume_rel_tol: float = 1e-4,
) -> VerificationReport:
    """Export BRepFinalized to STEP, re-import for round-trip verification, and mark BREP_ESCAPED."""
    if not isinstance(finalized, BRepFinalized):
        raise TypeError("export_finalized_step expects a BRepFinalized object")
    orig_inv = extract_topology_invariants(finalized.shape)
    if not orig_inv["is_valid"] or orig_inv["volume_mm3"] <= 1e-9:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            metrics=orig_inv,
            message="BRepFinalized failed BRepCheck_Analyzer or has zero volume.",
        )

    sf = pathlib.Path(step_path)
    sf.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(finalized.shape, STEPControl_AsIs)
    if writer.Write(str(sf)) != IFSelect_RetDone:
        return VerificationReport(status=Status.FAIL, reason=FailReason.TOPOLOGY_INVALID, message="STEP write failed")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(sf)) != IFSelect_RetDone:
        return VerificationReport(status=Status.FAIL, reason=FailReason.TOPOLOGY_INVALID, message="STEP re-read failed")
    reader.TransferRoots()
    reimported = reader.OneShape()
    re_inv = extract_topology_invariants(reimported)
    vol_rel_err = abs(re_inv["volume_mm3"] - orig_inv["volume_mm3"]) / max(orig_inv["volume_mm3"], 1e-12)
    if not re_inv["is_valid"] or vol_rel_err > volume_rel_tol:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.TOPOLOGY_INVALID,
            metrics={"orig": orig_inv, "reimported": re_inv, "volume_rel_err": vol_rel_err},
            message="BRepFinalized STEP round-trip topology/volume check failed.",
        )

    if craft_json_path is not None:
        cf = pathlib.Path(craft_json_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(
            json.dumps(
                {
                    "certification_tier": finalized.tier.value,
                    "irreversible_reason": finalized.irreversible_reason,
                    "applied_ops": finalized.applied_ops,
                    "step_path": str(sf),
                    "topology": re_inv,
                    "volume_rel_err": vol_rel_err,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return VerificationReport(
        status=Status.PASS,
        metrics={
            "certification_tier": finalized.tier.value,
            "irreversible_reason": finalized.irreversible_reason,
            "topology": re_inv,
            "volume_rel_err": vol_rel_err,
        },
        message="BREP_ESCAPED STEP export & round-trip verified.",
    )
