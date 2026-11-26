"""M7 E2E Verification: Type-isolated irreversible BRepFinalized terminal escape hatch."""

from __future__ import annotations

import json
import pathlib
import warnings

import pytest

from reify.ast import Box, Intersect, Param, Part, SmoothUnion, Subtract, Transform, Union
from reify.brep_compiler import export_step
from reify.features import Offset, Shell, TaperedExtrude
from reify.finalize import (
    BRepFinalized,
    LocalChamferEdges,
    LocalFilletEdges,
    export_finalized_step,
    finalize_brep,
    import_step_terminal,
    verify_brep_clearance,
)
from reify.sketch import Loop2D, Sketch2D
from reify.solve import solve
from reify.spec import Spec
from reify.types import CertificationTier, FailReason, Status
from reify.units import deg, mm


def test_m7_type_isolation_and_consent_guards() -> None:
    """I-M7-1 & I-M7-2: Reject missing consent/reason and block BRepFinalized re-entry into AST/solve."""
    box = Box(mm(20.0), mm(20.0), mm(10.0))

    with pytest.raises(ValueError, match="irreversible=True"):
        finalize_brep(box, [LocalChamferEdges(mm(1.0))], irreversible=False, reason="test")

    with pytest.raises(ValueError, match="non-empty reason"):
        finalize_brep(box, [LocalChamferEdges(mm(1.0))], irreversible=True, reason="   ")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fin = finalize_brep(
            box,
            [LocalChamferEdges(mm(1.0))],
            irreversible=True,
            reason="Cosmetic edge break on outer flange",
        )
    assert any("BREP_ESCAPED" in str(w.message) for w in rec)
    assert isinstance(fin, BRepFinalized)
    assert fin.tier == CertificationTier.BREP_ESCAPED

    # Strict TypeError on any attempt to re-insert into Class 1 AST or solve()
    for fn in (
        lambda: Union([fin, box]),  # type: ignore[list-item]
        lambda: Intersect([fin, box]),  # type: ignore[list-item]
        lambda: Subtract(fin, box),  # type: ignore[arg-type]
        lambda: Transform(fin, translation=(mm(1.0), mm(0.0), mm(0.0))),  # type: ignore[arg-type]
        lambda: SmoothUnion(fin, box, k=mm(1.0)),  # type: ignore[arg-type]
        lambda: Part(root=fin),  # type: ignore[arg-type]
        lambda: Offset(fin, mm(0.5)),  # type: ignore[arg-type]
        lambda: Shell(fin, mm(1.0)),  # type: ignore[arg-type]
        lambda: solve(fin, [Param("x", 1.0, 0.0, 2.0, "mm")], Spec()),  # type: ignore[arg-type]
    ):
        with pytest.raises(TypeError, match="BRepFinalized"):
            fn()


def test_m7_terminal_fillet_step_and_vendor_clearance(tmp_path: pathlib.Path) -> None:
    """I-M7-3: Local fillet+chamfer STEP (BREP_ESCAPED) + vendor STEP import & 4-state clearance."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    def _edge_z_mid(e) -> float:
        c = BRepAdaptor_Curve(e)
        p = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
        return float(p.Z())

    bracket = TaperedExtrude(
        Sketch2D(Loop2D.rounded_rect(mm(30.0), mm(20.0), mm(3.0))),
        height=mm(10.0),
        draft_angle=deg(2.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fin = finalize_brep(
            bracket,
            [
                LocalFilletEdges(mm(0.5), edge_filter=lambda e: _edge_z_mid(e) > 4.0),
                LocalChamferEdges(mm(0.3), edge_filter=lambda e: _edge_z_mid(e) < -4.0),
            ],
            irreversible=True,
            reason="Local cosmetic fillet and edge chamfer after drafted body",
        )

    step_p = tmp_path / "fin_bracket.step"
    craft_p = tmp_path / "fin_bracket.craft.json"
    rep = export_finalized_step(fin, step_p, craft_json_path=craft_p)
    assert rep.status == Status.PASS, rep.message
    craft = json.loads(craft_p.read_text(encoding="utf-8"))
    assert craft["certification_tier"] == "BREP_ESCAPED"
    assert "cosmetic fillet" in craft["irreversible_reason"]

    # Export a vendor STEP (far vs colliding) and import via import_step_terminal
    vendor_far_step = tmp_path / "vendor_far.step"
    export_step(Transform(Box(mm(10.0), mm(10.0), mm(10.0)), translation=(mm(30.0), mm(0.0), mm(0.0))), vendor_far_step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        vendor_far = import_step_terminal(vendor_far_step, irreversible=True, reason="Vendor motor housing STEP")

    c_pass = verify_brep_clearance(fin, vendor_far, min_clearance_mm=5.0)
    assert c_pass.status == Status.PASS

    vendor_hit_step = tmp_path / "vendor_hit.step"
    export_step(Transform(Box(mm(10.0), mm(10.0), mm(10.0)), translation=(mm(14.0), mm(0.0), mm(0.0))), vendor_hit_step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        vendor_hit = import_step_terminal(vendor_hit_step, irreversible=True, reason="Vendor colliding bolt STEP")

    c_fail = verify_brep_clearance(fin, vendor_hit, min_clearance_mm=2.0)
    assert c_fail.status == Status.FAIL
    assert c_fail.reason in (FailReason.PENETRATION, FailReason.CLEARANCE_VIOLATION)
