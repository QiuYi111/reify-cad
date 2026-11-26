"""Example Reify Class 1 constrained assembly for closed-loop CLI/Viewer/certify/autosize verification."""

from __future__ import annotations

from reify import (
    Box,
    CircularPattern,
    ClearanceRule,
    Cylinder,
    Loop2D,
    Param,
    Part,
    RevolveSketch,
    Shell,
    Sketch2D,
    Spec,
    Subtract,
    TaperedExtrude,
    Transform,
    Union,
    deg,
    mm,
)
from reify.cad_bridge import reify_model

DRAFT_DEG = Param("draft_deg", 1.0, 0.5, 10.0, "deg")
PARAMS = [DRAFT_DEG]


def build_part() -> Part:
    outer_loop = Loop2D.rounded_rect(mm(36.0), mm(36.0), mm(5.0))
    body = TaperedExtrude(Sketch2D(outer_loop), height=mm(16.0), draft_angle=DRAFT_DEG)
    enclosure = Shell(body, thickness=mm(2.5), open_top=True)

    boss_profile = Loop2D.polygon([(mm(2.0), mm(-8.0)), (mm(5.0), mm(-8.0)), (mm(4.5), mm(-2.0)), (mm(2.0), mm(-2.0))])
    boss = RevolveSketch(Sketch2D(boss_profile), angle_deg=deg(360.0))

    holes = Transform(CircularPattern(Cylinder(mm(1.5), mm(6.0)), count=4, radius=mm(10.0)), translation=(mm(0.0), mm(0.0), mm(-8.0)))
    return Part(root=Subtract(Union([enclosure, boss]), holes), name="drafted_enclosure")


def build_spec() -> Spec:
    # Keep-out obstacle near top outer +X edge (x = 17.5..23.5, z = 2..10); draft >= ~5 deg clears it
    obs = Transform(Box(mm(6.0), mm(12.0), mm(8.0)), translation=(mm(20.5), mm(0.0), mm(6.0)))
    return Spec(obstacles=[ClearanceRule(obstacle=obs, min_clearance_mm=1.0)])


@reify_model(name="drafted_enclosure")
def enclosure_model() -> Part:
    return build_part()


if __name__ == "__main__":
    enclosure_model()
