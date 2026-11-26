---
name: reify
description: Primary Reify certified parametric CAD workflow. Use FIRST for all 3D modeling, sketches/splines, drafted/shelled/revolved/patterned parts, 3D spline sweeps, NURBS surfaces, clearance certification, multi-start sizing solve(), visual snapshot review, and STEP/GLB/STL/3MF export. Only fall back to `reify build123d` or `finalize_brep` when Class 1 cannot express the geometry.
license: MIT
---

# Reify Primary CAD Workflow

Reify is the primary CAD path. Write models using **Class 1 (`CORE_CERTIFIED`)** parametric primitives (`Sketch2D`, `ExtrudeSketch`, `TaperedExtrude`, `RevolveSketch`, `Shell`, `Offset`, `LinearPattern`, `CircularPattern`, `SweepAlongSpline`, `LoftSketches`, `BSplineSurfaceShell`, `Param`, `Spec`) decorated with `@reify_model`.

## Standard Closed Loop

1. **Start warm server once**:
   ```bash
   reify server start
   ```
2. **Build & view live in 3D Viewer**:
   ```bash
   reify run model.py
   ```
   Prints document `id:` and `view:` URL.
3. **Inspect visually (headless multi-view / section / exploded)**:
   ```bash
   reify snapshot "$ID" tmp/iso.png
   reify snapshot "$ID" tmp/top.png --camera top
   reify snapshot "$ID" tmp/section.png --display '{"clip":{"enabled":true,"axis":"z","at":0.0}}'
   ```
4. **Certify clearances & freezes (`PASS / FAIL / UNKNOWN`)**:
   ```bash
   reify certify model.py --snapshot-dir tmp/certify
   ```
   If `FAIL`, inspect `tmp/certify/violation_overlay.png` (red spheres mark exact witness points).
5. **Auto-size parameters to clear obstacles**:
   ```bash
   reify autosize model.py --snapshot-dir tmp/solved
   ```
   Runs multi-start `solve()`, verifies in `float64`, pushes solved geometry to the Viewer, and renders multi-view PNGs.
6. **Export deliverables**:
   ```bash
   reify export "$ID" out/part.step out/part.glb out/part.stl out/part.3mf
   ```

## Hard Boundaries

- **Default to Class 1 (`CORE_CERTIFIED`)**: Keep the entire parametric tree inside Reify Class 1 so `autosize` gradients, interval lower bounds, and `certify` stay intact.
- **Irreversible terminal escape hatch (`BREP_ESCAPED`)**: Only call `finalize_brep(..., irreversible=True, reason="...")` at the very end for local edge fillets/chamfers or external STEP imports when Class 1 cannot express them. Never pass `BRepFinalized` back into AST operators or `solve()`.
- **Secondary fallback**: Use `reify build123d <subcommand>` (`skills/cad/SKILL.md`) only for legacy `build123d` scripts or workflows not yet wrapped in Reify.

## Minimal `@reify_model` Template

```python
from reify import Box, ClearanceRule, Loop2D, Param, Part, Shell, Sketch2D, Spec, TaperedExtrude, Transform, deg, mm
from reify.cad_bridge import reify_model

DRAFT = Param("draft_deg", 1.0, 0.5, 12.0, "deg")
PARAMS = [DRAFT]

def build_part() -> Part:
    body = TaperedExtrude(Sketch2D(Loop2D.rounded_rect(mm(36), mm(36), mm(5))), height=mm(16), draft_angle=DRAFT)
    return Part(root=Shell(body, thickness=mm(2.5), open_top=True), name="enclosure")

def build_spec() -> Spec:
    obs = Transform(Box(mm(6), mm(12), mm(8)), translation=(mm(20.5), mm(0), mm(6)))
    return Spec(obstacles=[ClearanceRule(obstacle=obs, min_clearance_mm=1.0)])

@reify_model(name="enclosure")
def enclosure_model() -> Part:
    return build_part()

if __name__ == "__main__":
    enclosure_model()
```
