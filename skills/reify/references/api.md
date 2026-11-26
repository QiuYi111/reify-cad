# Reify Class 1 API Reference

## Coordinate Conventions (Read First!)
- **Z-Centered Solids**: `Box`, `Cylinder`, `ExtrudeSketch`, `TaperedExtrude`, and `Frustum` are ALL centered at `z = 0` (`z ∈ [-height/2, +height/2]`) and `(x, y) = (0, 0)`.
  - Example: For an enclosure of `height=mm(20)` (`z ∈ [-10, +10]`) with `Shell(..., thickness=mm(2.5), open_top=True)`, the bottom floor lies in `z ∈ [-10, -7.5]`. To drill through-holes in the floor, translate cylinders to `z = mm(-8.5)` with `height = mm(6.0)`.
- **RevolveSketch**: Maps 2D sketch `(x, y)` into the XZ plane `(r = x, z = y)` and revolves around the `+Z` axis.
- **Units & Parameters**:
  - Lengths: `mm(val)` (`Quantity`). Angles: `deg(val)` or `rad(val)`.
  - Differentiable optimization variable: `Param(name, init_val, min_val, max_val, unit)` where `unit` is `"mm"` or `"deg"`.

## Class 1 Primitives (`CORE_CERTIFIED`)
| Category | Symbol & Signature | Notes |
| :--- | :--- | :--- |
| **3D Primitives** | `Box(lx, ly, lz)`, `Cylinder(radius, height)`, `Sphere(radius)`, `Frustum(r_bottom, r_top, height)` | Z-centered at origin |
| **2D Sketches** | `Loop2D.rect(lx, ly)`, `Loop2D.rounded_rect(lx, ly, r)`, `Loop2D.polygon(pts)`, `Loop2D.circle(r)` | Closed CCW loops |
| | `LineSeg2D(p0, p1)`, `Arc2D(center, radius, start_deg, end_deg)`, `QuadBezier2D(...)`, `CubicBezier2D(...)`, `BSpline2D(ctrl_pts, degree=3, periodic=False)` | Segment primitives for `Loop2D(segments)` |
| | `Sketch2D(outer, holes=[...])` | Multi-hole 2D profile |
| **Features** | `ExtrudeSketch(sketch, height)`, `TaperedExtrude(sketch, height, draft_angle)` | Inward draft angle `>= 0` deg |
| | `RevolveSketch(sketch, angle_deg=deg(360))`, `Offset(child, radius)`, `Shell(child, thickness, open_top=False)` | `Shell` requires `L=1` child |
| | `LinearPattern(child, count, spacing=(dx, dy, dz))`, `CircularPattern(child, count, radius, start_angle_deg=deg(0))` | Array patterns |
| **3D Curves & NURBS** | `BSplineCurve3D(control_points, degree=3)` | 3D spine for sweeps |
| | `SweepAlongSpline(path, profile)` | `profile` can be `Quantity\|Param` (circle radius) OR `Sketch2D` |
| | `LoftSketches(bottom_sketch, top_sketch, height)` | Ruled loft `z ∈ [-h/2, +h/2]` |
| | `BSplineSurface(control_grid, degree_u=3, degree_v=3)`, `BSplineSurfaceShell(surface, thickness)` | Differentiable NURBS shell |
| **Booleans & Transforms** | `Union([...])`, `Subtract(base, tool)`, `Intersect([...])`, `SmoothUnion(a, b, k)` | `SmoothUnion` works in pure-SDF hot loop! |
| | `Transform(child, translation=(x,y,z), rotation=(rx,ry,rz))` | Rotation in `deg(...)` |
| **Specs & Part** | `Part(root, name="part")`, `Spec(obstacles=[ClearanceRule(obstacle=obs, min_clearance_mm=1.0)])` | Used by `certify` / `autosize` |

## Class 2 Escape Hatch (`BREP_ESCAPED`, Irreversible)
- `finalize_brep(node, irreversible=True, reason="...")` → `BRepFinalized` (supports `LocalFilletEdges`, `LocalChamferEdges`, `LocalDraftFaces`, `import_step_terminal`).
- Never pass `BRepFinalized` back into AST operators or `solve()`.

## Pure-SDF Hot Loop vs STEP Export
- `reify run`, `reify snapshot`, `reify certify`, `reify autosize`, and `reify export ... .glb/.stl/.3mf` run **100% in pure SDF** (Dual Contouring + Three.js GLB). Zero OCCT B-Rep failures can break viewing, certifying, or solving.
- `reify export <id> out.step` compiles and validates exact OCCT B-Rep (`BRepCheck_Analyzer` + `MAT2d` `MakeEvolved` + dual-backend Hausdorff check) on demand at export time.
