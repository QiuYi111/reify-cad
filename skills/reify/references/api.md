# Reify Class 1 API Reference

## Coordinate Conventions (Read First!)
- **Z-Centered Solids**: `Box`, `Cylinder`, `ExtrudeSketch`, `TaperedExtrude`, and `Frustum` are ALL centered at `z = 0` (`z ∈ [-height/2, +height/2]`) and `(x, y) = (0, 0)`.
  - Example: For an enclosure of `height=mm(20)` (`z ∈ [-10, +10]`) with `Shell(..., thickness=mm(2.5), open_top=True)`, the bottom floor lies in `z ∈ [-10, -7.5]`. To drill through-holes in the floor, translate cylinders to `z = mm(-8.5)` with `height = mm(6.0)`.
- **RevolveSketch**: Maps 2D sketch `(x, y)` into the XZ plane `(r = x, z = y)` and revolves around the `+Z` axis.
- **Units & Parameters**:
  - Lengths: `mm(val)` (`Quantity`). Angles: `deg(val)` or `rad(val)`.
  - Differentiable optimization variable: `Param(name, init_val, min_val, max_val, unit="mm"|"deg")`.

## Class 1 Primitives (`CORE_CERTIFIED`)
| Category | Exact Signature | Notes |
| :--- | :--- | :--- |
| **3D Primitives** | `Box(lx, ly, lz)` | Z-centered at origin |
| | `Cylinder(radius, height)` | Z-centered at origin |
| | `Sphere(radius)` | Centered at origin |
| | `Frustum(r_bottom, r_top, height)` | Z-centered cone frustum |
| **2D Sketches** | `Loop2D.rounded_rect(lx, ly, fillet_r)` | Closed CCW rounded rectangle |
| | `Loop2D.polygon(points, fillet_r=None, chamfer_d=None)` | `points`: `[(x, y), ...]` |
| | `Loop2D.circle(center, radius)` | `center`: `(x, y)` `Quantity` tuple |
| | `LineSeg2D(p0, p1)`, `Arc2D(center, radius, start_deg, end_deg)` | Segment primitives |
| | `QuadBezier2D(p0, p1, p2)`, `CubicBezier2D(p0, p1, p2, p3)` | Bézier segments |
| | `BSpline2D(control_points, degree=3, knots=None, closed_periodic=False)` | 2D B-spline |
| | `Sketch2D(outer, holes=None)` | Multi-hole 2D profile |
| **Features** | `ExtrudeSketch(sketch, height)` | Z-centered extrusion |
| | `TaperedExtrude(sketch, height, draft_angle)` | Inward draft `>= 0` deg |
| | `RevolveSketch(sketch, angle_deg=deg(360.0))` | Revolve `(r=x, z=y)` around `+Z` |
| | `Offset(child, radius)`, `Shell(child, thickness, open_top=False)` | `Shell` requires `L=1` child |
| | `LinearPattern(child, count, spacing=(dx, dy, dz))` | Linear array |
| | `CircularPattern(child, count, radius=mm(0), start_angle_deg=deg(0))` | Circular array around `+Z` |
| **3D Curves & NURBS** | `BSplineCurve3D(control_points, degree=3, knots=None)` | 3D spine for sweeps |
| | `SweepAlongSpline(path, profile)` | `profile`: `Quantity\|Param` OR `Sketch2D` |
| | `LoftSketches(sections, z_levels)` | Ruled loft across `z_levels` |
| | `BSplineSurface(control_net, degree_u=3, degree_v=3)` | 2D grid of 3D points |
| | `BSplineSurfaceShell(surface, thickness)` | Differentiable NURBS shell |
| **Booleans & Transforms** | `Union(children)`, `Subtract(base, tool)`, `Intersect(children)` | Exact min/max CSG |
| | `SmoothUnion(a, b, k, kind="quad_smin")` | Pure-SDF smooth blend |
| | `Transform(child, translation=(x,y,z), rotation_deg=(rx,ry,rz))` | Intrinsic XYZ `deg(...)` |
| **Specs & Part** | `Part(root, name="part")`, `Spec(obstacles=(ClearanceRule(obstacle, min_clearance_mm=0.0), ...))` | Used by `certify` / `autosize` |

## Class 2 Escape Hatch (`BREP_ESCAPED`, Irreversible)
- `finalize_brep(source, ops, *, irreversible=True, reason="...", params=None)` → `BRepFinalized` (supports `LocalFilletEdges`, `LocalChamferEdges`, `LocalDraftFaces`, `import_step_terminal`).
- Never pass `BRepFinalized` back into AST operators or `solve()`.

## Pure-SDF Rendering, Viewer & DFM vs STEP Export
- `reify snapshot`, `reify certify --snapshot-dir`, and `reify autosize --snapshot-dir` render **directly via native PyTorch SDF Sphere Tracing** (zero mesh/GLB/B-Rep conversion, pixel-exact resolution).
- `reify viewer-sdf <script.py> [--out viewer.html]` compiles Class 1 AST directly to a **60fps WebGL2 GLSL SDF raymarching** HTML viewer (zero triangle meshes, live clip plane & overhang heatmap).
- `reify export <id> out.glb/.stl/.3mf` uses **Manifold Dual Contouring** (Schaefer-Ju-Warren 2007 + SVD-QEF; `.glb` auto-converts Z-up → glTF Y-up).
- `reify export <id> out.step` compiles and validates exact OCCT B-Rep (`BRepCheck_Analyzer` + `MAT2d` `MakeEvolved` + dual-backend Hausdorff check) on demand at export time.
- Extra pure-SDF helpers: `check_draft_and_overhang(...)`, `wost_solve_point(...)` (Walk-on-Stars mixed Dirichlet+Neumann), `GeometryQuery.distance_query(...)` (Two-stage 3D+6D Lagrangian-Affine B&B), `get_compute_device(...)`, `device_info()`.
- CLI utilities: `reify doctor` (CUDA/MPS/CPU & OCCT diagnostics), `reify snapshot <id|script.py> <out.png>`, `reify export <id> <out.step>` (supports single `Part` and multi-part `dict[str, Part]` assemblies), `reify viewer-sdf <script.py> --out viewer.html` (universal WebGL2 GLSL + `sampler3D` volume viewer).
