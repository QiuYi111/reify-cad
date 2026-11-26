"""Native WebGL2 GLSL ES 3.00 SDF interactive raymarching HTML viewer (zero triangle meshes)."""

from __future__ import annotations

import math
import pathlib
from typing import Any

from reify.ast import (
    Box,
    Cylinder,
    ExtrudeRoundedRect,
    Intersect,
    Node,
    Part,
    SmoothUnion,
    Sphere,
    Subtract,
    Transform,
    Union,
    _val_deg_float,
    _val_mm_float,
)
from reify.features import CircularPattern, ExtrudeSketch, Frustum, LinearPattern, Offset, Shell, TaperedExtrude

_PALETTE_RGB = (
    (0.290, 0.435, 0.647),
    (0.357, 0.549, 0.353),
    (0.557, 0.424, 0.533),
    (0.831, 0.639, 0.451),
    (0.239, 0.353, 0.502),
    (0.878, 0.478, 0.373),
    (0.420, 0.565, 0.502),
    (0.710, 0.514, 0.553),
)


def _glsl_mat3_inv_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> str:
    """Return GLSL mat3 R^T where R = Rx @ Ry @ Rz (so p_local = R_T * (p - t))."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = ((1, 0, 0), (0, cx, -sx), (0, sx, cx))
    Ry = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
    Rz = ((cz, -sz, 0), (sz, cz, 0), (0, 0, 1))

    def mm3(A: Any, B: Any) -> Any:
        return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))

    R = mm3(Rx, mm3(Ry, Rz))
    # GLSL mat3 constructor is column-major; passing R rows as columns gives R^T!
    cols = []
    for i in range(3):
        cols.extend(f"{R[i][j]:.7f}" for j in range(3))
    return f"mat3({', '.join(cols)})"


def _emit_glsl_node(
    node: Node,
    p_expr: str,
    params: dict[str, Any] | None,
    lines: list[str],
    counter: list[int],
) -> str:
    """Emit GLSL float SDF expression variable name for a Class 1 AST node."""
    vid = f"d_{counter[0]}"
    counter[0] += 1

    if isinstance(node, Transform):
        tx = _val_mm_float(node.translation[0], params)
        ty = _val_mm_float(node.translation[1], params)
        tz = _val_mm_float(node.translation[2], params)
        rx = _val_deg_float(node.rotation[0], params)
        ry = _val_deg_float(node.rotation[1], params)
        rz = _val_deg_float(node.rotation[2], params)
        p_loc = f"p_{counter[0]}"
        counter[0] += 1
        t_glsl = f"vec3({tx:.6f}, {ty:.6f}, {tz:.6f})"
        if abs(rx) + abs(ry) + abs(rz) > 1e-7:
            rt = _glsl_mat3_inv_xyz(rx, ry, rz)
            lines.append(f"  vec3 {p_loc} = {rt} * ({p_expr} - {t_glsl});")
        else:
            lines.append(f"  vec3 {p_loc} = {p_expr} - {t_glsl};")
        ch_v = _emit_glsl_node(node.child, p_loc, params, lines, counter)
        lines.append(f"  float {vid} = {ch_v};")
        return vid

    if isinstance(node, Box):
        hx = 0.5 * _val_mm_float(node.lx, params)
        hy = 0.5 * _val_mm_float(node.ly, params)
        hz = 0.5 * _val_mm_float(node.lz, params)
        lines.append(f"  float {vid} = sdBox({p_expr}, vec3({hx:.6f}, {hy:.6f}, {hz:.6f}));")
        return vid

    if isinstance(node, Cylinder):
        r = _val_mm_float(node.radius, params)
        hz = 0.5 * _val_mm_float(node.height, params)
        lines.append(f"  float {vid} = sdCylinderZ({p_expr}, {r:.6f}, {hz:.6f});")
        return vid

    if isinstance(node, Sphere):
        r = _val_mm_float(node.radius, params)
        lines.append(f"  float {vid} = length({p_expr}) - {r:.6f};")
        return vid

    if isinstance(node, Frustum):
        r1 = _val_mm_float(node.r_bottom, params)
        r2 = _val_mm_float(node.r_top, params)
        hz = 0.5 * _val_mm_float(node.height, params)
        lines.append(f"  float {vid} = sdFrustumZ({p_expr}, {r1:.6f}, {r2:.6f}, {hz:.6f});")
        return vid

    if isinstance(node, ExtrudeRoundedRect):
        hx = 0.5 * _val_mm_float(node.lx, params)
        hy = 0.5 * _val_mm_float(node.ly, params)
        r = _val_mm_float(node.r_corner, params)
        hz = 0.5 * _val_mm_float(node.lz, params)
        lines.append(f"  float {vid} = sdRoundedBoxZ({p_expr}, {hx:.6f}, {hy:.6f}, {r:.6f}, {hz:.6f});")
        return vid

    if isinstance(node, ExtrudeSketch):
        ar = getattr(node.sketch.outer, "_analytic_rounded_rect", None)
        if ar is not None and not node.sketch.holes:
            hx = 0.5 * _val_mm_float(ar[0], params)
            hy = 0.5 * _val_mm_float(ar[1], params)
            r = _val_mm_float(ar[2], params)
            hz = 0.5 * _val_mm_float(node.height, params)
            lines.append(f"  float {vid} = sdRoundedBoxZ({p_expr}, {hx:.6f}, {hy:.6f}, {r:.6f}, {hz:.6f});")
            return vid

    if isinstance(node, TaperedExtrude):
        ar = getattr(node.sketch.outer, "_analytic_rounded_rect", None)
        if ar is not None and not node.sketch.holes:
            hx = 0.5 * _val_mm_float(ar[0], params)
            hy = 0.5 * _val_mm_float(ar[1], params)
            r = _val_mm_float(ar[2], params)
            hz = 0.5 * _val_mm_float(node.height, params)
            th = math.radians(_val_deg_float(node.draft_angle, params))
            lines.append(
                f"  float {vid} = sdTaperedRoundedZ({p_expr}, {hx:.6f}, {hy:.6f}, {r:.6f}, {hz:.6f}, {math.cos(th):.7f}, {math.sin(th):.7f});"
            )
            return vid

    if isinstance(node, Offset):
        r = _val_mm_float(node.radius, params)
        ch_v = _emit_glsl_node(node.child, p_expr, params, lines, counter)
        lines.append(f"  float {vid} = {ch_v} - {r:.6f};")
        return vid

    if isinstance(node, Shell):
        t_w = _val_mm_float(node.thickness, params)
        if node.open_top and isinstance(node.child, TaperedExtrude):
            te = node.child
            ar = getattr(te.sketch.outer, "_analytic_rounded_rect", None)
            if ar is not None and not te.sketch.holes:
                hx = 0.5 * _val_mm_float(ar[0], params)
                hy = 0.5 * _val_mm_float(ar[1], params)
                r = _val_mm_float(ar[2], params)
                hz = 0.5 * _val_mm_float(te.height, params)
                th = math.radians(_val_deg_float(te.draft_angle, params))
                lines.append(
                    f"  float {vid} = sdShellTaperedOpenZ({p_expr}, {hx:.6f}, {hy:.6f}, {r:.6f}, {hz:.6f}, {math.cos(th):.7f}, {math.sin(th):.7f}, {t_w:.6f});"
                )
                return vid
        ch_v = _emit_glsl_node(node.child, p_expr, params, lines, counter)
        lines.append(f"  float {vid} = max({ch_v}, -({ch_v} + {t_w:.6f}));")
        return vid

    if isinstance(node, Union):
        ch_vars = [_emit_glsl_node(ch, p_expr, params, lines, counter) for ch in node.children]
        cur = ch_vars[0]
        for cv in ch_vars[1:]:
            cur = f"min({cur}, {cv})"
        lines.append(f"  float {vid} = {cur};")
        return vid

    if isinstance(node, Intersect):
        ch_vars = [_emit_glsl_node(ch, p_expr, params, lines, counter) for ch in node.children]
        cur = ch_vars[0]
        for cv in ch_vars[1:]:
            cur = f"max({cur}, {cv})"
        lines.append(f"  float {vid} = {cur};")
        return vid

    if isinstance(node, Subtract):
        vb = _emit_glsl_node(node.base, p_expr, params, lines, counter)
        vt = _emit_glsl_node(node.tool, p_expr, params, lines, counter)
        lines.append(f"  float {vid} = max({vb}, -({vt}));")
        return vid

    if isinstance(node, SmoothUnion):
        va = _emit_glsl_node(node.a, p_expr, params, lines, counter)
        vb = _emit_glsl_node(node.b, p_expr, params, lines, counter)
        k = _val_mm_float(node.k, params)
        lines.append(f"  float {vid} = sminQuad({va}, {vb}, {k:.6f});")
        return vid

    if isinstance(node, LinearPattern):
        sx = _val_mm_float(node.spacing[0], params)
        sy = _val_mm_float(node.spacing[1], params)
        sz = _val_mm_float(node.spacing[2], params)
        acc = []
        for k in range(node.count):
            pk = f"p_{counter[0]}"
            counter[0] += 1
            lines.append(f"  vec3 {pk} = {p_expr} - vec3({k*sx:.6f}, {k*sy:.6f}, {k*sz:.6f});")
            acc.append(_emit_glsl_node(node.child, pk, params, lines, counter))
        expr = acc[0]
        for a in acc[1:]:
            expr = f"min({expr}, {a})"
        lines.append(f"  float {vid} = {expr};")
        return vid

    if isinstance(node, CircularPattern):
        rad = _val_mm_float(node.radius, params)
        a0 = math.radians(_val_deg_float(node.start_angle_deg, params))
        acc = []
        for k in range(node.count):
            ang = a0 + 2.0 * math.pi * k / node.count
            ca, sa = math.cos(ang), math.sin(ang)
            pk = f"p_{counter[0]}"
            counter[0] += 1
            lines.append(
                f"  vec3 {pk} = mat3({ca:.7f}, {-sa:.7f}, 0.0, {sa:.7f}, {ca:.7f}, 0.0, 0.0, 0.0, 1.0) * ({p_expr} - vec3({rad*ca:.6f}, {rad*sa:.6f}, 0.0));"
            )
            acc.append(_emit_glsl_node(node.child, pk, params, lines, counter))
        expr = acc[0]
        for a in acc[1:]:
            expr = f"min({expr}, {a})"
        lines.append(f"  float {vid} = {expr};")
        return vid

    # Fallback AABB box for unsupported complex spline/NURBS leaf inside WebGL shader
    bmin, bmax = node.local_bounds(params)
    cx, cy, cz = (0.5 * (bmin[i] + bmax[i]) for i in range(3))
    hx, hy, hz = (0.5 * max(bmax[i] - bmin[i], 0.5) for i in range(3))
    lines.append(f"  float {vid} = sdBox({p_expr} - vec3({cx:.6f}, {cy:.6f}, {cz:.6f}), vec3({hx:.6f}, {hy:.6f}, {hz:.6f}));")
    return vid


def export_sdf_webgl_html(
    obj: Node | Part | dict[str, Node | Part],
    out_html: str | pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
    title: str = "Reify WebGL2 Native SDF Viewer",
) -> pathlib.Path:
    """Compile Class 1 AST directly to a standalone 60fps WebGL2 GLSL SDF raymarching HTML page."""
    out = pathlib.Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = list(obj.items()) if isinstance(obj, dict) else [(obj.name if isinstance(obj, Part) else "part", obj)]
    bmins, bmaxs = [], []
    map_lines: list[str] = ["  float bestD = 1e9;", "  float bestId = 0.0;"]
    counter = [0]
    colors_glsl = []

    for idx, (_name, item) in enumerate(items):
        root = item.root if isinstance(item, Part) else item
        mn, mx = root.local_bounds(params)
        bmins.append(mn)
        bmaxs.append(mx)
        lip = max(1.0, float(root.lipschitz_bound(params)))
        dv = _emit_glsl_node(root, "p", params, map_lines, counter)
        map_lines.append(f"  float ds_{idx} = {dv} / {lip:.4f};")
        map_lines.append(f"  if (ds_{idx} < bestD) {{ bestD = ds_{idx}; bestId = {float(idx):.1f}; }}")
        c = _PALETTE_RGB[idx % len(_PALETTE_RGB)]
        colors_glsl.append(f"  if (id < {idx + 0.5:.1f}) return vec3({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f});")

    s_bmin = [min(b[i] for b in bmins) for i in range(3)]
    s_bmax = [max(b[i] for b in bmaxs) for i in range(3)]
    cx, cy, cz = (0.5 * (s_bmin[i] + s_bmax[i]) for i in range(3))
    diag = math.sqrt(sum((s_bmax[i] - s_bmin[i]) ** 2 for i in range(3)))
    diag = max(diag, 10.0)

    map_body = "\n".join(map_lines)
    color_body = "\n".join(colors_glsl)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
  canvas {{ width: 100vw; height: 100vh; display: block; }}
  #hud {{ position: fixed; top: 14px; left: 14px; background: rgba(255,255,255,0.92); backdrop-filter: blur(8px);
          padding: 12px 14px; border-radius: 10px; box-shadow: 0 4px 16px rgba(15,23,42,0.12); font-size: 13px; color: #0f172a; }}
  #hud label {{ display: flex; align-items: center; gap: 8px; margin-top: 6px; }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">
  <strong>{title}</strong> (60fps Pure-SDF Raymarching)<br/>
  <label>Mode
    <select id="mode">
      <option value="0">Studio CAD</option>
      <option value="1">Surface Normals</option>
      <option value="2">Draft / Overhang</option>
    </select>
  </label>
  <label>Clip Axis
    <select id="clipAx">
      <option value="-1">Off</option>
      <option value="0">X</option>
      <option value="1">Y</option>
      <option value="2">Z</option>
    </select>
    <input id="clipAt" type="range" min="{min(s_bmin):.1f}" max="{max(s_bmax):.1f}" step="0.5" value="{cz:.1f}" />
    <span id="clipVal">{cz:.1f}</span>
  </label>
  <div style="margin-top:6px;color:#64748b;font-size:11px;">Drag: Orbit · Wheel: Zoom · Shift+Drag: Pan</div>
</div>
<script>
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2');
const vs = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main() {{ vUv = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }}`;
const fs = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uCamPos, uTarget;
uniform float uHalfH;
uniform int uMode, uClipAx;
uniform float uClipAt;

float sdBox(vec3 p, vec3 b) {{ vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }}
float sdCylinderZ(vec3 p, float r, float hz) {{ vec2 d = vec2(length(p.xy) - r, abs(p.z) - hz); return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }}
float sdFrustumZ(vec3 p, float r1, float r2, float hz) {{
  vec2 q = vec2(length(p.xy), p.z);
  vec2 k1 = vec2(r2, hz); vec2 k2 = vec2(r2 - r1, 2.0 * hz);
  vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - hz);
  vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / max(dot(k2, k2), 1e-12), 0.0, 1.0);
  float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}}
float sdRounded2D(vec2 p, float hx, float hy, float r) {{
  vec2 q = abs(p) - vec2(hx - r, hy - r);
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}}
float sdRoundedBoxZ(vec3 p, float hx, float hy, float r, float hz) {{
  vec2 d = vec2(sdRounded2D(p.xy, hx, hy, r), abs(p.z) - hz);
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}}
float sdTaperedRoundedZ(vec3 p, float hx, float hy, float r, float hz, float cth, float sth) {{
  float d2 = sdRounded2D(p.xy, hx, hy, r);
  return max(d2 * cth + (p.z + hz) * sth, abs(p.z) - hz);
}}
float sdShellTaperedOpenZ(vec3 p, float hx, float hy, float r, float hz, float cth, float sth, float t) {{
  float d2 = sdRounded2D(p.xy, hx, hy, r);
  float ds = d2 * cth + (p.z + hz) * sth;
  float fo = max(ds, abs(p.z) - hz);
  float fi = max(ds, -hz - p.z);
  return max(fo, -(fi + t));
}}
float sminQuad(float a, float b, float k) {{
  float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}}
vec3 partColor(float id) {{
{color_body}
  return vec3(0.4, 0.5, 0.7);
}}
vec3 mapScene(vec3 p) {{
{map_body}
  float isCap = 0.0;
  if (uClipAx >= 0) {{
    float dc = (uClipAx == 0 ? p.x : (uClipAx == 1 ? p.y : p.z)) - uClipAt;
    if (dc > bestD) {{ bestD = dc; isCap = 1.0; }}
  }}
  return vec3(bestD, bestId, isCap);
}}
vec3 calcNormal(vec3 p, float h) {{
  vec2 e = vec2(h, -h);
  return normalize(e.xyy * mapScene(p + e.xyy).x + e.yyx * mapScene(p + e.yyx).x +
                   e.yxy * mapScene(p + e.yxy).x + e.xxx * mapScene(p + e.xxx).x);
}}
void main() {{
  vec3 fwd = normalize(uTarget - uCamPos);
  vec3 right = normalize(cross(fwd, vec3(0.0, 0.0, 1.0)));
  vec3 up = cross(right, fwd);
  float aspect = uRes.x / max(1.0, uRes.y);
  vec3 ro = uCamPos + (vUv.x * uHalfH * aspect) * right + (vUv.y * uHalfH) * up;
  vec3 rd = fwd;
  float t = 0.0; float pid = 0.0; float isCap = 0.0; bool hit = false;
  float maxT = {diag * 4.0:.2f};
  for (int i = 0; i < 110; i++) {{
    vec3 res = mapScene(ro + t * rd);
    if (res.x < 0.002 * uHalfH) {{ hit = true; pid = res.y; isCap = res.z; break; }}
    t += max(res.x, 0.001 * uHalfH);
    if (t > maxT) break;
  }}
  vec3 col = vec3(0.945, 0.96, 0.98);
  if (hit) {{
    vec3 p = ro + t * rd;
    vec3 n = calcNormal(p, 0.0015 * uHalfH);
    if (uMode == 1) {{ col = 0.5 + 0.5 * n; }}
    else if (uMode == 2) {{
      float ov = clamp(-n.z, 0.0, 1.0);
      col = mix(vec3(0.25, 0.55, 0.85), vec3(0.90, 0.25, 0.20), smoothstep(0.5, 0.85, ov));
    }} else {{
      vec3 base = partColor(pid);
      if (isCap > 0.5) base = mix(base, vec3(0.22, 0.26, 0.32), 0.35);
      vec3 key = normalize(right * 0.5 + up * 0.7 - fwd * 0.5);
      float diff = max(dot(n, key), 0.0);
      float ao = clamp(mapScene(p + n * {diag * 0.03:.3f}).x / {diag * 0.03:.3f}, 0.25, 1.0);
      col = base * (0.35 * ao + 0.65 * diff);
    }}
  }}
  fragColor = vec4(col, 1.0);
}}`;

function compile(t, s) {{ const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh); return sh; }}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(prog); gl.useProgram(prog);
const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const loc = gl.getAttribLocation(prog, 'aPos'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

let az = -0.78, el = 0.58, target = [{cx:.3f}, {cy:.3f}, {cz:.3f}], halfH = {diag * 0.55:.3f};
let drag = false, shift = false, lx = 0, ly = 0;
canvas.onmousedown = e => {{ drag = true; shift = e.shiftKey; lx = e.clientX; ly = e.clientY; }};
window.onmouseup = () => drag = false;
window.onmousemove = e => {{
  if (!drag) return;
  const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
  if (shift) {{
    const fwd = [Math.cos(el)*Math.cos(az), Math.cos(el)*Math.sin(az), Math.sin(el)];
    const r = [Math.sin(az), -Math.cos(az), 0];
    target[0] += r[0]*dx*halfH*0.002; target[1] += r[1]*dx*halfH*0.002; target[2] += dy*halfH*0.002;
  }} else {{ az -= dx * 0.008; el = Math.max(-1.5, Math.min(1.5, el + dy * 0.008)); }}
}};
canvas.onwheel = e => {{ e.preventDefault(); halfH *= Math.exp(e.deltaY * 0.001); }};
document.getElementById('clipAt').oninput = e => document.getElementById('clipVal').textContent = e.target.value;

function frame() {{
  canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio;
  gl.viewport(0, 0, canvas.width, canvas.height);
  const dist = {diag * 2.0:.2f};
  const cp = [target[0]+dist*Math.cos(el)*Math.cos(az), target[1]+dist*Math.cos(el)*Math.sin(az), target[2]+dist*Math.sin(el)];
  gl.uniform2f(gl.getUniformLocation(prog, 'uRes'), canvas.width, canvas.height);
  gl.uniform3fv(gl.getUniformLocation(prog, 'uCamPos'), cp);
  gl.uniform3fv(gl.getUniformLocation(prog, 'uTarget'), target);
  gl.uniform1f(gl.getUniformLocation(prog, 'uHalfH'), halfH);
  gl.uniform1i(gl.getUniformLocation(prog, 'uMode'), +document.getElementById('mode').value);
  gl.uniform1i(gl.getUniformLocation(prog, 'uClipAx'), +document.getElementById('clipAx').value);
  gl.uniform1f(gl.getUniformLocation(prog, 'uClipAt'), +document.getElementById('clipAt').value);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(frame);
}}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out
