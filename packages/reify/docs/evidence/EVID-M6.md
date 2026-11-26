---
title: "M6 验收证据：RMF 3D 样条扫掠、放样与可微认证 NURBS 自由曲面壳"
id: EVID-M6
type: evidence
status: active
result: pass
baseline: v2.0
produced_by: ['[[ACT-T-M6]]']
supports: ['[[REQ-013]]', '[[DES-007]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m6_e2e.py']
tags: [edac/evidence, m6, nurbs, sweep, loft]
---

# Evidence (M6)

## Result
- `uv run pytest tests/ -v` 中 `tests/test_m6_e2e.py` **2/2 全部通过**（3D 扫掠+放样 `3.21s`、NURBS 5k 盒+梯度+`solve()` `3.62s`；含于全量 `21 passed in 62.41s`）。

## Interpretation
1. **I-M6-1（3D RMF 扫掠与多截面放样）**：S 型 3D 样条管道 `SweepAlongSpline` 通过原生 STEP 回读、双向 Hausdorff 校验（$< 0.05\text{ mm}$）、M2 `check_kinematic_sweep` 运动防撞与 M3 `Gmsh` 体网格生成；三截面 `LoftSketches` 过渡体通过 STEP 回读与双向 Hausdorff 校验（$< 0.25\text{ mm}$）。
2. **I-M6-2 / I-M6-3（NURBS 自由曲面壳下界、包络求导与 `solve()` 避障）**：双三次 `BSplineSurfaceShell` 通过原生 NURBS STEP 回读与双向 Hausdorff 校验（$< 0.35\text{ mm}$）；5,000 个随机 3D 盒控制网 BVH 距离下界 **0 违背**；3D 控制网点包络定理梯度与 `float64` 中心差分相对误差 $< 10^{-4}$；`solve()` 自动鼓包避让内部障碍物并通过 `float64` `verify() == PASS`。
