---
title: "M6 考题：3D 样条扫掠、放样与可微认证 NURBS 壳优化 E2E"
id: ACT-T-M6
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-013]]', '[[DES-007]]']
depends_on: ['[[ACT-T-M5]]']
supersedes: []
tags: [edac/activity, tdd, m6, nurbs, sweep, loft]
source_location: "conversation://74c93056-b59d-4c27-ba31-4f0a4a62be35"
---

# Verification Test Card (M6)

## 1. 核心不变量
- **I-M6-1**：`SweepAlongSpline`、`LoftSketches` 与 `BSplineSurfaceShell` 通过原生 NURBS STEP 回读与双向 Hausdorff 核验。
- **I-M6-2**：5,000 个随机 3D 盒上 `BSplineSurface` 区间下界零违背；控制网点梯度与中心差分一致。
- **I-M6-3**：NURBS 控制网点经 `solve()` 自动鼓包避让内部障碍物并通过 `float64` `verify() == PASS`。
