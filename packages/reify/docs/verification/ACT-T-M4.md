---
title: "M4 考题：2D 草图/样条精确度、区间下界零违背、包络定理梯度与 OCCT 编译 E2E"
id: ACT-T-M4
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-011]]', '[[DES-005]]']
depends_on: []
supersedes: []
tags: [edac/activity, tdd, m4, sketch, spline]
source_location: "conversation://74c93056-b59d-4c27-ba31-4f0a4a62be35"
---

# Verification Test Card (M4)

## 1. 核心不变量
- **I-M4-1**：`Sketch2D`（含圆弧、Bézier、B-Spline 与内孔）距离场与 10,000 点高密暴力真值偏差 $< 10^{-5}\text{ mm}$。
- **I-M4-2**：10,000 个随机 2D 盒上的 `interval_bound_2d` 零违背。
- **I-M4-3**：包络定理梯度与 `float64` 中心差分相对误差 $< 10^{-5}$。
- **I-M4-4**：样条控制点经 `solve()` 自动优化避障并通过 `verify() == PASS` 与 STEP 回读。
