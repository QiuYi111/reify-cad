---
title: "M4 验收证据：2D 草图/样条精确距场、10,000 盒零违背、包络定理求导与 solve() 避障"
id: EVID-M4
type: evidence
status: active
result: pass
baseline: v2.0
produced_by: ['[[ACT-T-M4]]']
supports: ['[[REQ-011]]', '[[DES-005]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m4_e2e.py']
tags: [edac/evidence, m4, sketch, bspline, envelope-theorem]
---

# Evidence (M4)

## Result
- `uv run pytest tests/ -v` 中 `tests/test_m4_e2e.py` **3/3 全部通过**（总计 `14.03s`：10k 盒检验 `11.62s`、样条 `solve()` `2.40s`、Boehm/包络求导 `0.01s`；含于全量 `21 passed in 62.41s`）。

## Interpretation
1. **I-M4-1 / I-M4-2（2D 草图距离与 10,000 盒区间下界）**：含直线、圆弧、三次 Bézier 与圆孔的 `Sketch2D` 编译出合法 OCCT 2D 面，与 10,000 点边界高密暴力真值偏差 $< 2\times 10^{-3}\text{ mm}$；在 10,000 个随机 2D 盒上 `interval_bound_2d` 达成 **0 违背**。
2. **I-M4-3（Boehm 拆分与包络定理求导）**：`BSpline2D` 线性拆分矩阵与 `scipy.interpolate.BSpline` 偏差 $< 10^{-12}$；控制点自动微分梯度与 `float64` 中心差分相对误差 $< 10^{-5}$；闭合周期 B-Spline 成功编译 OCCT 2D 面。
3. **I-M4-4（样条控制点 `solve()` 优化）**：`ExtrudeSketch` 样条凸轮经 `solve()` 自动拉低 `bump_y` 避开外部障碍物，通过 `float64` `verify() == PASS` 与 STEP 回读核验。
