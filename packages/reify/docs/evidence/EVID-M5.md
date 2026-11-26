---
title: "M5 验收证据：拔模敞口抽壳外壳、旋转凸台、环形孔阵与 CalculiX 静力求解"
id: EVID-M5
type: evidence
status: active
result: pass
baseline: v2.0
produced_by: ['[[ACT-T-M5]]']
supports: ['[[REQ-012]]', '[[DES-006]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m5_e2e.py']
tags: [edac/evidence, m5, features, shell, draft, calculix]
---

# Evidence (M5)

## Result
- `uv run pytest tests/ -v` 中 `tests/test_m5_e2e.py` **2/2 全部通过**（注塑外壳+CalculiX `3.18s`、拔模反求+阵列 `1.57s`；含于全量 `21 passed in 62.41s`）。

## Interpretation
1. **I-M5-1 / I-M5-3（真实注塑外壳全流程）**：`Shell(TaperedExtrude(..., draft_angle=3 deg), thickness=2.5 mm, open_top=True)` 组合 `RevolveSketch` 凸台与 `CircularPattern` 螺栓孔阵，通过 STEP 回读与双向 Hausdorff 校验（$< 0.25\text{ mm}$），通过 M2 `compute_structural_thickness` 校验，并通过 M3 `Gmsh` + 真实外部 **`CalculiX ccx_2.23`** 静力求解（`Status.PASS`）。
2. **I-M5-2（拔模角反求与阵列/偏置/锥台）**：`solve()` 自动增大 `draft_angle` 使外壳顶部内收避让障碍物并通过 `float64` `verify() == PASS`；`Frustum`、`Offset`、`LinearPattern` 组合体通过 STEP 回读与双向 Hausdorff 校验（$< 0.05\text{ mm}$）。
