---
title: "M0 验收证据：55 棵随机 AST 双后端一致性、蜕变测试与退化门禁全绿"
id: EVID-M0
type: evidence
status: active
result: pass
baseline: v1.4
produced_by: ['[[ACT-T-M0]]']
supports: ['[[REQ-001]]', '[[REQ-002]]', '[[REQ-003]]', '[[DES-001]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m0_e2e.py']
tags: [edac/evidence, m0, occt, fuzz, step-roundtrip]
---

# Evidence (M0)

## Result
- `uv run pytest tests/test_m0_e2e.py -v`：**3 passed in 38.31s**（0 failed, 0 skipped）。

## Interpretation
1. **I-M0-1（内外分类与外部净空下界）**：55 棵随机嵌套布尔 AST × 60 空间点（共 3,300 次点查询）与独立 OCCT `BRepClass3d_SolidClassifier`、`BRepExtrema_DistShapeShape` 100% 吻合；仿射算术 `interval_bound` 全数包络子盒采样极值。
2. **I-M0-2（双后端同构与 STEP 回读）**：所有抽检树双向 Hausdorff 距离均 $< 10^{-3}\text{ mm}$、体积相对误差 $< 10^{-4}$、`BRepCheck_Analyzer` 全绿且工艺清单 JSON 导出完整。
3. **I-M0-3（蜕变测试、可微 Anchor 与 Hausdorff 冻结）**：`SmoothParam` 自动微分梯度与中心差分相对误差 $< 10^{-4}$；刚体平移旋转后体积与 `distance_query` 不变；$f_{\text{new}} = 2 f_{\text{old}}$ 零等值面缩放下 `world_shell` 冻结返回 `PASS`，$0.05\text{ mm}$ 外扩返回 `FAIL`；大尺度超界、非 B-Rep 导出与闭环 `JointGraph` 准确返回 `OUT_OF_DOMAIN`。
4. **退化几何门禁**：外切圆柱 `provenance` 准确返回 `MULTI_BRANCH`，`1e-4 mm` 近共面、`0.05 mm` 极薄壁及退化圆角均安全处理无崩溃。

## Raw evidence
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/jingyi/3d/reify
collected 3 items

tests/test_m0_e2e.py ...                                                 [100%]

============================== 3 passed in 38.31s ==============================
```
