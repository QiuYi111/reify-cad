---
title: "M7 验收证据：BRepFinalized 类型隔离与不可逆末端封口"
id: EVID-M7
type: evidence
status: active
result: pass
baseline: v2.0
produced_by: ['[[ACT-T-M7]]']
supports: ['[[REQ-014]]', '[[DES-008]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m7_e2e.py']
tags: [edac/evidence, m7, finalize-brep, isolation]
---

# Evidence (M7)

## Result
- `uv run pytest tests/ -v` 中 `tests/test_m7_e2e.py` **2/2 全部通过**（含于全量 21 passed in 224.64s）。

## Interpretation
1. **I-M7-1 / I-M7-2（类型硬隔离与显式同意）**：`finalize_brep` 严格拒绝 `irreversible=False` 或空 `reason`（`ValueError`），触发 `[BREP_ESCAPED / IRREVERSIBLE]` 告警；将 `BRepFinalized` 回插 `Union` / `Intersect` / `Subtract` / `Transform` / `SmoothUnion` / `Part` / `Offset` / `Shell` / `solve()` 全部在运行时抛出 `TypeError` 拦截。
2. **I-M7-3（末端局部倒圆/倒角与外部 STEP 间隙核验）**：拔模支架经 `LocalFilletEdges` + `LocalChamferEdges` 导出有效 STEP 并在 `craft.json` 明确记录 `BREP_ESCAPED` 与不可逆原因；通过 `import_step_terminal` 导入外部供应商 STEP 后，`verify_brep_clearance` 准确区分安全间隙（`PASS`）与碰撞违规（`FAIL`）。
