---
title: "M7 考题：BRepFinalized 类型隔离与不可逆末端封口 E2E"
id: ACT-T-M7
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-014]]', '[[DES-008]]']
depends_on: ['[[ACT-T-M5]]']
supersedes: []
tags: [edac/activity, tdd, m7, finalize-brep, isolation]
source_location: "conversation://74c93056-b59d-4c27-ba31-4f0a4a62be35"
---

# Verification Test Card (M7)

## 1. 核心不变量
- **I-M7-1**：`BRepFinalized` 无法回插任何解析 AST 组合算子或 `solve()`（严格 `TypeError` 拦截）。
- **I-M7-2**：未显式传 `irreversible=True` 或空 `reason` 必被拒绝（`ValueError`）。
- **I-M7-3**：末端局部倒角/倒圆及外部 STEP 导入后，`verify_brep_clearance` 准确输出四态认证与证人点，`craft.json` 明确记录 `BREP_ESCAPED`。
