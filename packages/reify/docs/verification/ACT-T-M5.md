---
title: "M5 考题：拔模敞口抽壳外壳、旋转凸台、环形孔阵全流程 E2E"
id: ACT-T-M5
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-012]]', '[[DES-006]]']
depends_on: ['[[ACT-T-M4]]']
supersedes: []
tags: [edac/activity, tdd, m5, features, calculix]
source_location: "conversation://74c93056-b59d-4c27-ba31-4f0a4a62be35"
---

# Verification Test Card (M5)

## 1. 核心不变量
- **I-M5-1**：拔模 + 敞口抽壳 + 旋转凸台 + 环形阵列的真实外壳通过 OCCT STEP 回读与双向 Hausdorff 核验。
- **I-M5-2**：拔模角与壁厚经 `solve()` 反求后通过 `float64` `verify() == PASS`。
- **I-M5-3**：抽壳外壳直接通过 M2 结构厚度校验与 M3 Channel A `Gmsh + ccx_2.23` 静力求解。
