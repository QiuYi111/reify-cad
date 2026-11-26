---
title: "M1 考题：细针/包含/保守下界未过三分支防撞、Hausdorff 冻结与 smin 支架闭环"
id: ACT-T-M1
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-004]]', '[[REQ-005]]', '[[DES-002]]']
depends_on: ['[[ACT-T-M0]]']
supersedes: []
tags: [edac/activity, tdd, m1, certified-clearance, hausdorff, smin, solve]
source_location: "ref/reify_system_spec_v1.4.md#L154"
---

# Verification Test Card (M1)

## 1. 核心不变量（Invariants）
- **I-M1-1（`PASS` 与 `FAIL` 双重零误判不变量）**：
  - 凡输出 `PASS`，经独立 OCCT 核验真实最小分离距离严格 $> m - \varepsilon_{\text{num}}$ 且无内部包含；
  - 凡输出 `FAIL`，必须携带经核验真实的穿透内点证人（`PENETRATION`）或满足 $\|p - q\| \le m + \varepsilon_{\text{num}}$ 的显式表面证人点对（`CLEARANCE_VIOLATION`），**严禁把仅因保守下界松弛导致的 $d_{\text{lower}} \le m$ 误判为 `FAIL`**。
- **I-M1-2（硬约束终审铁律）**：凡任一硬约束未通过切回硬布尔后的真实几何 `verify()` 复核，`solve()` 严禁返回 `FEASIBLE_FOUND`。

## 2. 场景卡（外部独立裁判：OCCT B-Rep 布尔交 & 精确距离）
1. **三分支完备性陷阱测试**：
   - **细针穿粗网**：$0.2\text{ mm}$ 细针刺入 $1.0\text{ mm}$ 粗采样板 $\to$ 自适应细分找到穿透内点证人，返回 `FAIL (PENETRATION)`；
   - **小件完全缩在大件内部**：$5\text{ mm}$ 方块置于 $50\text{ mm}$ 实体中心 $\to$ 连通分支内点检出 `INSIDE`，返回 `FAIL (PENETRATION)`；
   - **伪距离下界松弛陷阱**：布尔拐角处真实间隙 $0.5\text{ mm} > m = 0.2\text{ mm}$，但初始保守下界仅 $0.1\text{ mm} \le m$ $\to$ 限制细分深度为 0 时必须返回 `UNKNOWN`（绝不误报 `FAIL`），放开细分后证毕 `PASS`。
2. **双向 Hausdorff 冻结与多障碍支架 `solve()` 闭环**：在带 3 个 PCB 禁区与破壁风险的支架算例上，通过二次 $C^1$ `smin` 多起点求解收敛至 `FEASIBLE_FOUND` 并通过 STEP 回读核验。
