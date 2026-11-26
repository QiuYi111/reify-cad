---
title: "M1 验收证据：三分支间隙防撞陷阱、双作用域冻结与多障碍支架 solve() 全绿"
id: EVID-M1
type: evidence
status: active
result: pass
baseline: v1.4
produced_by: ['[[ACT-T-M1]]']
supports: ['[[REQ-004]]', '[[REQ-005]]', '[[DES-002]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m1_e2e.py']
tags: [edac/evidence, m1, certified-clearance, hausdorff, solve, occt]
---

# Evidence (M1)

## Result
- `uv run pytest tests/test_m0_e2e.py tests/test_m1_e2e.py -v`：**6 passed in 46.05s**（M0 3 项 + M1 3 项，0 failed, 0 skipped）。

## Interpretation
1. **I-M1-1（`PASS` 与 `FAIL` 双重零误判）**：
   - `0.2 mm` 细针刺入 `1.0 mm` 粗采样平板准确自适应细分返回 `FAIL (PENETRATION)`；
   - `5 mm` 小方块置于 `50 mm` 大障碍物正中心准确检出内部包含并返回 `FAIL (PENETRATION)`；
   - 保守下界 $d_{\text{lower}} = 0.1 \le m = 0.2$ 但真实间隙为 $0.5\text{ mm}$ 且限制细分深度为 0 时，准确返回 `UNKNOWN` 而绝不误判 `FAIL`；
   - `0.10 mm < 0.20 mm` 净空不足返回 `FAIL (CLEARANCE_VIOLATION)` 并附满足上界的双证人点对；`1.50 mm` 安全算例返回 `PASS`，经 OCCT `BRepAlgoAPI_Common`（交集体积 `0.0`）与 `BRepExtrema` 确认。
2. **双作用域冻结**：`DATUM_AXIS` 允许板厚 $6 \to 12\text{ mm}$ 变化并拦截 $0.2\text{ mm}$ 孔位偏移；`WORLD_SHELL` 基于双向 Hausdorff 距离拦截 $0.05\text{ mm}$ 外扩。
3. **I-M1-2（支架多起点 `solve()` 闭环）**：在 3 个 PCB 电容/IC 禁区与双冻结约束下，采用二次 $C^1$ `smin` 代理 + Sobol 多起点 `solve()` 收敛至 `FEASIBLE_FOUND`，经硬几何 `verify()` 复核、独立 OCCT 布尔交集零体积核验及 STEP 回读全数 `PASS`。

## Raw evidence
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/jingyi/3d/reify
collected 6 items

tests/test_m0_e2e.py ...                                                 [ 50%]
tests/test_m1_e2e.py ...                                                 [100%]

============================== 6 passed in 46.05s ==============================
```
