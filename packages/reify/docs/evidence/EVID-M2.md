---
title: "M2 验收证据：R/P 运动 CCD、含毛坯暴力对照 FFT、方向性壁厚与 IID 公差良率全绿"
id: EVID-M2
type: evidence
status: active
result: pass
baseline: v1.4
produced_by: ['[[ACT-T-M2]]']
supports: ['[[REQ-006]]', '[[REQ-007]]', '[[REQ-008]]', '[[DES-003]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m2_e2e.py']
tags: [edac/evidence, m2, ccd, fcl, stock-fft, wall-thickness, iid-yield, mmc]
---

# Evidence (M2)

## Result
- `uv run pytest tests/test_m0_e2e.py tests/test_m1_e2e.py tests/test_m2_e2e.py -v`：**9 passed in 48.90s**（M0 3 项 + M1 3 项 + M2 3 项，0 failed, 0 skipped）。

## Interpretation
1. **I-M2-1（R/P 串联路径 CCD 与工具包络）**：串联移动副 + 转动副 `0.3 mm` 薄片摆臂掠过障碍尖角，`check_kinematic_sweep` 检出碰撞参数 $s$ 与解析真值偏差 $< 5 \times 10^{-3}$；L 型内六角扳手 $60^\circ$ 摆动准确拦截高墙挡臂并放行安全布局；四连杆闭环机构立即返回 `OUT_OF_DOMAIN`。
2. **I-M2-2（含毛坯线性卷积等价性 & 壁厚分工）**：在含初始毛坯 $P_{\text{stock}}$、贴边凸台与短刃粗柄深槽算例上，零填充 Local-ROI 3D FFT 的 `f_cons_mask` 与 `m_removable_mask` 与空间暴力离散形态学结果**逐体素 100% 一致**；超显存上限准确返回 `OUT_OF_DOMAIN`；`check_directional_wall_thickness` 准确检出 $30^\circ$ 斜交 `1.0 mm` 薄壁与 T 型筋，并与 `compute_structural_thickness` 的 T 型交汇最大内切球尺度明确分离。
3. **I-M2-3（IID Clopper-Pearson 与 MMC/LMC）**：$\rho=0.7$ 相关二元正态装配模型 150 次独立 IID 试验下 95% 下界覆盖率达标；`MMC / LMC` 确定性销孔与偏移方盒极值判定与解析真值一致。

## Raw evidence
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/jingyi/3d/reify
collected 9 items

tests/test_m0_e2e.py ...                                                 [ 33%]
tests/test_m1_e2e.py ...                                                 [ 66%]
tests/test_m2_e2e.py ...                                                 [100%]

============================== 9 passed in 48.90s ==============================
```
