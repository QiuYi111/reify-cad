---
title: "M3 验收证据：FCM 三级 GCI、WoS 标准误收敛、平面应力 F_zz MPM 与双通道网格桥接全绿"
id: EVID-M3
type: evidence
status: active
result: pass
baseline: v1.4
produced_by: ['[[ACT-T-M3]]']
supports: ['[[REQ-009]]', '[[REQ-010]]', '[[DES-004]]']
refutes: []
supersedes: []
artifacts: ['tests/test_m3_e2e.py']
tags: [edac/evidence, m3, fcm, wos, mpm, gmsh, isosurface-stuffing, ngsolve]
---

# Evidence (M3)

## Result
- `uv run pytest tests/ -v`：**12 passed in 42.11s**（M0 3 项 + M1 3 项 + M2 3 项 + M3 3 项，0 failed, 0 skipped），已接入真实外部 **`CalculiX ccx_2.23`**（`/opt/homebrew/bin/ccx_2.23`）完成求解验证。

## Interpretation
1. **I-M3-1（FCM 三级 GCI、Kirsch 带孔板 & WoS 统计收敛阶）**：
   - 斜置 $25^\circ$ 铁木辛柯悬臂梁（不贴合背景笛卡尔网格）通过 $h \in \{4.0, 2.0, 1.0\}\text{ mm}$ 三级网格加密与 Richardson GCI 验收（$p_{\text{obs}} > 0.5$，细网格与铁木辛柯解析解偏差 $< 6\%$，与独立子进程 `NGSolve` 贴体参考解偏差 $< 5\%$）；Kirsch 带圆孔板柔度与独立子进程 `NGSolve` 贴体参考解偏差 $< 8\%$；极小切割单元无 `NaN` 奇异；
   - 同心球壳稳态导热算例上，WoS 标准误严格按 $O(N^{-1/2})$ 收敛，100 次独立 IID 试验 95% 正态近似 CI 覆盖率达标；双介质、Neumann 边界与域外点准确返回 `OUT_OF_DOMAIN`，阈值落入 CI 内返回 `UNKNOWN`；光路遮挡追迹准确检出挡光与无遮挡光线。
2. **I-M3-2（虚域伪模态能量筛查与 MAC）**：通过分别组装物理域/虚域刚度与质量矩阵并按物理域能量参与比 $\eta_{\text{phys}} \ge 0.85$ 过滤，滤除虚域局部伪模态后前 3 阶物理频率与欧拉-伯努利解析解吻合，MAC 对角元 $> 0.99$。
3. **I-M3-3（平面应力 $F_{zz}$ MPM 与 `Gmsh` 双通道桥接）**：
   - 平面应力准静态 ADR-MLS-MPM 在每个物质点求解 $\sigma_{zz}(F_{zz}) = 0$（残余相对误差 $< 10^{-6}$），且变形柔度显著大于同参数平面应变解；请求非光滑库仑摩擦梯度准确返回 `OUT_OF_DOMAIN`；
   - 通道 A（`Gmsh` OpenCASCADE `C3D4` & `C3D10`）保持 $< 10^{-4}\text{ mm}$ 锐边外廓，导出 `.inp` 经 `meshio.read()` 校验并通过真实 **`ccx_2.23`**（`require_ccx=True`）静力求解（含 `.dat` 位移解析与 Z 向挠度验证）；通道 B（BCC `Isosurface Stuffing`）网格化非 B-Rep `SmoothUnion` 实体，单元体积全正、二面角满足 $\in (10^\circ, 166^\circ)$，导出 `.inp` 经真实 **`ccx_2.23`** 求解通过。

## Raw evidence
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/jingyi/3d/reify
collected 12 items

tests/test_m0_e2e.py ...                                                 [ 25%]
tests/test_m1_e2e.py ...                                                 [ 50%]
tests/test_m2_e2e.py ...                                                 [ 75%]
tests/test_m3_e2e.py ...                                                 [100%]

============================= 12 passed in 42.11s ==============================
```
