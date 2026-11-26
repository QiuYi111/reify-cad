---
title: "M3 考题：FCM 离线三级 GCI 与虚域伪模态筛查、WoS O(N^-1/2) 标准误、平面应力 F_zz 与 Gmsh 桥接"
id: ACT-T-M3
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-009]]', '[[REQ-010]]', '[[DES-004]]']
depends_on: ['[[ACT-T-M0]]']
supersedes: []
tags: [edac/activity, tdd, m3, fcm, wos, mpm, gmsh, calculix]
source_location: "ref/reify_system_spec_v1.4.md#L156"
---

# Verification Test Card (M3)

## 1. 核心不变量（Invariants）
- **I-M3-1（FCM 三级 GCI 与 WoS 统计收敛阶）**：离线 Benchmark 随背景网格 $h \to h/2 \to h/4$ 加密按理论阶收敛；WoS 样本均值方差按 $O(1/N)$、**标准误 $\operatorname{SE}$ 按 $O(1/\sqrt{N})$** 收敛。
- **I-M3-2（虚域局部伪模态滤除）**：当含极小切割单元时，经物理域能量参与比筛查后的前 6 阶物理模态与贴体参考网格 `CalculiX` 模态的 MAC 对角元 $> 0.95$，无虚域局部伪模态混入。
- **I-M3-3（平面应力 $\sigma_{zz}=0$ 与 `Gmsh` 双通道可解性）**：平面应力 MPM 物质点处 $|\sigma_{zz}| / \|\boldsymbol{\sigma}\|_F < 10^{-6}$；通道 A（`Gmsh` C3D10 保留锐边）与通道 B（隐式四面体网格）导出的 `.inp` 均在真实 `CalculiX` 零报错求解吻合。

## 2. 场景卡（外部独立裁判：解析闭式解 & 真实 CalculiX）
1. **斜置 $25^\circ$ 铁木辛柯悬臂梁静力/前 6 阶模态与 Kirsch 带孔板**：验证 FCM 正向位移、柔度及滤除虚域模式后的固有频率与解析解、`CalculiX` 吻合。
2. **同心球壳稳态导热解析算例**：验证 WoS 标准误 $O(1/\sqrt{N})$ 斜率与 95% 正态近似 CI 覆盖率，双介质输入返回 `OUT_OF_DOMAIN`。
3. **薄片卡扣（平面应力，含 $F_{zz}$ Newton 求解）与厚缓冲梁（平面应变）压缩**：验证正向 ADR-MLS-MPM 力-位移曲线分别与 `CalculiX` CPS（平面应力）/ CPE（平面应变）非线性解吻合。
