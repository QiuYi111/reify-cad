---
title: "带误差界与 Gmsh 双通道体网格桥接的免贴体/免网格物理套件"
id: NEED-003
type: need
subtype: urd
status: approved
priority: P1
baseline: v1.4
depends_on: ['[[NEED-001]]']
supersedes: []
tags: [edac/need, m3, fcm, wos, mpm, gmsh, calculix]
source_location: "ref/reify_system_spec_v1.4.md#L117-L133"
---

# Need

## Need
优先复用与适配成熟科学计算库，为复杂参数化几何提供免除脆弱手工网格前处理的正向物理快筛与外部求解器桥接能力，覆盖线弹性静力/模态/导热（FCM）、单均匀介质局部点探针（WoS）、平面应力/平面应变大变形柔顺结构（MPM）、几何光路遮挡检查，以及基于 `Gmsh`（解析锐边件）与隐式体网格器（复杂有机件）的双通道 `CalculiX` 桥接。

## Validation
- 在斜置铁木辛柯悬臂梁、Kirsch 带孔板、同心球壳 Laplace 解析解及真实 `CalculiX` 求解对照测试上，物理量误差与网格/样本加密收敛律符合理论预期。
