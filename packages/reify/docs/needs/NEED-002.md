---
title: "受限域运动路径校验、含毛坯制造可达性与公差统计良率校验"
id: NEED-002
type: need
subtype: urd
status: approved
priority: P1
baseline: v1.4
depends_on: ['[[NEED-001]]']
supersedes: []
tags: [edac/need, m2, kinematics, fft-dfm, yield, mmc]
source_location: "ref/reify_system_spec_v1.4.md#L96-L115, #L140-L143"
---

# Need

## Need
在不重造全功能 CAM 或高维运动规划器的前提下，优先组合复用成熟几何/碰撞组件，为 Agent 提供带明确适用域守卫（Domain Guard）的开链转动/移动副路径防穿模、结构化工具包络、含初始毛坯的刀柄/刀刃制造可达性（含 Local-ROI 3D FFT）、方向性对向壁厚筛查，以及区分 IID 统计良率置信下界与极限实体条件（MMC/LMC）的公差评估能力。

## Validation
- 在带解析真值的转动/移动串联关节算例、暴力卷积对照的保守毛坯深腔算例、方向性壁厚算例及联合误差分布公差算例上，通过独立真值校验并在越界时准确返回 `OUT_OF_DOMAIN`。
