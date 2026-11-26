---
title: "Reify v2.0：第一类全量参数化几何表达与类型隔离末端 B-Rep 封口"
id: NEED-004
type: need
subtype: urd
status: approved
priority: P0
baseline: v2.0
depends_on: ['[[NEED-001]]']
supersedes: []
tags: [edac/need, v2.0, m4, m5, m6, m7, sketch, features, nurbs, escape-hatch]
source_location: "conversation://74c93056-b59d-4c27-ba31-4f0a4a62be35"
---

# Need

## Need
Engineering Agent 能在不破坏 Reify 核心体验（严格 Lipschitz 认证、区间下界、可微 `solve()` 与双后端 STEP 同构）的前提下，在第一类主路径中直接构建真实工程零件与自由曲面（2D 完整草图与样条、拔模/旋转/抽壳/偏置/阵列、3D 样条扫掠、参数化放样、可微 NURBS 自由曲面壳），并将存在拓扑突变的局部选边/选面操作严格隔离为不可逆末端封口 `BRepFinalized`。

## Legacy Problems Addressed（待解决的老问题）
1. **v1.4 原语过少**：仅 4 个基础体无法直接表达工业结构件、草图轮廓、拔模外壳与自由曲面。
2. **局部拓扑操作污染内核**：若允许局部倒角/自由补面回插解析树，会导致整棵树退化，丢失梯度与认证下界。

## Validation
- M4–M7 全部垂直切片通过对应独立 E2E 考题，保持 M0–M3 零回归，且 `BRepFinalized` 类型隔离在运行时强制生效。
