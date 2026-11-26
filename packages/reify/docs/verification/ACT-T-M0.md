---
title: "M0 考题：尺度误差预算、双后端几何/拓扑同构、蜕变测试与退化预警集"
id: ACT-T-M0
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-001]]', '[[REQ-002]]', '[[REQ-003]]', '[[DES-001]]']
depends_on: []
supersedes: []
tags: [edac/activity, tdd, m0, fuzz, metamorphic-testing, occt]
source_location: "ref/reify_system_spec_v1.4.md#L152-L153"
---

# Verification Test Card (M0)

## 1. 核心不变量（Invariants）
- **I-M0-1（内外分类与外部净空安全性）**：在规定尺度 $S \le S_{\max}$ 与 `float64` 认证精度下，凡 `classify_point(x) == OUTSIDE` 的点必满足 $x \notin \Omega$ 且 `external_clearance_lower(x) <= exact_external_distance_occt(x)`；全空间点必满足 `unsigned_boundary_distance_lower(x) <= exact_boundary_distance_occt(x)`。
- **I-M0-2（双后端几何与拓扑同构）**：双后端双向 Hausdorff 距离 $< 10^{-3}\text{ mm}$，连通分量数与孔洞亏格（Betti 数/欧拉示性数）100% 一致；STEP 回读拓扑有效且体积相对偏差 $< 10^{-4}$。
- **I-M0-3（刚体变换蜕变不变性与场缩放冻结抗扰性）**：任意三维平移旋转后体积、组件距离及局部 `Anchor` 相对关系不变；对同一零等值面施加 $f_{\text{new}} = 2 f_{\text{old}}$ 场值缩放时，基于双向 Hausdorff 距离的 `world_shell` 准确判定零位移（`PASS`）。

## 2. 分口径测试集（外部独立裁判：OCCT）
1. **几何逼近 Fuzz 套件**：自动生成 50+ 棵带多轴旋转、圆角拉伸、盲孔/通孔与嵌套布尔的标准 AST，逐一校验 I-M0-1..3。
2. **退化检测口径套件**：外切/内切相切圆柱、相差 $10^{-4}\text{ mm}$ 近共面切割、$0.05\text{ mm}$ 极薄壁与退化圆角，验证准确触发并列分支或退化预警而非静默崩溃。
