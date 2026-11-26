---
title: "Reify 项目与领域术语词典"
status: approved
---

# TERMEOLOGY（术语与路径表）

> 状态：`approved`（经 `[[DEC-001]]` 批准生效；后续扩充与变更需经人类决策者新审批）。
> 原则：词典聚焦项目核心域、概念与路径，保持清晰精简。

| 术语 / 名称 | 英文 / 简称 | 定义与边界 | 位置 / 备注 |
| :--- | :--- | :--- | :--- |
| Reify | `reify` | 面向 Coding / Engineering Agent 的轻量 Python 参数化几何编译、尺寸求解与受限验证内核 | `/Users/jingyi/3d/reify` |
| 规格书基准 | `ref/reify_system_spec_v1.4.md` | 终审收口版系统架构与受限算子规格书（复用成熟组件 + 严守算子组合边界） | `ref/reify_system_spec_v1.4.md` |
| 声明式参数 AST | `reify.ast` / AST | 携带显式物理单位、局部原点坐标、参数分类、可微锚点 `Anchor` 与开链 `R/P Joint` 的建模树 | Layer 0 |
| 只读约束规格 | `reify.spec.Spec` | 进程内只读 Python 约束对象，声明双向 Hausdorff 冻结区、禁区、初始毛坯与物理阈值 | Layer 0 / Layer 3 |
| 几何查询接口 | `GeometryQuery` | 严格分离内外分类、外部净空下界（仅 `OUTSIDE` 有效）、无符号边界距离下界、覆盖采样与距离保证等级 | Layer 1 |
| 伪距离场 | `Pseudo-SDF` | 由 `min/max` 布尔生成的零等值面精确、但非欧氏距离的隐式场；不可直接冒充精确间隙 | Layer 1 |
| 平滑布尔代理 | `SmoothBooleanProxy` | 仅在 `solve()` 内循环使用的光滑布尔近似（默认二次 $C^1$ 或三次 $C^2$ 多项式 `smin`），终审回切硬几何 | Layer 3 |
| 四态诚实诊断 | Four-State Verification | 统一返回 `PASS`（下界证毕）、`FAIL`（具显式上界或穿透证人）、`UNKNOWN`（跨阈值未决）、`OUT_OF_DOMAIN` | Layer 2 / Layer 3 |
| 免贴体/免网格物理 | Unfitted & Meshfree Physics | 包含有限胞元法（FCM）、球面随机游走（WoS）、平面应力/应变物质点法（MPM）与 `Gmsh` 双通道 CalculiX 桥接 | Layer 2 |
| 双后端一致交付 | Dual-Backend Export | 同一标准 AST 编译至隐式链与 OCCT B-Rep（STEP）链，强制经过双向 Hausdorff、拓扑不变量与回读核验 | Layer 1 / Layer 3 |
| 2D 草图与样条 | `Sketch2D` / `BSpline2D` | 支持直线、圆弧、Bézier、B-Spline 闭合外环与内孔的精确 1-Lipschitz 有符号距离与 OCCT 2D 面 | Layer 0 / Layer 1 |
| 工程特征体 | `TaperedExtrude` / `RevolveSketch` / `Shell` / `Pattern` | 保持第一类解析与可微特性的拔模拉伸、旋转体、等厚抽壳、偏置及线性/环形阵列原语 | Layer 0 / Layer 1 |
| 包络定理求导 | Envelope Theorem Autograd | 对样条/NURBS 最近点前向用保护牛顿法求解、反向直接按 Danskin 定理解析求导至控制点 | Layer 1 / Layer 3 |
| 3D 扫掠与 NURBS 壳 | `SweepAlongSpline` / `BSplineSurfaceShell` / `LoftSketches` | RMF 无扭转 3D 样条扫掠、多截面放样与带控制网 BVH 下界的可微张量积 NURBS 自由曲面壳 | Layer 0 / Layer 1 |
| 不可逆末端封口 | `BRepFinalized` (`finalize_brep`) | 类型系统硬隔离的末端 B-Rep 容器，仅用于局部倒圆/倒角/选面拔模与外部 STEP 导入，禁止回插解析 AST | Layer 3 |
| 双等级认证 | `CORE_CERTIFIED` / `BREP_ESCAPED` | 区分纯第一类全认证模型与经过不可逆末端封口降级后的 B-Rep 终审模型 | Layer 3 |
