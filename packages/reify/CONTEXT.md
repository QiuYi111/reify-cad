# Reify Context（EDAC 视图）

> 本文件由 `/Users/jingyi/3d/reify` 下的 EDAC 规格库派生，不替代源节点。旧原型已完整归档隔离至 `/Users/jingyi/3d/archive/reify-PCAD`。

- **当前状态**：`v1.4`（`[[DEC-001]]`）与 **`v2.0`（`[[DEC-002]]`）全量基线均已批准并完成 M0 → M7 全部实现与 E2E 验收**（`tests/` 全量 **21 项 E2E 测试全部通过**，`ACT-T-M0..M7` 全部 `done` 并产出 `EVID-M0..M7` 证据节点，EDAC 覆盖率 100%）。
- **Needs（4 项）**：
  - `NEED-001` (`P0`)：面向 Agent 的可认证参数化几何编译与闭环求解底座
  - `NEED-002` (`P1`)：受限域运动路径校验、含毛坯制造可达性与公差统计良率校验
  - `NEED-003` (`P1`)：带误差界与 `Gmsh` 双通道体网格桥接的免贴体/免网格物理套件
  - `NEED-004` (`P0`, `v2.0`)：第一类全量参数化几何表达与类型隔离末端 B-Rep 封口
- **Requirements（14 项）**：
  - M0–M3：`REQ-001`..`REQ-010`（v1.4 基础 AST、四态间隙认证、多起点 `solve()`、运动防撞、毛坯 3D FFT、公差良率、FCM/WoS/MPM/CalculiX）
  - M4：`REQ-011`（`Sketch2D` 直线/圆弧/Bézier/B-Spline、Boehm 线性拆分、10k 盒区间界与包络定理求导）
  - M5：`REQ-012`（`ExtrudeSketch`、`TaperedExtrude`、`Frustum`、`RevolveSketch`、`Offset`、`Shell`、`LinearPattern`、`CircularPattern`）
  - M6：`REQ-013`（RMF `SweepAlongSpline`、`LoftSketches` 与控制网 BVH 可微 `BSplineSurfaceShell`）
  - M7：`REQ-014`（类型硬隔离的不可逆末端封口 `BRepFinalized` 与 `CORE_CERTIFIED` / `BREP_ESCAPED` 双等级）
- **设计 TDD 考题（8 项）**：`ACT-T-M0`..`ACT-T-M7`（全部 `done`）
- **架构设计（8 项）**：`DES-001`..`DES-008`
