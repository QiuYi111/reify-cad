# Changelog

## v0.2.0 — Pure-SDF Certified Parametric CAD (2026-11-26)

### 纯 SDF 优先架构与核心升级

- **原生 PyTorch SDF 球面追踪渲染器（零网格 / 零 GLB 转换）**
  - 新增 `render_sdf_snapshot`、`render_violation_overlay_png`，`reify snapshot`、`reify certify --snapshot-dir`、`reify autosize --snapshot-dir` 全部直出原生 SDF 渲染图。
  - 8 件木牛装配出图实测 `2.92s`；19.89° 拔模壳全套 `autosize` + 4 视图 `21.95s`，零破面。
- **流形双等高线法（Manifold Dual Contouring）网格导出**
  - 实现 Schaefer–Ju–Warren (2007) 256 态体素拓扑表 + Ju (2002) SVD 截断 QEF + Rank-1 光滑等值面牛顿贴合 + 解析 SDF 梯度法线 + 自动 Z-up → glTF Y-up 转换。
  - 导出 `.glb` / `.stl` / `.3mf` 严格水密（`watertight=True`、0 条非流形边），并消除 `cad_bridge.py` 中的重复建网。
- **纯 SDF 距离查询与仿射算术修复**
  - 修复 `AffineForm.square()` 过度保守分支导致线性系数被清零的底层缺陷，新增 `SmoothUnion` 仿射形式与 `eval_affine_form`。
  - `GeometryQuery.distance_query` 采用多起点交替投影 + 两阶段 3D 拉格朗日仿射 B&B + Stage 3 稀疏 6D 配对，旋转重叠包围盒场景从 `9.34s / 误差 0.69 mm` 提升至 `0.165s / 误差 < 1e-10 mm`（提速约 57×），且原生支持双非精确隐式（`SmoothUnion` vs `SmoothUnion`）。
- **通用 60fps WebGL2 纯 SDF 交互预览器**
  - 新增 `reify viewer-sdf` CLI 与 `export_sdf_webgl_html`，将 Class 1 AST 编译为单文件 GLSL ES 3.00 实时光线步进 HTML。
  - 支持 `Scale` 解析 GLSL，并对 `FreeImplicit` / NURBS / 通用样条自动烘焙 `48×48×48` R8 3D 纹理，通过 `sampler3D` (`sdVolume3D`) 实现无死角硬件三线性插值预览。
- **纯 SDF 制造性检查、八叉树验证、同伦求解与 Walk-on-Stars**
  - `dfm.py`：`check_directional_wall_thickness` 改用 Lipschitz 安全球面追踪，`check_draft_and_overhang` 改用解析 SDF 法线。
  - `verify.py`：采用仿射算术八叉树收集内点候选。
  - `solve.py`：接入同伦延拓退火（Homotopy continuation）、`TaperedExtrude`/`Shell` 可微退化势垒与余弦学习率衰减。
  - `physics/wos.py`：新增 Walk-on-Stars（`wost_solve_point`, Sawhney et al. SIGGRAPH 2023）支持混合 Dirichlet/Neumann 边界 PDE 求解。
- **工业级 B-Rep 出口与 CLI 加固**
  - `nurbs.py`：`LoftSketches` 支持不等分段截面弧长重采样与最小扭转对齐。
  - `brep_compiler.py`：带孔 `TaperedExtrude` 改用 OCCT `MAT2d` `BRepOffsetAPI_MakeEvolved` 并经 `BRepCheck_Analyzer` 硬检查；`export_step` 与 `reify export` 现支持 `dict[str, Part]` 多体装配体（`TopoDS_Compound`）。
  - `cli.py`：新增 `reify doctor` 与 `reify snapshot <script.py> <out.png>` 直出。
- **统一计算设备后端（CUDA + Apple Metal MPS + CPU）**
  - 新增 `device.py`（`get_compute_device`、`sync_device`、`device_info`），接入 `sdf_renderer.py`、`sdf_mesh.py`、`dfm.py`、`physics/wos.py`。

### 测试验收

- `reify` 全量 E2E 测试：**32/32 passed**（`test_m0_e2e.py`–`test_m7_e2e.py` + `test_v2_1_algorithms_e2e.py`）。
- `reify-cad` 全量 E2E 测试：**12/12 passed**（`test_reify_cad_e2e.py` + `test_v2_1_algorithms_e2e.py`）。

### 已知限制

- **Apple Silicon Metal (MPS)**：全算子可在 `device="mps"` 下正确运行（与 CPU 像素误差 `0.000`），但受限于 PyTorch MPS 逐算子命令提交开销，实测耗时高于统一内存 CPU（渲染 `8.325s` vs `0.710s`；30 万点求值 `0.190s` vs `0.060s`）。因此 `REIFY_DEVICE=auto` 仅自动启用 NVIDIA CUDA，Mac 默认走 CPU，可通过 `REIFY_DEVICE=mps` 显式启用 Metal。
- **网格导出**：Manifold DC 针对细小斜向特征在默认分辨率下仍有轻微三角化；日常视觉检查请优先使用原生 SDF 的 `reify snapshot` 或 `reify viewer-sdf`。
