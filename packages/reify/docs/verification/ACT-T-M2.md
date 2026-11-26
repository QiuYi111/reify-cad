---
title: "M2 考题：含移动副 CCD、保守毛坯暴力卷积对照 FFT、方向性壁厚与 SciPy IID 良率"
id: ACT-T-M2
type: activity
subtype: verification
status: done
evaluates: ['[[REQ-006]]', '[[REQ-007]]', '[[REQ-008]]', '[[DES-003]]']
depends_on: ['[[ACT-T-M1]]']
supersedes: []
tags: [edac/activity, tdd, m2, ccd, fcl, stock-fft, directional-thickness, yield]
source_location: "ref/reify_system_spec_v1.4.md#L155"
---

# Verification Test Card (M2)

## 1. 核心不变量（Invariants）
- **I-M2-1（R/P 串联路径零穿模）**：含转动副与移动副的开链路径 $q(s)$ 若存在真实穿透区间，CCD 必须返回 `FAIL`（未证毕返回 `UNKNOWN`，绝不误报 `PASS`）。
- **I-M2-2（含毛坯线性卷积逐体素等价性）**：在相同体素网格下，含初始毛坯 $P_{\text{stock}}$ 的零填充 Local-ROI 3D FFT 可切除余料掩码 $M_{\text{removable}}$ 与空间暴力离散形态学结果逐体素 100% 一致。
- **I-M2-3（IID Clopper-Pearson 覆盖诚实性）**：解析多元联合分布下，基于 IID 伪随机抽样的 `scipy.stats.binomtest` 95% 下界在重复独立试验中实际覆盖率 $\ge 95\%$。

## 2. 场景卡（外部独立裁判：解析几何解 & 暴力离散卷积）
1. **滑台平移 + 摆臂旋转复合擦边与闭环拦截**：串联 `P + R` 机构掠过障碍尖角（解析碰撞参数 $s^*$ 已知），检出误差 $< 10^{-3}$；输入四连杆闭环机构立即返回 `OUT_OF_DOMAIN`。
2. **初始毛坯深槽撞刀柄算例**：目标深槽底部虽不碰刀刃，但未切除的初始毛坯侧壁会撞击粗刀柄 $T_{\text{nc}}$，验证保守毛坯形态学准确判定该深槽底部余料不可切。
3. **斜交薄壁方向性筛查与同基准联合正态公差**：验证 `check_directional_wall_thickness` 在法向锥内准确检出薄壁且不受 T 型筋内切球膨胀干扰，并与独立 `MMC` 销孔极值判定对照。
