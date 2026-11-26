---
title: "面向 Agent 的可认证参数化几何编译与闭环求解底座"
id: NEED-001
type: need
subtype: urd
status: approved
priority: P0
baseline: v1.4
depends_on: []
supersedes: []
tags: [edac/need, m0, m1, geometry-kernel, agent-loop, solve]
source_location: "ref/reify_system_spec_v1.4.md#L10-L45"
---

# Need

## Need
Engineering Agent 能通过声明式参数化代码与只读 Python `Spec` 对象，在复用成熟几何/数值组件的双路径底座上，自动完成多变体评估、带光滑代理引导的连续尺寸求解、上下界严格分离的安全间隙认证与双向 Hausdorff 冻结校验，并导出经独立重读核验的精确 STEP 与工艺清单。

## Legacy Problems Addressed（待解决的老问题）
1. **传统 CAD 脚本对 Agent 极脆**：裸写 B-Rep 脚本依赖易漂移的面/边索引，改参常触发拓扑崩溃或静默出错。
2. **LLM 改连续尺寸陷入乒乓死循环**：多约束耦合下，大模型靠看报错文本加减数字难以收敛。
3. **验证缺乏严格上下界与包含语义**：稀疏采样、无符号场值或用下界未过直接判败易导致漏检与误判。

## Validation
- 在带外部障碍物禁区、双向 Hausdorff 冻结接口与工具可达要求的转接支架任务上，Agent 仅定义拓扑与 `Param` 范围，即可自动求解出可行解并通过独立 OCCT/STEP 回读核验。
