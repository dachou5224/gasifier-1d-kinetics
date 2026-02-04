# 2026-02-04 项目回顾与后续工作计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标**: 整理当前散乱的项目目录，集中精力解决 Case 6 的点火问题，并完成动力学参数的最终校准与验证。

**架构**: 保持现有 `src/model` 结构，重点优化 `kinetics.py` 和 `gasifier_system.py` 中的点火逻辑。

---

## 任务结构

### Task 1: 项目目录整理 (Project Cleanup)

**Files:**
- Create: `results/` (directory)
- Move: `*.csv`, `*.png` (from root to `results/`)

**Step 1: 创建结果目录并移动文件**
将根目录下散落的诊断结果和图表移动到 `results/` 文件夹中，保持根目录整洁。

```bash
mkdir -p results
mv diagnostic_results.csv results/
mv homo_audit.csv results/
mv energy_audit.csv results/
mv dp_evolution_profile.png results/
mv dp_o2_evolution_profile.png results/
```

### Task 2: Case 6 点火问题诊断 (Debug Ignition)

**Files:**
- Run: `tests/debug_case6_ignition.py`
- Modify: `src/model/gasifier_system.py` (potential)

**Step 1: 运行诊断脚本**
运行现有的 case 6 调试脚本，获取详细的点火传播数据。

```bash
python tests/debug_case6_ignition.py
```

**Step 2: 分析输出**
检查 `Cell 0` 到 `Cell 10` 的温度分布和组分变化。
- 预期：在入口处有显著温升或 O2 消耗。
- 故障现象：温度维持在入口温度，或反应速率极低。

### Task 3: 动力学参数校准 (Calibrate Kinetics)

**Files:**
- Modify: `src/model/kinetics.py`
- Verify: `src/model/chemistry.py`

**Step 1: 审查 Jones-Lindstedt 参数**
确保所有反应速率常数 (A, E) 与文献一致，且单位转换为 SI (kmol, m3, s, K)。
重点检查：
- 甲烷氧化 (CH4 + 0.5 O2 -> CO + 2 H2)
- CO 氧化 (CO + 0.5 O2 -> CO2)
- H2 氧化 (H2 + 0.5 O2 -> H2O)

**Step 2: 修正代码**
如果在 Step 1 发现单位或数值错误，直接在 `kinetics.py` 中修正。

### Task 4: 综合验证 (Full Verification)

**Files:**
- Run: `tests/verify_cases.py`

**Step 1: 运行全量测试**
确保所有 Case (1-6) 均通过验证，误差在 5% 以内。

```bash
python tests/verify_cases.py
```

## 执行建议
建议按顺序执行 Task 1 -> Task 2 -> Task 3 -> Task 4。Task 1 为快速清理，Task 2 & 3 为核心调试工作。
