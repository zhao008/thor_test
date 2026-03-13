# MatMul No-WriteBack: 不写回 C 矩阵的 MatMul 性能实验

## 项目概述 (Project Overview)

本项目用于验证**不写回 C 矩阵数据**对矩阵乘法计算性能的影响。通过对比本工程与正常写回的 MatMul 的 NCU 分析结果，用于排除数据写入对读取路径（含 L2 evict/写放大）的干扰。

**核心挑战**: 简单删除 `ct.store(C, ...)` 会导致 cuTile 编译器的 DCE (Dead Code Elimination) 优化掉整个 MMA 计算——得到一个空 kernel。

**解决方案**: 对 accumulator tile 做 `ct.sum(accumulator, axis=None)` 归约为标量，然后将该标量存入一个微小的 checksum buffer（每个 block 只写 1 个 float32 值）。这样：
- `ct.sum` 消费了 accumulator → MMA 循环**不可被消除**
- `ct.store(checksum)` 消费了 sum 结果 → sum 也**不可被消除**
- 每 block 写 4 bytes vs 正常写 tm×tn×2 bytes（如 256×256×2=128KB）→ 写入量减少约 32,768 倍

This project provides a comprehensive framework for analyzing CUDA Tile matrix multiplication performance across different matrix dimensions, with a focus on understanding memory access patterns and L2 cache behavior.

## 主要功能 (Key Features)

- **No-WriteBack Kernel**: 不写回 C 矩阵，用 checksum 防止编译器 DCE
- **两种 Swizzle 策略**: M 维度和 N 维度两种 block swizzle
- **自动化性能分析**: 通过脚本自动化进行大规模 NCU 性能测试
- **DDR 比率分析**: 分析实际 DDR 访问与理论值的比率

## 文件结构 (File Structure)

```
thor_matmul_C_nowriteback/
├── NoWriteBackSwizzleM.py    # M 维度 swizzle 的 no-writeback kernel
├── NoWriteBackSwizzleN.py    # N 维度 swizzle 的 no-writeback kernel
├── MatMulPerf.py             # 主性能测试脚本
├── run_matmul_ncu.sh         # NCU 自动化测试脚本
├── parse_ncu_results.py      # NCU 报告解析工具
├── analyze_ddr_ratio.py      # DDR 访问比率分析和可视化工具
├── ncu_results/              # NCU 原始报告文件 (.ncu-rep)
└── parsed_results/           # 解析后的性能数据 (JSON/CSV)
```

## 核心组件 (Core Components)

### 1. No-WriteBack Kernel

#### NoWriteBackSwizzleM.py
M 维度 swizzle 的 no-writeback kernel。Block 按 M 维度分组（同组共享 B 矩阵列瓦片）。

#### NoWriteBackSwizzleN.py
N 维度 swizzle 的 no-writeback kernel。Block 按 N 维度分组（同组共享 A 矩阵行瓦片）。

**Kernel 工作流程**:
```
加载 A tile, B tile → MMA 累加 → ct.sum(accumulator) → ct.store(checksum[block_id])
```
- 完整执行所有 load + MMA 计算（与正常 kernel 一致）
- 不执行 `ct.store(C, ...)` — C 矩阵不写回
- 用 `ct.sum` 归约 + `ct.store(checksum)` 防止 DCE

```python
# 使用示例
cutile_matmul_no_writeback(A, B,
    matmul_kernel=NoWriteBackSwizzleM.matmul_kernel_m_no_writeback)
```

### 2. 性能测试框架 (Performance Testing Framework)

#### MatMulPerf.py
主性能测试脚本，支持：
- 自定义矩阵维度 (M, N, K)
- 可选的正确性检查（对比 checksum 与 torch 参考值）
- 同时测试 M-swizzle 和 N-swizzle 两种变体

```bash
python3 MatMulPerf.py --M 2048 --N 2048 --K 1024 --correctness-check
```

#### run_matmul_ncu.sh
自动化 NCU 性能测试脚本：
- 遍历 K 维度: 512 → 4096 (步长 256)，共 15 个 K
- 遍历 M=N 维度: 256 → 8192 (步长 256)，共 32 个尺寸
- 总计 480 个测试用例；每个 K 会生成独立目录 `ncu_results/K_<K>/`
- 自动生成 Nsight Compute 报告 (.ncu-rep)

### 3. 数据分析工具 (Analysis Tools)

#### parse_ncu_results.py
解析 NCU 报告文件，提取关键性能指标：
- Kernel 执行周期数
- DDR 读写字节数（实际值）
- 计算理论 DDR 读取量（A + B）
- 理论写入量仅为 checksum（极小，每 block 4 bytes）
- 输出为 JSON 和 CSV 格式

#### analyze_ddr_ratio.py
分析 DDR 访问比率并生成可视化：
- 计算实际/理论 DDR 读取比率
- 识别性能拐点（L2 缓存不足的临界点）
- 生成 3D 可视化图表 (M × K → DDR Ratio)

## 快速开始 (Quick Start)

### 前置要求 (Prerequisites)

- CUDA Toolkit (支持 CUDA Tile)
- Python 3.8+
- PyTorch with CUDA support
- NVIDIA Nsight Compute (ncu)
- 必需的 Python 包: `pandas`, `numpy`, `matplotlib`

### 基本使用 (Basic Usage)

1. **运行单个性能测试**:
```bash
python3 MatMulPerf.py --M 2048 --N 2048 --K 1024 --correctness-check
```

2. **运行完整的 NCU 分析**:
```bash
./run_matmul_ncu.sh
```

3. **解析 NCU 结果**:
```bash
python3 parse_ncu_results.py
```

4. **生成 DDR 比率分析**:
```bash
python3 analyze_ddr_ratio.py
```

## 性能分析流程 (Performance Analysis Workflow)

### 标准工作流程:

1. **收集性能数据**:
   ```bash
   ./run_matmul_ncu.sh
   ```
   - 生成大量不同维度组合的 NCU 报告
   - 结果按 K 值组织在 `ncu_results/K_*/` 目录下

2. **解析 NCU 报告**:
   ```bash
   python3 parse_ncu_results.py
   ```
   - 自动扫描 `ncu_results/K_*/` 下的全部 `.ncu-rep`
   - 生成 `parsed_results/` 目录下的 JSON / CSV（含汇总文件）

3. **分析和可视化**:
   ```bash
   python3 analyze_ddr_ratio.py
   ```
   - 计算 DDR 读取放大比率
   - 识别 L2 缓存容量不足的拐点
   - 生成 3D 可视化图表

## 关键概念 (Key Concepts)

### Block Swizzle 策略

**M-Swizzle** (NoWriteBackSwizzleM.py):
- Block 按 M 维度分组
- 同组内的 block 共享 B 矩阵的列瓦片

**N-Swizzle** (NoWriteBackSwizzleN.py):
- Block 按 N 维度分组
- 同组内的 block 共享 A 矩阵的行瓦片

### Anti-DCE 原理

```
正常 kernel:
  for k: load A, load B → MMA → accumulator
  ct.store(C, accumulator)    ← 全量写回 C (tm×tn 元素)

No-writeback kernel:
  for k: load A, load B → MMA → accumulator
  scalar = ct.sum(accumulator) ← 归约为 1 个标量
  ct.store(checksum, scalar)   ← 只写 1 个 float32 (4B)
```

cuTile 编译器的 DCE 规则：只有被 **side-effecting 操作**（如 `ct.store`）消费的计算才会保留。通过 `ct.sum → ct.store` 这条链路，所有 MMA 计算都被间接消费，因此无法被消除。

### DDR 访问比率

理论 DDR 读取量计算:
```
DDR_read_theoretical = (M * K + K * N) * element_size
```

当实际 DDR 读取量远大于理论值时，表明 L2 缓存容量不足，导致读取放大。

## 输出结果 (Output Results)

### parsed_results/ 目录:

- `ncu_results_K_*.json`: 各 K 值的详细性能数据
- `ncu_results_K_*.csv`: CSV 格式的性能数据
- `ncu_results_combined.csv`: 所有 K 值的汇总数据
- `last_normal_points.csv`: 各 K 值下 L2 缓存充足的最大 M 值
- `ddr_ratio_analysis.png`: DDR 比率 3D 可视化图表

## 对比分析 (How to Compare)

将本工程的 NCU 结果与**正常写回 C 矩阵**的 MatMul NCU 结果对比：

| 指标 | 正常模式 | No-WriteBack | 差异含义 |
|------|---------|-------------|---------|
| DDR Write Bytes | M×N×2 bytes | ~0 (仅 checksum) | C 写回的 DDR 开销 |
| L2 Write Sectors | 较多 | 极少 | C 写回的 L2 开销 |
| Elapsed Cycles | 较长 | 较短 | C 写回对总时间的影响 |
| DDR Read Bytes | 相同 | 相同 | 读取路径无变化 |

## 许可证 (License)

Apache-2.0

## English Summary

- **Goal**: Measure matmul compute + read cost without C write-back, isolating L2 evictions/write amplification from output stores. Accumulator is reduced (`ct.sum`) and only one float32 checksum per block is written to keep kernels alive (anti-DCE).
- **Kernels**: Two no-writeback variants with block swizzle on M or N (`NoWriteBackSwizzleM.py`, `NoWriteBackSwizzleN.py`).
- **Driver script**: `run_matmul_ncu.sh` sweeps K=512→4096 (step 256, 15 values) and M=N=256→8192 (step 256, 32 values), producing 480 Nsight Compute reports under `ncu_results/K_<K>/`.
- **Single test**: `python3 MatMulPerf.py --M 2048 --N 2048 --K 1024 --correctness-check` (runs both M/N swizzle kernels, optional checksum vs torch reference).
- **Parse reports**: `python3 parse_ncu_results.py` auto-scans `ncu_results/K_*/` and writes JSON/CSV (per-K and combined) to `parsed_results/`. Example files already present for K=512/768/1024.
- **DDR ratio analysis (optional)**: `python3 analyze_ddr_ratio.py` consumes `parsed_results/ncu_results_combined.csv`, computes actual/theoretical DDR read ratios, marks inflection points, and emits plots/CSVs in `parsed_results/`.
- **Key metric meanings**:
   - Theoretical read bytes: $(M\times K + K\times N)$ elements × bytes-per-element (fp16=2B).
   - Actual read bytes: from `lts__d_sectors_fill_sysmem.sum * 32` (m/n kernels).
   - Write bytes: checksum only (4B per block) versus full C (tm×tn×2B) in normal matmul.

## Full English Version

# MatMul No-WriteBack: MatMul Performance Without Writing Back C

## Project Overview

This project measures how omitting C write-back affects matrix-multiplication performance. By comparing Nsight Compute (NCU) results of this no-writeback kernel to a normal writeback kernel, we can isolate the impact of C write traffic on reads (including L2 evictions and write amplification).

**Core challenge**: Simply removing `ct.store(C, ...)` lets the cuTile compiler DCE the entire MMA loop, yielding an empty kernel.

**Solution**: Reduce the accumulator tile with `ct.sum(accumulator, axis=None)` and write a tiny checksum buffer (one float32 per block):
- `ct.sum` consumes the accumulator → the MMA loop cannot be eliminated.
- `ct.store(checksum)` consumes the sum → the reduction cannot be eliminated.
- Each block writes 4 bytes vs. tm×tn×2 bytes (e.g., 256×256×2 = 128 KB), reducing writes by ~32,768×.

This repo provides a full workflow for profiling CUDA Tile matmul across sizes, focusing on memory traffic and L2 behavior.

## Key Features

- **No-WriteBack kernel**: Prevents DCE while avoiding full C write-back.
- **Two swizzle strategies**: Block swizzle along M or N.
- **Automated profiling**: Mass NCU runs via script.
- **DDR ratio analysis**: Actual vs. theoretical memory traffic.

## File Structure

```
thor_matmul_C_nowriteback/
├── NoWriteBackSwizzleM.py    # No-writeback kernel with M swizzle
├── NoWriteBackSwizzleN.py    # No-writeback kernel with N swizzle
├── MatMulPerf.py             # Main perf driver
├── run_matmul_ncu.sh         # NCU batch runner
├── parse_ncu_results.py      # NCU report parser
├── analyze_ddr_ratio.py      # DDR ratio analysis & plots
├── ncu_results/              # Raw NCU reports (.ncu-rep)
└── parsed_results/           # Parsed JSON/CSV outputs
```

## Core Components

### 1) No-WriteBack Kernels

- **NoWriteBackSwizzleM.py**: Blocks swizzled along M; blocks in the same group share B-column tiles.
- **NoWriteBackSwizzleN.py**: Blocks swizzled along N; blocks in the same group share A-row tiles.

Workflow:
```
load A tile, load B tile → MMA accumulate → ct.sum(accumulator) → ct.store(checksum[block_id])
```
- All loads + MMA are executed (same compute path as normal).
- `ct.store(C, ...)` is skipped; only checksum is written.
- `ct.sum` + `ct.store(checksum)` prevents DCE.

Example:
```python
cutile_matmul_no_writeback(A, B,
      matmul_kernel=NoWriteBackSwizzleM.matmul_kernel_m_no_writeback)
```

### 2) Performance Test Driver (MatMulPerf.py)

- Custom M, N, K.
- Optional correctness check (checksum vs. torch reference sum).
- Runs both M-swizzle and N-swizzle variants.

Example:
```bash
python3 MatMulPerf.py --M 2048 --N 2048 --K 1024 --correctness-check
```

### 3) NCU Batch Runner (run_matmul_ncu.sh)

- K sweep: 512 → 4096, step 256 (15 K values).
- M=N sweep: 256 → 8192, step 256 (32 sizes).
- 480 total test cases; each K gets its own directory `ncu_results/K_<K>/`.
- Generates Nsight Compute reports (.ncu-rep).

Run:
```bash
./run_matmul_ncu.sh
```

### 4) Report Parsing (parse_ncu_results.py)

- Extracts: kernel cycles, actual DDR read/write bytes (m/n kernels), theoretical reads (A+B), checksum-only write bytes.
- Auto-scans `ncu_results/K_*/` and writes per-K and combined JSON/CSV to `parsed_results/`.

Run:
```bash
python3 parse_ncu_results.py
```

### 5) DDR Ratio Analysis (analyze_ddr_ratio.py)

- Computes actual/theoretical DDR read ratios (average of m/n kernels).
- Finds last “normal” points (≤1.05×) and inflection points (>1.1×).
- Produces 3D + 2D plots and markers; outputs to `parsed_results/`.

Run:
```bash
python3 analyze_ddr_ratio.py
```

## Quick Start

**Prerequisites**
- CUDA Toolkit (with CUDA Tile)
- Python 3.8+
- PyTorch with CUDA
- NVIDIA Nsight Compute (ncu)
- Python deps: `pandas`, `numpy`, `matplotlib`

**Single test**
```bash
python3 MatMulPerf.py --M 2048 --N 2048 --K 1024 --correctness-check
```

**Full NCU sweep**
```bash
./run_matmul_ncu.sh
```

**Parse reports**
```bash
python3 parse_ncu_results.py
```

**Analyze DDR ratios**
```bash
python3 analyze_ddr_ratio.py
```

## Workflow

1) Collect NCU data
```bash
./run_matmul_ncu.sh
```
- Reports organized under `ncu_results/K_*/`.

2) Parse reports
```bash
python3 parse_ncu_results.py
```
- Generates per-K and combined JSON/CSV in `parsed_results/`.

3) Analyze & visualize
```bash
python3 analyze_ddr_ratio.py
```
- Computes DDR read amplification, marks L2 capacity breakpoints, saves plots.

## Key Concepts

### Block Swizzle
- **M-swizzle**: blocks grouped along M, sharing B-column tiles.
- **N-swizzle**: blocks grouped along N, sharing A-row tiles.

### Anti-DCE Mechanism

```
Normal kernel:
   for k: load A, load B → MMA → accumulator
   ct.store(C, accumulator)    # full C write (tm×tn elements)

No-writeback kernel:
   for k: load A, load B → MMA → accumulator
   scalar = ct.sum(accumulator)  # reduce to 1 scalar
   ct.store(checksum, scalar)    # only 1 float32 (4B)
```

cuTile keeps computations only when consumed by side-effecting ops (`ct.store`). The `ct.sum → ct.store(checksum)` chain keeps the entire MMA path alive without writing full C.

### DDR Traffic
- **Theoretical read bytes**: $(M\times K + K\times N)$ elements × bytes-per-element (fp16=2B).
- **Actual read bytes**: collected via NCU (`lts__d_sectors_fill_sysmem.sum * 32`), per m/n kernels.
- **Write bytes**: checksum only (4B per block), versus full C write in normal matmul.

## Outputs

`parsed_results/` contains:
- `ncu_results_K_*.json` / `ncu_results_K_*.csv`: per-K parsed metrics (examples present for K=512/768/1024).
- `ncu_results_combined.json` / `ncu_results_combined.csv`: all parsed K merged.
- After running `analyze_ddr_ratio.py`: ratio plots and feature-point CSVs.

## Comparison Guidance

Compare against a normal writeback matmul to quantify C write impact:

| Metric | Normal | No-WriteBack | Meaning |
| --- | --- | --- | --- |
| DDR Write Bytes | M×N×2 bytes | ~0 (checksum only) | DDR cost of writing C |
| L2 Write Sectors | High | Minimal | L2 cost of writing C |
| Elapsed Cycles | Higher | Lower | Time impact from C write |
| DDR Read Bytes | Same | Same | Read path unchanged |

## License

Apache-2.0
