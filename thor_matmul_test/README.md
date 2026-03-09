# MatMul_K: CUDA Tile Matrix Multiplication Performance Analysis

## 项目概述 (Project Overview)

matmul_k 是一个用于分析和测试 CUDA Tile (cuTile) 矩阵乘法性能的工程。该项目专注于研究不同矩阵维度（特别是 K 维度）对矩阵乘法性能的影响，并通过 NVIDIA Nsight Compute (NCU) 进行深度性能分析。

This project provides a comprehensive framework for analyzing CUDA Tile matrix multiplication performance across different matrix dimensions, with a focus on understanding memory access patterns and L2 cache behavior.

## 主要功能 (Key Features)

- **多种 Swizzle 策略**: 实现了基于 M 维度和 N 维度的两种 block swizzle 策略
- **L2 缓存控制**: 使用 CUDA L2 Access Policy API 优化缓存行为，减少读取放大
- **自动化性能分析**: 通过脚本自动化进行大规模性能测试和数据收集
- **NCU 集成**: 深度集成 NVIDIA Nsight Compute 进行性能指标收集
- **DDR 比率分析**: 分析实际 DDR 访问与理论值的比率，识别性能拐点

## 文件结构 (File Structure)

```
matmul_k/
├── NormalSwizzleM.py         # M 维度 swizzle 的矩阵乘法 kernel
├── NormalSwizzleN.py         # N 维度 swizzle 的矩阵乘法 kernel
├── L2CacheCtrl.py            # L2 缓存访问策略控制工具
├── MatMulPerf.py             # 主性能测试脚本
├── run_matmul_ncu.sh         # NCU 自动化测试脚本
├── parse_ncu_results.py      # NCU 报告解析工具
├── analyze_ddr_ratio.py      # DDR 访问比率分析和可视化工具
├── ncu_results/              # NCU 原始报告文件 (.ncu-rep)
└── parsed_results/           # 解析后的性能数据 (JSON/CSV)
```

## 核心组件 (Core Components)

### 1. Kernel 实现 (Kernel Implementations)

#### NormalSwizzleM.py
实现基于 M 维度分组的 block swizzle 策略。多个 block 共享 B 矩阵的列瓦片，适合使用 L2 缓存持久化策略来缓存 B 矩阵。

```python
# 使用示例
cutile_matmul(A, B, matmul_kernel=NormalSwizzleM.matmul_kernel_m, l2_persist_input='B')
```

#### NormalSwizzleN.py
实现基于 N 维度分组的 block swizzle 策略。多个 block 共享 A 矩阵的行瓦片，适合使用 L2 缓存持久化策略来缓存 A 矩阵。

```python
# 使用示例
cutile_matmul(A, B, matmul_kernel=NormalSwizzleN.matmul_kernel_n, l2_persist_input='A')
```

### 2. L2 缓存控制 (L2 Cache Control)

[L2CacheCtrl.py](L2CacheCtrl.py) 提供了 CUDA L2 访问策略的 Python 接口：

- **Persisting 策略**: 将频繁访问的数据标记为持久化在 L2 缓存中
- **Streaming 策略**: 将一次性访问的数据（如输出矩阵）快速驱逐出 L2
- 支持 CUDA 计算能力 8.0+ 的设备

### 3. 性能测试框架 (Performance Testing Framework)

#### MatMulPerf.py
主性能测试脚本，支持：
- 自定义矩阵维度 (M, N, K)
- 可选的正确性检查
- L2 缓存持久化控制
- 支持 float16 和 float32 数据类型

```bash
# 运行示例
python3 MatMulPerf.py --M 2048 --N 2048 --K 1024 --correctness-check --l2-persist-input
```

#### run_matmul_ncu.sh
自动化 NCU 性能测试脚本：
- 遍历 K 维度: 512 → 4096 (步长 256)
- 遍历 M=N 维度: 256 → 8192 (步长 256)
- 为每个 K 值创建独立的结果目录
- 自动生成 NCU 报告文件

### 4. 数据分析工具 (Analysis Tools)

#### parse_ncu_results.py
解析 NCU 报告文件，提取关键性能指标：
- Kernel 执行周期数
- DDR 读写字节数（实际值）
- 计算理论 DDR 访问量
- 输出为 JSON 和 CSV 格式

#### analyze_ddr_ratio.py
分析 DDR 访问比率并生成可视化：
- 计算实际/理论 DDR 读取比率
- 识别性能拐点（L2 缓存不足的临界点）
- 生成 3D 可视化图表 (M × K → DDR Ratio)
- 输出关键数据点到 CSV

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

2. **使用 L2 缓存优化**:
```bash
python3 MatMulPerf.py --M 4096 --N 4096 --K 2048 --l2-persist-input
```

3. **运行完整的 NCU 分析**:
```bash
./run_matmul_ncu.sh
```

4. **解析 NCU 结果**:
```bash
# 解析单个 K 值的结果
python3 parse_ncu_results.py ncu_results/K_1024

# 解析所有 K 值的结果
python3 parse_ncu_results.py ncu_results
```

5. **生成 DDR 比率分析**:
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
   python3 parse_ncu_results.py ncu_results
   ```
   - 提取所有 .ncu-rep 文件的性能指标
   - 生成 `parsed_results/` 目录下的 JSON 和 CSV 文件

3. **分析和可视化**:
   ```bash
   python3 analyze_ddr_ratio.py
   ```
   - 计算 DDR 读取放大比率
   - 识别 L2 缓存容量不足的拐点
   - 生成 3D 可视化图表

## 关键概念 (Key Concepts)

### Block Swizzle 策略

**M-Swizzle** (NormalSwizzleM.py):
- Block 按 M 维度分组
- 同组内的 block 共享 B 矩阵的列瓦片
- 适合 L2 持久化 B 矩阵

**N-Swizzle** (NormalSwizzleN.py):
- Block 按 N 维度分组  
- 同组内的 block 共享 A 矩阵的行瓦片
- 适合 L2 持久化 A 矩阵

### L2 缓存策略

通过 CUDA L2 Access Policy Window API 控制缓存行为：

- **Persisting**: 输入矩阵 (A 或 B) → 保留在 L2 中以供重用
- **Normal**: 默认策略
- **Streaming**: 输出矩阵 (C) → 快速驱逐，不污染 L2

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

## 性能调优建议 (Performance Tuning Tips)

1. **选择合适的 Swizzle 策略**: 根据矩阵形状选择 M-swizzle 或 N-swizzle
2. **启用 L2 持久化**: 对于大矩阵乘法，使用 `--l2-persist-input` 可显著减少 DDR 访问
3. **关注 DDR 比率**: 比率 > 1.05 表示可能存在 L2 缓存不足
4. **调整 Tile 大小**: 在 `cutile_matmul()` 中根据数据类型自动选择最优 tile 大小

## 许可证 (License)

Apache-2.0

---

**注意**: 该项目是 cutile-python 示例的一部分，用于演示 CUDA Tile 编程模型的高级用法和性能分析技术。
