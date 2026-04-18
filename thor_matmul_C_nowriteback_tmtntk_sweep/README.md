# MatMul No-WriteBack tm/tn/tk Sweep

## 项目概述

`thor_matmul_C_nowriteback_tmtntk_sweep` 是和 `thor_matmul_tmtntk_sweep`
同风格的 no-writeback 平行实验目录，用来扫描：

```python
tm = tn in {64, 128, 256, 512}
tk = 64
```

和正常写回版不同，这个目录不会把完整 `C` 写回全局内存。每个 block 只把
一个 fp32 checksum 写回，用来防止 cuTile 把 MMA 计算做 DCE。

## 主要功能

- 保持 `M-swizzle` / `N-swizzle` 两个 no-writeback kernel
- 支持 `--tile-m` / `--tile-n` / `--tile-k`
- 批量扫描同步变化的 `tm=tn`
- 结果目录按 `tm_<tm>_tn_<tn>_tk_<tk>/K_<K>/...` 组织
- `parse_ncu_results.py` 会把 `tile_m` / `tile_n` / `tile_k` 写入结果
- 理论写回量按 checksum buffer 计算，而不是完整 `C`

## 核心 no-writeback 机制

```text
load A tile + load B tile -> mma accumulate -> sum(accumulator) -> store(checksum)
```

- 完整保留 load + mma 路径
- 不执行 `ct.store(C, ...)`
- 每个 block 只写 1 个 fp32 checksum

## 快速开始

单点测试示例：

```bash
python3 MatMulPerf.py --M 2048 --N 2048 --K 1024 --tile-m 128 --tile-n 128 --tile-k 64 --correctness-check
```

运行完整批量实验：

```bash
bash run_matmul_ncu.sh
```

解析结果：

```bash
python3 parse_ncu_results.py
```

绘制 DDR read ratio：

```bash
python3 analyze_ddr_ratio.py
```

## 输出组织

```text
ncu_results/
└── tm_256_tn_256_tk_64/
    ├── K_512/
    ├── K_1024/
    └── ...
```

单个报告文件命名格式：

```text
MatMulPerf_tm256_tn256_tk64_M2048_N2048_K1024.ncu-rep
```

## 说明

- 本目录是 no-writeback 版，不是正常写回版
- 默认仍使用 float16 输入和 float32 累加
- `--l2-persist-input` 可选打开，与正常 sweep 目录保持同类接口
