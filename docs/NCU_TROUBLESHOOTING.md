# Nsight Compute (`ncu`) 排障与固定配置

本文件记录当前仓库在这台机器上可用的 `ncu` 配置、常见故障现象以及已验证的解决方案。

适用场景：

- `ncu` 启动后卡住
- `ncu` 报驱动不兼容
- WSL / Python / PyTorch 路径下无法正常采样
- 以后需要重新做 matmul / transpose 的 `ncu` 分析

## 当前已验证可用的组合

机器现状（本次验证时）：

- GPU: `NVIDIA GeForce RTX 4070`
- Driver: `576.88`
- `nvidia-smi` 报告 CUDA: `12.9`
- PyTorch: `2.8.0+cu128`
- `nvcc`: `12.8`

结论：

- `nsight-compute 2025.3.1` 在当前驱动上会失败
- `nsight-compute 2025.2.1` 可以正常使用

## 已知坏配置

以下组合在当前机器上失败：

- `nsight-compute 2025.3.1`

典型报错：

- `Cuda driver is not compatible with Nsight Compute`
- 随后可能伴随 Python / PyTorch 进程段错误

不要再直接用：

- `/home/yuliu/miniconda3/pkgs/nsight-compute-2025.3.1.4-hcd14d4a_0/nsight-compute-2025.3.1/ncu`

## 推荐固定配置

统一使用：

- `ncu`:
  - `/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu`
- 环境变量:
  - `NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds`
- 推荐参数:
  - `--profile-from-start off`
  - `--target-processes application-only`

如果目标脚本使用 `cudaProfilerStart/Stop` 包围 profile window，必须配合：

- `--profile-from-start off`

否则常见结果是：

- `==WARNING== No kernels were profiled.`

## 通用排障顺序

### 1. 先确认驱动和工具版本

```bash
nvidia-smi
/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu --version
/home/yuliu/miniconda3/pkgs/nsight-compute-2025.3.1.4-hcd14d4a_0/nsight-compute-2025.3.1/ncu --version
```

### 2. 先用旧版 `ncu` 跑一个 native CUDA smoke test

```bash
NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds \
/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu \
  --section LaunchStats \
  --section SpeedOfLight \
  --section Occupancy \
  -o /tmp/ncu_native_smoke \
  ./build/transpose_benchmark 512 512 1 1
```

如果这个都不通，先不要怀疑 matmul 脚本，优先怀疑：

- `ncu` 版本
- sandbox / 权限
- 驱动兼容性

### 3. 再跑 Python / PyTorch 路径

推荐模板：

```bash
NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds \
/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu \
  --profile-from-start off \
  --target-processes application-only \
  --section LaunchStats \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  -o results/<report_name> \
  /home/yuliu/miniconda3/bin/python profile_target.py \
    --impl <impl> \
    --m <M> --n <N> --k <K> \
    --warmup 1 \
    --iters 3 \
    --init-on-cpu \
    --profile-window \
    --skip-correctness
```

## 本仓库已验证可用的 matmul `ncu` 命令

在 `experiments/matmul_ptx` 目录下运行：

### PyTorch 2048

```bash
NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds \
/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu \
  --profile-from-start off \
  --target-processes application-only \
  --section LaunchStats \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  -o results/ncu_stage5_pytorch_2048 \
  /home/yuliu/miniconda3/bin/python profile_target.py \
    --impl pytorch \
    --m 2048 --n 2048 --k 2048 \
    --warmup 1 \
    --iters 3 \
    --init-on-cpu \
    --profile-window \
    --skip-correctness
```

### PTX `cpasync_128` 2048

```bash
NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds \
/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu \
  --profile-from-start off \
  --target-processes application-only \
  --section LaunchStats \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  -o results/ncu_stage6_cpasync128_2048 \
  /home/yuliu/miniconda3/bin/python profile_target.py \
    --impl cpasync_128 \
    --m 2048 --n 2048 --k 2048 \
    --warmup 1 \
    --iters 3 \
    --init-on-cpu \
    --profile-window \
    --skip-correctness
```

## 为什么这些参数重要

- `NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds`
  - 避免 WSL / 本地连接路径问题
- `--profile-from-start off`
  - 让 `cudaProfilerStart/Stop` 真正控制采样窗口
- `--target-processes application-only`
  - 减少无关子进程干扰
- `--init-on-cpu`
  - 把初始化放在 CPU 端，再显式拷到 GPU，减少 profile 前杂波
- `--skip-correctness`
  - 做 `ncu` 时只保留目标 kernel，避免 correctness 路径引入额外 kernel

## 遇到相似问题时的快速判断

### 现象 1

- 报 `Cuda driver is not compatible with Nsight Compute`

处理：

- 直接切到 `ncu 2025.2.1`
- 不要继续重试 `2025.3.1`

### 现象 2

- `ncu` 看似连接成功，但长时间卡住

处理：

- 确认是否在沙箱内运行
- 优先用脱离沙箱的方式运行
- 加上 `NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds`

### 现象 3

- `No kernels were profiled`

处理：

- 检查是否用了 `cudaProfilerStart/Stop`
- 如果用了，就必须加 `--profile-from-start off`

### 现象 4

- Python/PyTorch 路径有问题，但 native CUDA 程序正常

处理：

- 先保持旧版 `ncu`
- 用 `profile_target.py --init-on-cpu --profile-window --skip-correctness`
- 缩到 `512` 或 `1024` 小尺寸先跑通

## 参考结果文件

本轮新生成和验证过的报告：

- `experiments/matmul_ptx/results/ncu_stage5_pytorch_2048.ncu-rep`
- `experiments/matmul_ptx/results/ncu_stage5_cpasync128_2048.ncu-rep`
- `experiments/matmul_ptx/results/ncu_stage6_cpasync128_2048.ncu-rep`

相关 benchmark：

- `experiments/matmul_ptx/results/bench_stage6_2048.json`
- `experiments/matmul_ptx/results/bench_stage6_4096.json`

## 后续约定

以后仓库里凡是要跑 `ncu`：

- 默认先查本文件
- 默认优先使用旧版 `ncu 2025.2.1`
- 默认带上 `NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE=uds`
- 若脚本中使用 `profile window`，默认带 `--profile-from-start off`
