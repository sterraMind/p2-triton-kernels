# p2-triton：P2-1 Triton 算子练习产物

目的：补 7 份 JD 里 4 份要求的 CUDA/Triton 缺口（job-search-ops.md P2 第 1 项）。
环境：Windows + triton-windows 3.9.0.post30 + torch 2.11.0+cu128，GPU RTX 3070 Ti Laptop 8GB（sm_86）。
venv：`resume\.tmp-run\p2-triton-20261117-0001\.venv`（不进仓库，重建命令见下）。

```
uv venv --python 3.12 resume\.tmp-run\p2-triton-20261117-0001\.venv
uv pip install --python <venv>\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python <venv>\Scripts\python.exe triton-windows numpy
```

## 产物

| 文件 | 内容 | 结果 |
|---|---|---|
| check_env.py | 环境自检 | torch 2.11.0+cu128 / triton 3.9.0 / sm 8.6 / 8.0GB |
| smoke_add.py | 向量加法冒烟 | max diff 0.0 |
| rmsnorm.py | 单 pass RMSNorm fused 算子（归约 + rstd + 仿射） | n_cols=512/4096/8192，max diff ≤2.9e-06，全 allclose |
| bench_rmsnorm.py | (4096,8192) fp32 ×200 iters | Triton 734.4us（365.5GB/s，81.6% 峰值带宽）vs torch 1209.0us，**1.65×** |
| sweep_warps.py | num_warps 4/8/16/32 扫描 | nw=8 最优 726.7us/369.4GB/s；nw=32 反而降速（790.1us） |
| paged_attn.py | 分页 KV-cache 解码注意力（PagedAttention 解码阶段复刻） | 32 批 ×1024 上下文 max diff 1.2e-04，allclose(1e-2) |
| bench_paged.py | 与 torch 参考实现对比 | S32/L1024 40.0×，S64/L2048 15.4×，S128/L4096 11.2×（参考实现是逐序列 Python 循环，倍数仅证明算子本身能跑，不代表对 vLLM 的优势） |

## 已知边界

- Nsight Compute 2026.4.0 已装（winget），能连进程；硬件计数器被 Windows 默认权限挡住（ERR_NVGPUCTRPERM，RmProfilingAdminOnly=1），用户选择暂不开，调优走手工计时 + num_warps 扫描。
- paged attention 的参考实现是逐序列 Python 循环，40×/15×/11× 只证明算子相对朴素实现有效；要跟 vLLM 比需要接真 PagedAttention（列在 P2-1 后续）。
