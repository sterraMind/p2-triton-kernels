# Triton kernels: fused RMSNorm and paged KV-cache decode attention

Two kernels used in LLM inference, written in Triton, verified against PyTorch references and benchmarked on a single consumer GPU. The goal is to keep the numbers reproducible and to state the limits of each measurement.

## Environment

- Windows 11, Python 3.12
- torch 2.11.0+cu128, triton-windows 3.9.0.post30, numpy 2.5.5
- GPU: RTX 3070 Ti Laptop, 8 GB, sm_86 (peak HBM bandwidth used for the utilization figures: 448 GB/s)

Set up with uv:

```
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe triton-windows numpy
python check_env.py
```

## Files

| File | What it does | Result on the GPU above |
|---|---|---|
| `check_env.py` | prints torch / triton / GPU capability | torch 2.11.0+cu128, triton 3.9.0, sm 8.6, 8.0 GB |
| `smoke_add.py` | vector add, checks the compile-and-run path | max diff 0.0 |
| `rmsnorm.py` | single-pass fused RMSNorm (sum of squares, rstd, affine) | n_cols 512 / 4096 / 8192: max diff ≤ 2.9e-06, allclose |
| `bench_rmsnorm.py` | (4096, 8192) fp32, 200 iterations | Triton 734.4 us / 365.5 GB/s / 81.6% of peak vs torch 1209.0 us (222 GB/s), 1.65x |
| `sweep_warps.py` | num_warps 4 / 8 / 16 / 32 | 8 warps best (726.7 us, 369.4 GB/s); 32 warps slower (790.1 us) |
| `paged_attn.py` | paged KV-cache decode attention: block table indirection, online softmax, grouped-query attention, tail masking | 32 sequences x 1024 context, GQA 32/8, head dim 128: max diff 1.2e-04, allclose at 1e-2 |
| `bench_paged.py` | paged decode against a per-sequence reference loop | 40.0x at S32/L1024, 15.4x at S64/L2048, 11.2x at S128/L4096 |

## How the kernels work

`rmsnorm.py` reads a row once, accumulates the sum of squares in fp32, computes `rstd = 1/sqrt(mean(x^2) + eps)` and writes `x * rstd * weight` in the same pass. One row per program, `BLOCK_N` rounded up to the next power of two, the tail masked. Throughput is bandwidth-bound at this shape, so the measured 81.6% of peak is the number that matters, not occupancy.

`paged_attn.py` implements the decode phase of paged attention. The KV cache is stored as fixed-size pages in a physical pool; a `block_table` maps each sequence's logical positions to physical pages, so sequences of different lengths can share one pool without contiguous allocation. Each program handles one query head of one sequence: it walks logical positions, resolves each to a physical page through the block table, loads K and V, and accumulates with online softmax so the full attention row never has to be materialized. Grouped-query attention is handled by mapping a query head to its KV head with `head // group_size`. Batch construction in `bench_paged.py` uses a random permutation of physical pages so the layout looks like a fragmented pool.

## Reproducing

```
python check_env.py
python smoke_add.py
python rmsnorm.py
python bench_rmsnorm.py
python sweep_warps.py
python paged_attn.py
python bench_paged.py
```

Each script prints its own reference comparison and, for the benchmarks, microseconds, achieved GB/s and the ratio to the PyTorch implementation.

## Limits of these measurements

- The paged-attention reference is a per-sequence Python loop, not vLLM. The 40x / 15x / 11x figures describe this kernel against a naive loop; they do not say anything about performance relative to vLLM's implementation.
- Hardware counters are not available on this machine: Nsight Compute connects to the process but `ERR_NVGPUCTRPERM` blocks the profiling counters, and enabling them needs an administrator change to the driver setting plus a reboot. Tuning here is therefore done with wall-clock timing and a `num_warps` sweep, not with memory-throughput counters.
- `rmsnorm.py` covers fp32 only. fp16 and bf16 paths are not implemented.
- The decode kernel handles a single query token per sequence, which is the decode case. Prefill (many query tokens, chunked) is not covered.
- Absolute numbers come from one laptop GPU with a shared power budget; they are useful for comparing implementations on the same machine, not for cross-hardware claims.
