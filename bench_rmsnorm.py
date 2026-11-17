import time, torch, triton
import triton.language as tl
from rmsnorm import rmsnorm

torch.manual_seed(0)
rows, n_cols = 4096, 8192
x = torch.randn(rows, n_cols, device="cuda", dtype=torch.float32)
w = torch.randn(n_cols, device="cuda", dtype=torch.float32)

def bench(fn, iters=200, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

bytes_moved = rows * n_cols * 4 * 2  # read x + write y
t_triton = bench(lambda: rmsnorm(x, w))
t_torch  = bench(lambda: torch.nn.functional.rms_norm(x, (n_cols,), weight=w, eps=1e-6))
bw_triton = bytes_moved / t_triton / 1e9
bw_torch  = bytes_moved / t_torch / 1e9
print(f"triton rmsnorm : {t_triton*1e6:8.1f} us  | {bw_triton:6.1f} GB/s")
print(f"torch rms_norm : {t_torch*1e6:8.1f} us  | {bw_torch:6.1f} GB/s")
print(f"speedup (torch/triton): {t_torch/t_triton:.2f}x")
print(f"RTX 3070 Ti Laptop peak bandwidth ~448 GB/s; utilization {bw_triton/448*100:.1f}%")
