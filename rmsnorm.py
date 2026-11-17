import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr,
                   n_cols, eps,
                   BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * n_cols
    out_ptr += row * n_cols
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + offs, y, mask=mask)

def rmsnorm(x, w, eps=1e-6):
    # x: (rows, n_cols); w: (n_cols,)
    out = torch.empty_like(x)
    rows, n_cols = x.shape
    BLOCK_N = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(rows,)](x, w, out, n_cols, eps, BLOCK_N=BLOCK_N, num_warps=8)
    return out

torch.manual_seed(0)
for n_cols in (512, 4096, 8192):
    rows = 4096
    x = torch.randn(rows, n_cols, device="cuda", dtype=torch.float32)
    w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
    # 对齐 nn.RMSNorm（无 weight 的纯归一化 + 手动乘 w）
    ref = torch.nn.functional.rms_norm(x, (n_cols,), weight=w, eps=1e-6)
    got = rmsnorm(x, w)
    diff = (got - ref).abs().max().item()
    print(f"n_cols={n_cols}: max diff {diff:.3e}  allclose={torch.allclose(got, ref, atol=1e-4, rtol=1e-4)}")
