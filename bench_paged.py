import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paged_attn import paged_decode, reference, make_batch

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us

configs = [
    # (seqs, heads, kv_heads, head_dim, max_len)
    (32, 32, 8, 128, 1024),
    (64, 32, 8, 128, 2048),
    (128, 32, 8, 128, 4096),
]
print(f"{'S':>4} {'H':>3} {'KVH':>4} {'D':>4} {'maxL':>5} | {'triton us':>10} | {'torch us':>9} | speedup")
for S, H, KVH, D, L in configs:
    q, k, v, bt, sl, bs = make_batch(num_seqs=S, num_heads=H, num_kv_heads=KVH,
                                     head_dim=D, max_len=L)
    t_tri = bench(lambda: paged_decode(q, k, v, bt, sl, bs))
    t_ref = bench(lambda: reference(q, k, v, bt, sl, bs), iters=20)
    print(f"{S:>4} {H:>3} {KVH:>4} {D:>4} {L:>5} | {t_tri:>10.1f} | {t_ref:>9.1f} | {t_ref/t_tri:>6.1f}x")
