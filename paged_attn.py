"""Paged KV-cache decode attention, Triton 实现。

对应 vLLM 的 PagedAttention 解码阶段：KV 缓存按固定大小 block 分页存放，
每个序列持有一张 block_table（逻辑页 -> 物理页），算子按页遍历，不要求
KV 在显存里连续。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr, o_ptr,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_bt_s, stride_bt_b,
    stride_os, stride_oh, stride_od,
    group_size, head_dim, block_size,
    scale,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // group_size

    off_d = tl.arange(0, BLOCK_D)
    d_mask = off_d < head_dim
    q = tl.load(q_ptr + seq * stride_qs + head * stride_qh + off_d * stride_qd,
                mask=d_mask, other=0.0)

    seq_len = tl.load(sl_ptr + seq)
    num_blocks = tl.cdiv(seq_len, block_size)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    off_t = tl.arange(0, BLOCK_T)
    t_mask = off_t < block_size

    for bi in range(0, num_blocks):
        phys = tl.load(bt_ptr + seq * stride_bt_s + bi * stride_bt_b)
        k_ptrs = (k_ptr + phys * stride_kb + off_t[:, None] * stride_ks
                  + kv_head * stride_kh + off_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=t_mask[:, None] & d_mask[None, :], other=0.0)
        v_ptrs = (v_ptr + phys * stride_vb + off_t[:, None] * stride_vs
                  + kv_head * stride_vh + off_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=t_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q[None, :], tl.trans(k)) * scale
        qk = tl.reshape(qk, [BLOCK_T])

        pos = bi * block_size + off_t
        qk = tl.where(pos < seq_len, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    out = acc / l_i
    tl.store(o_ptr + seq * stride_os + head * stride_oh + off_d * stride_od,
             out, mask=d_mask)


def paged_decode(q, k_cache, v_cache, block_tables, seq_lens, block_size, scale=None):
    """q: [S, H, D]  k_cache/v_cache: [NB, BS, HKV, D]  block_tables: [S, MB] int32"""
    S, H, D = q.shape
    NB, BS, HKV, _ = k_cache.shape
    MB = block_tables.shape[1]
    group = H // HKV
    out = torch.empty_like(q)
    if scale is None:
        scale = D ** -0.5
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_T = triton.next_power_of_2(block_size)
    _paged_decode_kernel[(S, H)](
        q, k_cache, v_cache, block_tables, seq_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        group, D, block_size, scale,
        BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T,
        num_warps=4, num_stages=2,
    )
    return out


def reference(q, k_cache, v_cache, block_tables, seq_lens, block_size):
    """按 block_table 把 KV 收集成连续张量，再用 torch 做同样的注意力计算。"""
    outs = []
    S, H, D = q.shape
    _, _, HKV, _ = k_cache.shape
    group = H // HKV
    for s in range(S):
        L = int(seq_lens[s])
        nb = (L + block_size - 1) // block_size
        phys = block_tables[s, :nb].long()
        k = k_cache[phys].reshape(-1, HKV, D)[:L]      # [L, HKV, D]
        v = v_cache[phys].reshape(-1, HKV, D)[:L]
        qs = q[s].float()                               # [H, D]
        kk = k.float().repeat_interleave(group, dim=1)   # [L, H, D]
        vv = v.float().repeat_interleave(group, dim=1)
        score = torch.einsum("hd,lhd->hl", qs, kk) * (D ** -0.5)
        p = torch.softmax(score, dim=-1)
        outs.append(torch.einsum("hl,lhd->hd", p, vv))
    return torch.stack(outs).to(q.dtype)


def make_batch(num_seqs=32, num_heads=32, num_kv_heads=8, head_dim=128,
               block_size=16, max_len=1024, seed=0):
    torch.manual_seed(seed)
    max_blocks = (max_len + block_size - 1) // block_size
    num_phys = num_seqs * max_blocks
    total_blocks = (num_phys + block_size - 1) // block_size * block_size
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                          device="cuda", dtype=torch.float16) * 0.1
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(num_seqs, num_heads, head_dim, device="cuda", dtype=torch.float16)
    seq_lens = torch.randint(1, max_len + 1, (num_seqs,), device="cuda", dtype=torch.int32)
    # 每序列分配互不重叠的物理页，模拟分页后的碎片布局
    perm = torch.randperm(total_blocks, device="cuda")
    tables = torch.zeros(num_seqs, max_blocks, device="cuda", dtype=torch.int32)
    idx = 0
    for s in range(num_seqs):
        nb = int((int(seq_lens[s]) + block_size - 1) // block_size)
        tables[s, :nb] = perm[idx:idx + nb]
        idx += nb
    return q, k_cache, v_cache, tables, seq_lens, block_size


if __name__ == "__main__":
    q, k, v, bt, sl, bs = make_batch()
    out = paged_decode(q, k, v, bt, sl, bs)
    ref = reference(q, k, v, bt, sl, bs)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"max diff: {diff:.3e}  allclose(1e-2): {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
