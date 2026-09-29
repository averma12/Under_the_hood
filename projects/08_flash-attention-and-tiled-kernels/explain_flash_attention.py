"""
Companion code for FLASH_ATTENTION_EXPLAINED.md.

Every claim in the explainer that can be checked numerically is checked here:

1. Streaming (online) softmax gives exactly the same answer as ordinary softmax.
2. Forgetting to rescale the running accumulator gives a silently wrong answer.
3. Softmax states can be merged in any order. Flash-Decoding (split-K) and
   FlashAttention-2's parallelism both rely on this.
4. A simple HBM-traffic model for naive vs. FlashAttention on an A100,
   including arithmetic intensity and roofline time estimates.
5. The backward pass only needs the per-row logsumexp. P is rebuilt exactly
   from it, and the gradients match autograd.

Run:
    python explain_flash_attention.py
"""

from __future__ import annotations

import math

import torch

# A100 SXM 40GB figures from NVIDIA's A100 datasheet (dense, no sparsity).
A100 = {
    "name": "A100 40GB SXM",
    "hbm_bytes_per_s": 1.555e12,
    "tensor_fp16_flops": 312e12,
    "fp32_flops": 19.5e12,
    "sms": 108,
    "smem_per_sm_bytes": 164 * 1024,
}


# ---------------------------------------------------------------------------
# 1. Three ways to compute the same softmax
# ---------------------------------------------------------------------------


def softmax_three_pass(x: torch.Tensor) -> torch.Tensor:
    """Safe softmax as most textbooks write it: find max, sum, normalize."""
    m = x.max()  # pass 1: read every element
    l = torch.exp(x - m).sum()  # pass 2: read every element again
    return torch.exp(x - m) / l  # pass 3: read again, write output


def softmax_online(x: torch.Tensor, block: int) -> torch.Tensor:
    """Online softmax: max and sum in ONE streaming pass, then normalize."""
    m = torch.tensor(float("-inf"), dtype=x.dtype)
    l = torch.tensor(0.0, dtype=x.dtype)
    for start in range(0, x.numel(), block):
        xb = x[start : start + block]
        m_new = torch.maximum(m, xb.max())
        # The old sum was computed relative to the old max. Re-express it
        # relative to the new max before adding this block's terms.
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new).sum()
        m = m_new
    return torch.exp(x - m) / l


def attention_row_streaming(
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block: int,
    rescale: bool = True,
    trace: list[dict] | None = None,
) -> torch.Tensor:
    """One query row of attention in a single pass over K/V blocks.

    Holds only (m, l, o): one running max, one running sum, one d-vector.
    With rescale=False this reproduces the classic hand-written-kernel bug.
    """
    d = q.numel()
    m = torch.tensor(float("-inf"), dtype=q.dtype)
    l = torch.tensor(0.0, dtype=q.dtype)
    o = torch.zeros(V.shape[1], dtype=q.dtype)
    for start in range(0, K.shape[0], block):
        s = (K[start : start + block] @ q) / math.sqrt(d)  # this block's scores
        m_new = torch.maximum(m, s.max())
        alpha = torch.exp(m - m_new) if rescale else torch.tensor(1.0, dtype=q.dtype)
        p = torch.exp(s - m_new)
        l = l * alpha + p.sum()
        o = o * alpha + p @ V[start : start + block]
        if trace is not None:
            trace.append(
                {"scores": s.tolist(), "m_old": m.item(), "m_new": m_new.item(),
                 "alpha": alpha.item(), "l": l.item()}
            )
        m = m_new
    return o / l  # normalize once at the end (FlashAttention-2 style)


def naive_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    S = Q @ K.T / math.sqrt(Q.shape[-1])
    if causal:
        T = S.shape[0]
        S = S.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    return torch.softmax(S, dim=-1) @ V


# ---------------------------------------------------------------------------
# 2. Softmax states are mergeable (the basis of split-K / Flash-Decoding)
# ---------------------------------------------------------------------------


def partial_state(
    q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(m, l, o_unnormalized) for one query over one chunk of keys."""
    s = K @ q / math.sqrt(q.numel())
    m = s.max()
    p = torch.exp(s - m)
    return m, p.sum(), p @ V


def merge_states(
    a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine two partial softmax states. Associative and commutative."""
    m_a, l_a, o_a = a
    m_b, l_b, o_b = b
    m = torch.maximum(m_a, m_b)
    ca, cb = torch.exp(m_a - m), torch.exp(m_b - m)
    return m, l_a * ca + l_b * cb, o_a * ca + o_b * cb


def split_kv_attention(
    q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_splits: int, reverse: bool = False
) -> torch.Tensor:
    """Flash-Decoding idea: each split could run on a different SM, then merge."""
    chunks = list(zip(K.chunk(num_splits), V.chunk(num_splits)))
    if reverse:
        chunks = chunks[::-1]
    states = [partial_state(q, k, v) for k, v in chunks]
    total = states[0]
    for s in states[1:]:
        total = merge_states(total, s)
    _, l, o = total
    return o / l


# ---------------------------------------------------------------------------
# 3. HBM traffic model
# ---------------------------------------------------------------------------


def attention_flops(N: int, d: int, causal: bool = False) -> float:
    """Matmul FLOPs of the forward pass for one head: QK^T plus PV."""
    flops = 4.0 * N * N * d
    return flops / 2 if causal else flops


def hbm_bytes_naive(N: int, d: int, bytes_per_el: int = 2) -> float:
    """Unfused PyTorch-style forward: GEMM -> softmax -> GEMM, each via HBM.

    GEMM 1   reads Q, K            writes S  (N x N)
    softmax  reads S               writes P  (N x N)
    GEMM 2   reads P, V            writes O
    Scale, mask and dropout kernels would each add another 2*N^2.
    """
    elements = (2 * N * d + N * N) + (N * N + N * N) + (N * N + N * d + N * d)
    return float(elements * bytes_per_el)


def hbm_bytes_flash(
    N: int,
    d: int,
    block_q: int = 128,
    bytes_per_el: int = 2,
    causal: bool = False,
    kv_from_l2: bool = False,
) -> float:
    """FlashAttention-2 forward: each Q block reads Q once and streams K, V.

    kv_from_l2=False is an upper bound: every Q block re-reads K/V from HBM.
    kv_from_l2=True is a lower bound: K/V leave HBM once and every re-read
    hits the L2 cache. Real kernels land in between.
    """
    num_q_blocks = 1 if kv_from_l2 else math.ceil(N / block_q)
    kv_passes = num_q_blocks * (0.5 if causal and not kv_from_l2 else 1.0)
    elements = N * d + 2 * N * d * kv_passes + N * d  # Q in, K/V streamed, O out
    return float(elements * bytes_per_el + N * 4)  # + fp32 logsumexp per row


def roofline_seconds(flops: float, hbm_bytes: float, gpu: dict = A100) -> float:
    """Lower bound on time: whichever of compute or memory traffic is slower."""
    return max(flops / gpu["tensor_fp16_flops"], hbm_bytes / gpu["hbm_bytes_per_s"])


# ---------------------------------------------------------------------------
# 4. Backward from logsumexp only
# ---------------------------------------------------------------------------


def flash_backward_from_lse(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, dO: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Gradients using only O and the logsumexp L saved by the forward pass.

    FlashAttention does this blockwise. It is written dense here so each
    equation is readable; the point is WHAT gets saved, not the tiling.
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = Q @ K.T * scale
    L = torch.logsumexp(S, dim=-1)  # the only per-row statistic saved: N floats
    O = torch.exp(S - L[:, None]) @ V

    # Backward: recompute P from Q, K, and L. No stored N x N matrix.
    P = torch.exp(Q @ K.T * scale - L[:, None])
    dV = P.T @ dO
    dP = dO @ V.T
    D = (dO * O).sum(dim=-1)  # rowsum(dO * O) == rowsum(P * dP)
    dS = P * (dP - D[:, None])  # softmax Jacobian applied row by row
    dQ = dS @ K * scale
    dK = dS.T @ Q * scale
    return dQ, dK, dV


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    torch.manual_seed(0)

    _section("1. Streaming softmax on scores [3, 1, 5, 2], blocks of 2")
    x = torch.tensor([3.0, 1.0, 5.0, 2.0], dtype=torch.float64)
    m, l = float("-inf"), 0.0
    for start in (0, 2):
        xb = x[start : start + 2]
        m_new = max(m, xb.max().item())
        alpha = math.exp(m - m_new) if m != float("-inf") else 0.0
        l = l * alpha + torch.exp(xb - m_new).sum().item()
        print(f"  block {xb.tolist()}: m {m} -> {m_new}, alpha={alpha:.4f}, l={l:.4f}")
        m = m_new
    print(f"  direct sum exp(x - 5) = {torch.exp(x - 5).sum().item():.4f}  (matches l)")
    print(f"  softmax 3-pass : {softmax_three_pass(x).numpy().round(4)}")
    print(f"  softmax online : {softmax_online(x, block=2).numpy().round(4)}")

    _section("2. One query row, attention in one pass (T=256, d=64, block=32)")
    T, d = 256, 64
    Q, K, V = (torch.randn(T, d, dtype=torch.float64) for _ in range(3))
    # Put the largest score in a late block so rescaling actually matters.
    K[200] = Q[0] * 3
    ref = naive_attention(Q[:1], K, V)[0]
    good = attention_row_streaming(Q[0], K, V, block=32)
    bad = attention_row_streaming(Q[0], K, V, block=32, rescale=False)
    print(f"  with rescale   : max |diff| vs naive = {(good - ref).abs().max():.2e}")
    print(f"  WITHOUT rescale: max |diff| vs naive = {(bad - ref).abs().max():.2e}  <- silent bug")

    _section("3. Merge partial states in any order (split-K / Flash-Decoding)")
    q = torch.randn(d, dtype=torch.float64)
    ref = naive_attention(q[None], K, V)[0]
    for splits in (1, 4, 16):
        fwd = split_kv_attention(q, K, V, splits)
        rev = split_kv_attention(q, K, V, splits, reverse=True)
        print(f"  {splits:>2} splits: fwd diff {(fwd - ref).abs().max():.1e}, "
              f"reversed-order diff {(rev - ref).abs().max():.1e}")

    _section(f"4. HBM traffic per head, forward, fp16, {A100['name']}")
    print(f"  ridge point = {A100['tensor_fp16_flops'] / A100['hbm_bytes_per_s']:.0f} "
          "FLOP/byte (below this, memory-bound)")
    print("  flash-hi: every Q block re-reads K/V from HBM (upper bound)")
    print("  flash-lo: K/V leave HBM once, re-reads hit L2 (lower bound)")
    print(f"  {'N':>6} {'d':>4} | {'naive MB':>8} {'AI':>5} {'t_min us':>8} | "
          f"{'flash-hi MB':>11} {'AI':>5} {'t_min us':>8} | {'flash-lo MB':>11} {'AI':>5} "
          f"{'t_min us':>8}")
    for N, dh in ((1024, 64), (4096, 64), (4096, 128), (16384, 128)):
        f = attention_flops(N, dh)
        bn = hbm_bytes_naive(N, dh)
        hi = hbm_bytes_flash(N, dh, block_q=128)
        lo = hbm_bytes_flash(N, dh, kv_from_l2=True)
        cols = [f"{b / 1e6:>{w}.1f} {f / b:>5.0f} {roofline_seconds(f, b) * 1e6:>8.1f}"
                for b, w in ((bn, 8), (hi, 11), (lo, 11))]
        print(f"  {N:>6} {dh:>4} | " + " | ".join(cols))
    print("  AI = arithmetic intensity (FLOP per HBM byte).")
    print("  naive AI ~ d/2 (stuck below ridge); flash-hi AI ~ block_q; flash-lo AI ~ N/2.")

    _section("5. Backward from logsumexp only (no stored N x N matrix)")
    T, d = 64, 32
    Qg, Kg, Vg = (torch.randn(T, d, dtype=torch.float64, requires_grad=True) for _ in range(3))
    dO = torch.randn(T, d, dtype=torch.float64)
    naive_attention(Qg, Kg, Vg).backward(dO)
    ours = flash_backward_from_lse(Qg.detach(), Kg.detach(), Vg.detach(), dO)
    for name, mine, auto in zip(("dQ", "dK", "dV"), ours, (Qg.grad, Kg.grad, Vg.grad)):
        print(f"  {name}: max |ours - autograd| = {(mine - auto).abs().max():.2e}")
    print(f"  saved for backward: naive P = {T * T} floats, flash L = {T} floats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
