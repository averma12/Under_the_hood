# FlashAttention, explained from the GPU up

> My notes from working through Chapter 8 of *Under the Hood* ([buy the book](https://leanpub.com/under-the-hood)). The chapter's [`build.py`](build.py) shows the tiled schedule on CPU. These notes go one level lower. They cover what the attention equation physically does on a GPU, why the ordinary version is slow, and what FlashAttention changes.
>
> Every numeric claim below is checked by [`explain_flash_attention.py`](explain_flash_attention.py) and its tests in [`tests/test_explain_flash_attention.py`](tests/test_explain_flash_attention.py). Run it yourself:
>
> ```bash
> python projects/08_flash-attention-and-tiled-kernels/explain_flash_attention.py
> ```

**The one-sentence version:** FlashAttention computes *exactly* the same numbers as normal attention. It reorders the work so the big `N × N` score matrix never travels to or from GPU main memory. The trick that makes the reordering legal is the **streaming (online) softmax**.

---

## Contents

1. [The equation and its shapes](#1-the-equation-and-its-shapes)
2. [A GPU in ten minutes](#2-a-gpu-in-ten-minutes)
3. [How normal attention actually runs on a GPU](#3-how-normal-attention-actually-runs-on-a-gpu)
4. [Streaming softmax](#4-streaming-softmax)
5. [FlashAttention: the tiled forward pass](#5-flashattention-the-tiled-forward-pass)
6. [The backward pass: recompute instead of store](#6-the-backward-pass-recompute-instead-of-store)
7. [FlashAttention-2 and -3: what changed](#7-flashattention-2-and--3-what-changed)
8. [Inference: decoding and Flash-Decoding](#8-inference-decoding-and-flash-decoding)
9. [Other GPU concepts worth knowing](#9-other-gpu-concepts-worth-knowing)
10. [How this connects to my experiments](#10-how-this-connects-to-my-experiments)
11. [Common misconceptions](#11-common-misconceptions)
12. [References](#12-references)

---

## 1. The equation and its shapes

For one attention head:

```
S = Q Kᵀ / √d          scores         (N × N)
P = softmax(S)         row-wise       (N × N)
O = P V                output         (N × d)
```

| Symbol | Shape | Meaning |
|---|---|---|
| `N` | – | sequence length (tokens) |
| `d` | – | head dimension, usually 64 or 128 |
| `Q, K, V` | `N × d` | queries, keys, values |
| `S, P` | `N × N` | scores and attention weights |
| `O` | `N × d` | output |

Two facts drive everything else:

- **Work grows as N²d.** `QKᵀ` costs `2N²d` FLOPs (one multiply and one add per term) and `PV` costs another `2N²d`. So the forward pass is about `4N²d` FLOPs per head.
- **The inputs and output are small, but the middle is huge.** `Q`, `K`, `V`, and `O` are `N × d`. `S` and `P` are `N × N`. With `N = 8192` and `d = 64`, `S` is 128× bigger than `Q`.

The middle matrices are temporary. Nobody needs `S` or `P` after the output is computed. The goal of FlashAttention is to never write them to main memory at all.

---

## 2. A GPU in ten minutes

### 2.1 The compute side: SMs, warps, threads

```
GPU (A100: 108 SMs)
├── SM 0  ─┬─ 4 warp schedulers
│          ├─ tensor cores  (matrix-multiply units)
│          ├─ FP32 / INT units, special-function units (exp, log)
│          ├─ register file  256 KB
│          └─ L1 / shared memory  192 KB (up to 164 KB usable as shared)
├── SM 1 ...
└── SM 107
        │
     L2 cache  40 MB  (shared by all SMs)
        │
     HBM  40–80 GB  ~1.5–2.0 TB/s  ("GPU memory", "global memory")
```

- A **thread** is one lane of execution.
- A **warp** is 32 threads that execute the same instruction in lockstep (SIMT: single instruction, multiple threads).
- A **thread block** (CUDA calls it a CTA) is a group of warps that runs on *one* SM and can share that SM's **shared memory**.
- A **kernel** is one GPU program launch. It runs as a *grid* of thread blocks spread over all SMs.

When people say "SRAM" in the FlashAttention paper they mean the on-chip memory: registers plus shared memory. When they say "HBM" they mean off-chip GPU main memory.

### 2.2 The memory hierarchy is the whole story

| Level | Size (A100) | Bandwidth | Rough latency | Who can see it |
|---|---|---|---|---|
| Registers | 256 KB per SM | highest | ~1 cycle | one thread |
| Shared memory | up to 164 KB per SM | ~19 TB/s aggregate | ~20–30 cycles | one thread block |
| L2 cache | 40 MB | several TB/s | ~200 cycles | all SMs |
| HBM | 40–80 GB | 1.5–2.0 TB/s | ~400–600 cycles | everyone |

Latencies are rough orders of magnitude, not datasheet numbers. The 19 TB/s shared-memory figure is the FlashAttention paper's estimate.

On-chip memory is roughly **10× faster** than HBM and **thousands of times smaller** (about 20 MB of shared memory across all 108 SMs, vs. 40–80 GB of HBM). Fast kernels load data from HBM once, do as much work as possible on-chip, and write back once.

### 2.3 Roofline: are you compute-bound or memory-bound?

Every kernel has an **arithmetic intensity**:

```
arithmetic intensity (AI) = FLOPs performed / bytes moved to or from HBM
```

The GPU has two speed limits:

```
time ≥ FLOPs / peak_FLOPs        (compute limit)
time ≥ bytes / HBM_bandwidth     (memory limit)
```

On an A100 with FP16 tensor cores, `312 TFLOP/s ÷ 1.555 TB/s ≈ 201 FLOP/byte`. This is the **ridge point**:

- AI **below ~200**: the kernel is **memory-bound**. Tensor cores sit idle waiting for data. Adding FLOPs is nearly free; moving bytes is not.
- AI **above ~200**: the kernel is **compute-bound**. You are getting your money's worth.

Elementwise ops (add, scale, mask, exp, dropout) have AI around 1. Large matrix multiplies have AI in the hundreds or thousands. **Softmax is elementwise-like, so it is badly memory-bound.**

### 2.4 Tensor cores and "non-matmul FLOPs are expensive"

Tensor cores only do one thing: small matrix multiply-accumulates (for example `16×8×16` tiles), with FP16/BF16 inputs and FP32 accumulation. On A100:

- tensor-core FP16 matmul: **312 TFLOP/s**
- ordinary FP32 math (exp, max, scaling): **19.5 TFLOP/s**

That is a 16× gap. A FLOP spent in softmax is much more expensive than a FLOP spent in `QKᵀ`. FlashAttention-2 was largely about removing non-matmul FLOPs for this reason ([section 7](#7-flashattention-2-and--3-what-changed)).

### 2.5 Kernel launches and fusion

Every separate PyTorch op is usually a separate kernel. Each kernel reads its inputs from HBM and writes its outputs back to HBM. **Kernel fusion** means merging several ops into one kernel so intermediates stay in registers or shared memory. FlashAttention is, at heart, a kernel fusion of `matmul → scale → mask → softmax → dropout → matmul`. The streaming softmax is what makes that fusion possible.

---

## 3. How normal attention actually runs on a GPU

Here is the textbook PyTorch version:

```python
S = (Q @ K.transpose(-2, -1)) / math.sqrt(d)   # kernel 1: GEMM, kernel 2: scale
S = S.masked_fill(mask, float("-inf"))         # kernel 3: mask
P = torch.softmax(S, dim=-1)                   # kernel 4: softmax
P = dropout(P)                                 # kernel 5: dropout (training)
O = P @ V                                      # kernel 6: GEMM
```

Each line is a round trip through HBM:

```
          HBM                             SMs
 Q,K ──────────────────────────────▶ GEMM ──▶ S (N×N) ──▶ HBM
 S  ───────────────────────────────▶ scale ─▶ S (N×N) ──▶ HBM
 S  ───────────────────────────────▶ mask ──▶ S (N×N) ──▶ HBM
 S  ───────────────────────────────▶ softmax ▶ P (N×N) ──▶ HBM
 P  ───────────────────────────────▶ dropout ▶ P (N×N) ──▶ HBM
 P,V ──────────────────────────────▶ GEMM ──▶ O (N×d) ──▶ HBM
```

Even the minimal three-kernel version (GEMM, softmax, GEMM) moves `4N² + 4Nd` elements. For large `N` the `4N²` term dominates. The FLOPs are `4N²d`. So:

```
AI_naive ≈ 4N²d / (4N² × 2 bytes) = d / 2 FLOP/byte
```

With `d = 64` that is **~32 FLOP/byte**, far below the ~200 ridge. **Normal attention is memory-bound on the N² matrices**, no matter how fast the tensor cores are. The runnable table (section 4 of the script output):

```
       N    d | naive MB    AI t_min us
    1024   64 |      8.9    30      5.7
    4096   64 |    136.3    32     87.7
    4096  128 |    138.4    62     89.0
   16384  128 |   2164.3    64   1391.8
```

This is one head's forward pass. A real model multiplies it by batch × heads × layers.

**Training makes it worse.** Autograd saves `P` for the backward pass. That is `N²` values per head per layer held in HBM until backward runs. At `N = 8192`, FP16, 32 heads, one layer's `P` is 4 GB for a single sequence. This is why long-context training ran out of memory before FlashAttention.

### Why not just fuse the three kernels?

The obvious fix is one kernel that computes a block of `S`, softmaxes it, and multiplies by `V`, all on-chip. The problem is that **softmax needs the whole row before it can normalize**:

```
softmax(s)ᵢ = exp(sᵢ − max(s)) / Σⱼ exp(sⱼ − max(s))
                     ▲                    ▲
          needs the max of the     needs the sum over
          ENTIRE row               the ENTIRE row
```

A thread block handling queries `i..i+127` sees keys in chunks. It cannot know the row's max or sum until it has seen every key. Streaming softmax removes that dependency.

---

## 4. Streaming softmax

### 4.1 Why subtract the max at all?

`exp(12)` already overflows FP16 (max ≈ 65504), and `exp(89)` overflows FP32. Subtracting the max makes every exponent ≤ 0, so every `exp` is in `(0, 1]`. The result is mathematically identical, because the `exp(−m)` factor cancels between numerator and denominator:

```
exp(sᵢ − m) / Σⱼ exp(sⱼ − m)  =  exp(sᵢ)·e⁻ᵐ / (Σⱼ exp(sⱼ)·e⁻ᵐ)  =  exp(sᵢ) / Σⱼ exp(sⱼ)
```

That cancellation is the key. **Any** reference value `m` works, as long as numerator and denominator use the same one.

### 4.2 Three passes → two passes → one pass

**Safe softmax, three passes over the row:**

```
pass 1:  m = max_j s_j
pass 2:  l = Σ_j exp(s_j − m)
pass 3:  p_j = exp(s_j − m) / l
```

**Online softmax, two passes** ([Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867)). Compute `m` and `l` together in one pass. When a new block raises the max, fix the old sum:

```
m_new = max(m_old, max(block))
l_new = l_old · exp(m_old − m_new) + Σ_{j∈block} exp(s_j − m_new)
                ▲
                re-express the old sum relative to the new max
```

The correction factor `α = exp(m_old − m_new)` is always ≤ 1. It shrinks the old terms by exactly the amount they were too big.

**Attention needs only one pass.** We never need the normalized `P` itself, only `O = PV`. `O` is also a sum over keys, so it can carry the same correction:

```
for each K/V block j:
    s      = q · K_jᵀ / √d                  # this block's scores
    m_new  = max(m, max(s))
    α      = exp(m − m_new)                  # fix-up for everything seen so far
    p      = exp(s − m_new)
    l      = α · l + Σ p
    o      = α · o + p · V_j                 # unnormalized output
    m      = m_new
O = o / l                                    # normalize ONCE at the end
```

The only state per query row is one number `m`, one number `l`, and one `d`-vector `o`. That fits in registers.

### 4.3 Worked example

Scores `[3, 1, 5, 2]`, processed in blocks of two. From the script:

```
block [3.0, 1.0]: m -inf -> 3.0, alpha=0.0000, l=1.1353
block [5.0, 2.0]: m 3.0 -> 5.0, alpha=0.1353, l=1.2034
direct sum exp(x - 5) = 1.2034  (matches l)
softmax 3-pass : [0.1125 0.0152 0.831  0.0414]
softmax online : [0.1125 0.0152 0.831  0.0414]
```

Step by step:

1. Block 1: `m = 3`, `l = e⁰ + e⁻² = 1 + 0.1353 = 1.1353`.
2. Block 2 has a bigger max, 5. The old sum was measured against 3, so multiply it by `α = e^(3−5) = 0.1353`. That gives `0.1536`. Add the new terms `e⁰ + e⁻³ = 1.0498`. Result: `l = 1.2034`.
3. Direct check: `e⁻² + e⁻⁴ + e⁰ + e⁻³ = 1.2034`. ✓

### 4.4 The classic bug: forgetting to rescale

If you drop `α` (keep it at 1), nothing crashes. The output is just wrong. The script puts the largest score in a late block:

```
with rescale   : max |diff| vs naive = 1.78e-15
WITHOUT rescale: max |diff| vs naive = 2.15e+00  <- silent bug
```

This is why the chapter's tests compare tiled attention against naive attention across many block sizes. See [`tests/test_unit.py`](tests/test_unit.py).

### 4.5 Softmax states can be merged in any order

A partial result over some set of keys is a triple `(m, l, o)`. Two partial results over *disjoint* key sets merge like this:

```
m = max(m_a, m_b)
l = l_a·exp(m_a − m) + l_b·exp(m_b − m)
o = o_a·exp(m_a − m) + o_b·exp(m_b − m)
```

This merge is associative and commutative (tested in `test_merge_is_associative`). That means:

- the K/V blocks can be processed **in any order**
- different SMs can process **different key ranges in parallel**, then merge

The script splits keys into 1, 4, and 16 chunks and merges them forward and in reverse. All results match naive attention to about `1e-16`. This property is the basis of Flash-Decoding ([section 8](#8-inference-decoding-and-flash-decoding)).

---

## 5. FlashAttention: the tiled forward pass

### 5.1 The picture

```
                     K/V blocks (Bc keys each) →
                 j=0    j=1    j=2    j=3
               ┌──────┬──────┬──────┬──────┐
 Q block i=0   │  ▣   │  ·   │  ·   │  ·   │   ▣ = diagonal block: compute + mask
 (Br queries)  ├──────┼──────┼──────┼──────┤   ■ = full block: compute
 Q block i=1   │  ■   │  ▣   │  ·   │  ·   │   · = entirely in the future:
               ├──────┼──────┼──────┼──────┤       SKIPPED under causal masking
 Q block i=2   │  ■   │  ■   │  ▣   │  ·   │
               ├──────┼──────┼──────┼──────┤
 Q block i=3   │  ■   │  ■   │  ■   │  ▣   │
               └──────┴──────┴──────┴──────┘
       Each Q block is one thread block. It walks left to right,
       keeping (m, l, o) for its Br rows on-chip. S is never written to HBM.
```

### 5.2 The algorithm (FlashAttention-2 loop order)

```
parallel for each Q block i (one thread block per (batch, head, i)):
    load Q_i  (Br × d) from HBM into shared memory          # once
    m = -inf, l = 0, o = 0   (in registers)
    for each K/V block j  (skip if entirely in the future):
        load K_j, V_j (Bc × d) from HBM into shared memory
        S_ij = Q_i K_jᵀ / √d      (Br × Bc)   tensor cores, stays on-chip
        mask S_ij if on the diagonal
        m_new = max(m, rowmax(S_ij))
        P_ij  = exp(S_ij − m_new)                          # on-chip only
        α     = exp(m − m_new)
        l     = α·l + rowsum(P_ij)
        o     = α·o + P_ij V_j        (Br × d)   tensor cores
        m     = m_new
    O_i = o / l                  → write to HBM              # once
    L_i = m + log(l)             → write to HBM (N floats, for backward)
```

What lives where:

| Data | Where | Size example (Br=128, Bc=64, d=64) |
|---|---|---|
| `Q_i` | shared memory | 128×64 × 2 B = 16 KB |
| `K_j`, `V_j` | shared memory | 8 KB each |
| `S_ij`, `P_ij` | registers | 128×64 FP32 = 32 KB, spread across warps |
| `o` accumulator | registers | 128×64 FP32 = 32 KB |
| `m`, `l` | registers | 128 floats each |
| `S`, `P` (full) | **nowhere** | would be N×N |

Everything fits in one SM. Block sizes are picked for exactly this: the largest tiles that fit in shared memory and registers without spilling.

### 5.3 How much HBM traffic does this save?

Each Q block reads `Q_i` once, streams all of `K` and `V`, and writes `O_i` once.

- **Upper bound** (every K/V re-read goes to HBM): `2Nd` elements of K/V per Q block, and `N/Br` Q blocks. That gives `AI ≈ Br`, about 128 FLOP/byte.
- **Lower bound** (K/V re-reads hit L2, because many Q blocks of the same head run at the same time): K and V leave HBM once. That gives `AI ≈ N/2`, which is way above the ridge.

The FlashAttention paper's formal result: standard attention needs `Θ(Nd + N²)` HBM accesses, and FlashAttention needs `Θ(N²d²/M)`, where `M` is the on-chip memory size. For `d` = 64–128, `d²` is 4K–16K elements while `M` is on the order of 100K, so FlashAttention needs several times fewer HBM accesses.

From the script (one head, forward, FP16, A100):

```
       N    d | naive MB    AI t_min us | flash-hi MB    AI t_min us | flash-lo MB    AI t_min us
    1024   64 |      8.9    30      5.7 |         2.4   114      1.5 |         0.5   508      0.9
    4096   64 |    136.3    32     87.7 |        34.6   124     22.3 |         2.1  2032     13.8
    4096  128 |    138.4    62     89.0 |        69.2   124     44.5 |         4.2  2040     27.5
   16384  128 |   2164.3    64   1391.8 |      1082.2   127    695.9 |        16.8  8160    440.5
```

`t_min` is the roofline lower bound, not a measured time. How to read it:

- Naive attention is stuck at `AI ≈ d/2`. It is memory-bound at every size.
- Even the pessimistic Flash model is 2–4× less traffic. The realistic case is near the lower bound, where Flash becomes **compute-bound**. From there, the only way to go faster is better tensor-core use. That is what FlashAttention-2 and -3 chase.
- **Peak memory** drops from `O(N²)` to `O(N)`. That is the part that makes long context *possible*, not just faster. The chapter's [`build.py`](build.py) measures it directly.

### 5.4 Causal masking is a free 2× on FLOPs

Blocks entirely above the diagonal are skipped: no load, no compute. Only diagonal blocks need an element-wise mask. So causal attention does about half the FLOPs of full attention. Naive attention computes the full `N × N` and then throws half away with `-inf`.

A small detail: a fully masked row inside a diagonal block gives `exp(−inf − (−inf)) = NaN`. Real kernels guard against it. The chapter's CPU reference uses `nan_to_num` for the same reason.

---

## 6. The backward pass: recompute instead of store

Normal autograd saves `P` (`N × N`) for backward. FlashAttention saves only:

- `O` (`N × d`, needed anyway)
- `L = m + log(l)` per row (`N` floats: the **logsumexp**)

In backward, it recomputes `S` block by block from `Q` and `K`, and rebuilds `P` exactly:

```
P_ij = exp(S_ij − L_i)
```

Then the gradients are:

```
dV = Pᵀ dO
dP = dO Vᵀ
D_i = rowsum(dO ∘ O)_i            # = rowsum(P ∘ dP)_i, but costs only N×d
dS = P ∘ (dP − D)                  # softmax backward
dQ = dS K / √d
dK = dSᵀ Q / √d
```

The `D` trick matters. The softmax Jacobian needs `Σ_j P_ij dP_ij` for each row. Computed directly that is an `N × N` reduction. But it equals `dO_i · O_i`, which only touches `N × d` data you already have.

The script checks this against PyTorch autograd:

```
dQ: max |ours - autograd| = 7.77e-16
dK: max |ours - autograd| = 8.88e-16
dV: max |ours - autograd| = 3.89e-16
saved for backward: naive P = 4096 floats, flash L = 64 floats
```

**Recomputation costs extra FLOPs but still wins.** Recomputing `QKᵀ` is a tensor-core matmul, which is cheap. Reading `P` back from HBM is a memory-bound transfer, which is expensive. On a GPU, trading FLOPs for bytes is usually a good deal (the roofline again). The same idea at model scale is called *activation checkpointing*.

---

## 7. FlashAttention-2 and -3: what changed

**FlashAttention-1** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) introduced the tiled, IO-aware schedule above. Its outer loop ran over K/V blocks and its inner loop over Q blocks. That meant `O` was read and written in HBM on every outer iteration, and parallelism came only from batch × heads.

**FlashAttention-2** ([Dao, 2023](https://arxiv.org/abs/2307.08691)) kept the math and changed the work partitioning:

1. **Fewer non-matmul FLOPs.** Keep `o` unnormalized and divide by `l` once at the end, instead of rescaling by `1/l` every iteration. Save only `L` (logsumexp) for backward instead of both `m` and `l`. This matters because of the 16× matmul vs. non-matmul gap ([section 2.4](#24-tensor-cores-and-non-matmul-flops-are-expensive)).
2. **Parallelize over sequence length.** Swap the loops so the outer loop is over Q blocks, each an independent thread block. With long sequences and a small batch, batch × heads alone may be fewer than 108 SMs. Splitting over Q blocks keeps every SM busy.
3. **Split Q across warps, not K.** FA1 split K/V across the 4 warps of a block. Each warp then held a partial result that had to be combined through shared memory. FA2 gives each warp its own rows of Q, so warps never need to talk to each other.

**FlashAttention-3** ([Shah et al., 2024](https://arxiv.org/abs/2407.08608)) targets Hopper (H100) hardware features:

- **Asynchrony.** The Tensor Memory Accelerator (TMA) copies tiles HBM → shared memory in the background. *Producer* warps issue loads while *consumer* warps compute. This is called warp specialization.
- **Overlap softmax with matmul.** "Ping-pong" scheduling lets one warpgroup do its `exp` work while another uses the tensor cores. `exp` runs on slow special-function units, so hiding it matters.
- **FP8** with block-wise scaling and incoherent processing (a random orthogonal transform that spreads out outliers) to limit quantization error.

The paper reports up to about 75% of H100's theoretical FP16 throughput.

The pattern across all three versions: **the math never changes, only the schedule.** Each version finds the next bottleneck (HBM traffic, then non-matmul FLOPs and parallelism, then synchronous execution) and removes it.

---

## 8. Inference: decoding and Flash-Decoding

Generation has two phases, and they behave very differently on a GPU.

- **Prefill** (processing the prompt): `N` queries against `N` keys. This is the same as training forward, and FlashAttention applies directly.
- **Decode** (one new token at a time): **1 query** against all cached keys and values (the **KV cache**). Now `Q` is `1 × d`:

```
FLOPs ≈ 4Nd         bytes ≈ 2Nd × 2 (read K and V from the cache)
AI ≈ 1 FLOP/byte    →  completely memory-bound
```

During decode, speed is set by how fast you can read the KV cache from HBM. That is why KV cache size (multi-query and grouped-query attention, KV quantization) matters so much for inference speed.

There is also a parallelism problem. With one query per head and a small batch, FA2's "one thread block per Q block" gives maybe a few dozen thread blocks. That leaves most of the 108 SMs idle.

**Flash-Decoding** ([Dao et al., 2023](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)) splits the *keys* instead. Each SM handles one chunk of the KV cache and produces a partial `(m, l, o)`. A small final kernel merges the partials using the rule from [section 4.5](#45-softmax-states-can-be-merged-in-any-order). This is "split-K" parallelism, and it is correct only because the softmax merge is associative. The script's `split_kv_attention` is a CPU version of exactly this.

**PagedAttention** (vLLM) is a related idea about *memory management*, not math. It stores the KV cache in fixed-size pages, like virtual memory, so sequences of different lengths don't waste memory.

---

## 9. Other GPU concepts worth knowing

| Concept | What it means | Why attention kernels care |
|---|---|---|
| **Occupancy** | How many warps are resident on an SM at once. | When one warp waits on memory, the scheduler switches to another. More resident warps hide more latency. |
| **Register pressure** | Each thread has a limited number of registers. If you need more, values *spill* to slow local memory. | The `o` accumulator is `Br × d` FP32. Large `d` (256) is hard because the accumulator stops fitting, so block sizes must shrink. |
| **Coalesced access** | A warp's 32 loads are fast when they hit consecutive addresses in one transaction. | Tiles are loaded row-major and contiguous. Layout matters, which is why `head_dim` is the last, contiguous dimension. |
| **Shared-memory bank conflicts** | Shared memory has 32 banks. Two threads hitting the same bank at the same time get serialized. | Kernels swizzle (permute) tile layouts so transposed reads of `K` don't conflict. |
| **Async copy / double buffering** | Start loading tile `j+1` while computing on tile `j`. `cp.async` on Ampere, TMA on Hopper. | This overlaps memory latency with compute. It is the core of FA3's speedup. |
| **Tensor-core shapes** | MMA instructions work on fixed tile shapes, such as multiples of 8 or 16. | Head dims like 64 and 128 map cleanly. Odd sizes get padded, and supported head dims are limited. |
| **Mixed precision** | FP16/BF16 inputs, FP32 accumulation. | `S`, `m`, `l`, and `o` are kept in FP32 on-chip. Only `Q`, `K`, `V`, and `O` are 16-bit in HBM. BF16 has FP32's exponent range, so it overflows less than FP16. |
| **Kernel launch overhead** | Each launch costs a few microseconds. | For small shapes (like my `T = 128` run) launch overhead and other layers dominate, so Flash's advantage is small. |
| **Special-function units** | `exp` and `log` run on separate, slower units. | Attention does `N²` exponentials. FA3 overlaps them with matmuls, and newer kernels go further to reduce `exp` cost. |

---

## 10. How this connects to my experiments

- The chapter's [`build.py`](build.py) is the CPU version of [section 5.2](#52-the-algorithm-flashattention-2-loop-order). It shows the `O(N²) → O(N)` peak-memory drop: 16× at `T = 128`.
- My [mHC + Flash experiment](EXPERIMENT_MHC_FLASH.md) calls the real fused kernel through `torch.nn.functional.scaled_dot_product_attention` with only the Flash backend allowed. The profiler confirmed `pytorch_flash::flash_fwd_kernel` ran on the A100. At context 128, the `N × N` matrix is tiny (128×128), so that run verifies the fused path works. **It is not a long-context speed or memory test.** Per [section 5.3](#53-how-much-hbm-traffic-does-this-save), Flash's advantage grows with `N`.
- My [RoPE experiment](../07_the-details-that-matter/README.md) changes `Q` and `K` *before* the dot product. FlashAttention changes how that dot product is scheduled. They are independent and compose. Real kernels often fuse RoPE into the Q/K load.

- My GPT's `generate()` now uses a KV cache, with an optional split-KV (Flash-Decoding) decode step that merges `(m, l, o)` partials exactly as in [section 4.5](#45-softmax-states-can-be-merged-in-any-order). The design, tests, and CPU timings are in [Chapter 5 notes](../05_your-gpt-from-a-blank-file/NOTES.md#faster-generation-kv-cache-and-split-kv-decoding).

**Natural next steps:**

1. Benchmark SDPA's `math` vs `flash` backends on an A100 for `N ∈ {512, 2k, 8k, 32k}`. Measure time and `torch.cuda.max_memory_allocated()`, and see where the curves split.
2. Write the forward kernel in [Triton](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html). Triton exposes block tiling and shared memory without raw CUDA, and its tutorial builds exactly this kernel.
3. Profile both backends with Nsight Compute and compare achieved HBM bandwidth against tensor-core utilization. That is the roofline measured, not modeled.

---

## 11. Common misconceptions

- **"FlashAttention is an approximation."** No. It is exact up to floating-point reordering. The CPU reference matches naive attention to about `1e-7` in FP32 and about `1e-15` in FP64.
- **"It reduces FLOPs."** No. Non-causal forward FLOPs are the same, and backward does *more* FLOPs because of recomputation. It reduces **memory traffic** and **peak memory**. The causal-block skip is a separate saving that naive attention simply doesn't exploit.
- **"It makes attention O(N)."** Compute is still `O(N²d)`. Only *memory* goes from `O(N²)` to `O(N)`. Linear-attention methods change the math; FlashAttention does not.
- **"It only matters for long context."** Memory savings grow with `N`. The fusion itself, fewer kernel launches with no HBM round trips for scale, mask, and dropout, helps at moderate lengths too.

---

## 12. References

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Maxim Milakov, Natalia Gimelshein. *Online normalizer calculation for softmax.* 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- Markus N. Rabe, Charles Staats. *Self-attention Does Not Need O(n²) Memory.* 2021. [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)
- Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov. *Flash-Decoding for long-context inference.* 2023. [Stanford CRFM blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- Horace He. *Making Deep Learning Go Brrrr From First Principles.* [horace.io/brrr_intro.html](https://horace.io/brrr_intro.html). The best short intro to compute-bound vs. memory-bound vs. overhead-bound.
- NVIDIA. *A100 Tensor Core GPU Architecture* whitepaper, for the SM, memory, and tensor-core figures.
- Ramchand Kumaresan. *Under the Hood*, Chapter 8. [leanpub.com/under-the-hood](https://leanpub.com/under-the-hood)
